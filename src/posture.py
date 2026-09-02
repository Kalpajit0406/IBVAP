"""
posture.py -- skeleton-based anomaly classifier.

Two classes:
  PoseEstimator   -- thin wrapper around yolo26n-pose.pt; runs on person crops
  PoseClassifier  -- pure-Python geometry rules; no GPU, no model, fully testable

Rule 5 (`chest_aim`) is a WEAPON-READY POSTURE heuristic, not object-level
weapon detection. It reasons purely about where the wrists/elbows sit relative
to the shoulders and hips in 2D image space, so it cannot see a weapon and
cannot tell a rifle from a broomstick. It is deliberately conservative: the
geometry must describe a held, two-handed, forward, chest/chin-height grip and
must persist for several frames before the flag is raised. A real firearm
classifier (custom-trained object model) is the roadmap replacement.
"""
from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# -- COCO 17-keypoint indices --------------------------------------------------
NOSE          = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHLDR, R_SHLDR = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP     = 11, 12
L_KNEE, R_KNEE   = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Skeleton connectivity for drawing
SKELETON_EDGES = [
    (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
    (L_SHLDR, R_SHLDR),
    (L_SHLDR, L_ELBOW), (R_SHLDR, R_ELBOW),
    (L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST),
    (L_SHLDR, L_HIP),   (R_SHLDR, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE),    (R_HIP, R_KNEE),
    (L_KNEE, L_ANKLE),  (R_KNEE, R_ANKLE),
]


@dataclass
class PostureFlags:
    """One instance per tracked person per detection frame."""
    track_id:  int
    keypoints: Optional[np.ndarray] = None  # shape (17, 3): x, y, visibility
    lying:     bool = False
    crouching: bool = False
    vigorous:  bool = False   # head oscillating -- vigorous look-around
    arms_up:   bool = False   # wrists above shoulders (surrender / overhead raise)
    chest_aim: bool = False   # HELD two-handed forward chest/chin-height grip (weapon-ready posture)

    @property
    def any_anomaly(self) -> bool:
        return self.lying or self.crouching or self.vigorous or self.arms_up or self.chest_aim

    @property
    def label(self) -> str:
        tags = []
        if self.lying:      tags.append("LYING")
        if self.crouching:  tags.append("CROUCH")
        if self.vigorous:   tags.append("SCAN")
        if self.arms_up:    tags.append("ARMS-UP")
        if self.chest_aim:  tags.append("AIM")
        return " ".join(tags) if tags else ""

    @property
    def risk_weight(self) -> float:
        """0.0-1.0 anomaly severity fed into risk_engine._behaviour()."""
        if self.lying:      return 1.0
        if self.chest_aim:  return 0.95   # active weapon aim — highest non-lying threat
        if self.arms_up:    return 0.9
        if self.crouching:  return 0.7
        if self.vigorous:   return 0.5
        return 0.0



class PoseEstimator:
    """
    Loads yolo26n-pose.pt once and runs inference on person bounding-box crops.

    Running on crops rather than full frames is dramatically cheaper:
    a 160x320 px person crop at imgsz=256 takes ~2-3 ms vs ~14 ms for a
    full 640x640 frame through the pose head.
    """

    def __init__(self, weights: str = "yolo26n-pose.pt",
                 device: str = "0",
                 half: bool = True,
                 kp_conf: float = 0.5) -> None:
        logger.info("Loading pose model: %s", weights)
        self._model   = YOLO(weights)
        self._device  = device
        self._kp_conf = kp_conf
        # quantize=16 for FP16; half= is deprecated in ultralytics 8.4.x
        self._q = 16 if half else None
        # One warm-up pass so cuDNN is ready before the first real frame
        dummy = np.zeros((160, 80, 3), dtype=np.uint8)
        self._model.predict(dummy, imgsz=256, quantize=self._q,
                            device=self._device, verbose=False)
        logger.info("Pose model ready  device=%s  kp_conf=%.2f", device, kp_conf)

    def run(self, frame: np.ndarray,
            person_bboxes: list[tuple[int, int, int, int, int]]
            ) -> dict[int, np.ndarray]:
        """
        Run pose estimation on person crops extracted from frame.

        Args:
            frame:         Full camera frame (BGR).
            person_bboxes: list of (x1, y1, x2, y2, track_id) for each person.

        Returns:
            dict mapping track_id to keypoints ndarray of shape (17, 3):
            columns are x, y, visibility.  Joints below kp_conf are zeroed.
        """
        if not person_bboxes:
            return {}

        h, w = frame.shape[:2]
        crops: list[np.ndarray]        = []
        meta:  list[tuple[int, int, int]] = []   # (track_id, crop_x1, crop_y1)

        for x1, y1, x2, y2, tid in person_bboxes:
            cx1 = max(0, x1);  cy1 = max(0, y1)
            cx2 = min(w, x2);  cy2 = min(h, y2)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            crops.append(frame[cy1:cy2, cx1:cx2])
            meta.append((tid, cx1, cy1))

        if not crops:
            return {}

        results = self._model.predict(
            crops, imgsz=256, quantize=self._q,
            device=self._device, verbose=False,
        )

        out: dict[int, np.ndarray] = {}
        for (tid, ox, oy), res in zip(meta, results):
            if res.keypoints is None or res.keypoints.xy is None:
                continue
            xy   = res.keypoints.xy.cpu().numpy()    # (n_dets, 17, 2)
            conf = res.keypoints.conf.cpu().numpy()  # (n_dets, 17)
            if len(xy) == 0:
                continue
            # Pick the detection with the highest mean keypoint confidence
            best     = int(conf.max(axis=1).argmax())
            kp       = np.zeros((17, 3), dtype=np.float32)
            kp[:, 0] = xy[best, :, 0] + ox   # restore to full-frame coords
            kp[:, 1] = xy[best, :, 1] + oy
            kp[:, 2] = conf[best]
            kp[kp[:, 2] < self._kp_conf, :] = 0.0   # zero low-confidence joints
            out[tid] = kp

        return out


class PoseClassifier:
    """
    Stateful, per-track geometry rule engine.  Needs per-frame history for the
    vigorous-scan rule; everything else is single-frame geometry.
    All thresholds tunable via config.yaml pose: block.
    """

    LYING_RATIO_THRESH = 0.35   # skeleton h/w ratio below this -> lying
    CROUCH_HIP_RATIO   = 0.60   # body height < 60% of standing baseline -> crouching
    KNEE_HIP_DELTA_PX  = 30     # knee.y within this of hip.y -> crouching
    SCAN_WINDOW        = 10     # frames of nose_x history per track
    SCAN_STD_THRESH    = 12.0   # px std-dev of nose_x to flag vigorous scan
    ARMS_UP_MARGIN_PX  = 20     # px wrist must be above shoulder line

    # -- Rule 5: weapon-ready posture. All ratios are of shoulder width or of
    #    body height (shoulder->hip). Tuned to CATCH a real two-handed gun hold
    #    (high-ready to low-ready, aimed at or across the camera) while still
    #    rejecting hands-at-the-belt, one-hand-up, and folded arms.
    AIM_BAND_ABOVE       = 0.60  # wrist band top: this * body_h ABOVE shoulder (~forehead)
    AIM_BAND_BELOW       = 0.55  # wrist band bottom: this * body_h BELOW shoulder (~navel)
    AIM_WRIST_SEP_RATIO  = 0.85  # wrists within this * shoulder-width of each other
    AIM_WRIST_LEVEL_RATIO = 0.60 # |left wrist.y - right wrist.y| within this * shoulder-width
    AIM_CENTER_MARGIN    = 0.55  # both wrists within this * sh-width of the shoulder span
    AIM_WRIST_ABOVE_HIP  = 0.10  # each wrist at least this * body_h above the hip (arms off the thighs)
    AIM_HOLD_FRAMES      = 2     # raw geometry must persist this many detection frames before firing

    def __init__(self, config: dict | None = None) -> None:
        p = (config or {}).get("pose", {})
        self.LYING_RATIO_THRESH  = float(p.get("lying_ratio",       self.LYING_RATIO_THRESH))
        self.CROUCH_HIP_RATIO    = float(p.get("crouch_ratio",      self.CROUCH_HIP_RATIO))
        self.KNEE_HIP_DELTA_PX   = float(p.get("knee_hip_delta",    self.KNEE_HIP_DELTA_PX))
        self.SCAN_WINDOW         = int  (p.get("scan_window",        self.SCAN_WINDOW))
        self.SCAN_STD_THRESH     = float(p.get("scan_std",           self.SCAN_STD_THRESH))
        self.ARMS_UP_MARGIN_PX   = float(p.get("arms_up_margin",    self.ARMS_UP_MARGIN_PX))
        self.AIM_BAND_ABOVE       = float(p.get("aim_band_above",     self.AIM_BAND_ABOVE))
        self.AIM_BAND_BELOW       = float(p.get("aim_band_below",     self.AIM_BAND_BELOW))
        self.AIM_WRIST_SEP_RATIO  = float(p.get("aim_wrist_sep",      self.AIM_WRIST_SEP_RATIO))
        self.AIM_WRIST_LEVEL_RATIO = float(p.get("aim_wrist_level",   self.AIM_WRIST_LEVEL_RATIO))
        self.AIM_CENTER_MARGIN    = float(p.get("aim_center_margin",  self.AIM_CENTER_MARGIN))
        self.AIM_WRIST_ABOVE_HIP  = float(p.get("aim_wrist_above_hip", self.AIM_WRIST_ABOVE_HIP))
        self.AIM_HOLD_FRAMES      = int  (p.get("aim_hold_frames",    self.AIM_HOLD_FRAMES))
        self._nose_history:    dict[int, collections.deque] = {}
        self._height_baseline: dict[int, float]             = {}
        self._aim_streak:      dict[int, int]               = {}   # consecutive raw-aim frames per track


    def classify(self, track_id: int, kp: np.ndarray) -> PostureFlags:
        """
        kp: (17, 3) array -- x, y, visibility.  visibility == 0 means absent.
        Returns PostureFlags for this person this frame.
        """
        flags = PostureFlags(track_id=track_id, keypoints=kp)

        def pt(idx: int) -> Optional[tuple[float, float]]:
            return (float(kp[idx, 0]), float(kp[idx, 1])) if kp[idx, 2] > 0 else None

        def mid(a: int, b: int) -> Optional[tuple[float, float]]:
            pa, pb = pt(a), pt(b)
            if pa and pb:
                return ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
            return pa or pb

        shoulders = mid(L_SHLDR, R_SHLDR)
        hips      = mid(L_HIP,   R_HIP)
        knees     = mid(L_KNEE,  R_KNEE)
        nose      = pt(NOSE)
        l_wrist   = pt(L_WRIST);  r_wrist = pt(R_WRIST)
        l_shldr   = pt(L_SHLDR);  r_shldr = pt(R_SHLDR)
        visible   = kp[kp[:, 2] > 0]

        # -- Rule 1: Lying on ground -------------------------------------------
        # Skeleton is much wider than tall when lying horizontally.
        if len(visible) >= 4:
            kp_h = float(visible[:, 1].max() - visible[:, 1].min())
            kp_w = float(visible[:, 0].max() - visible[:, 0].min())
            if kp_w > 10:
                flags.lying = (kp_h / kp_w) < self.LYING_RATIO_THRESH

        # -- Rule 2: Crouching -------------------------------------------------
        # Two signals: (a) knees near same y as hips, (b) compressed body height.
        if not flags.lying and shoulders and hips and knees:
            sh_y   = shoulders[1]
            hip_y  = hips[1]
            knee_y = knees[1]
            body_h = abs(hip_y - sh_y)

            # Update standing-height baseline when clearly upright
            if body_h > 30 and knee_y > hip_y + 20:
                self._height_baseline[track_id] = body_h

            baseline        = self._height_baseline.get(track_id)
            knee_near_hip   = abs(knee_y - hip_y) < self.KNEE_HIP_DELTA_PX
            height_reduced  = baseline is not None and body_h < baseline * self.CROUCH_HIP_RATIO
            flags.crouching = knee_near_hip or height_reduced

        # -- Rule 3: Vigorous head scan ----------------------------------------
        # High std-dev of nose_x over a sliding window = rapid head panning.
        if nose:
            hist = self._nose_history.setdefault(
                track_id, collections.deque(maxlen=self.SCAN_WINDOW))
            hist.append(nose[0])
            if len(hist) >= self.SCAN_WINDOW // 2:
                flags.vigorous = float(np.std(hist)) > self.SCAN_STD_THRESH

        # -- Rule 4: Arms raised / surrender posture ---------------------------
        # In image coords y increases downward; wrist above shoulder = lower y.
        arms_up = 0
        if l_wrist and l_shldr and l_wrist[1] < l_shldr[1] - self.ARMS_UP_MARGIN_PX:
            arms_up += 1
        if r_wrist and r_shldr and r_wrist[1] < r_shldr[1] - self.ARMS_UP_MARGIN_PX:
            arms_up += 1
        flags.arms_up = arms_up >= 1

        # -- Rule 5: Weapon-ready posture (2D-skeleton heuristic) -------------
        # A two-handed gun hold: both hands on one object, held out from the
        # body somewhere between forehead and navel. Clauses:
        #   in_band       wrists between ~forehead and ~navel height
        #   wrists_level   hands roughly the same height (two on one object)
        #   wrists_close   hands within ~0.85 shoulder-width of each other
        #   wrists_centred hands in front of the torso (rejects folded arms,
        #                  where a wrist sits out past the far shoulder)
        #   off_thighs     arms raised off the legs
        #   arms_engaged   at least one elbow up near torso height, i.e. the
        #                  arm is bent/forward, not hanging at the side
        raw_aim = False
        l_elb = pt(L_ELBOW);  r_elb = pt(R_ELBOW)
        if l_wrist and r_wrist and l_shldr and r_shldr:
            sh_y     = (l_shldr[1] + r_shldr[1]) / 2
            sh_xmin  = min(l_shldr[0], r_shldr[0])
            sh_xmax  = max(l_shldr[0], r_shldr[0])
            sh_width = max(sh_xmax - sh_xmin, 20)
            hip_y    = hips[1] if hips else sh_y + sh_width * 2
            body_h   = max(abs(hip_y - sh_y), 30)

            band_top   = sh_y - body_h * self.AIM_BAND_ABOVE
            band_bot   = sh_y + body_h * self.AIM_BAND_BELOW
            wrist_avg_y = (l_wrist[1] + r_wrist[1]) / 2
            m = sh_width * self.AIM_CENTER_MARGIN

            in_band       = band_top <= wrist_avg_y <= band_bot
            wrists_level  = abs(l_wrist[1] - r_wrist[1]) < sh_width * self.AIM_WRIST_LEVEL_RATIO
            wrists_close  = abs(l_wrist[0] - r_wrist[0]) < sh_width * self.AIM_WRIST_SEP_RATIO
            wrists_centred = (sh_xmin - m <= l_wrist[0] <= sh_xmax + m and
                              sh_xmin - m <= r_wrist[0] <= sh_xmax + m)
            off_thighs    = (l_wrist[1] < hip_y - body_h * self.AIM_WRIST_ABOVE_HIP and
                             r_wrist[1] < hip_y - body_h * self.AIM_WRIST_ABOVE_HIP)

            elbows = [e for e in (l_elb, r_elb) if e is not None]
            if elbows:
                # engaged = elbow no lower than mid-torso (bent/forward arm).
                arms_engaged = any(e[1] <= sh_y + body_h * 0.55 for e in elbows)
            else:
                arms_engaged = True                            # no elbow data — don't block on it

            raw_aim = (in_band and wrists_level and wrists_close
                       and wrists_centred and off_thighs and arms_engaged)

        streak = self._aim_streak.get(track_id, 0) + 1 if raw_aim else 0
        self._aim_streak[track_id] = streak
        flags.chest_aim = streak >= self.AIM_HOLD_FRAMES

        return flags


    def flush_track(self, track_id: int) -> None:
        """Remove stale per-track state when a ByteTrack ID is lost."""
        self._nose_history.pop(track_id, None)
        self._height_baseline.pop(track_id, None)
        self._aim_streak.pop(track_id, None)


# -- Skeleton overlay drawing --------------------------------------------------
def draw_skeleton(canvas: np.ndarray, flags: PostureFlags,
                  dim: bool = False) -> np.ndarray:
    """Draw 17-point skeleton + anomaly label onto canvas in-place."""
    kp = flags.keypoints
    if kp is None:
        return canvas

    # Red for any anomaly; cyan for normal; dimmed on carried (non-infer) frames
    colour: tuple[int, int, int] = (0, 0, 220) if flags.any_anomaly else (220, 200, 0)
    if dim:
        colour = tuple(int(c * 0.6) for c in colour)

    # Joints
    for i in range(17):
        if kp[i, 2] > 0:
            cv2.circle(canvas, (int(kp[i, 0]), int(kp[i, 1])), 4, colour, -1)

    # Limb edges
    for a, b in SKELETON_EDGES:
        if kp[a, 2] > 0 and kp[b, 2] > 0:
            cv2.line(canvas,
                     (int(kp[a, 0]), int(kp[a, 1])),
                     (int(kp[b, 0]), int(kp[b, 1])),
                     colour, 2, cv2.LINE_AA)

    # Anomaly text label above the person
    label = flags.label
    if label:
        visible = kp[kp[:, 2] > 0]
        if len(visible):
            tx = int(visible[:, 0].min())
            ty = max(int(visible[:, 1].min()) - 8, 14)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas,
                          (tx - 2, ty - th - 4), (tx + tw + 4, ty + 2),
                          (0, 0, 0), -1)
            cv2.putText(canvas, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 255), 1, cv2.LINE_AA)

    return canvas
