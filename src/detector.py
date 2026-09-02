"""
Batched multi-stream detector.

Three techniques keep GPU cost sub-linear in the number of cameras:

  1. Batched inference — frames from every camera due for a detection pass are
     stacked into one forward pass, the way DeepStream's nvstreammux does. One
     batched call on N frames costs far less than N single-frame calls, because
     a single 640x640 image leaves most of the GPU's SMs idle.

  2. Motion gating — a camera whose scene has not changed never enters the
     batch at all (see motion_gate.py).

  3. Adaptive skipping with full-rate tracking — detection runs at a target
     rate below the stream frame rate, while tracks are carried forward on
     every frame by linear extrapolation, so boxes still move smoothly on the
     monitor between detector passes.
"""
from __future__ import annotations

import copy
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from .motion_gate import MotionGate
from .posture import PoseEstimator, PoseClassifier, PostureFlags, draw_skeleton

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=".*ByteTrack.*", category=FutureWarning)

_PERSON = 0
_VEHICLES = {2, 3, 5, 7}          # car, motorcycle, bus, truck


@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    confidence: float
    is_person: bool
    is_vehicle: bool
    velocity: tuple[float, float] = (0.0, 0.0)   # px/frame, for extrapolation
    posture: Optional[PostureFlags] = None        # set by pose pass; None on non-persons


@dataclass
class StreamResult:
    cam_id: int
    timestamp: float
    frame: np.ndarray
    detections: list[Detection]
    person_count: int
    vehicle_count: int
    annotated_frame: Optional[np.ndarray] = None
    inferred: bool = False        # True when the detector actually ran
    gated: bool = False           # True when motion gating skipped it


@dataclass
class _CamState:
    """State kept across frames for one camera stream."""
    tracker: sv.ByteTrack
    gate: MotionGate
    frames: int = 0
    last_detect_tick: int = -999
    last_gate_pass_tick: int = -999
    detections: list[Detection] = field(default_factory=list)
    prev_centres: dict[int, tuple[float, float]] = field(default_factory=dict)
    person_count: int = 0
    vehicle_count: int = 0
    infer_count: int = 0
    gated_count: int = 0



@dataclass
class PipelineStats:
    frames_in: int = 0
    detector_passes: int = 0      # batched calls issued
    frames_inferred: int = 0      # frames that went through the model
    frames_gated: int = 0         # skipped by motion gate
    frames_rate_skipped: int = 0  # skipped by the detection-rate limiter
    batch_total: int = 0          # sum of batch sizes, for the mean
    last_batch: int = 0
    padded_frames: int = 0        # blanks added to reach a fixed batch shape
    infer_ms: float = 0.0

    @property
    def mean_batch(self) -> float:
        if not self.detector_passes:
            return 0.0
        return self.batch_total / self.detector_passes

    @property
    def gpu_saving_pct(self) -> float:
        """Share of incoming frames the GPU never had to look at."""
        if not self.frames_in:
            return 0.0
        return 100.0 * (self.frames_gated + self.frames_rate_skipped) / self.frames_in


class Detector:
    """One YOLO model shared by every camera; per-camera tracker and motion gate."""

    @classmethod
    def from_profile(cls, config: dict, profile: str, num_cameras: int = 0) -> "Detector":
        """Build a Detector using a named model_profiles entry."""
        profiles = config.get("model_profiles", {})
        if profile not in profiles:
            raise ValueError(f"Unknown profile {profile!r}. Available: {list(profiles)}")
        p = profiles[profile]
        patched = copy.deepcopy(config)
        patched.setdefault("model", {})
        patched["model"]["weights"] = p["det_weights"]
        patched["model"]["max_batch"] = p.get("max_batch", patched["model"].get("max_batch", 8))
        patched.setdefault("pose", {})
        patched["pose"]["weights"] = p["pose_weights"]
        return cls(patched, num_cameras=num_cameras)

    def __init__(self, config: dict, num_cameras: int = 0) -> None:
        m = config["model"]
        self._weights = str(m["weights"])
        # A TensorRT .engine (or .onnx) is already bound to its device and only
        # supports predict/val — .to(), .train() etc. raise. Track this so the
        # PyTorch-only calls below are skipped.
        self._is_engine = self._weights.lower().endswith((".engine", ".onnx", ".plan"))
        self._model = YOLO(self._weights)
        self._conf = m.get("confidence", 0.40)
        self._iou = m.get("iou", 0.50)
        self._imgsz = m.get("image_size", 640)
        self._half = bool(m.get("half", False))
        self._max_batch = int(m.get("max_batch", 16))

        # Detection rate: how many detector passes per second per camera.
        # Streams run at 24-25 fps; detecting at 8 is plenty when tracking
        # carries the boxes between passes.
        self._detect_fps = float(m.get("detect_fps", 8.0))
        self._stream_fps = float(m.get("stream_fps", 24.0))
        self._detect_every = max(1, round(self._stream_fps / max(self._detect_fps, 0.1)))

        self._motion_gating = bool(m.get("motion_gating", True))
        gate_cfg = m.get("motion", {}) or {}
        # force_every / hold_frames count gate calls, not raw frames: the gate
        # only sees frames that already cleared the rate cap.
        self._gate_kwargs = {
            "pixel_threshold": gate_cfg.get("pixel_threshold", 18),
            "area_threshold": gate_cfg.get("area_threshold", 0.002),
            "force_every": gate_cfg.get("force_every", max(1, int(self._detect_fps))),
            "hold_frames": gate_cfg.get("hold_frames", 3),
        }

        # Fixed batch sizes we are willing to submit; anything between is padded
        # up to the next one. Set before the device warm-up, which primes
        # exactly these shapes. See _bucket for why this matters so much.
        #
        # A TensorRT engine is built with ONE fixed input shape
        # (max_batch, 3, imgsz, imgsz), so there is exactly one legal bucket:
        # always pad to max_batch. A PyTorch model tolerates any shape, so it
        # gets the finer ladder to avoid wasting compute on small batches.
        if self._is_engine:
            self._buckets = [self._max_batch]
        else:
            self._buckets = [b for b in (1, 2, 4, 8, 16, 32) if b <= self._max_batch]
            if not self._buckets:
                self._buckets = [self._max_batch]
            elif self._buckets[-1] < self._max_batch:
                self._buckets.append(self._max_batch)
        self._pad_frame: Optional[np.ndarray] = None

        self._device = self._resolve_device(str(m.get("device", "0")))

        self._cams: dict[int, _CamState] = {}
        self._box_ann = sv.BoxAnnotator(thickness=2)
        self._lbl_ann = sv.LabelAnnotator(text_scale=0.45, text_thickness=1)
        self.stats = PipelineStats()
        # One tick per batch, shared by every camera. Scheduling detection on a
        # global clock rather than each camera's own frame count is what makes
        # cameras come due together — and a batch only pays if its members
        # arrive in the same pass.
        self._tick = 0

        # ── Pose estimation (off when pose.enabled is false) ──────────────────
        pose_cfg = config.get("pose", {})
        self._pose_enabled = bool(pose_cfg.get("enabled", True))
        if self._pose_enabled:
            self._pose = PoseEstimator(
                weights=pose_cfg.get("weights", "yolo26n-pose.pt"),
                device=self._device,
                half=self._half,
                kp_conf=float(pose_cfg.get("kp_conf", 0.5)),
            )
            self._classifier = PoseClassifier(config)
        else:
            self._pose       = None
            self._classifier = None
            logger.info("Pose estimation disabled (pose.enabled: false)")

        logger.info(
            "Detector ready: %s | device=%s | imgsz=%d | conf=%.2f | fp16=%s | "
            "detect every %d frames (%.0f fps of %.0f) | motion gating=%s | "
            "batch buckets %s",
            m["weights"], self._device, self._imgsz, self._conf, self._half,
            self._detect_every, self._detect_fps, self._stream_fps,
            self._motion_gating, self._buckets,
        )

    # ── Setup ────────────────────────────────────────────────────────────────
    def _resolve_device(self, requested: str) -> str:
        if requested != "cpu" and not torch.cuda.is_available():
            logger.warning(
                "CUDA unavailable (torch=%s) — falling back to CPU. Install the CUDA "
                "build: pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu126", torch.__version__)
            return "cpu"

        if requested != "cpu":
            logger.info("CUDA active: %s | torch=%s | VRAM=%.1f GB | backend=%s",
                        torch.cuda.get_device_name(0), torch.__version__,
                        torch.cuda.get_device_properties(0).total_memory / 1e9,
                        "TensorRT engine" if self._is_engine else "PyTorch")
            if not self._is_engine:
                # PyTorch model: move it to the GPU. An engine is already there
                # and .to() would raise.
                self._model.to(f"cuda:{requested}" if requested.isdigit() else requested)
            self._warm_up(requested)
        return requested

    def _warm_up(self, device: str) -> None:
        """
        Prime every batch shape the pipeline will actually submit.

        cuDNN autotunes kernels per input shape on first use, and that tuning
        costs far more than a steady-state pass. Motion gating makes the batch
        size vary from frame to frame, so warming only one shape leaves the
        rest to be tuned mid-flight — which shows up as a slow, jittery first
        few seconds on the monitor.
        """
        t0 = time.perf_counter()
        blank = np.zeros((self._imgsz, self._imgsz, 3), np.uint8)
        kw = dict(device=device, imgsz=self._imgsz, verbose=False)
        if not self._is_engine:
            kw["quantize"] = 16 if self._half else None
        # Only the bucket sizes are ever submitted; prime each a few times so
        # TRT/cuDNN lazy kernel init is done before the first real frame.
        for n in self._buckets:
            for _ in range(5 if self._is_engine else 1):
                self._model.predict([blank] * n, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("GPU warm-up complete in %.0f ms (batch shapes %s)",
                    (time.perf_counter() - t0) * 1000,
                    ", ".join(str(b) for b in self._buckets))

    @property
    def device(self) -> str:
        return self._device

    def _cam(self, cam_id: int) -> _CamState:
        st = self._cams.get(cam_id)
        if st is None:
            st = _CamState(tracker=sv.ByteTrack(),
                           gate=MotionGate(**self._gate_kwargs))
            self._cams[cam_id] = st
            logger.info("CAM-%02d registered with the pipeline", cam_id)
        return st

    # ── Main entry point ─────────────────────────────────────────────────────
    def process_batch(
        self, items: list[tuple[int, float, np.ndarray]]
    ) -> list[StreamResult]:
        """
        Take one frame from each active camera and return a result for each.

        Frames that clear both the rate limiter and the motion gate are stacked
        into a single batched forward pass; the rest are carried forward by
        track extrapolation, which costs no GPU at all.
        """
        if not items:
            return []

        self.stats.frames_in += len(items)
        self._tick += 1

        to_infer: list[tuple[int, float, np.ndarray]] = []
        to_carry: list[tuple[int, float, np.ndarray]] = []

        for cam_id, ts, frame in items:
            st = self._cam(cam_id)
            st.frames += 1

            # 1. Detection-rate limiter, on the shared tick so every camera
            #    becomes due in the same pass and the batch stays full.
            if self._tick - st.last_detect_tick < self._detect_every:
                self.stats.frames_rate_skipped += 1
                to_carry.append((cam_id, ts, frame))
                continue

            # 2. Motion gate — sub-millisecond, still far cheaper than the GPU.
            if self._motion_gating and not st.gate.should_detect(frame):
                self.stats.frames_gated += 1
                st.gated_count += 1
                # Advance the schedule as though a pass had happened, so a
                # gated camera waits a full interval instead of retrying on the
                # next frame and drifting out of phase with the others.
                st.last_detect_tick = self._tick
                to_carry.append((cam_id, ts, frame))
                continue

            # Only real motion (not the keep-alive sweep) marks a camera "live".
            if not self._motion_gating or st.gate.last_pass_was_motion:
                st.last_gate_pass_tick = self._tick
            to_infer.append((cam_id, ts, frame))

        results: list[StreamResult] = []

        # 3. One batched forward pass for every camera that needs one.
        for start in range(0, len(to_infer), self._max_batch):
            chunk = to_infer[start:start + self._max_batch]
            results.extend(self._infer_batch(chunk))

        for cam_id, ts, frame in to_carry:
            results.append(self._carry_forward(cam_id, ts, frame))

        results.sort(key=lambda r: r.cam_id)
        return results

    # ── Batched inference ────────────────────────────────────────────────────
    def _bucket(self, n: int) -> int:
        """Round a batch up to the next size we are willing to submit.

        Changing the batch shape is expensive: Ultralytics reconfigures its
        predictor and cuDNN re-tunes kernels, and measured on an RTX 3050 a
        run of varying batches averaged 304 ms per pass where a constant
        batch of 12 took 46 ms. Motion gating makes the natural batch size
        jitter with however many cameras happen to be moving, so we snap it
        to a handful of fixed sizes and pad the difference with a blank
        frame. The padding is wasted GPU work, but far cheaper than the
        reconfiguration it avoids.
        """
        for b in self._buckets:
            if n <= b:
                return b
        return self._buckets[-1]

    def _infer_batch(self, chunk: list[tuple[int, float, np.ndarray]]) -> list[StreamResult]:
        frames = [f for _, _, f in chunk]
        real = len(frames)

        target = self._bucket(real)
        if target > real:
            if self._pad_frame is None or self._pad_frame.shape != frames[0].shape:
                self._pad_frame = np.zeros_like(frames[0])
            frames = frames + [self._pad_frame] * (target - real)

        # An engine's precision is baked in at build time; passing quantize= to
        # it makes Ultralytics do extra per-call work and adds latency jitter
        # (measured: p90 39 ms -> 31 ms, max 66 ms -> 33 ms at batch 8).
        predict_kw = dict(conf=self._conf, iou=self._iou, device=self._device,
                          imgsz=self._imgsz, verbose=False)
        if not self._is_engine:
            predict_kw["quantize"] = 16 if self._half else None

        t0 = time.perf_counter()
        outputs = self._model.predict(frames, **predict_kw)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.stats.detector_passes += 1
        self.stats.frames_inferred += real
        self.stats.padded_frames += target - real
        self.stats.batch_total += real
        self.stats.last_batch = real
        self.stats.infer_ms = elapsed_ms / max(real, 1)

        results = []
        for (cam_id, ts, frame), out in zip(chunk, outputs[:real]):
            results.append(self._finish_detection(cam_id, ts, frame, out))
        return results

    def _finish_detection(self, cam_id: int, ts: float,
                          frame: np.ndarray, out) -> StreamResult:
        st = self._cam(cam_id)
        st.last_detect_tick = self._tick
        st.infer_count += 1

        sv_det = sv.Detections.from_ultralytics(out)
        sv_det = st.tracker.update_with_detections(sv_det)

        detections: list[Detection] = []
        centres: dict[int, tuple[float, float]] = {}

        for i in range(len(sv_det)):
            cid = int(sv_det.class_id[i])
            tid = int(sv_det.tracker_id[i]) if sv_det.tracker_id is not None else -1
            x1, y1, x2, y2 = sv_det.xyxy[i].astype(int)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            # Velocity from the previous detection pass, so carried frames can
            # move the box instead of freezing it.
            vx = vy = 0.0
            prev = st.prev_centres.get(tid)
            if prev is not None and self._detect_every > 0:
                vx = (cx - prev[0]) / self._detect_every
                vy = (cy - prev[1]) / self._detect_every
            centres[tid] = (cx, cy)

            detections.append(Detection(
                track_id=tid, class_id=cid, class_name=out.names[cid],
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(sv_det.confidence[i]),
                is_person=(cid == _PERSON), is_vehicle=(cid in _VEHICLES),
                velocity=(vx, vy),
            ))

        st.prev_centres = centres
        st.detections = detections
        st.person_count = sum(1 for d in detections if d.is_person)
        st.vehicle_count = sum(1 for d in detections if d.is_vehicle)

        # ── Pose estimation on person crops ───────────────────────────────────
        if self._pose_enabled and self._pose and self._classifier:
            person_boxes = [
                (d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.track_id)
                for d in detections if d.is_person and d.track_id >= 0
            ]
            if person_boxes:
                kp_by_tid = self._pose.run(frame, person_boxes)
                for d in detections:
                    if d.is_person and d.track_id in kp_by_tid:
                        d.posture = self._classifier.classify(
                            d.track_id, kp_by_tid[d.track_id])
            # Drop per-track classifier state (nose history, aim streak, height
            # baseline) for IDs ByteTrack no longer reports — otherwise those
            # dicts grow without bound as track IDs churn, and a stale aim
            # streak could carry over to a reused ID.
            live_tids = {tid for *_, tid in person_boxes}
            for stale in [t for t in self._classifier._aim_streak
                          if t not in live_tids]:
                self._classifier.flush_track(stale)

        return StreamResult(
            cam_id, ts, frame, detections, st.person_count, st.vehicle_count,
            self._annotate(frame, detections), inferred=True,
        )

    # ── Carrying tracks between detector passes ──────────────────────────────
    def _carry_forward(self, cam_id: int, ts: float, frame: np.ndarray) -> StreamResult:
        """Advance known boxes by their velocity — no GPU work at all."""
        st = self._cam(cam_id)
        age = max(0, self._tick - st.last_detect_tick)

        carried: list[Detection] = []
        h, w = frame.shape[:2]
        for d in st.detections:
            dx, dy = d.velocity[0] * age, d.velocity[1] * age
            x1, y1, x2, y2 = d.bbox
            nx1, ny1 = int(x1 + dx), int(y1 + dy)
            nx2, ny2 = int(x2 + dx), int(y2 + dy)
            # Drop tracks that have drifted off-frame rather than pinning them
            # to the edge, which would show a phantom target on the wall.
            if nx2 <= 0 or ny2 <= 0 or nx1 >= w or ny1 >= h:
                continue
            carried.append(Detection(
                d.track_id, d.class_id, d.class_name,
                (max(nx1, 0), max(ny1, 0), min(nx2, w), min(ny2, h)),
                d.confidence, d.is_person, d.is_vehicle, d.velocity,
                posture=d.posture,   # carry last known pose — dimmed in draw_skeleton
            ))

        return StreamResult(
            cam_id, ts, frame, carried, st.person_count, st.vehicle_count,
            self._annotate(frame, carried, dim=True),
            inferred=False, gated=True,
        )

    # ── Drawing ──────────────────────────────────────────────────────────────
    def _annotate(self, frame: np.ndarray, detections: list[Detection],
                  dim: bool = False) -> np.ndarray:
        canvas = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            if d.is_person:
                colour = (60, 220, 60)
            elif d.is_vehicle:
                colour = (240, 170, 40)
            else:
                colour = (170, 170, 170)
            if dim:                       # carried box: slightly muted
                colour = tuple(int(c * 0.72) for c in colour)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            label = f"#{d.track_id} {d.class_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(y1 - 4, th + 4)
            cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), colour, -1)
            cv2.putText(canvas, label, (x1 + 3, ty - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)
            # Skeleton overlay on persons that have pose data
            if d.is_person and d.posture is not None:
                draw_skeleton(canvas, d.posture, dim=dim)
        return canvas

    # ── Introspection ────────────────────────────────────────────────────────
    # A camera counts as "live" (motion is currently reaching YOLO) if the
    # motion gate passed one of its frames within this many ticks. Otherwise
    # it is "idle" — a static scene the GPU is not being spent on.
    LIVE_WITHIN_TICKS = 12

    def gate_stats(self) -> dict[int, dict]:
        return {
            cam_id: {
                "considered": st.gate.stats.considered,
                "skipped": st.gate.stats.skipped,
                "skip_pct": round(st.gate.stats.skip_pct, 1),
                "last_motion": round(st.gate.stats.last_ratio, 5),
                "infer_count": st.infer_count,
                "gated_count": st.gated_count,
                # live = motion gating is currently passing frames through
                "live": (self._tick - st.last_gate_pass_tick) <= self.LIVE_WITHIN_TICKS,
            }
            for cam_id, st in self._cams.items()
        }

    # ── Backwards-compatible single-frame path ───────────────────────────────
    def process(self, cam_id: int, timestamp: float, frame: np.ndarray) -> StreamResult:
        return self.process_batch([(cam_id, timestamp, frame)])[0]
