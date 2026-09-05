"""
anpr.py — Automatic Number-Plate Recognition, wired into the IBVAP pipeline.

Two-stage, ported from https://github.com/anindya-mukhopadhyay/ANPR (MIT,
© 2023 BAPPY AHMED):

  1. a YOLOv8 model trained only on `license_plate` (models/license_plate_detector.pt)
     runs on each *vehicle crop* the main detector already found — far cheaper
     and higher-recall than scanning whole frames;
  2. EasyOCR reads the characters off the plate crop, on a background worker
     thread so it never touches the real-time budget.

Preprocessing (upscale small ROIs, greyscale, CLAHE) and the OCR fallback
ladder are from that project; the plumbing — per-track dedup, region-format
cleanup, evidence/CSV logging, graceful-degrade when EasyOCR isn't installed —
is IBVAP's.

Set `anpr.enabled: true` in config.yaml and `pip install easyocr`. If either is
missing the engine disables itself with one warning and the rest of the
pipeline is unaffected.
"""
from __future__ import annotations

import csv
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("ibvap.anpr")

# Region plate-format patterns (post-cleanup: A–Z0–9 only, no spaces).
_REGION_RE = {
    # India: SS RR L(L)(L) NNNN  e.g. WB06AB1234, DL3CAB123, KA01A9999
    "IN": re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"),
    # Generic: 4–10 alphanumerics with at least one letter and one digit
    "XX": re.compile(r"^(?=.*[A-Z])(?=.*[0-9])[A-Z0-9]{4,10}$"),
}


@dataclass
class PlateReading:
    text: str
    conf: float
    bbox: tuple[int, int, int, int]      # plate box in full-frame coords
    cam_id: int
    track_id: int
    ts: float
    valid: bool = False                  # matched the region format


@dataclass
class _Stats:
    plates_detected: int = 0             # plate boxes the YOLO stage found
    ocr_runs: int = 0
    plates_logged: int = 0
    queue_drops: int = 0


class AnprEngine:
    def __init__(self, cfg: dict, device: str = "0") -> None:
        self.available = False
        self._cfg = cfg or {}
        self._device = device
        self._region = str(self._cfg.get("region", "IN")).upper()
        self._yolo_conf = float(self._cfg.get("yolo_conf", 0.30))
        self._ocr_min_conf = float(self._cfg.get("ocr_min_conf", 0.20))
        self._min_area = int(self._cfg.get("min_plate_area", 300))
        self._detect_every = max(1, int(self._cfg.get("plate_detect_every", 3)))
        self._cooldown = float(self._cfg.get("cooldown_s", 8.0))
        self._plate_imgsz = int(self._cfg.get("plate_imgsz", 320))
        n_workers = max(1, int(self._cfg.get("workers", 1)))

        self.stats = _Stats()
        self._plates: dict[int, PlateReading] = {}     # track_id → best reading
        self._plates_lock = threading.Lock()
        self._events: list[PlateReading] = []
        self._events_lock = threading.Lock()
        self._last_logged: dict[str, float] = {}       # plate text → last CSV time
        self._q: queue.Queue = queue.Queue(maxsize=8)
        self._stop = threading.Event()

        weights = self._cfg.get("weights", "models/license_plate_detector.pt")
        if not Path(weights).exists():
            logger.warning("ANPR disabled — plate model not found at %s "
                           "(see docs/ANPR.md)", weights)
            return
        try:
            import easyocr                              # noqa: F401
            from ultralytics import YOLO
        except ImportError as e:
            logger.warning("ANPR disabled — %s. Run: pip install easyocr", e)
            return

        try:
            self._plate_model = YOLO(str(weights))
            gpu = str(device) != "cpu"
            logger.info("ANPR: loading EasyOCR (gpu=%s) — first run downloads "
                        "~64 MB of OCR models", gpu)
            self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        except Exception as e:
            logger.error("ANPR disabled — failed to initialise (%s)", e)
            return

        self._csv_path = Path("data/plates/plates.csv")
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["timestamp", "cam_id", "track_id", "plate", "confidence",
                     "valid", "crop"])

        self._workers = [
            threading.Thread(target=self._ocr_worker, daemon=True,
                             name=f"anpr-ocr-{i}")
            for i in range(n_workers)
        ]
        for w in self._workers:
            w.start()
        self.available = True
        logger.info("ANPR ready — plate model %s, region %s, every %d ticks, "
                    "%d OCR worker(s)", Path(weights).name, self._region,
                    self._detect_every, n_workers)

    # ── stage 1: plate detection on vehicle crops (caller's thread) ──────────
    def submit(self, cam_id: int, frame: np.ndarray,
               vehicles: list[tuple[int, int, int, int, int]], tick: int) -> None:
        """vehicles: list of (x1, y1, x2, y2, track_id)."""
        if not self.available or tick % self._detect_every != 0 or not vehicles:
            return
        h, w = frame.shape[:2]
        now = time.time()
        crops, meta = [], []
        for (x1, y1, x2, y2, tid) in vehicles:
            # already have a solid, recent reading for this vehicle → skip it
            with self._plates_lock:
                r = self._plates.get(tid)
            if r is not None and r.conf >= 0.55 and (now - r.ts) < self._cooldown:
                continue
            # pad the vehicle box a little; plates sit low/front/rear
            px1 = max(0, x1 - 8); py1 = max(0, y1 - 8)
            px2 = min(w, x2 + 8); py2 = min(h, y2 + 8)
            if px2 - px1 < 20 or py2 - py1 < 20:
                continue
            crops.append(frame[py1:py2, px1:px2])
            meta.append((tid, px1, py1))

        if not crops:
            return
        try:
            outs = self._plate_model.predict(
                crops, conf=self._yolo_conf, imgsz=self._plate_imgsz,
                device=self._device, verbose=False)
        except Exception as e:
            logger.debug("ANPR plate-detect failed: %s", e)
            return

        for (tid, ox, oy), out in zip(meta, outs):
            for b in out.boxes:
                bx1, by1, bx2, by2 = (int(v) for v in b.xyxy[0])
                if (bx2 - bx1) * (by2 - by1) < self._min_area:
                    continue
                self.stats.plates_detected += 1
                fx1, fy1, fx2, fy2 = bx1 + ox, by1 + oy, bx2 + ox, by2 + oy
                plate_img = frame[max(0, fy1):fy2, max(0, fx1):fx2].copy()
                if plate_img.size == 0:
                    break
                item = (cam_id, tid, plate_img, (fx1, fy1, fx2, fy2), now)
                try:
                    self._q.put_nowait(item)
                except queue.Full:
                    self.stats.queue_drops += 1
                break                                   # one plate per vehicle

    # ── stage 2: OCR (worker thread) ───────────────────────────────────────
    @staticmethod
    def _preprocess(roi: np.ndarray):
        h, w = roi.shape[:2]
        if h < 80 or w < 160:
            s = max(80.0 / max(h, 1), 160.0 / max(w, 1), 2.0)
            roi = cv2.resize(roi, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        return roi, gray, enhanced

    def _clean(self, raw: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", raw).upper()

    # OCR confuses these in both directions depending on font/blur.
    _TO_DIGIT = str.maketrans("OQDILZSBG", "000112586")
    _TO_ALPHA = str.maketrans("0125868", "OIZSBGB")

    def _coerce_in(self, t: str) -> tuple[str, bool]:
        """Force an Indian plate into  SS RR L(L)(L) NNNN  by swapping
        digit/letter look-alikes per position. Returns (text, matched)."""
        if _REGION_RE["IN"].match(t):
            return t, True
        if not (6 <= len(t) <= 11):
            return t, False
        s = list(t)
        # positions 0-1: state letters
        for i in (0, 1):
            if i < len(s):
                s[i] = s[i].translate(self._TO_ALPHA) if s[i].isdigit() else s[i]
        # positions 2..3: RTO digits
        for i in (2, 3):
            if i < len(s) and s[i].isalpha():
                s[i] = s[i].translate(self._TO_DIGIT)
        # last 1-4: number → digits
        for i in range(max(4, len(s) - 4), len(s)):
            if s[i].isalpha():
                s[i] = s[i].translate(self._TO_DIGIT)
        out = "".join(s)
        return (out, True) if _REGION_RE["IN"].match(out) else (t, False)

    def _ocr_worker(self) -> None:
        allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        while not self._stop.is_set():
            try:
                cam_id, tid, roi, bbox, ts = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                roi_s, gray, enh = self._preprocess(roi)
                self.stats.ocr_runs += 1
                res = (self._reader.readtext(enh, allowlist=allow)
                       or self._reader.readtext(gray, allowlist=allow)
                       or self._reader.readtext(roi_s))
                parts, confs = [], []
                for item in res:
                    c = self._clean(item[1])
                    if len(c) >= 2:
                        parts.append(c)
                        confs.append(float(item[2]))
                if not parts:
                    continue
                text = "".join(parts)
                conf = sum(confs) / len(confs)

                if self._region == "IN":
                    text, valid = self._coerce_in(text)
                else:
                    valid = bool(_REGION_RE["XX"].match(text))
                # A format-valid read is worth more than its raw OCR score.
                score = min(1.0, conf + 0.15) if valid else conf

                pr = PlateReading(text, score, bbox, cam_id, tid, time.time(), valid)
                with self._plates_lock:
                    prev = self._plates.get(tid)
                    if prev is None or score >= prev.conf or \
                            (time.time() - prev.ts) > 4.0:
                        self._plates[tid] = pr
                self._maybe_log(pr, roi)
            except Exception as e:
                logger.debug("ANPR OCR error: %s", e)
            finally:
                self._q.task_done()

    def _maybe_log(self, pr: PlateReading, crop=None) -> None:
        if pr.conf < self._ocr_min_conf:
            return
        now = time.time()
        if now - self._last_logged.get(pr.text, 0.0) < self._cooldown:
            return
        self._last_logged[pr.text] = now
        self.stats.plates_logged += 1

        crop_path = ""
        try:
            crop_dir = self._csv_path.parent
            safe = re.sub(r"[^A-Z0-9]", "_", pr.text) or "UNKNOWN"
            crop_path = str(crop_dir / f"{datetime.now():%Y%m%d_%H%M%S}_"
                                       f"cam{pr.cam_id}_{safe}.jpg")
            if crop is not None and getattr(crop, "size", 0):
                cv2.imwrite(crop_path, crop)
        except Exception:
            pass
        try:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [datetime.now().isoformat(timespec="seconds"), pr.cam_id,
                     pr.track_id, pr.text, f"{pr.conf:.2f}", int(pr.valid),
                     crop_path])
        except Exception as e:
            logger.debug("ANPR CSV write failed: %s", e)

        with self._events_lock:
            self._events.append(pr)
        logger.info("PLATE  CAM-%02d  %s  (%.0f%%%s)", pr.cam_id, pr.text,
                    pr.conf * 100, "" if pr.valid else " unverified")

    # ── consumer interface (caller's thread) ───────────────────────────────
    def readings_for_cam(self, cam_id: int, max_age: float = 5.0) -> dict[int, PlateReading]:
        now = time.time()
        with self._plates_lock:
            return {tid: r for tid, r in self._plates.items()
                    if r.cam_id == cam_id and (now - r.ts) < max_age}

    def flush_track(self, track_id: int) -> None:
        with self._plates_lock:
            self._plates.pop(track_id, None)

    def drain_events(self) -> list[PlateReading]:
        with self._events_lock:
            out, self._events = self._events, []
        return out

    def status(self) -> dict:
        with self._plates_lock:
            active = [
                {"cam": r.cam_id, "track": tid, "plate": r.text,
                 "conf": round(r.conf, 2), "valid": r.valid}
                for tid, r in sorted(self._plates.items())
                if (time.time() - r.ts) < 5.0
            ]
        return {
            "enabled": self.available,
            "region": self._region,
            "plates_detected": self.stats.plates_detected,
            "ocr_runs": self.stats.ocr_runs,
            "plates_logged": self.stats.plates_logged,
            "queue_drops": self.stats.queue_drops,
            "active": active,
        }

    def stop(self) -> None:
        self._stop.set()
