"""
Motion gating — the cheap pre-filter in front of the detector.

Border footage is static most of the time: an empty fence line, an unchanging
road at 03:00. Running YOLO on those frames burns GPU for a guaranteed-empty
result. This module answers "did anything actually change?" in well under a
millisecond per frame, so the detector only ever sees frames worth looking at.

The check is a downscaled greyscale frame difference:
    * shrink to ~160x90  (≈100x fewer pixels than 720p)
    * blur to kill sensor noise
    * absdiff against the previous frame
    * threshold, then measure what fraction of pixels moved

Two safeguards keep it from ever hiding a real intrusion:

  * force_every — a frame is always sent to the detector at least this often,
    so a target that creeps in slower than the pixel threshold, or that was
    already present when the gate initialised, is still picked up.
  * hold_frames — once motion is seen the gate stays open for a short tail,
    so a target that pauses mid-scene keeps being detected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class GateStats:
    considered: int = 0        # frames offered to the gate
    passed: int = 0            # frames sent on to the detector
    skipped: int = 0           # frames the detector never saw
    forced: int = 0            # passed only because of the keep-alive
    last_ratio: float = 0.0    # fraction of pixels that changed, last frame

    @property
    def skip_pct(self) -> float:
        return 100.0 * self.skipped / self.considered if self.considered else 0.0


class MotionGate:
    """Per-camera motion detector. One instance per stream."""

    WIDTH, HEIGHT = 160, 90     # analysis resolution, 16:9

    def __init__(
        self,
        pixel_threshold: int = 18,     # per-pixel intensity delta that counts as change
        area_threshold: float = 0.002, # fraction of pixels that must change (0.2%)
        force_every: int = 24,         # never skip more than this many in a row
        hold_frames: int = 6,          # keep the gate open this long after motion
    ) -> None:
        self._pixel_threshold = pixel_threshold
        self._area_threshold = area_threshold
        self._force_every = max(1, force_every)
        self._hold_frames = max(0, hold_frames)

        self._prev: np.ndarray | None = None
        self._since_pass = 0
        self._hold = 0
        self.stats = GateStats()
        # True when the last pass was driven by real motion (or its hold tail),
        # False when it was only the keep-alive sweep. The dashboard uses this
        # to distinguish a genuinely active camera from a swept static one.
        self.last_pass_was_motion = False

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (self.WIDTH, self.HEIGHT),
                           interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(grey, (5, 5), 0)

    def should_detect(self, frame: np.ndarray) -> bool:
        """True when this frame is worth a detector pass."""
        self.stats.considered += 1
        current = self._prepare(frame)

        # First frame of a stream: always detect, we have no baseline yet.
        if self._prev is None:
            self._prev = current
            self._since_pass = 0
            self.stats.passed += 1
            self.stats.forced += 1
            self.last_pass_was_motion = False
            return True

        delta = cv2.absdiff(self._prev, current)
        _, mask = cv2.threshold(delta, self._pixel_threshold, 255, cv2.THRESH_BINARY)
        ratio = float(np.count_nonzero(mask)) / mask.size
        self.stats.last_ratio = ratio
        self._prev = current

        moved = ratio >= self._area_threshold
        if moved:
            self._hold = self._hold_frames

        self._since_pass += 1
        forced = self._since_pass >= self._force_every

        if moved or self._hold > 0 or forced:
            self.last_pass_was_motion = moved or self._hold > 0
            if self._hold > 0 and not moved:
                self._hold -= 1
            if forced and not moved:
                self.stats.forced += 1
            self._since_pass = 0
            self.stats.passed += 1
            return True

        self.stats.skipped += 1
        return False

    def reset(self) -> None:
        self._prev = None
        self._hold = 0
        self._since_pass = 0
