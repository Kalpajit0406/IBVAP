"""
rtsp_capture.py — pull a real CCTV stream (RTSP / MJPEG / webcam / file) into the
same pipeline the phones feed.

`RtspCapture` is duck-compatible with `WebSocketCapture`: same `.read()`,
`.connected`, `.socket_open`, `.delivered_fps`, `.latency_ms`, `.info()`,
`.frames_received`, `.stop()` — so `server.py`'s muxer, dashboard and prune
logic treat an NVR channel exactly like a phone.

Design notes for real cameras (see docs/CCTV_INTEGRATION.md):

  * One decode thread per camera. A frozen camera must never block the muxer,
    so decoding happens here and `.read()` only pops the newest decoded frame.
  * RTSP over **TCP**. UDP RTSP tears on any packet loss. Forced via
    OPENCV_FFMPEG_CAPTURE_OPTIONS (set once, below).
  * `grab()` every loop (cheap — drains the socket, no buffer bloat), but
    `retrieve()` (the expensive decode) only at `decode_fps`. Detection runs at
    8 fps and CCTV sub-streams are often 12–15 fps anyway.
  * Auto-reconnect with backoff; a stall watchdog reopens a stream that stops
    delivering frames without erroring.
  * Every frame normalised to (norm_w, norm_h) so the detector sees one size.
  * Credentials in the URL are redacted everywhere they are logged or exposed.
"""
from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger("ibvap.rtsp")

# Force TCP + sane socket/read timeouts for every FFmpeg-backed capture. Must be
# set before the first cv2.VideoCapture opens; importing this module does that.
#   rtsp_transport;tcp   — reliable, no UDP smear
#   stimeout;5000000     — 5 s socket timeout (µs) so a dead host fails fast
#   max_delay;500000     — cap reorder buffering at 0.5 s
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|max_delay;500000",
)

_CRED_RE = re.compile(r"://([^:/@]+):([^@/]+)@")


def redact(url: str) -> str:
    """rtsp://admin:secret@10.0.0.5/... → rtsp://admin:***@10.0.0.5/..."""
    return _CRED_RE.sub(lambda m: f"://{m.group(1)}:***@", str(url))


class RtspCapture:
    STALE_AFTER = 3.0     # matches WebSocketCapture so the muxer's cutoff is uniform

    def __init__(self, cam_id: int, url: str, *,
                 name: str | None = None,
                 norm_w: int = 1280, norm_h: int = 720,
                 decode_fps: float = 15.0,
                 transport: str = "tcp",
                 reconnect_delay: float = 3.0,
                 stall_timeout: float = 8.0,
                 open_timeout: float = 8.0) -> None:
        self.cam_id = cam_id
        self._url = str(url)
        # "0"/"1" → local webcam index; everything else → URL / file path.
        self._source: int | str = int(url) if str(url).isdigit() else str(url)
        self._is_file = isinstance(self._source, str) and bool(
            re.search(r"\.(mp4|avi|mov|mkv|webm)$", self._source, re.I))

        self.label = (name or f"CAM-{cam_id:02d}")[:40]
        self.kind = "rtsp" if str(self._source).lower().startswith("rtsp") else \
                    "file" if self._is_file else \
                    "http" if str(self._source).lower().startswith("http") else \
                    "webcam"
        self._transport = transport
        self._decode_period = 1.0 / max(decode_fps, 1.0)
        self._reconnect_delay = reconnect_delay
        self._stall_timeout = stall_timeout
        self._open_timeout = open_timeout

        self._norm_w, self._norm_h = norm_w, norm_h
        self._resized_count = 0

        self._queue: queue.Queue = queue.Queue(maxsize=1)   # depth-1, drop-oldest
        self._last_frame_at = 0.0
        self._last_arrival = 0.0
        self._fps_ema = 0.0
        self.frames_received = 0
        self.connected_at = 0.0

        self.neg_w = self.neg_h = 0
        self.neg_fps = 0.0
        self.reconnects = 0
        self.last_error = ""

        self._stop = threading.Event()
        self._reopen = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"rtsp-{cam_id}")

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> "RtspCapture":
        self.connected_at = time.time()
        self._thread.start()
        logger.info("CAM-%02d (%s) pulling %s", self.cam_id, self.kind,
                    redact(self._url))
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3)

    def request_reconnect(self) -> None:
        """Force a drop + reopen — handy from the dashboard if a feed freezes."""
        self._reopen.set()

    # ── WebSocketCapture-compatible stubs (a phone hitting this id is ignored)
    def open(self) -> int:
        return 0

    def close(self, generation: int) -> bool:
        return False

    def set_hello(self, info: dict) -> float | None:
        return None

    def push_frame(self, data: bytes) -> None:
        if not getattr(self, "_warned_ws", False):
            self._warned_ws = True
            logger.warning("CAM-%02d is an RTSP camera — ignoring a WebSocket "
                           "frame pushed to the same id", self.cam_id)

    # ── the decode thread ──────────────────────────────────────────────────
    def _open_capture(self) -> cv2.VideoCapture | None:
        api = cv2.CAP_FFMPEG if isinstance(self._source, str) else cv2.CAP_ANY
        try:
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self._open_timeout * 1000),
                      cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self._stall_timeout * 1000)]
            cap = cv2.VideoCapture(self._source, api, params)
        except Exception:
            cap = cv2.VideoCapture(self._source, api)   # older OpenCV: no params arg
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.neg_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.neg_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.neg_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return cap

    def _loop(self) -> None:
        while not self._stop.is_set():
            cap = self._open_capture()
            if cap is None:
                self.last_error = "cannot open"
                logger.warning("CAM-%02d cannot open %s — retry in %.0fs",
                               self.cam_id, redact(self._url), self._reconnect_delay)
                self._sleep(self._reconnect_delay)
                continue

            logger.info("CAM-%02d connected: %dx%d @ %.0f fps",
                        self.cam_id, self.neg_w, self.neg_h, self.neg_fps)
            self._reopen.clear()
            last_decode = 0.0
            last_ok = time.monotonic()

            while not self._stop.is_set() and not self._reopen.is_set():
                if not cap.grab():
                    if self._is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop the file
                        continue
                    self.last_error = "grab failed"
                    break
                now = time.monotonic()
                if now - last_decode < self._decode_period:
                    continue                      # keep draining, skip the decode
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    if now - last_ok > self._stall_timeout:
                        self.last_error = "stalled (no frames)"
                        logger.warning("CAM-%02d stalled — reconnecting", self.cam_id)
                        break
                    continue
                last_decode = last_ok = now
                self._publish(frame, now)

            cap.release()
            if self._stop.is_set():
                break
            self.reconnects += 1
            self._sleep(self._reconnect_delay if not self._reopen.is_set() else 0.2)

    def _publish(self, frame: np.ndarray, now: float) -> None:
        h, w = frame.shape[:2]
        if (w, h) != (self._norm_w, self._norm_h):
            frame = cv2.resize(frame, (self._norm_w, self._norm_h),
                               interpolation=cv2.INTER_AREA if w > self._norm_w
                               else cv2.INTER_LINEAR)
            self._resized_count += 1

        if self._last_arrival:
            dt = now - self._last_arrival
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema == 0.0 else \
                    0.9 * self._fps_ema + 0.1 * inst
        self._last_arrival = now

        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put((now, frame))
        self._last_frame_at = now
        self.frames_received += 1

    def _sleep(self, secs: float) -> None:
        self._stop.wait(timeout=secs)

    # ── consumer interface ─────────────────────────────────────────────────
    def read(self) -> tuple[float, np.ndarray] | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def connected(self) -> bool:
        return (self._thread.is_alive()
                and (time.monotonic() - self._last_frame_at) < self.STALE_AFTER)

    @property
    def socket_open(self) -> bool:
        # For the muxer's prune check: True while the puller thread lives, so a
        # briefly-down configured camera is never pruned.
        return self._thread.is_alive() and not self._stop.is_set()

    @property
    def delivered_fps(self) -> float:
        if self._last_arrival and (time.monotonic() - self._last_arrival) > 2.0:
            return 0.0
        return round(self._fps_ema, 1)

    @property
    def latency_ms(self) -> float:
        # No capture-time clock from an RTSP camera; "is it stalled" is covered
        # by `connected` / `delivered_fps`. Report the gap only once a live
        # stream goes quiet, so the dashboard can flag a freeze.
        if self._last_frame_at and self._thread.is_alive():
            gap = (time.monotonic() - self._last_frame_at) * 1000.0
            return round(gap) if gap > 1500 else 0
        return 0

    def info(self) -> dict:
        return {
            "cam_id": self.cam_id,
            "label": self.label,
            "kind": self.kind,
            "connected": self.connected,
            "socket_open": self.socket_open,
            "connected_at": round(self.connected_at, 1),
            "frames_received": self.frames_received,
            "requested": "—",
            "negotiated": (f"{self.neg_w}x{self.neg_h}@{self.neg_fps:.0f}"
                           if self.neg_w else "connecting…"),
            "delivered_fps": self.delivered_fps,
            "latency_ms": self.latency_ms,
            "resized": self._resized_count,
            "normalised_to": f"{self._norm_w}x{self._norm_h}",
            "rtsp": {
                "url": redact(self._url),
                "transport": self._transport,
                "reconnects": self.reconnects,
                "last_error": self.last_error,
            },
        }
