from __future__ import annotations

import logging
import queue
import struct
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger("ibvap.capture")


class WebSocketCapture:
    """
    Drop-in replacement for StreamCapture when frames arrive from a browser
    via WebSocket rather than from a local camera URL.

    Frames are pushed in by the FastAPI WebSocket handler (async context)
    and consumed by the detection loop (sync thread) — the Queue bridges them.

    A mobile device may not grant the resolution it was asked for, so every
    frame is normalised to (norm_w, norm_h) here, before it reaches the
    pipeline. The pipeline downstream therefore never has to care whether a
    stream came from a 4K bullet camera or a phone that quietly fell back to
    640x480.
    """

    # A camera is considered live if a frame landed within this window.
    STALE_AFTER = 3.0

    def __init__(self, cam_id: int, queue_size: int = 1,
                 norm_w: int = 1280, norm_h: int = 720) -> None:
        self.cam_id = cam_id
        # Depth 1 + drop-oldest: the consumer always gets the freshest frame,
        # and a slow consumer can never build a backlog of stale frames.
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._last_frame_at = 0.0
        self._lock = threading.Lock()
        # Incremented on every new socket; a closing socket only clears the
        # flag if it still owns the current generation, so a reconnect that
        # races with the old socket's teardown is not marked disconnected.
        self._generation = 0
        self._open = False
        self.frames_received = 0

        # ── Normalisation target ────────────────────────────────────────────
        self._norm_w = norm_w
        self._norm_h = norm_h
        self._resized_count = 0        # frames we had to rescale server-side
        self._logged_deviation = False

        # ── Device metadata (from the browser's "hello" message) ────────────
        self.label: str = f"CAM-{cam_id:02d}"
        self.kind: str = "unknown"    # "mobile" | "rtsp" | "replay" | "unknown"
        self.req_w = self.req_h = self.req_fps = 0     # what the browser asked for
        self.neg_w = self.neg_h = 0                    # what it actually got
        self.neg_fps = 0.0
        self.connected_at = 0.0

        # ── Delivered frame-rate estimate (EMA of arrival intervals) ────────
        self._fps_ema = 0.0
        self._last_arrival = 0.0

        # ── End-to-end latency (phone capture → server receive) ─────────────
        # The browser prepends an 8-byte LE millisecond capture timestamp to
        # every JPEG frame. clock_offset_ms aligns the phone's clock to the
        # server's (set from the hello). latency_ms is an EMA in milliseconds.
        self._clock_offset_ms = 0.0
        self._have_offset = False
        self._latency_ms = 0.0
        self._latency_raw_ms = 0.0

    # ── Connection lifecycle (called from the WebSocket handler) ──────────────
    def open(self) -> int:
        with self._lock:
            self._generation += 1
            self._open = True
            self.connected_at = time.time()
            self._fps_ema = 0.0
            self._last_arrival = 0.0
            return self._generation

    def close(self, generation: int) -> bool:
        """Mark disconnected only if this socket is still the current one."""
        with self._lock:
            if generation == self._generation:
                self._open = False
                return True
            return False

    def set_hello(self, info: dict) -> float | None:
        """Record what the browser negotiated with the phone's camera.

        Returns the server-clock millisecond value the caller should echo back
        in a ``synced`` reply, or None if the hello carried no ``t0``.
        """
        self.label = str(info.get("label") or self.label)[:40]
        self.kind = str(info.get("kind") or "mobile")
        self.req_w = int(info.get("reqWidth") or 0)
        self.req_h = int(info.get("reqHeight") or 0)
        self.req_fps = int(info.get("reqFps") or 0)
        self.neg_w = int(info.get("width") or 0)
        self.neg_h = int(info.get("height") or 0)
        try:
            self.neg_fps = float(info.get("fps") or 0.0)
        except (TypeError, ValueError):
            self.neg_fps = 0.0
        # The browser re-sends the hello every ~10 s to re-sync the clock; only
        # log the first one (and any that actually changed) to keep logs quiet.
        sig = (self.label, self.req_w, self.req_h, self.neg_w, self.neg_h)
        if sig != getattr(self, "_hello_sig", None):
            self._hello_sig = sig
            logger.info(
                "CAM-%02d hello: %s | requested %dx%d@%d, negotiated %dx%d@%.0f",
                self.cam_id, self.label, self.req_w, self.req_h, self.req_fps,
                self.neg_w, self.neg_h, self.neg_fps,
            )
        if (self.neg_w, self.neg_h) != (self.req_w, self.req_h) and self.req_w and sig != getattr(self, "_dev_warned", None):
            self._dev_warned = sig
            logger.warning(
                "CAM-%02d resolution not honoured: asked %dx%d, got %dx%d — "
                "frames will be resized server-side to %dx%d",
                self.cam_id, self.req_w, self.req_h, self.neg_w, self.neg_h,
                self._norm_w, self._norm_h,
            )

        # Clock sync: offset = server_now - phone_t0 (ignores the tens-of-ms
        # one-way hello delay — negligible for a "is it seconds behind" gauge).
        try:
            t0 = float(info.get("t0") or 0.0)
        except (TypeError, ValueError):
            t0 = 0.0
        server_now_ms = time.time() * 1000.0
        if t0 > 0:
            self._clock_offset_ms = server_now_ms - t0
            self._have_offset = True
            return server_now_ms
        return None

    @property
    def connected(self) -> bool:
        """Open socket AND recent frames — an idle socket is not a live camera."""
        if not self._open:
            return False
        return (time.monotonic() - self._last_frame_at) < self.STALE_AFTER

    @property
    def socket_open(self) -> bool:
        return self._open

    @property
    def delivered_fps(self) -> float:
        # Decay toward zero if frames stopped arriving.
        if self._last_arrival and (time.monotonic() - self._last_arrival) > 2.0:
            return 0.0
        return round(self._fps_ema, 1)

    @property
    def latency_ms(self) -> float:
        """Smoothed phone-capture → server-receive latency, or 0 if unknown."""
        if not self._have_offset:
            return 0.0
        # A stalled stream: report the growing gap since the last frame.
        if self._last_arrival and (time.monotonic() - self._last_arrival) > 1.0:
            return round(self._latency_ms + (time.monotonic() - self._last_arrival) * 1000.0)
        return round(self._latency_ms)

    def info(self) -> dict:
        """Everything the dashboard needs to describe this device."""
        return {
            "cam_id": self.cam_id,
            "label": self.label,
            "kind": self.kind,
            "connected": self.connected,
            "socket_open": self._open,
            "connected_at": round(self.connected_at, 1),
            "frames_received": self.frames_received,
            "requested": (f"{self.req_w}x{self.req_h}@{self.req_fps}"
                          if self.req_w else "—"),
            "negotiated": (f"{self.neg_w}x{self.neg_h}@{self.neg_fps:.0f}"
                           if self.neg_w else "—"),
            "delivered_fps": self.delivered_fps,
            "latency_ms": self.latency_ms,
            "resized": self._resized_count,
            "normalised_to": f"{self._norm_w}x{self._norm_h}",
        }

    # ── Frame flow ───────────────────────────────────────────────────────────
    def push_frame(self, data: bytes) -> None:
        # Frame wire format from the browser: [8-byte LE capture-ms | JPEG].
        # A bare JPEG starts with the SOI marker FF D8; a timestamped frame has
        # FF D8 at offset 8. Require the SOI in the right place so a timestamp
        # whose low bytes happen to be FF D8 can't be misread as a bare JPEG.
        cap_ms = None
        if (len(data) > 10 and data[8] == 0xFF and data[9] == 0xD8
                and not (data[0] == 0xFF and data[1] == 0xD8)):
            cap_ms = struct.unpack_from("<Q", data, 0)[0]
            jpeg_bytes = data[8:]
        else:
            jpeg_bytes = data

        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        if cap_ms is not None and self._have_offset:
            raw = time.time() * 1000.0 - (cap_ms + self._clock_offset_ms)
            self._latency_raw_ms = raw
            # Clamp absurd values from a phone with a wildly wrong clock.
            raw = max(0.0, min(raw, 30000.0))
            self._latency_ms = raw if self._latency_ms == 0.0 else \
                0.7 * self._latency_ms + 0.3 * raw

        # ── Server-side normalisation ──────────────────────────────────────
        h, w = frame.shape[:2]
        if (w, h) != (self._norm_w, self._norm_h):
            frame = cv2.resize(frame, (self._norm_w, self._norm_h),
                               interpolation=cv2.INTER_AREA
                               if w > self._norm_w else cv2.INTER_LINEAR)
            self._resized_count += 1
            if not self._logged_deviation:
                self._logged_deviation = True
                logger.info("CAM-%02d normalising %dx%d → %dx%d (first frame)",
                            self.cam_id, w, h, self._norm_w, self._norm_h)

        now = time.monotonic()
        if self._last_arrival:
            dt = now - self._last_arrival
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema == 0.0 else \
                    0.9 * self._fps_ema + 0.1 * inst
        self._last_arrival = now

        if self._queue.full():
            try:
                self._queue.get_nowait()   # drop the stale frame, keep latest
            except queue.Empty:
                pass
        self._queue.put((now, frame))
        self._last_frame_at = now
        self.frames_received += 1

    def read(self) -> tuple[float, np.ndarray] | None:
        """Non-blocking: the detection loop polls every camera each pass, so a
        blocking read here would multiply latency by the camera count."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        pass  # lifecycle managed by the WebSocket handler
