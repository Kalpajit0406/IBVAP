"""
IBVAP Web Server — browser-based camera intake + live monitor dashboard.

Setup:
    pip install fastapi uvicorn[standard] aiofiles

Run:
    python server.py

Local:
    Phone opens  → http://<your-LAN-ip>:8000/camera/0
    Monitor at   → http://<your-LAN-ip>:8000/monitor

Over the internet (requires ngrok):
    ngrok http 8000
    Phone opens  → https://<ngrok-id>.ngrok-free.app/camera/0   (HTTPS required for iOS)
    Monitor at   → https://<ngrok-id>.ngrok-free.app/monitor

Each stream ID (0, 1, 2 …) is a separate camera. Share /camera/N links to each phone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

from src.detector import Detector
from src.event_store import EventStore
from src.evidence import EvidenceChain
from src.risk_engine import RiskEngine
from src.ws_capture import WebSocketCapture

# Windows consoles default to cp1252 and mangle non-ASCII log output
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ibvap.server")

# ── Shared state (module-level so detection thread can reach it) ───────────────
captures: dict[int, WebSocketCapture] = {}
monitor_clients: set[WebSocket] = set()
_loop: Optional[asyncio.AbstractEventLoop] = None
_detector = None                       # set by the detection thread once built
cfg: dict = {}

# Ingest normalisation target — phones that don't honour the requested
# resolution are rescaled to this before entering the pipeline.
NORM_W, NORM_H = 1280, 720

# Latest raw annotated frame per camera (ndarray) — used to build the mosaic
_latest_bgr: dict[int, "np.ndarray"] = {}
# Latest metadata per camera
_camera_meta: dict[int, dict] = {}

# Hand-off from the muxer thread to the inference worker thread. Depth 1: the
# worker always gets the freshest snapshot; if it falls behind, the muxer drops
# the stale one rather than letting a backlog grow. This is what keeps the
# blocking GPU predict() (≈28 ms for the TensorRT engine) off the muxer, so the
# muxer holds its 24 Hz tick and never backpressures the camera sockets.
_infer_q: "queue.Queue[list]" = queue.Queue(maxsize=1)
_switch_q: "queue.Queue[str]" = queue.Queue(maxsize=1)

STATIC = Path("static")



# ── Lifecycle ─────────────────────────────────────────────────────────────────
_detection_started = False


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Both the HTTP and HTTPS listeners share this app, so the lifespan fires
    # twice — the detection thread must only ever start once.
    global cfg, _loop, _detection_started, NORM_W, NORM_H
    _loop = asyncio.get_running_loop()
    if not _detection_started:
        _detection_started = True
        cfg = yaml.safe_load(open("config.yaml"))
        ing = cfg.get("ingest", {}) or {}
        NORM_W = int(ing.get("normalise_width", NORM_W))
        NORM_H = int(ing.get("normalise_height", NORM_H))
        for s in cfg.get("streams", []):
            captures[s["id"]] = WebSocketCapture(s["id"], norm_w=NORM_W, norm_h=NORM_H)
        threading.Thread(target=_inference_worker, daemon=True, name="infer").start()
        threading.Thread(target=_muxer_loop, daemon=True, name="muxer").start()
        logger.info("IBVAP ready — muxer + inference worker started "
                    "(ingest normalised to %dx%d)", NORM_W, NORM_H)
    yield


app = FastAPI(title="IBVAP", lifespan=_lifespan)


# ── Camera intake ─────────────────────────────────────────────────────────────
# Both /cam/N and /camera/N serve the phone page. The page always opens the
# WebSocket at /ws/camera/N (cam_id is injected server-side), so the short path
# works behind a custom domain without any client change.
@app.get("/cam/{cam_id}", response_class=HTMLResponse)
@app.get("/camera/{cam_id}", response_class=HTMLResponse)
async def camera_page(cam_id: int) -> HTMLResponse:
    html = (STATIC / "camera.html").read_text()
    return HTMLResponse(html.replace("{{CAM_ID}}", str(cam_id)))


@app.websocket("/ws/camera/{cam_id}")
async def camera_ws(ws: WebSocket, cam_id: int) -> None:
    await ws.accept()
    cap = captures.setdefault(
        cam_id, WebSocketCapture(cam_id, norm_w=NORM_W, norm_h=NORM_H))
    generation = cap.open()
    logger.info("CAM-%02d connected", cam_id)
    try:
        while True:
            # A message is either a JSON text "hello" (device metadata) or a
            # binary JPEG frame. The browser sends exactly one hello, first.
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                cap.push_frame(data)
            elif (text := msg.get("text")) is not None:
                try:
                    info = json.loads(text)
                    if info.get("type") == "hello":
                        server_ms = cap.set_hello(info)
                        if server_ms is not None:
                            # Echo the server clock so the phone can align its
                            # own timestamps and show a live latency figure.
                            await ws.send_text(json.dumps({
                                "type": "synced",
                                "serverTime": server_ms,
                                "t0": info.get("t0"),
                            }))
                except (ValueError, TypeError):
                    logger.debug("CAM-%02d: non-JSON text frame ignored", cam_id)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("CAM-%02d socket error: %s", cam_id, e)
    finally:
        # Only the current socket may mark the camera down; a reconnect that
        # raced ahead of this teardown keeps its own live state.
        if cap.close(generation):
            logger.info("CAM-%02d disconnected (%d frames received)",
                        cam_id, cap.frames_received)


# ── Monitor ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/monitor")


@app.get("/monitor", response_class=HTMLResponse)
async def monitor_page() -> HTMLResponse:
    return HTMLResponse(
        (STATIC / "monitor.html").read_text(),
        headers={"Cache-Control": "no-store"},
    )


@app.websocket("/ws/monitor")
async def monitor_ws(ws: WebSocket) -> None:
    await ws.accept()
    monitor_clients.add(ws)
    logger.info("Monitor connected (%d client(s))", len(monitor_clients))
    try:
        while True:
            await asyncio.sleep(20)
    except WebSocketDisconnect:
        pass
    finally:
        monitor_clients.discard(ws)
        logger.info("Monitor disconnected (%d client(s))", len(monitor_clients))


# ── MJPEG mosaic — every camera in one grid, one connection ──────────────────
# The grid is composed here, in the request's own worker thread (run_in_executor),
# NOT on the detection thread. Compositing 8x 720p frames + JPEG encoding costs
# ~15-25 ms; doing it on the detection loop stole enough time to drop the muxer
# below real-time. The detector only keeps _latest_bgr fresh.
@app.get("/stream")
async def mjpeg_mosaic():
    logger.info("MJPEG mosaic requested")
    loop = asyncio.get_running_loop()

    async def generate():
        try:
            while True:
                if _latest_bgr:
                    jpeg = await loop.run_in_executor(None, _build_mosaic)
                    if jpeg:
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n"
                               b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                               + jpeg + b"\r\n")
                await asyncio.sleep(0.045)    # ~22 fps mosaic (fine for 2-4 cams)
        except asyncio.CancelledError:
            logger.info("MJPEG mosaic closed")
            raise

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache", "Connection": "close"},
    )


# ── MJPEG stream — one camera, for debugging (dashboard uses the mosaic) ──────
@app.get("/stream/{cam_id}")
async def mjpeg_stream(cam_id: int):
    logger.info("MJPEG stream requested for CAM-%02d", cam_id)
    loop = asyncio.get_running_loop()

    def _encode(cid: int):
        f = _latest_bgr.get(cid)
        if f is None:
            return None
        ok, jpeg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpeg.tobytes() if ok else None

    async def generate():
        try:
            while True:
                jpeg = await loop.run_in_executor(None, _encode, cam_id)
                if jpeg:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                           + jpeg + b"\r\n")
                await asyncio.sleep(0.045)   # ~22 fps single-cam stream
        except asyncio.CancelledError:
            raise

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache",
                 "Connection": "close"},
    )


@app.get("/meta/{cam_id}")
async def cam_meta(cam_id: int):
    from fastapi.responses import JSONResponse
    return JSONResponse(_camera_meta.get(cam_id, {}))


# ── Detection loop (background thread) ────────────────────────────────────────
# Shared counters for /status endpoint
_stats: dict = {"frames": 0, "broadcasts": 0}


def _device_rows() -> list[dict]:
    """Per-device view for the dashboard: identity, link state, live/idle, res/fps."""
    gate = _detector.gate_stats() if _detector is not None else {}
    rows = []
    for cid, cap in sorted(captures.items()):
        g = gate.get(cid, {})
        base = cap.info()
        # "live" here means motion gating is currently letting frames reach
        # YOLO. A connected-but-static camera is "idle" — no GPU spent on it.
        base["state"] = ("idle" if base["connected"] and not g.get("live", False)
                         else "live" if base["connected"] else "offline")
        base["infer_count"] = g.get("infer_count", 0)
        base["gated_count"] = g.get("gated_count", 0)
        base["gate_skip_pct"] = g.get("skip_pct", 0.0)
        base["last_motion"] = g.get("last_motion", 0.0)
        rows.append(base)
    return rows


@app.get("/devices")
async def devices():
    from fastapi.responses import JSONResponse
    return JSONResponse({"count": len(captures), "devices": _device_rows()})


@app.post("/api/switch-model", status_code=202)
async def switch_model(payload: dict = Body(...)):
    profile = str(payload.get("profile", "")).strip().lower()
    profiles = cfg.get("model_profiles", {})
    if profile not in profiles:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": f"Unknown profile '{profile}'. Valid profiles: {list(profiles.keys())}"}, status_code=400)
    try:
        # Clear out any stale switch request and push new one
        try:
            _switch_q.get_nowait()
        except queue.Empty:
            pass
        _switch_q.put_nowait(profile)
        _stats["switching"] = True
        _stats["switch_target"] = profile
        logger.info("Model switch requested: %s", profile)
    except Exception as e:
        logger.warning("Failed to queue model switch: %s", e)
    return {"queued": profile}


@app.get("/api/models")
async def list_models():
    from fastapi.responses import JSONResponse
    profiles = cfg.get("model_profiles", {})
    return JSONResponse({
        "active": _stats.get("active_model", "nano"),
        "switching": _stats.get("switching", False),
        "profiles": {k: v.get("label", k) for k, v in profiles.items()},
    })


@app.get("/status")
async def status():
    from fastapi.responses import JSONResponse
    return JSONResponse({
        "device": _stats.get("device", "unknown"),
        "active_model": _stats.get("active_model", "nano"),
        "switching": _stats.get("switching", False),
        "cameras": {cid: {"live": cap.connected,
                          "socket_open": cap.socket_open,
                          "frames_received": cap.frames_received}
                    for cid, cap in captures.items()},
        "devices": _device_rows(),
        "monitor_clients": len(monitor_clients),
        "frames_processed": _stats["frames"],
        "broadcasts_sent": _stats["broadcasts"],
        "inference_ms": round(_stats.get("infer_ms", 0.0), 1),
        "fps": round(_stats.get("fps", 0.0), 1),
        "pipeline": {
            "muxer_tps": round(_stats.get("mux_tps", 0.0), 1),
            "worker_bps": round(_stats.get("worker_bps", 0.0), 1),
            "snapshots_dropped": _stats.get("dropped", 0),
            "mean_batch": round(_stats.get("mean_batch", 0.0), 2),
            "last_batch": _stats.get("last_batch", 0),
            "detector_passes": _stats.get("detector_passes", 0),
            "frames_inferred": _stats.get("frames_inferred", 0),
            "frames_gated": _stats.get("frames_gated", 0),
            "frames_rate_skipped": _stats.get("frames_rate_skipped", 0),
            "gpu_saving_pct": round(_stats.get("gpu_saving_pct", 0.0), 1),
        },
        "buffered_bgr": {cid: True for cid in _latest_bgr},
        "meta": _camera_meta,
    })



# ── Thread 1: muxer ──────────────────────────────────────────────────────────
# Drains every camera's queue continuously into a one-deep latest-frame slot,
# then on a FIXED tick snapshots all live slots and hands the whole set to the
# inference worker. It never runs the model, ByteTrack, or annotation, so it
# holds its 24 Hz tick regardless of how long a detection pass takes — which is
# what stops slow inference from backpressuring the camera WebSockets.
def _muxer_loop() -> None:
    tick_period = 1.0 / max(float(cfg["model"].get("stream_fps", 24.0)), 1.0)
    latest: dict[int, tuple[float, "np.ndarray"]] = {}
    configured = {s["id"] for s in cfg.get("streams", [])}
    last_prune = time.monotonic()
    next_tick = time.monotonic()
    tick_window_start = time.monotonic()
    ticks = 0
    idle_logged = 0

    def drain() -> None:
        # cap.read() pops one frame; the inner loop drains any backlog and keeps
        # only the newest, so draining once per tick is equivalent to doing it
        # continuously — without the busy-poll that was starving other threads
        # (and the load-test client process) of CPU.
        for cam_id, cap in list(captures.items()):
            fd = cap.read()
            while fd is not None:
                latest[cam_id] = fd
                fd = cap.read()

    while True:
        if not captures:
            time.sleep(0.1)
            continue

        # Sleep until the tick is due (self-correcting accumulator), then drain.
        slack = next_tick - time.monotonic()
        if slack > 0:
            time.sleep(min(slack, tick_period))
        next_tick += tick_period
        if time.monotonic() - next_tick > 0.5:    # fell badly behind; resync
            next_tick = time.monotonic()
        drain()

        ticks += 1
        tw = time.monotonic() - tick_window_start
        if tw >= 1.0:
            _stats["mux_tps"] = ticks / tw
            ticks = 0
            tick_window_start = time.monotonic()

        # Prune dynamic devices (phones, replay) gone a while, so the dashboard
        # reflects who is actually here. Configured RTSP/webcam streams stay.
        if time.monotonic() - last_prune > 5.0:
            last_prune = time.monotonic()
            for cid in [c for c, cap in captures.items()
                        if c not in configured and not cap.socket_open
                        and (time.monotonic() - cap._last_frame_at) > 20.0]:
                captures.pop(cid, None)
                latest.pop(cid, None)
                _latest_bgr.pop(cid, None)
                _camera_meta.pop(cid, None)
                logger.info("CAM-%02d pruned from device list (gone > 20s)", cid)

        now = time.monotonic()
        batch = [(cid, ts, frm) for cid, (ts, frm) in list(latest.items())
                 if (now - ts) < WebSocketCapture.STALE_AFTER]

        if not batch:
            idle_logged += 1
            if idle_logged == 300:
                connected = [cid for cid, c in captures.items() if c.connected]
                if connected:
                    logger.warning("No frames for ~%ds from connected camera(s) %s — "
                                   "the sender may have stalled",
                                   int(300 * tick_period), connected)
                else:
                    logger.info("Waiting for a camera to connect…")
                idle_logged = 0
            continue
        idle_logged = 0

        # Hand off. Depth-1 queue: if the worker is still busy, drop the stale
        # snapshot and enqueue the fresh one — process the newest frames, never
        # a backlog.
        try:
            _infer_q.put_nowait(batch)
        except queue.Full:
            try:
                _infer_q.get_nowait()
                _stats["dropped"] = _stats.get("dropped", 0) + 1
            except queue.Empty:
                pass
            try:
                _infer_q.put_nowait(batch)
            except queue.Full:
                pass


# ── Thread 2: inference worker ───────────────────────────────────────────────
# Owns the Detector (and therefore all per-camera ByteTrack / motion-gate
# state — single-threaded, no locks needed) plus risk scoring and evidence.
# Runs at its own pace: a slow pass just means it consumes fewer snapshots,
# the muxer keeps ticking, and video/detection degrade gracefully instead of
# stalling the ingest.
def _inference_worker() -> None:
    global _detector
    detector = Detector(cfg, num_cameras=0)
    _detector = detector                      # expose for /status and /devices
    _stats["device"] = detector.device
    init_w = str(cfg.get("model", {}).get("weights", ""))
    _stats["active_model"] = "medium" if "26m" in init_w else "nano"
    _stats["switching"] = False
    risk_engine = RiskEngine(cfg)
    evidence = EvidenceChain(cfg["evidence"]["hash_chain_path"])
    store = EventStore(cfg["evidence"]["db_path"])
    prev_levels: dict[int, str] = {}
    log_levels = {"High", "Critical"}

    win_start = time.monotonic()
    win_frames = 0
    win_batches = 0

    while True:
        # Check for pending model-switch request
        try:
            profile = _switch_q.get_nowait()
            logger.info("Hot-swapping model profile to '%s'...", profile)
            _stats["switching"] = True
            old_det = detector
            try:
                new_det = Detector.from_profile(cfg, profile, num_cameras=0)
                detector = new_det
                _detector = detector
                _stats["device"] = detector.device
                _stats["active_model"] = profile
                logger.info("Model successfully hot-swapped to '%s' on %s", profile, detector.device)
                del old_det
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error("Failed to switch model profile to '%s': %s", profile, e)
                detector = old_det
            finally:
                _stats["switching"] = False
        except queue.Empty:
            pass

        try:
            batch = _infer_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if not batch:
            continue

        results = detector.process_batch(batch)
        _stats["frames"] += len(batch)
        win_frames += len(batch)
        win_batches += 1


        elapsed = time.monotonic() - win_start
        if elapsed >= 1.0:
            ds = detector.stats
            _stats["fps"] = win_frames / elapsed
            _stats["worker_bps"] = win_batches / elapsed
            _stats["infer_ms"] = ds.infer_ms
            _stats["mean_batch"] = ds.mean_batch
            _stats["last_batch"] = ds.last_batch
            _stats["gpu_saving_pct"] = ds.gpu_saving_pct
            _stats["frames_inferred"] = ds.frames_inferred
            _stats["frames_gated"] = ds.frames_gated
            _stats["frames_rate_skipped"] = ds.frames_rate_skipped
            _stats["detector_passes"] = ds.detector_passes
            logger.info(
                "PIPE  %d cams | mux %.1f tps | worker %.0f fps (%.1f bps) | "
                "batch %.1f | %.1f ms/frame | GPU saw %d of %d (%.0f%% skipped) | "
                "dropped %d | %s",
                len(captures), _stats.get("mux_tps", 0.0), _stats["fps"],
                _stats["worker_bps"], ds.mean_batch, ds.infer_ms,
                ds.frames_inferred, ds.frames_in, ds.gpu_saving_pct,
                _stats.get("dropped", 0), _stats.get("device", "?"),
            )
            win_start = time.monotonic()
            win_frames = 0
            win_batches = 0

        for sr in results:
            ra = risk_engine.assess(sr)

            prev = prev_levels.get(sr.cam_id, "Normal")
            if ra.level != prev:
                prev_levels[sr.cam_id] = ra.level
                if ra.level in log_levels:
                    event = {
                        "cam_id": sr.cam_id, "level": ra.level,
                        "score": round(ra.score, 2),
                        "persons": sr.person_count, "vehicles": sr.vehicle_count,
                    }
                    ev_hash = evidence.append(event)
                    store.log(sr.cam_id, ra.level, ra.score,
                              sr.person_count, sr.vehicle_count, event, ev_hash)
                    logger.warning("CAM-%02d %s  score=%.1f  hash=%s…",
                                   sr.cam_id, ra.level, ra.score, ev_hash[:12])

            cap = captures.get(sr.cam_id)
            latency_ms = int(cap.latency_ms) if cap is not None else 0

            _camera_meta[sr.cam_id] = {
                "cam_id": sr.cam_id,
                "level": ra.level,
                "score": round(ra.score, 1),
                "persons": sr.person_count,
                "vehicles": sr.vehicle_count,
                "weapon": ra.weapon,
                "latency_ms": latency_ms,
                "anomalies": [
                    d.posture.label
                    for d in sr.detections
                    if d.is_person and d.posture is not None and d.posture.any_anomaly
                ],
            }

            # Draw the overlay and stash the annotated frame. NO JPEG encoding
            # here — the /stream endpoints encode in their own worker threads.
            frame_out = sr.annotated_frame if sr.annotated_frame is not None else sr.frame
            colour = ((0, 200, 0) if ra.level == "Normal"
                      else (0, 165, 255) if ra.level == "High" else (0, 0, 255))
            tag = "LIVE" if sr.inferred else "trk"
            label = (f"CAM-{sr.cam_id:02d}  {ra.level} {ra.score:.0f}  "
                     f"P:{sr.person_count} V:{sr.vehicle_count}  [{tag}]")
            _draw_label(frame_out, label, (8, 24), 0.55, colour)

            # Latency badge, top-right. Green when snappy, red when the feed is
            # running behind (network or phone uplink).
            if latency_ms:
                late = latency_ms >= 1500
                txt = (f"NET {latency_ms/1000:.1f}s  DELAYED" if late
                       else f"NET {latency_ms} ms")
                scale = 0.62 if late else 0.5
                (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
                _draw_label(frame_out, txt, (frame_out.shape[1] - tw - 12, 24),
                            scale, (0, 0, 255) if late else (0, 210, 120))

            # Weapon banner — impossible to miss.
            if ra.weapon:
                _draw_banner(frame_out, "WEAPON  -  CRITICAL")

            _latest_bgr[sr.cam_id] = frame_out

            if monitor_clients and _loop is not None:
                fut = asyncio.run_coroutine_threadsafe(
                    _broadcast(json.dumps(_camera_meta[sr.cam_id])), _loop)
                fut.add_done_callback(_log_broadcast_error)
                _stats["broadcasts"] += 1


def _draw_label(img, text: str, org, scale: float, colour) -> None:
    """Text with a black outline so it reads on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, 1, cv2.LINE_AA)


def _draw_banner(img, text: str) -> None:
    """A full-width red alert bar across the middle of the frame."""
    h, w = img.shape[:2]
    y0, y1 = int(h * 0.42), int(h * 0.58)
    strip = img[y0:y1].copy()
    cv2.rectangle(strip, (0, 0), (w, y1 - y0), (0, 0, 200), -1)
    cv2.addWeighted(strip, 0.55, img[y0:y1], 0.45, 0, img[y0:y1])
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    org = ((w - tw) // 2, (y0 + y1) // 2 + th // 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)


def _build_mosaic() -> bytes | None:
    """Tile every camera's latest annotated frame into one grid JPEG.

    Runs in a request worker thread (see /stream), never on the detection loop.
    """
    ids = sorted(_latest_bgr)
    if not ids:
        return None
    n = len(ids)
    cols = 1 if n == 1 else 2 if n <= 4 else 3 if n <= 9 else 4
    rows = (n + cols - 1) // cols
    tw, th = 480, 270                       # tile size
    grid = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, cid in enumerate(ids):
        f = _latest_bgr.get(cid)
        if f is None:
            continue
        r, c = divmod(i, cols)
        grid[r*th:(r+1)*th, c*tw:(c+1)*tw] = cv2.resize(f, (tw, th))
    ok, jpeg = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpeg.tobytes() if ok else None


def _log_broadcast_error(fut) -> None:
    exc = fut.exception()
    if exc is not None and _stats.get("broadcast_errors", 0) < 5:
        _stats["broadcast_errors"] = _stats.get("broadcast_errors", 0) + 1
        logger.error("Monitor broadcast failed: %r", exc)


async def _broadcast(meta: str) -> None:
    dead: set[WebSocket] = set()
    for ws in list(monitor_clients):
        try:
            await ws.send_text(meta)
        except Exception as e:
            if _stats.get("send_errors", 0) < 5:
                _stats["send_errors"] = _stats.get("send_errors", 0) + 1
                logger.warning("Monitor send failed, dropping client: %r", e)
            dead.add(ws)
    monitor_clients.difference_update(dead)


# ── Entry point ───────────────────────────────────────────────────────────────
# Two listeners share the same app and detection state:
#   HTTPS :8443 — phones (getUserMedia demands a secure context)
#   HTTP  :8080 — monitor dashboard on this machine (no cert warning)
HTTPS_PORT = 8443
HTTP_PORT = 8090      # 8080 is commonly taken (Steam webhelper, Jenkins, Tomcat)


def _lan_ip() -> str:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))     # no packets sent; just picks the route
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _port_taken(port: int) -> bool:
    """Windows lets two sockets share a port, so probe for a live responder."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.4)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


if __name__ == "__main__":
    import uvicorn
    from pathlib import Path as _Path

    ip = _lan_ip()
    have_cert = _Path("cert.pem").exists() and _Path("key.pem").exists()

    for _p, _label in ((HTTP_PORT, "HTTP"), (HTTPS_PORT, "HTTPS")):
        if _port_taken(_p):
            logger.error("Port %d (%s) is already answering — another process owns it. "
                         "Stop it, or change the port in server.py.", _p, _label)
            raise SystemExit(1)

    logger.info("+-- IBVAP ------------------------------------------------")
    logger.info("|  MONITOR (this PC, no cert warning)")
    logger.info("|     http://localhost:%d/monitor", HTTP_PORT)
    if have_cert:
        logger.info("|  CAMERA (phone, accept the certificate once)")
        logger.info("|     https://%s:%d/camera/0", ip, HTTPS_PORT)
        logger.info("|     https://%s:%d/camera/1   (second phone)", ip, HTTPS_PORT)
    else:
        logger.warning("|  No cert.pem — phone cameras need HTTPS. Run: python gen_cert.py")
    logger.info("|  STATUS   http://localhost:%d/status", HTTP_PORT)
    logger.info("+---------------------------------------------------------")

    # proxy_headers + forwarded_allow_ips: when the server sits behind a tunnel
    # (tunnel.py -> cloudflared / ngrok), trust the X-Forwarded-Proto/-For it
    # sets so request.url.scheme is "https" and client IPs are the real ones.
    # Safe here because the only untunnelled exposure is the LAN.
    _common = dict(log_level="info", access_log=False,
                   proxy_headers=True, forwarded_allow_ips="*")

    async def _serve() -> None:
        servers = [
            uvicorn.Server(uvicorn.Config(
                app, host="0.0.0.0", port=HTTP_PORT, **_common,
            ))
        ]
        if have_cert:
            servers.append(uvicorn.Server(uvicorn.Config(
                app, host="0.0.0.0", port=HTTPS_PORT,
                ssl_certfile="cert.pem", ssl_keyfile="key.pem", **_common,
            )))
        await asyncio.gather(*(s.serve() for s in servers))

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        logger.info("Shutting down")
