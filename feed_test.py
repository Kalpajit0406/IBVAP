"""
feed_test.py — replay video files into IBVAP as if they were live CCTV feeds.

Run AFTER server.py is up:

    python feed_test.py --src test_videos --cams 4          # 4 clips, native fps
    python feed_test.py --src test_videos --cams 8 --fps 24 # 8-device demo load
    python feed_test.py --src E:\\my_footage --cams 6
    python feed_test.py --static --cams 1                   # frozen frame (no-motion test)

Each camera gets its own video file (cycling if there are fewer files than
cameras) and streams it in real time, looping at the end. Every client first
sends the same JSON "hello" a phone browser sends, then streams JPEG frames
over the WebSocket — the server cannot tell replayed footage from a real phone.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import websockets

# Set by main() from --latency-sim: milliseconds to backdate each frame's
# capture timestamp, so the server's latency badge / "DELAYED" overlay can be
# exercised without a real slow uplink.
LATENCY_SIM_MS = 0
SEND_TIMESTAMP = True     # prepend the 8-byte capture-ms header a phone sends

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

SERVER = "wss://127.0.0.1:8443"
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

DEFAULT_SRC = Path(__file__).parent / "test_videos"

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def encode(frame, quality: int = 80) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    jpeg = buf.tobytes()
    if not SEND_TIMESTAMP:
        return jpeg
    cap_ms = int(time.time() * 1000) - LATENCY_SIM_MS
    return struct.pack("<Q", cap_ms) + jpeg


async def send_hello(ws, cam_id: int, w: int, h: int, fps: float,
                     kind: str = "replay") -> None:
    """Mirror the browser handshake so the server logs negotiated res/fps."""
    await ws.send(json.dumps({
        "type": "hello", "kind": kind,
        "label": f"replay-{cam_id:02d}" if kind == "replay" else f"static-{cam_id:02d}",
        "reqWidth": 1280, "reqHeight": 720, "reqFps": 24,
        "width": w, "height": h, "fps": fps,
        "t0": int(time.time() * 1000),      # clock-sync anchor for latency
    }))


def static_frame(cam_id: int) -> np.ndarray:
    """A fixed 1280x720 scene with no moving content — for the no-motion test."""
    img = np.full((720, 1280, 3), 60, dtype=np.uint8)
    cv2.rectangle(img, (80, 80), (1200, 640), (45, 45, 45), -1)
    cv2.putText(img, f"STATIC CAM-{cam_id:02d}  (no motion)", (120, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (110, 110, 110), 3)
    return img


async def stream_static(ws, cam_id: int, fps: float, counter: dict) -> None:
    """Push one unchanging frame forever — the motion gate should skip nearly all."""
    jpeg = encode(static_frame(cam_id), quality=80)
    period = 1.0 / fps
    next_at = time.perf_counter()
    while True:
        if jpeg:
            await ws.send(jpeg)
            counter["sent"] += 1
            if counter["sent"] % 240 == 0:
                el = time.perf_counter() - counter["start"]
                print(f"  CAM-{cam_id:02d} static {counter['sent']} frames "
                      f"({counter['sent']/el:.1f} fps out)", flush=True)
        next_at += period
        delay = next_at - time.perf_counter()
        await asyncio.sleep(delay if delay > 0 else 0)
        if delay <= 0:
            next_at = time.perf_counter()


async def stream_video(ws, path: Path, cam_id: int, fps: float | None,
                       counter: dict) -> None:
    """Play one file through the socket, pacing to wall-clock frame rate."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  CAM-{cam_id:02d} cannot open {path.name}", flush=True)
        return

    native = cap.get(cv2.CAP_PROP_FPS) or 24.0
    target = fps or native
    period = 1.0 / target
    next_at = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break                       # end of clip; caller loops

            jpeg = encode(frame)
            if jpeg:
                await ws.send(jpeg)
                counter["sent"] += 1
                if counter["sent"] % 240 == 0:
                    elapsed = time.perf_counter() - counter["start"]
                    print(f"  CAM-{cam_id:02d} {counter['sent']} frames "
                          f"({counter['sent'] / elapsed:.1f} fps out)", flush=True)

            # Pace to the target rate; drop the deficit if we fall behind so a
            # slow moment does not turn into an ever-growing backlog.
            next_at += period
            delay = next_at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_at = time.perf_counter()
    finally:
        cap.release()


async def stream_images(ws, images: list[Path], cam_id: int, fps: float,
                        offset: int, counter: dict) -> None:
    period = 1.0 / fps
    rotated = images[offset:] + images[:offset]
    for img_path in rotated:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        frame = cv2.resize(frame, (1280, 720))
        jpeg = encode(frame)
        if jpeg:
            await ws.send(jpeg)
            counter["sent"] += 1
        await asyncio.sleep(period)


async def run_camera(cam_id: int, videos: list[Path], images: list[Path],
                     fps: float | None, offset: int, static: bool = False) -> None:
    url = f"{SERVER}/ws/camera/{cam_id}"
    counter = {"sent": 0, "start": time.perf_counter()}
    my_video = None if static else (videos[cam_id % len(videos)] if videos else None)

    while True:
        try:
            async with websockets.connect(url, ssl=_CTX, max_size=8_000_000,
                                          ping_interval=20) as ws:
                await send_hello(ws, cam_id, 1280, 720, fps or 24.0,
                                 kind="static" if static else "replay")
                if static:
                    print(f"  CAM-{cam_id:02d} streaming STATIC frame", flush=True)
                    await stream_static(ws, cam_id, fps or 24.0, counter)
                    continue
                src_name = my_video.name if my_video else f"{len(images)} images"
                print(f"  CAM-{cam_id:02d} streaming {src_name}", flush=True)
                while True:                 # loop the source forever
                    if my_video:
                        await stream_video(ws, my_video, cam_id, fps, counter)
                    else:
                        await stream_images(ws, images, cam_id, fps or 10.0,
                                            offset, counter)

        except websockets.exceptions.WebSocketException as e:
            print(f"  CAM-{cam_id:02d} socket closed ({e}) - reconnecting in 2s",
                  flush=True)
            await asyncio.sleep(2)
        except OSError as e:
            print(f"  CAM-{cam_id:02d} cannot reach {SERVER} ({e}) - "
                  f"is server.py running? Retrying in 2s", flush=True)
            await asyncio.sleep(2)
        except Exception:
            import traceback
            print(f"  CAM-{cam_id:02d} FEEDER ERROR:", flush=True)
            traceback.print_exc()
            return


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="folder of videos (or images) to replay")
    ap.add_argument("--cams", type=int, default=4)
    ap.add_argument("--fps", type=float, default=None,
                    help="override source frame rate")
    ap.add_argument("--static", action="store_true",
                    help="stream a frozen frame per camera (no-motion gate test)")
    ap.add_argument("--latency-sim", type=int, default=0, metavar="MS",
                    help="backdate each frame's timestamp by MS ms to exercise "
                         "the server's latency badge / DELAYED overlay")
    args = ap.parse_args()

    global LATENCY_SIM_MS
    LATENCY_SIM_MS = max(0, args.latency_sim)
    if LATENCY_SIM_MS:
        print(f"Simulating {LATENCY_SIM_MS} ms of capture latency on every frame.")

    if args.static:
        print(f"Feeding {args.cams} STATIC camera(s) at {args.fps or 24} fps "
              f"— motion gating should skip almost every frame.  (Ctrl+C to stop)\n")
        await asyncio.gather(*[
            run_camera(cam, [], [], args.fps, 0, static=True)
            for cam in range(args.cams)
        ])
        return 0

    if not args.src.exists():
        print(f"Source not found: {args.src}")
        print("Build the test clips first:  python make_test_videos.py")
        return 1

    if args.src.is_file():
        videos = [args.src] if args.src.suffix.lower() in VID_EXT else []
        images = [args.src] if args.src.suffix.lower() in IMG_EXT else []
    else:
        videos = sorted(p for p in args.src.rglob("*") if p.suffix.lower() in VID_EXT)
        images = sorted(p for p in args.src.rglob("*") if p.suffix.lower() in IMG_EXT)

    if not videos and not images:
        print(f"No videos or images under {args.src}")
        print("Build the test clips first:  python make_test_videos.py")
        return 1

    print(f"Source : {args.src}")
    if videos:
        print(f"Videos : {len(videos)} file(s) - each camera gets one, cycling")
    else:
        print(f"Images : {len(images)} (no videos found; replaying stills)")
    rate = f"{args.fps} fps" if args.fps else "native frame rate"
    print(f"Feeding {args.cams} camera(s) at {rate}   (Ctrl+C to stop)\n")

    stride = max(1, len(images) // max(args.cams, 1)) if images else 1
    await asyncio.gather(*[
        run_camera(cam, videos, images, args.fps,
                   (cam * stride) % max(len(images), 1))
        for cam in range(args.cams)
    ])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nStopped.")
