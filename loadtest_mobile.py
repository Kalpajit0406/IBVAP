"""
loadtest_mobile.py — prove the 8-device target, with measurements.

Runs entirely against a live server (start it first: python server.py).
Spins up N WebSocket clients that behave exactly like phone browsers — each
sends the JSON hello, then streams 1280x720 JPEG frames at a target fps — then
samples /status and nvidia-smi once a second and prints a report.

    python loadtest_mobile.py                       # 8 moving streams, 30 s
    python loadtest_mobile.py --cams 8 --seconds 45
    python loadtest_mobile.py --static              # 8 frozen streams (no-motion test)
    python loadtest_mobile.py --mix 5               # 5 moving + 3 static

Reports per-stream delivered fps, aggregate pipeline fps, inference ms/frame,
motion-gate skip %, GPU utilisation and VRAM — measured, not assumed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import websockets

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

WSS = "wss://127.0.0.1:8443"
HTTP = "http://127.0.0.1:8090"
VIDEO_DIR = Path(__file__).parent / "test_videos"

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

_STOP = False


def jpg(frame: np.ndarray, q: int = 55) -> bytes:
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def load_clip(idx: int, want: int, fps: int) -> list[np.ndarray]:
    vids = sorted(VIDEO_DIR.glob("*.mp4"))
    frames: list[np.ndarray] = []
    if vids:
        cap = cv2.VideoCapture(str(vids[idx % len(vids)]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, (idx * 41) % 200)
        while len(frames) < want:
            ok, f = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = cap.read()
                if not ok:
                    break
            frames.append(cv2.resize(f, (1280, 720)))
        cap.release()
    if not frames:  # fall back to a moving synthetic frame
        for i in range(want):
            img = np.full((720, 1280, 3), 50, np.uint8)
            x = 100 + (i * 12) % 1000
            cv2.rectangle(img, (x, 300), (x + 120, 560), (200, 200, 200), -1)
            frames.append(img)
    return frames


def static_frame(idx: int) -> np.ndarray:
    img = np.full((720, 1280, 3), 60, np.uint8)
    cv2.rectangle(img, (80, 80), (1200, 640), (45, 45, 45), -1)
    cv2.putText(img, f"STATIC CAM-{idx:02d}", (150, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (110, 110, 110), 3)
    return img


def _prep_blobs(cam_id: int, fps: int, static: bool) -> list[bytes]:
    """Blocking: decode + JPEG-encode a short replay loop. Runs in a thread."""
    if static:
        return [jpg(static_frame(cam_id))]
    # A ~4 s loop is plenty and keeps memory + prep time small.
    return [jpg(f) for f in load_clip(cam_id, want=fps * 4, fps=fps)]


async def client(cam_id: int, fps: int, static: bool, stats: dict,
                 start_delay: float = 0.0) -> None:
    url = f"{WSS}/ws/camera/{cam_id}"
    period = 1.0 / fps

    # Stagger + off-loop prep so 8 concurrent starts don't block the handshake.
    await asyncio.sleep(start_delay)
    blobs = await asyncio.to_thread(_prep_blobs, cam_id, fps, static)

    try:
        async with websockets.connect(url, ssl=_CTX, max_size=8_000_000,
                                      open_timeout=15, ping_interval=20) as ws:
            await ws.send(json.dumps({
                "type": "hello",
                "kind": "static" if static else "mobile",
                "label": f"{'static' if static else 'phone'}-{cam_id:02d}",
                "reqWidth": 1280, "reqHeight": 720, "reqFps": 24,
                "width": 1280, "height": 720, "fps": float(fps),
            }))
            i = 0
            next_at = time.perf_counter()
            while not _STOP:
                await ws.send(blobs[i % len(blobs)])
                stats["sent"][cam_id] += 1
                i += 1
                next_at += period
                d = next_at - time.perf_counter()
                await asyncio.sleep(d if d > 0 else 0)
                if d <= 0:
                    next_at = time.perf_counter()
    except Exception as e:
        print(f"  CAM-{cam_id:02d} client error: {e}", flush=True)


def gpu_sample() -> tuple[float, float]:
    """(utilisation %, VRAM used MiB) via nvidia-smi, or (nan, nan)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2).strip().splitlines()[0]
        u, m = (x.strip() for x in out.split(","))
        return float(u), float(m)
    except Exception:
        return float("nan"), float("nan")


def http_status() -> dict:
    import urllib.request
    with urllib.request.urlopen(f"{HTTP}/status", timeout=3) as r:
        return json.loads(r.read().decode())


async def sampler(seconds: int, samples: list) -> None:
    for _ in range(seconds):
        await asyncio.sleep(1.0)
        try:
            s = http_status()
        except Exception as e:
            print(f"  status poll failed: {e}", flush=True)
            continue
        gu, gm = gpu_sample()
        p = s.get("pipeline", {})
        devs = s.get("devices", [])
        samples.append({
            "t": time.time(),
            "fps_in": s.get("fps", 0.0),
            "infer_ms": s.get("inference_ms", 0.0),
            "mean_batch": p.get("mean_batch", 0.0),
            "gpu_saving_pct": p.get("gpu_saving_pct", 0.0),
            "frames_inferred": p.get("frames_inferred", 0),
            "frames_gated": p.get("frames_gated", 0),
            "gpu_util": gu, "vram_mb": gm,
            "dev_fps": {d["cam_id"]: d.get("delivered_fps", 0.0) for d in devs},
            "dev_state": {d["cam_id"]: d.get("state") for d in devs},
            "dev_infer": {d["cam_id"]: d.get("infer_count", 0) for d in devs},
            "dev_kind": {d["cam_id"]: d.get("kind") for d in devs},
        })
        last = samples[-1]
        print(f"  t+{len(samples):2d}s  in={last['fps_in']:6.1f}fps  "
              f"batch={last['mean_batch']:.1f}  {last['infer_ms']:5.1f}ms/f  "
              f"gate_skip={last['gpu_saving_pct']:4.0f}%  "
              f"gpu={last['gpu_util']:4.0f}%  vram={last['vram_mb']:.0f}MB",
              flush=True)


async def main() -> int:
    global _STOP
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, default=8)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--static", action="store_true",
                    help="all streams frozen — no-motion gate test")
    ap.add_argument("--mix", type=int, default=None,
                    help="first N cams moving, the rest static")
    args = ap.parse_args()

    try:
        http_status()
    except Exception:
        print(f"Server not reachable at {HTTP}. Start it first:  python server.py")
        return 1

    def is_static(cam: int) -> bool:
        if args.static:
            return True
        if args.mix is not None:
            return cam >= args.mix
        return False

    kinds = ["static" if is_static(c) else "moving" for c in range(args.cams)]
    print(f"\nLoad test: {args.cams} streams @ {args.fps} fps for {args.seconds}s")
    print(f"  {kinds.count('moving')} moving, {kinds.count('static')} static")
    print(f"  target aggregate = {args.cams * args.fps} fps to keep up\n")

    stats = {"sent": {c: 0 for c in range(args.cams)}}
    t0 = time.perf_counter()

    clients = [asyncio.create_task(
                   client(c, args.fps, is_static(c), stats, start_delay=0.15 * c))
               for c in range(args.cams)]
    samples: list = []
    # Wait for every stream to connect + prep + the server to settle.
    await asyncio.sleep(5.0 + 0.15 * args.cams)
    print("  --- sampling ---")
    await sampler(args.seconds, samples)

    _STOP = True
    await asyncio.sleep(0.5)
    for t in clients:
        t.cancel()
    elapsed = time.perf_counter() - t0

    if not samples:
        print("\nNo samples collected — is the server healthy?")
        return 1

    # ── Report ──────────────────────────────────────────────────────────────
    def mean(key):
        vals = [s[key] for s in samples if not _isnan(s[key])]
        return statistics.mean(vals) if vals else float("nan")

    def _isnan(x):
        return isinstance(x, float) and x != x

    warm = samples[len(samples) // 3:]     # drop the first third
    def wmean(key):
        vals = [s[key] for s in warm if not _isnan(s[key])]
        return statistics.mean(vals) if vals else float("nan")

    sent_total = sum(stats["sent"].values())
    per_stream_sent = sent_total / args.cams / elapsed

    last = samples[-1]
    print("\n" + "=" * 76)
    print(f"{'RESULT — ' + str(args.cams) + ' mobile streams':<40}"
          f"{'measured (steady-state mean)':>36}")
    print("-" * 76)
    print(f"{'Client send rate (per stream)':<40}{per_stream_sent:>30.1f} fps")
    print(f"{'Pipeline ingest rate (aggregate)':<40}{wmean('fps_in'):>30.1f} fps")
    print(f"{'Pipeline ingest (per stream)':<40}{wmean('fps_in')/args.cams:>30.1f} fps")
    print(f"{'Inference cost':<40}{wmean('infer_ms'):>27.1f} ms/frame")
    print(f"{'Mean inference batch':<40}{wmean('mean_batch'):>30.1f}")
    print(f"{'Motion gate — frames skipped':<40}{wmean('gpu_saving_pct'):>30.0f} %")
    print(f"{'GPU utilisation':<40}{wmean('gpu_util'):>30.0f} %")
    print(f"{'VRAM used':<40}{wmean('vram_mb'):>27.0f} MB")
    print("-" * 76)
    need = args.cams * args.fps
    got = wmean("fps_in")
    verdict = "MET" if got >= need * 0.95 else "NOT MET"
    print(f"{'Real-time target ' + str(need) + ' fps':<40}{verdict:>30}")

    if args.static or args.mix is not None:
        span_s = max(samples[-1]["t"] - samples[0]["t"], 1e-6)
        static_ids = [c for c in range(args.cams) if is_static(c)]
        moving_ids = [c for c in range(args.cams) if not is_static(c)]

        def infer_delta(ids):
            d0, d1 = samples[0]["dev_infer"], samples[-1]["dev_infer"]
            return sum(max(d1.get(c, 0) - d0.get(c, 0), 0) for c in ids)

        gg0, gg1 = samples[0]["frames_gated"], samples[-1]["frames_gated"]
        print("-" * 76)
        print(f"{'NO-MOTION CHECK over %.0fs window' % span_s:<40}")
        print(f"{'  frames motion-gated away (all streams)':<40}{gg1 - gg0:>30}")
        if static_ids:
            si = infer_delta(static_ids)
            per = si / len(static_ids) / span_s
            print(f"{'  YOLO passes on STATIC streams':<40}{si:>30}")
            print(f"{'  -> per static stream':<40}"
                  f"{'%.2f fps  (vs %d sent)' % (per, args.fps):>30}")
            print(f"\n  Expect ≈ 1 fps/stream — only the force_every keep-alive "
                  f"sweep.\n  {'PASS' if per < 3.0 else 'CHECK'}: "
                  f"static scene is {100*(1 - per/args.fps):.0f}% cheaper than "
                  f"running every frame.")
        if moving_ids and static_ids:
            mi = infer_delta(moving_ids)
            print(f"\n{'  YOLO passes on MOVING streams (context)':<40}{mi:>30}"
                  f"  ({mi/len(moving_ids)/span_s:.1f} fps/stream)")
    print("=" * 76)

    print("\nPer-stream delivered fps (last sample):")
    for cam in sorted(last["dev_fps"]):
        print(f"  CAM-{cam:02d}  {last['dev_fps'][cam]:5.1f} fps  [{last['dev_state'][cam]}]")

    return 0 if verdict == "MET" else 2


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        _STOP = True
        print("\nStopped.")
