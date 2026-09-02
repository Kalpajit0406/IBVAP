"""
benchmark.py — measure what the three optimisations actually buy us.

Runs the same footage through four configurations and reports throughput:

    baseline    one frame at a time, every frame, no gating   (the naive loop)
    +batching   all cameras in one forward pass
    +rate       detection at detect_fps, tracks carried between passes
    +motion     static scenes never reach the GPU

    python benchmark.py                       # 4 streams from test_videos
    python benchmark.py --cams 8 --seconds 20
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.detector import Detector

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_frames(videos: list[Path], cams: int, seconds: int, fps: int) -> list[list[np.ndarray]]:
    """Decode once, up front, so decoding never pollutes the timings."""
    want = seconds * fps
    per_cam: list[list[np.ndarray]] = []
    for cam in range(cams):
        path = videos[cam % len(videos)]
        cap = cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        # Stagger each camera into the clip so the streams are not identical.
        cap.set(cv2.CAP_PROP_POS_FRAMES, (cam * 37) % 200)
        while len(frames) < want:
            ok, f = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = cap.read()
                if not ok:
                    break
            frames.append(f)
        cap.release()
        per_cam.append(frames)
        print(f"  CAM-{cam:02d}  {len(frames)} frames from {path.name}")
    return per_cam


def run(name: str, cfg: dict, per_cam: list[list[np.ndarray]],
        batched: bool, warm_frames: int = 48) -> dict:
    det = Detector(cfg)
    n_cams = len(per_cam)
    n_frames = min(len(f) for f in per_cam)

    def feed(i: int) -> None:
        items = [(cam, time.monotonic(), per_cam[cam][i]) for cam in range(n_cams)]
        if batched:
            det.process_batch(items)
        else:
            for it in items:
                det.process_batch([it])       # one call per camera

    # Untimed warm-up. cuDNN tunes kernels per input shape on first use, so
    # without this the first configuration measured absorbs the tuning cost of
    # every batch shape and looks far slower than it really is.
    for i in range(min(warm_frames, n_frames)):
        feed(i)
    det.stats = type(det.stats)()             # reset counters after warm-up

    t0 = time.perf_counter()
    for i in range(n_frames):
        feed(i)
    elapsed = time.perf_counter() - t0

    total = n_frames * n_cams
    s = det.stats
    return {
        "name": name,
        "elapsed": elapsed,
        "total_frames": total,
        "throughput": total / elapsed,
        "per_cam_fps": (total / elapsed) / n_cams,
        "gpu_frames": s.frames_inferred,
        "passes": s.detector_passes,
        "mean_batch": s.mean_batch,
        "gated": s.frames_gated,
        "rate_skipped": s.frames_rate_skipped,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=15)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--src", type=Path, default=Path(__file__).parent / "test_videos")
    args = ap.parse_args()

    videos = sorted(p for p in args.src.rglob("*") if p.suffix.lower() in VID_EXT)
    if not videos:
        print(f"No videos under {args.src}")
        print("Build them first:  python make_test_videos.py")
        return 1

    base = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
    h, w = None, None

    print(f"\nDecoding {args.cams} x {args.seconds}s @ {args.fps} fps ...")
    per_cam = load_frames(videos, args.cams, args.seconds, args.fps)
    if not per_cam or not per_cam[0]:
        print("No frames decoded.")
        return 1
    h, w = per_cam[0][0].shape[:2]

    offered = min(len(f) for f in per_cam) * args.cams
    print(f"\nResolution : {w}x{h}")
    print(f"Streams    : {args.cams}")
    print(f"Frames     : {offered} total offered to the pipeline")
    print(f"Real time  : {args.cams * args.fps} fps needed to keep up\n")

    def variant(**over):
        c = copy.deepcopy(base)
        c["model"].update(over)
        return c

    configs = [
        ("baseline   (1 frame/call, every frame)",
         variant(motion_gating=False, detect_fps=args.fps, stream_fps=args.fps), False),
        ("+ batching (all cams, one pass)",
         variant(motion_gating=False, detect_fps=args.fps, stream_fps=args.fps), True),
        ("+ rate cap (detect 8fps, track 24fps)",
         variant(motion_gating=False, detect_fps=8, stream_fps=args.fps), True),
        ("+ motion gate (skip static scenes)",
         variant(motion_gating=True, detect_fps=8, stream_fps=args.fps), True),
    ]

    results = []
    for name, cfg, batched in configs:
        print(f"Running {name} ...", flush=True)
        results.append(run(name, cfg, per_cam, batched))

    need = args.cams * args.fps
    print("\n" + "=" * 92)
    print(f"{'configuration':<40} {'fps':>8} {'per-cam':>9} {'GPU frames':>12} "
          f"{'batch':>7} {'real-time':>11}")
    print("-" * 92)
    for r in results:
        verdict = "YES" if r["per_cam_fps"] >= args.fps else "no"
        print(f"{r['name']:<40} {r['throughput']:>8.1f} {r['per_cam_fps']:>9.1f} "
              f"{r['gpu_frames']:>12} {r['mean_batch']:>7.1f} {verdict:>11}")
    print("=" * 92)

    b, best = results[0], results[-1]
    print(f"\nBaseline    : {b['throughput']:.1f} fps aggregate "
          f"({b['per_cam_fps']:.1f} per camera), GPU saw {b['gpu_frames']} frames")
    print(f"Optimised   : {best['throughput']:.1f} fps aggregate "
          f"({best['per_cam_fps']:.1f} per camera), GPU saw {best['gpu_frames']} frames")
    print(f"Speed-up    : {best['throughput'] / b['throughput']:.2f}x")
    print(f"GPU work    : {100 * (1 - best['gpu_frames'] / max(b['gpu_frames'], 1)):.0f}% fewer frames inferred")
    print(f"Real time   : need {need} fps for {args.cams} streams @ {args.fps} fps -> "
          f"{'MET' if best['throughput'] >= need else 'NOT MET'}")

    headroom = best["throughput"] / args.fps
    print(f"Headroom    : this GPU could carry ~{headroom:.0f} streams at {args.fps} fps\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
