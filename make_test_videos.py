"""
make_test_videos.py — build 720p / 24 fps clips to exercise the pipeline.

Real CCTV footage is mostly static with occasional activity, which is exactly
the pattern motion gating is designed to exploit. These clips reproduce that:
long still stretches interrupted by movement, so the gate's skip rate is
representative rather than flattering.

Frames come from the Kaggle human-detection dataset (real people and vehicles),
upscaled to 1280x720 and animated with slow pans so tracking has something to
follow.

    python make_test_videos.py                 # 4 clips, 60 s each
    python make_test_videos.py --clips 6 --seconds 30
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

DATASET = Path(
    r"C:\Users\kalpa\.cache\kagglehub\datasets\constantinwerner"
    r"\human-detection-dataset\versions\5\human detection dataset"
)
OUT_DIR = Path(__file__).parent / "test_videos"
W, H, FPS = 1280, 720, 24


def fit_720p(img: np.ndarray) -> np.ndarray:
    """Scale to cover 1280x720, then centre-crop — no letterboxing."""
    ih, iw = img.shape[:2]
    scale = max(W / iw, H / ih)
    resized = cv2.resize(img, (int(iw * scale + 0.5), int(ih * scale + 0.5)),
                         interpolation=cv2.INTER_CUBIC)
    rh, rw = resized.shape[:2]
    x, y = (rw - W) // 2, (rh - H) // 2
    return resized[y:y + H, x:x + W]


def build_clip(out_path: Path, images: list[Path], seconds: int, seed: int) -> dict:
    rng = random.Random(seed)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {out_path}")

    total = seconds * FPS
    written = static_frames = 0

    while written < total:
        src = cv2.imread(str(images[rng.randrange(len(images))]))
        if src is None:
            continue
        base = fit_720p(src)

        # A scene = a still hold, then a slow pan. Mirrors real CCTV: nothing
        # happens for a while, then something moves through frame.
        hold = rng.randint(FPS * 2, FPS * 5)          # 2-5 s of no motion
        pan = rng.randint(FPS * 2, FPS * 4)           # 2-4 s of movement

        for _ in range(min(hold, total - written)):
            # Sensor noise only — below the motion gate's threshold.
            noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
                0, 1.1, (H, W, 1)).astype(np.int16)
            writer.write(np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8))
            written += 1
            static_frames += 1
        if written >= total:
            break

        # Ken-Burns pan: crop a window and walk it across the frame.
        zoom = 1.18
        cw, ch = int(W / zoom), int(H / zoom)
        x0, y0 = rng.randint(0, W - cw), rng.randint(0, H - ch)
        x1, y1 = rng.randint(0, W - cw), rng.randint(0, H - ch)
        for i in range(min(pan, total - written)):
            t = i / max(pan - 1, 1)
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)
            window = base[y:y + ch, x:x + cw]
            writer.write(cv2.resize(window, (W, H), interpolation=cv2.INTER_LINEAR))
            written += 1

    writer.release()
    return {
        "path": out_path,
        "frames": written,
        "static_pct": 100.0 * static_frames / written if written else 0.0,
        "size_mb": out_path.stat().st_size / 1e6,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--src", type=Path, default=DATASET)
    args = ap.parse_args()

    if not args.src.exists():
        print(f"Source images not found: {args.src}")
        return 1

    images = sorted(p for p in args.src.rglob("*")
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        print(f"No images under {args.src}")
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    print(f"Source   : {len(images)} images")
    print(f"Building : {args.clips} clips, {args.seconds}s each, {W}x{H} @ {FPS} fps")
    print(f"Output   : {OUT_DIR}\n")

    for i in range(args.clips):
        out = OUT_DIR / f"cam{i:02d}_720p24.mp4"
        print(f"  cam{i:02d} ... ", end="", flush=True)
        info = build_clip(out, images, args.seconds, seed=1000 + i)
        print(f"{info['frames']} frames, {info['static_pct']:.0f}% static, "
              f"{info['size_mb']:.1f} MB")

    print(f"\nDone. Stream them with:\n"
          f"  python feed_test.py --src test_videos --cams {args.clips} --fps {FPS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
