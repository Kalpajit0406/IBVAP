"""
nms_check.py --- verify YOLO26n is using the NMS-free inference path.

Run after any Ultralytics upgrade or after switching to a new model format:
    python nms_check.py

Healthy output (NMS-free path active):
    preprocess : ~2-5 ms
    inference  : ~5-15 ms   (varies by batch, GPU warm-up state, and FP16)
    postprocess: ~0-1 ms    <- near-zero is the signal

If postprocess time is non-trivial (> 5 ms), YOLO26n may have fallen back to
legacy NMS handling. Check:
  - Ultralytics version (pinned in requirements.txt)
  - Model format (.pt vs .engine --- both should show near-zero postprocess)
"""
from __future__ import annotations

import sys
import numpy as np
import yaml


def main() -> None:
    cfg = yaml.safe_load(open("config.yaml"))
    m   = cfg["model"]

    weights = m.get("weights", "yolo26n.pt")
    imgsz   = int(m.get("image_size", 640))
    half    = bool(m.get("half", False))
    device  = str(m.get("device", "0"))
    # ultralytics >= 8.4.x: quantize=16 means FP16, quantize=None means FP32
    quantize = 16 if half else None

    print(f"\nModel    : {weights}")
    print(f"imgsz    : {imgsz}")
    print(f"quantize : {quantize}  (16=FP16, None=FP32)")
    print(f"device   : {device}")

    from ultralytics import YOLO
    model = YOLO(weights)

    # One untimed warm-up pass so timing reflects steady-state, not cuDNN tuning.
    blank = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model.predict(blank, imgsz=imgsz, quantize=quantize, device=device, verbose=False)

    results = model.predict(blank, imgsz=imgsz, quantize=quantize, device=device, verbose=False)
    spd = results[0].speed

    pre  = spd.get("preprocess",  0.0)
    inf  = spd.get("inference",   0.0)
    post = spd.get("postprocess", 0.0)

    print(f"\n  preprocess : {pre:.2f} ms")
    print(f"  inference  : {inf:.2f} ms")
    print(f"  postprocess: {post:.2f} ms")
    print(f"  total      : {pre + inf + post:.2f} ms")

    if post < 5.0:
        print("\nNMS-free path is active (postprocess < 5 ms).")
    else:
        print("\nWARN: postprocess time is non-trivial --- NMS fallback may be active.")
        print("  Check: ultralytics version, model format (.pt vs .engine).")
        sys.exit(1)


if __name__ == "__main__":
    main()
