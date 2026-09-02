"""
export_engine.py --- one-time TensorRT engine export for the current GPU.

Run once per machine, then update config.yaml:
    python export_engine.py

Output: yolo26n.engine  (FP16, FIXED shape N x 3 x imgsz x imgsz, built for this GPU)

Next step --- update config.yaml:
    weights: "yolo26n.engine"

IMPORTANT
  * The .engine is hardware-specific: it only runs on the GPU model it was built
    on. Every teammate must run this on their own machine. Keep yolo26n.pt for
    training, fine-tuning, and any non-CUDA box.
  * The engine has a FIXED input shape of (max_batch, 3, imgsz, imgsz). The
    detector always submits exactly max_batch frames, padding short batches
    with blanks — so every GPU call is the identical shape and TRT never
    reconfigures. A *dynamic* engine was tried and rejected: on a 6 GB card its
    profile reserved ~4.6 GB and shape changes stalled the live pipeline.
  * Because the shape is fixed at max_batch, a single moving camera still costs
    one full max_batch pass. At batch 8 that is ~23 ms on an RTX 3050 — cheap
    at the 8 fps detection rate, and it covers up to 8 cameras at once.

Typical export time on RTX 3050: 5-10 minutes.
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import torch
import yaml

PT_WEIGHTS = "yolo26n.pt"
CONFIG     = "config.yaml"


def main() -> None:
    # Load image size from config so export and inference always agree.
    cfg = yaml.safe_load(open(CONFIG))
    m   = cfg["model"]
    imgsz     = int(m.get("image_size", 640))
    device    = int(m.get("device", 0)) if str(m.get("device", "0")).isdigit() else 0
    max_batch = int(m.get("max_batch", 16))

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. TensorRT export requires a CUDA GPU.")
        print("       On CPU-only machines keep weights: 'yolo26n.pt' in config.yaml.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(device)
    vram_gb  = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"\nGPU       : {gpu_name}")
    print(f"VRAM      : {vram_gb:.1f} GB")
    print(f"imgsz     : {imgsz}  (from {CONFIG})")
    print(f"batch     : fixed {max_batch}  (from model.max_batch — detector pads to this)")
    print(f"\nExporting {PT_WEIGHTS} -> TensorRT FP16 engine ...")
    print("(TRT profiles a tactic per layer — 5-10 min is normal)\n")

    from ultralytics import YOLO
    model = YOLO(PT_WEIGHTS)
    model.export(
        format="engine",
        device=device,
        half=True,           # export() still uses half= (quantize= is for predict only)
        imgsz=imgsz,         # must match config.yaml image_size
        dynamic=False,       # FIXED shape — see module docstring for why
        batch=max_batch,    # fixed batch = config.yaml model.max_batch
        workspace=4,         # TRT builder RAM cap in GB; safe on 6 GB VRAM
        verbose=True,
    )

    engine_path = Path(PT_WEIGHTS).with_suffix(".engine")
    if engine_path.exists():
        size_mb = engine_path.stat().st_size / 1e6
        print(f"\n* Engine written: {engine_path}  ({size_mb:.1f} MB)")
        print("\nNext step --- edit config.yaml line 20:")
        print('  weights: "yolo26n.engine"')
        print("\nThen verify with:  python nms_check.py")
    else:
        print("\n! Export finished but .engine file not found where expected.")
        print("  Check the Ultralytics output above for the actual output path.")


if __name__ == "__main__":
    main()
