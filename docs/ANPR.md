# ANPR — number-plate recognition

Ported from **https://github.com/anindya-mukhopadhyay/ANPR** (MIT, © 2023 BAPPY
AHMED). That project's YOLOv8 plate model, ROI preprocessing, and EasyOCR
fallback ladder are reused; the pipeline integration is IBVAP's.

## How it runs

```
main detector ──► vehicle boxes (car/motorcycle/bus/truck, with track ids)
                        │
                        ▼   every 3rd detection tick (~2.7 fps)
     YOLOv8 "license_plate" on each VEHICLE CROP  (models/license_plate_detector.pt)
                        │  plate box → mapped back to full-frame coords
                        ▼   bounded queue (drop-oldest)
     EasyOCR worker thread  ── upscale · greyscale · CLAHE ──► readtext()
                        │  clean → region-format check (IN / XX)
                        ▼
     per-track best reading  ──► drawn under the vehicle box
                             ──► data/plates/plates.csv + crop image
                             ──► SHA-256 hash-chained evidence log
                             ──► dashboard alert-log row + "plates" counter
```

Everything after the vehicle boxes is **asynchronous** — the OCR worker never
touches the real-time detection budget. A vehicle that already has a confident
reading (≥ 0.55) is skipped until the cooldown elapses.

## Setup

```bash
pip install easyocr          # ~200 MB of deps; first OCR run downloads ~64 MB of models
```

`config.yaml` → `anpr.enabled: true` (already the default). If `easyocr` or
`models/license_plate_detector.pt` is missing, `AnprEngine` logs one warning and
disables itself — nothing else breaks.

## Config (`config.yaml` `anpr:`)

| key | meaning |
|---|---|
| `enabled` | master switch |
| `weights` | plate-detector weights (shipped in `models/`) |
| `region` | `IN` = Indian format `SS RR L(L)(L) NNNN` (e.g. `WB06AB1234`); `XX` = any 4–10 alphanumerics with a letter and a digit. A format match boosts the stored confidence by 0.15. |
| `yolo_conf` | plate-detector confidence (0.30) |
| `ocr_min_conf` | min mean OCR confidence to **log** a plate (0.20) |
| `min_plate_area` | px² floor on plate boxes (300) |
| `plate_detect_every` | run the plate stage every Nth detection tick (3) |
| `plate_imgsz` | plate-detector input size (320 — plates are small) |
| `cooldown_s` | don't re-log the same string within this window (8 s) |
| `workers` | EasyOCR worker threads (1) |

## Output

- **`data/plates/plates.csv`** — `timestamp, cam_id, track_id, plate, confidence, valid, crop`
- **`data/plates/<ts>_cam<n>_<PLATE>.jpg`** — the plate crop (best effort)
- **Evidence chain** — a `{"type": "plate", ...}` record per confirmed plate
- **`/status` `anpr`** — `plates_detected / ocr_runs / plates_logged / queue_drops` + `active` readings
- **Dashboard** — yellow `PLATE` rows in the alert log, a `plates` figure in the header

## Tuning notes

- **Missing plates?** Lower `yolo_conf` to 0.20, raise `plate_detect_every` to 2,
  make sure the camera's sub-stream is sharp enough — plates need ~80 px width
  to OCR. A main stream gives more pixels but costs CPU.
- **Wrong characters?** Indian plates: `0/O`, `1/I`, `8/B`, `5/S` confusions are
  common. The `region: IN` check rejects strings that can't be a valid plate,
  but doesn't correct them. A plate-specific OCR model (e.g. fast-plate-ocr) is
  the upgrade path.
- **CPU hot with many vehicles?** Raise `plate_detect_every`, keep `workers: 1`,
  or gate ANPR to specific cameras (a `cameras: [0, 2]` filter is a small
  addition to `AnprEngine.submit`).

## Limitations

2-D, single-frame OCR with no plate-super-resolution and no multi-frame voting.
Good for a demo ("that car's plate is WB06AB1234, logged at 14:32, here's the
crop"); not production ANPR. The evidence record is the valuable part — it's
tamper-evident and timestamped.
