# IBVAP — AI-Based Intelligent Video Analytics Platform for Border Surveillance

SIH 2026 · PS 26187 · Ministry of Home Affairs / Sashastra Seema Bal (SSB)

A software-only analytics layer on top of existing CCTV. Multiple camera
streams (RTSP or phones via a browser link — no app install) are batched
through one **YOLO26n** detector on a single GPU, tracked with **ByteTrack**,
scored for risk, and shown on a live command dashboard. Events are written to a
SHA-256 hash-chained evidence log.

## Highlights

- **Batched multi-stream inference** — every camera's frame goes through the
  model in one forward pass (the nvstreammux idea). Measured 8×720p@24fps at
  ~192 fps aggregate on an RTX 3050 6 GB, GPU ~35 %.
- **Two-thread pipeline** — a fixed-tick muxer feeds a separate inference
  worker, so a slow GPU pass never backpressures the camera sockets.
- **Motion gating** — a cheap frame-difference pre-filter; a static scene never
  reaches the GPU (~80 % of frames skipped on real footage).
- **TensorRT FP16 engine** (`yolo26n.engine`) — the default backend, 4.8× the
  batch-1 latency of PyTorch.
- **Real CCTV ingestion** — RTSP / NVR channels / ONVIF cameras pulled on their
  own decode threads (TCP, auto-reconnect, stall watchdog), in the same process
  and pipeline as the phones. `rtsp_probe.py` + `discover_cameras.py` to set up.
- **Phone ingestion with latency control** — browser `getUserMedia` → JPEG over
  WebSocket, adaptive to uplink backpressure, with a per-stream latency badge.
- **Posture rules** — lying / crouch / scan / arms-up / two-handed weapon-ready
  grip (the last forces a Critical alert).
- **ANPR** — YOLOv8 plate detector on vehicle crops + threaded EasyOCR, Indian
  plate-format cleanup, logged to a CSV and the hash-chained evidence trail.
  (Ported from [anindya-mukhopadhyay/ANPR](https://github.com/anindya-mukhopadhyay/ANPR), MIT.)
- **Off-LAN access** — `tunnel.py` (ngrok or cloudflared), including your own
  domain via a cloudflared named tunnel (see `docs/CUSTOM_DOMAIN.md`).

## Quick start

```bash
# 1. PyTorch with CUDA 12.6, then the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2. one-time assets
python gen_cert.py            # self-signed cert for phone camera access on the LAN
python make_test_videos.py    # 4× 720p/24fps demo clips
python export_engine.py       # TensorRT engine for THIS GPU (optional; falls back to .pt)

# 3. run the demo — server + 4 replayed streams + dashboard
python run_demo.py --cams 4
#   dashboard  → http://localhost:8090/monitor
#   phone (LAN)→ https://<lan-ip>:8443/cam/0
#   phone (any network) → add --tunnel, open the URL it prints
```

## Layout

| path | |
|---|---|
| `server.py` | FastAPI: WS + RTSP intake, muxer + inference-worker threads, MJPEG mosaic, `/status` |
| `src/rtsp_capture.py` | CCTV/RTSP/NVR puller — one decode thread per camera, TCP, auto-reconnect |
| `src/anpr.py` | number-plate recognition — plate detector on vehicle crops + threaded EasyOCR |
| `rtsp_probe.py` · `discover_cameras.py` | validate a camera URL · find ONVIF cameras on the LAN |
| `src/detector.py` | batched YOLO26n + per-camera ByteTrack + track carry-forward + pose pass |
| `src/motion_gate.py` | frame-difference pre-filter |
| `src/posture.py` | pose model on person crops + geometry rules |
| `src/risk_engine.py` | zone × time × behaviour → 0–100 score |
| `src/evidence.py` · `src/event_store.py` | SHA-256 hash chain · SQLite store-and-forward |
| `run_demo.py` | one command: server + replay + browser (`--tunnel` for off-LAN) |
| `feed_test.py` · `loadtest_mobile.py` · `benchmark.py` · `diagnose.py` | test / measure tools |
| `tunnel.py` · `cloudflared.example.yml` | off-LAN phone access |
| `config.yaml` | streams, model, throughput, risk, pose thresholds |

## Connecting real CCTV

```bash
python discover_cameras.py --user admin --pass <pw>     # find ONVIF cameras on the LAN
python rtsp_probe.py "rtsp://admin:pass@10.0.0.50:554/Streaming/Channels/102"   # test one URL
```

Then add the (sub-stream) URLs to `config.yaml` `streams:` and run
`python server.py`. Full guide — how CCTV/NVRs are wired, vendor URL tables,
what to ask college IT for — is **`docs/CCTV_INTEGRATION.md`**.

## Deep docs

**`CLAUDE.md` has the full architecture, measured numbers, and design rationale.**
Model weights, the TensorRT engine, generated test videos, the TLS cert and
runtime data are `.gitignore`d — regenerate them with the commands above.

## Constraints (see `docs/`)

Detection + ByteTrack + ANPR/face are demo-live; multi-camera Re-ID, criminal-DB
face matching, and a trained weapon object model are roadmap. Weapon posture is
a 2D-skeleton heuristic, **not** gun detection. AGPL-3.0 applies (Ultralytics).
