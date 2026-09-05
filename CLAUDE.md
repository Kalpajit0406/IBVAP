# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**IBVAP** — AI-Based Intelligent Video Analytics Platform for Border Surveillance
SIH 2026, PS 26187 · Ministry of Home Affairs / Sashastra Seema Bal (SSB) · Theme: Blockchain & Cybersecurity

Software-only AI analytics layer on top of existing CCTV infrastructure. No proprietary FRS/ANPR hardware. 1–2 week build window, team of 6 with coursework-level CV/ML experience.

---

## Commands

```bash
# Install PyTorch with CUDA 12.6 first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**Mode A — Local OpenCV display** (webcam/file testing)
```bash
python main.py               # reads config.yaml
python main.py --verify-chain
```
Set `url: "0"` / `"1"` in `config.yaml` for laptop webcam, or a `.mp4` path.

**Mode B — Web server** (the main demo path)
```bash
python run_demo.py --cams 4        # server + 4 replayed CCTV streams + dashboard
python run_demo.py --cams 6 --src E:\my_footage
python run_demo.py --no-feed       # server only, for real phones
```

Or run the pieces separately:
```bash
python make_test_videos.py            # build 4x 720p/24fps clips from the Kaggle dataset
python server.py                      # HTTP :8090 (monitor) + HTTPS :8443 (phones)
python tunnel.py                      # public URL for phones off the LAN (cloudflared/ngrok)
python feed_test.py --src test_videos --cams 8       # replay 8 streams (phone stand-in)
python feed_test.py --static --cams 8                # frozen frames (no-motion gate test)
python loadtest_mobile.py --cams 8 --seconds 40      # 8-device load test WITH measurements
python loadtest_mobile.py --static                   # no-motion test with skip-count report
python benchmark.py --cams 4          # four-way comparison of the throughput optimisations
python diagnose.py                    # end-to-end pipeline health check
python tests/test_posture.py          # PoseClassifier geometry checks (no GPU)
```

- **Monitor** → `http://localhost:8090/monitor` (plain HTTP — no certificate warning)
- **Phone on the same WiFi** → `https://<LAN-IP>:8443/camera/0` (`getUserMedia`
  needs HTTPS; accept the self-signed cert once). Run `python gen_cert.py` first.
- **Phone on any other network** (mobile data, different WiFi) →
  `python tunnel.py`, then open the `…/camera/0` URL it prints. Real cert, no
  warning. `run_demo.py --tunnel` does this alongside the server.

Two listeners share one app and one detection loop. Port 8080 is deliberately
avoided — Steam's webhelper squats on it and silently answers requests instead
of the server.

## CCTV / RTSP ingestion (`src/rtsp_capture.py`)

`server.py` opens every entry in `config.yaml` `streams:` on startup, **in the
same process and pipeline as the phones**. A row with a real `url`
(`rtsp://…`, `http://…mjpg`, a webcam index, or a `.mp4` to loop) becomes an
`RtspCapture`; a row with `url: ws` (or no url) stays a WebSocket slot a phone
connects to. Both land in the `captures` dict and flow through the batched
muxer → detector → dashboard identically — an NVR channel and a phone are
indistinguishable downstream.

`RtspCapture` is built for real, flaky cameras: **one decode thread each** (a
frozen camera never stalls the muxer), **RTSP forced over TCP** with a 5 s
socket timeout, **`grab()` every loop but `retrieve()` (decode) only at
`cctv.decode_fps`** (15) to keep CPU cost low across 8–10 streams,
**auto-reconnect with backoff + a stall watchdog**, per-frame **resolution
normalise** to 1280×720, and **credential redaction** in every log line and in
`/status` / `/devices`. `POST /api/reconnect/<id>` force-cycles a frozen feed.
It is duck-compatible with `WebSocketCapture` (`.read()`, `.connected`,
`.info()`, `.latency_ms`, …) so nothing else changed.

Full operator guide — how college CCTV is wired, vendor RTSP URL tables,
main vs sub-stream, what to ask IT for, `rtsp_probe.py` /
`discover_cameras.py` — is **`docs/CCTV_INTEGRATION.md`**.

## Mobile ingestion (phones as demo cameras)

Phones stream into that same pipeline — no app install.

**How a phone connects.** It opens `https://<lan-ip>:8443/camera/<id>` (or
`/cam/<id>`) in its browser. `static/camera.html` calls `getUserMedia`, reads
what the phone *actually* granted from `track.getSettings()`, sends that as a
JSON `hello` over a WebSocket, then streams frames on the same socket.
`src/ws_capture.py` decodes each frame and **normalises it to 1280×720
server-side** (phones rarely honour the request).

**Latency control — the phone is the bottleneck, not the pipeline.** A mobile
uplink is ~1–5 Mbps; 720p JPEG at 24 fps is ~10 Mbps, so with a naive sender
the WebSocket send buffer grows without bound and every frame the server sees
is *seconds* old (the "feed is 4–5 s behind" bug). `camera.html` fixes this by:
downscaling to a ≤960 px send canvas before encoding; capping the send rate;
and — the key part — **skipping the capture whenever `ws.bufferedAmount` still
holds the previous frame**, so latency stays flat instead of growing. When it
stays congested it drops to 10 fps / q0.40 and recovers automatically. Measured
localhost capture→receive is 2–6 ms; the server pipeline adds ~120–180 ms
(muxer tick + worker + mosaic).

**Latency indicator.** Each frame carries an 8-byte little-endian capture-ms
header; the `hello` carries `t0` for a coarse clock-sync (`serverTime` echoed
back in a `synced` reply). `ws_capture` reports a smoothed
`latency_ms` per camera in `/status` and `/devices`. The worker paints a badge
in the **top-right of each stream tile** — green `NET 120 ms`, or red
`NET 3.2s DELAYED` at ≥ 1.5 s — and the dashboard device panel shows the same.
`feed_test.py --latency-sim <ms>` backdates timestamps to exercise it without a
real slow link.

**Why WebSocket-JPEG, not MediaMTX/WHIP/WebRTC.** The brief recommended
`getUserMedia → WHIP → MediaMTX → RTSP → pipeline`. We use a direct
browser→WebSocket→pipeline push instead. Trade-off: WHIP/WebRTC gives H.264
hardware encode on the phone (less phone CPU, ~half the bandwidth) and ~150 ms
lower latency, and yields a standard RTSP URL. But it adds a separate MediaMTX
process to supervise, WebRTC ICE negotiation that some campus/corporate WiFi
blocks, and it would force the pipeline to *pull* RTSP with a decode thread per
stream — re-introducing exactly the per-stream cost the batched muxer is built
to avoid. The WebSocket path is one moving part, works through the cert already
set up, and was measured carrying 8×720p@24 with the GPU ~32% busy. If phone CPU
or bandwidth becomes the limit, MediaMTX is the upgrade path.

**Two ways in, pick by where the phone is:**

* **Same WiFi** → self-signed HTTPS on `:8443` (`gen_cert.py`, SANs for every
  local IP). Fully offline, one command — the demo LAN may have no internet.
  Cost: one "Advanced → Proceed" tap per phone. If the laptop's IP changes
  (new WiFi), re-run `gen_cert.py`.

* **Any network — mobile data, a different WiFi** → `python tunnel.py`. It
  runs `cloudflared` (no signup) or `ngrok` (one free `authtoken` — the script
  prints the exact setup steps if it's not configured) against the **HTTP**
  port `:8090`. The tunnel terminates TLS with a real, publicly-trusted cert,
  so the phone opens a normal `https://…/camera/0` with **no certificate
  warning at all**, and the WebSocket upgrades to `wss://` through it
  unchanged. The script prints the public `/monitor` + `/camera/N` links and a
  scannable QR. `run_demo.py --tunnel` launches it alongside the server; a
  failed tunnel never takes the LAN demo down.

  **The link is stable, not per-run.** ngrok's free plan gives every account
  one permanent static domain (e.g. `adjective-adjective-noun.ngrok-free.dev`)
  and plain `ngrok http 8090` binds to it — the URL only changes if the
  authtoken changes. `tunnel.py --domain <name>` (or `$IBVAP_TUNNEL_DOMAIN`)
  only renames it to something memorable, which needs a one-time free claim at
  dashboard.ngrok.com/domains. Buying a domain is never required for a fixed
  link; cloudflared's *free* quick tunnels, by contrast, are always random.

* **Your own domain** (e.g. `stream.mathswithsd.in/cam/0`) → a **cloudflared
  named tunnel**. `python tunnel.py --named <tunnel> --hostname <host>` (or set
  `$IBVAP_CF_TUNNEL` / `$IBVAP_CF_HOSTNAME` and use `run_demo.py --tunnel`).
  One-time setup — move the zone to Cloudflare DNS (the Netlify/other site
  keeps working with the same records), `cloudflared tunnel create/route`, fill
  `cloudflared.example.yml` — is written out in **`docs/CUSTOM_DOMAIN.md`**.
  `server.py` serves both `/cam/N` and `/camera/N`.

  `server.py`'s uvicorn runs with `proxy_headers=True` /
  `forwarded_allow_ips="*"` so it trusts the tunnel's `X-Forwarded-Proto/-For`.
  **The tunnel link is public and unauthenticated** — anyone with it can view
  the dashboard or push camera frames. Demo only; Ctrl+C to close it.

### Two-thread pipeline (muxer + inference worker)

`server.py` splits the work across two threads so a slow GPU pass can never
backpressure the camera sockets:

* **Muxer thread** (`_muxer_loop`) — sleeps to a fixed 24 Hz tick, drains each
  camera's queue into a one-deep latest-frame slot, snapshots all live slots,
  and hands the set to a depth-1 queue (`_infer_q`). It runs no model, no
  ByteTrack, no annotation, so it holds 24 tps regardless of inference cost.
  A busy-poll here (`sleep(0.001)` loop) was starving the other threads — it
  now sleeps the whole slack interval; the accumulator self-corrects jitter.
* **Inference worker** (`_inference_worker`) — owns the `Detector` and all
  per-camera ByteTrack / motion-gate state (single-threaded, no locks), plus
  risk scoring and evidence. Pulls the freshest snapshot and runs the full
  `process_batch` at its own pace. If it falls behind, the muxer **drops** the
  stale snapshot rather than growing a backlog — video and detection degrade
  gracefully to a lower rate while ingest stays real-time.

The `PIPE` log line reports both: `mux <tps>` (should stay 24) and
`worker <fps> (<bps>)` with a `dropped` counter.

### Measured — 8 mobile streams, RTX 3050 Laptop 6 GB, YOLO26n TensorRT FP16

| metric | value |
|---|---|
| Aggregate ingest | **192.0 fps** (8 × 24, 0 drops) — real-time target MET |
| Per-stream ingest | 24.0 fps sustained |
| Muxer tick rate | 24.0 tps (never blocked by inference) |
| Inference cost | ~8–24 ms/frame (TensorRT engine, batch ~2) |
| Frames the GPU never saw | ~78–86% (rate cap + motion gate) |
| GPU utilisation | ~32% mean, 45% peak (`nvidia-smi` 2 Hz under load) |
| VRAM | 1.0 GB of 6 GB |
| Power | ~26 W · Temp ~64 °C |

Also verified: **6 moving + 2 static** → 191 fps, ~1 drop/sec; the two-thread
split is what let the fixed-shape engine hit real-time — on the old single
thread it managed ~160 fps. `.pt` FP16 on the split lands ~176–192 fps (a bit
client-starved in the load test); the engine is the better choice now.

**No-motion check** (`python loadtest_mobile.py --static`): 8 frozen streams
sent 192 fps; YOLO ran **~1 fps per stream** — only the deliberate `force_every`
keep-alive sweep (so a target already in frame at startup, or one creeping
slower than the pixel threshold, is still caught). ~96% of frames were
motion-gated away. It is not literally zero by design; raise `motion.force_every`
if you want it lower, but that trades away the safety sweep.

**Dashboard — connected devices.** The right rail lists every device: label,
CAM id, link state (live / idle / offline), negotiated resolution@fps, delivered
fps, and motion-gate skip %. **live** = motion gating is currently passing that
stream's frames to YOLO; **idle** = connected but static, no GPU spent on it.
The video is a single server-composited MJPEG **mosaic** (`/stream`), so N tiles
cost one HTTP connection — browsers cap at ~6 per host, which stalled tiles at
8 individual `/stream/<id>` connections.

## Code Structure

```
src/
├── capture.py      # StreamCapture: threaded auto-reconnecting capture (used by main.py)
├── rtsp_capture.py # RtspCapture: CCTV/RTSP/NVR puller for server.py — decode thread, TCP, reconnect
├── ws_capture.py   # WebSocketCapture: browser/replay frames; generation-guarded reconnect
├── motion_gate.py  # MotionGate: cheap frame-difference pre-filter in front of the GPU
├── detector.py     # Detector: batched YOLO26n + per-camera ByteTrack + track carry-forward + pose pass
├── posture.py      # PoseEstimator (yolo26n-pose on person crops) + PoseClassifier geometry rules
├── risk_engine.py  # RiskEngine: zone × time_of_day × behaviour → 0–100 score
├── evidence.py     # EvidenceChain: append-only SHA-256 hash-chain JSONL
├── event_store.py  # EventStore: SQLite store-and-forward (survives network outage)
└── display.py      # GridDisplay: OpenCV multi-stream grid with risk overlay
server.py           # FastAPI: WS + RTSP intake, muxer + inference-worker threads, MJPEG mosaic, /status
rtsp_probe.py       # Validate one camera/NVR URL: open time, res/fps/codec, jitter, verdict
discover_cameras.py # ONVIF WS-Discovery: list LAN cameras, print a paste-ready streams: block
tunnel.py           # Off-LAN phones: ngrok / cloudflared quick tunnel, or --named for your own domain
cloudflared.example.yml # Named-tunnel ingress template (custom domain) — see docs/CUSTOM_DOMAIN.md
run_demo.py         # One command: server + replay + browser  (--tunnel adds a public link)
feed_test.py        # Replays video / static frames as if they were phones (--static, --cams N)
loadtest_mobile.py  # 8-device load test: spins up phone-like WS clients, measures fps + GPU
make_test_videos.py # Builds 720p/24fps clips with realistic static/motion mix
benchmark.py        # Four-way comparison of the throughput optimisations
diagnose.py         # Checks each pipeline stage end to end
export_engine.py    # One-time TensorRT export (needs `pip install tensorrt`) — see note below
tests/test_posture.py # Synthetic-skeleton geometry checks for PoseClassifier (Rule 5)
static/camera.html  # Phone page: getUserMedia, negotiated-settings hello, JPEG-over-WS
static/monitor.html # Command dashboard: MJPEG mosaic + connected-devices panel + /status poll
config.yaml         # Stream URLs, ingest normalisation, model params, throughput, risk
data/               # Auto-created: events.db + hash_chain.jsonl
```

## Architecture

Two-track pipeline:

**Track A — Detection & Risk**
```
CCTV (RTSP/ONVIF) → Camera Tamper/Health Monitor + Edge Store-and-Forward
    → Video Ingestion & Frame Sampling (Day/Night/Thermal)
    → AI Analysis (YOLO26n + ByteTrack)
    → [Person (face detect) | Vehicle (ANPR) | Threat Behaviour | Virtual Fence]
    → Event & Risk Engine (zone × time × behaviour → 0–100 score)
    → Command Dashboard (React + Leaflet GIS)
```

**Track B — Verification & Evidence**
```
Human Verify → Confirm → Intercept Dispatch
    → Event + Face Evidence (DB)
    → SHA-256 Hash + Blockchain Ledger
    → Secure Audit Log
```

False-alarm dismissals feed an Active Learning retrain loop back into the AI pipeline.

---

## Posture / behaviour analysis (`src/posture.py`)

When `pose.enabled`, `yolo26n-pose.pt` runs on each person **crop** (not the
full frame — ~2–3 ms at imgsz 256) and `PoseClassifier` applies pure-Python
geometry rules to the 17 COCO keypoints. Output feeds `risk_engine._behaviour()`
and draws a skeleton + label. Rules: `LYING` (skeleton wider than tall),
`CROUCH` (knees near hips / compressed height), `SCAN` (nose-x oscillating),
`ARMS-UP` (wrist above shoulder), `AIM` (Rule 5, below).

**Rule 5 `chest_aim` / "AIM" — a weapon-READY POSTURE heuristic, not gun
detection.** It has no view of any weapon; it only asks whether the wrists and
elbows form a held two-handed grip. Clauses (all must hold, then persist
`aim_hold_frames` = **2** detection frames): both wrists between ~forehead and
~navel height, level with each other, within ~0.85 shoulder-width of each
other, in front of the torso (rejects folded arms — a folded wrist sits out
past the far shoulder), off the thighs, and at least one elbow no lower than
mid-torso (arm bent/forward, not hanging). This band is deliberately wide so a
real hold — high-ready, aiming across camera, low-ready muzzle-down — fires;
the earlier version was tight enough that a genuine toy-gun hold never
triggered. `tests/test_posture.py` (12 cases) checks the fire cases
(chest / low-ready / angled) and the reject cases (belt-clasp, folded arms,
one-hand, wide hands, surrender).

**A detected AIM forces the risk level to Critical** (`risk_engine.assess`:
`level = "Critical"`, `score = max(score, 92)`, `RiskAssessment.weapon = True`).
Previously a confirmed aim scored ~67 in daytime — under the 70 threshold — so
it only went Critical at night. The worker also paints a full-width red
`WEAPON — CRITICAL` banner across that tile.

Known limits of a 2D-skeleton approach: a one-handed pistol grip won't fire
(needs two hands), and a person holding a clipboard/phone two-handed at chest
height is geometrically an aim. **The real fix is a custom-trained weapon
object-detection model** — roadmap, not more geometry tuning.

Per-track classifier state (nose history, height baseline, aim streak) is
flushed in `detector.py` for any ByteTrack id that drops out, so the dicts
don't grow and a stale streak can't carry into a reused id.

---

## Multi-stream throughput — how the pipeline stays real-time

Five mechanisms, in the order a frame meets them. `benchmark.py` measures 1–4
in isolation on the dev RTX 3050; the live 8-stream numbers are under
"Two-thread pipeline" earlier.

**1. Fixed-tick muxer + inference worker** (`server.py`)
`server.py` runs two threads. The **muxer** sleeps to a 24 Hz tick, drains every
camera's queue into a one-deep slot, and hands the whole set of live slots to a
depth-1 queue. The **inference worker** pulls the freshest set and runs the
model + tracking + risk at its own pace; if it falls behind, the muxer drops
the stale snapshot. Because every camera contributes to every tick, the batch
is always "all connected cameras" (nvstreammux's job) — and because inference
is off the muxer thread, a slow GPU pass never backpressures the sockets.

**2. Detection-rate cap on a global tick** (`detect_fps` = 8, `stream_fps` = 24)
The muxer ticks at 24 Hz. Every 3rd tick a fresh YOLO + ByteTrack pass runs
(8 fps); on the two ticks between, each box is advanced by its last measured
per-tick velocity, so the annotated stream still updates at the full 24 fps —
smooth boxes, one third of the GPU cost. The schedule uses one shared tick
rather than each camera's own frame counter, so all cameras come due in the
same pass and the batch stays full. (This is velocity extrapolation, not a
Kalman predict — fine for people at demo range; a real 24 fps Kalman step is a
possible future refinement.)

**3. Motion gating** (`motion_gating`, `src/motion_gate.py`)
A 160×90 greyscale frame difference (0.93 ms) decides whether a scene changed
at all. Static footage never reaches the GPU. A gated camera advances its
schedule as though it had run, so it stays in phase with the others. Two
safeguards prevent a missed intrusion: `force_every` sweeps every camera at
least once a second regardless, and `hold_frames` keeps the gate open briefly
after motion stops.

**4. Fixed batch shapes** (`_bucket` in `src/detector.py`)
**This one is not optional and is easy to regress.** Ultralytics/cuDNN re-tune
kernels whenever the batch size changes (a run of varying batches averaged
**304 ms/pass** vs **46 ms** for a constant batch of 12 — 6× slower), and a
TensorRT engine simply *cannot* take a shape other than the one it was built
for. So batches are padded up to a fixed ladder — `[1, 2, 4, 8, 16]` for a
`.pt` model, or `[max_batch]` only for a `.engine` — and the padded outputs
discarded. The warm-up primes exactly these shapes.

**5. Motion-vs-forced live state** (`src/motion_gate.py`)
The gate records whether its last pass was real motion or just the keep-alive
sweep, so the dashboard's live/idle state reflects genuine activity — a swept
static camera stays "idle".

### Measured results (4 × 720p @ 24 fps, RTX 3050 6 GB)

| configuration | aggregate fps | per camera | GPU frames | real-time |
|---|---|---|---|---|
| baseline (1 frame/call, every frame) | 75.6 | 18.9 | 1152 | no |
| + batching | 203.2 | 50.8 | 1152 | yes |
| + rate cap (detect 8 fps, track 24 fps) | 468.1 | 117.0 | 384 | yes |
| + motion gating | 466.6 | 116.7 | 213 | yes |

**6.2× faster, 82% fewer frames inferred.** 96 fps is needed to keep up; the
pipeline sustains ~467, leaving headroom for roughly 19 streams. At 12 streams
it holds 526 fps against the 288 fps needed.

Benchmark numbers are only meaningful after a warm-up run — cuDNN autotuning
made the first configuration measured look 4× slower than it was, which briefly
made motion gating appear to be a regression. `benchmark.py` now discards a
warm-up pass; keep it that way.

---

## Locked Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Detection | **YOLO26n** (Ultralytics ≥8.3.0) | 40.9 mAP, 1.7 ms T4 TRT, 2.4M params |
| Tracker | **ByteTrack** | 80.3 MOTA; faster than DeepSORT, default in Ultralytics pipeline |
| ANPR | EasyOCR (primary) / PaddleOCR (fallback) | On cropped vehicle regions only, not full frame |
| Face detection | RetinaFace | Detection only — no live matching in demo |
| Re-ID | OSNet | Architecture/roadmap target; demo uses timestamp + visual heuristic |
| Backend | FastAPI + WebSocket + PostgreSQL/PostGIS | |
| Frontend | React 18 + Tailwind CSS + Leaflet | |
| Blockchain | SHA-256 hash-chain ledger | Evidence integrity, not a distributed network |
| Edge buffer | SQLite store-and-forward queue | 72-hour offline alert buffer |
| Deployment | Docker + NVIDIA Jetson Orin Nano/NX | |

**Python:** 3.11.9 (NOT 3.13/3.14 — PyTorch unsupported)
**CUDA:** 12.6 · **Inference precision:** FP16 for deployment, FP32 for dev
**Dev GPU:** RTX 3050 laptop, 6GB VRAM — this is the minimum hardware everything must run on

### TensorRT engine

`tensorrt-cu12 == 10.13.3.9` **is installed** (GPU-only library; the 1.5 GB
`tensorrt_cu12_libs` wheel is the CUDA-12 payload — matched to torch's cu126,
*not* the newest 11.x, which changed the builder API Ultralytics 8.4 expects).
`yolo26n.engine` **is built** — `export_engine.py`, FP16, fixed `(8, 3, 640,
640)` shape, TensorRT 10.13, specific to this RTX 3050.

**It is the pipeline default:** `weights: "yolo26n.engine"` in config.yaml.
The two-thread split (above) is what makes this work — on the old single
thread the fixed-shape engine's per-tick 8-image pass dropped the muxer to
~160 fps; with inference on its own worker the muxer holds 24 tps and the
engine sustains 192.

| | .pt PyTorch FP16 | .engine TensorRT FP16 |
|---|---|---|
| detections | reference | **identical** (batch 1/2/4/8 all match) |
| batch-1 latency (isolated) | 27 ms | **5.6 ms — 4.8× faster** |
| batch-8 latency (isolated) | 37 ms | 28 ms |
| **8-stream, two-thread pipeline** | ~176–192 fps | **192 fps, 0 drops** |
| VRAM | 0.9 GB | 1.0 GB (fixed shape); 4.6 GB if built `dynamic=True` — rejected |
| warm-up | ~11.6 s (cuDNN autotune) | ~0.5–1.7 s |

The engine's input shape is fixed at `max_batch`, so the detector pads every
detection tick to a full `max_batch`-image pass — cheap once inference is off
the muxer thread. A `dynamic=True` engine avoids the pad but reserved ~4.6 GB
on the 6 GB card and stalled on shape changes — rejected.

**Fall back to `weights: "yolo26n.pt"`** only when the `.engine` is missing
(not exported on this machine yet) or on a non-CUDA box. `src/detector.py`
detects `.engine`/`.onnx` (`self._is_engine`) and skips the PyTorch-only
`.to()` / `quantize=` calls automatically. Re-export per GPU (the engine is
hardware-specific): `python export_engine.py` — reads `image_size` and
`max_batch` from config, builds FP16, ~5–10 min.

---

## Hard Numbers — Do Not Change Without Flagging

- **AI inference stream:** 720p (1280×720) @ 15–25 fps via RTSP/ONVIF substream — never the primary 4K recording stream
- **Effective inference rate:** 8–10 fps (`detect_fps: 8`), with ByteTrack carrying boxes at the full 24 fps
- **Measured capacity:** 4 × 720p/24fps at ~467 fps aggregate on an RTX 3050 6 GB; ~19 streams of headroom
- **Risk score:** 0–100; zone sensitivity 40% + time-of-day 20% + behaviour 40%; threshold **≥70** = Critical Alert
- **YOLO26n COCO mAP:** 40.9 (50–95); latency 1.7 ms T4 TRT / 38.9 ms CPU ONNX
- **Evidence retention:** 90 days local, then archived to central command
- **False-positive target:** <15% at launch, tuned via active learning
- **License:** AGPL-3.0 (all Ultralytics models) — acknowledge openly, do not hide

---

## Hard Constraints

- **Criminal face matching:** No live matching against any criminal database in the demo. Use consented mock watchlist (team photos or LFW). Frame any match as a lead for human verification. Real deployment = API call to NCRB CrPI.
- **Camera stream:** Always run inference on the secondary/AI substream (720p), never primary 4K. ~4–9x compute saving with no detection quality loss.
- **GIS terminology:** Call it a "GIS Digital Map," not "Digital Twin" — a twin implies live 3D sync that is not being built.
- **Re-ID:** Do not claim OSNet multi-camera Re-ID is live. Present it as roadmap. Demo uses simplified heuristic.
- **TensorRT:** Only PyTorch, TorchScript, and TensorRT actually use the Jetson GPU — all other export formats are CPU-only.
- **YOLO + tracker naming must be consistent across all slides and all code.** The locked choices are YOLO26n + ByteTrack.

---

## Detection Range Tiers (state explicitly, not as a limitation)

| Tier | Camera | Range | Capability |
|---|---|---|---|
| Full analytics | Fixed bullet/dome 2–4MP | 0–80m | Detection + ANPR + face + behaviour |
| Detection only | Long-range PTZ 30–45x | 80–300m | Person/vehicle reliable; ANPR/face degrade past ~150m |
| Thermal presence | Bi-spectrum thermal | 1.5–8km | Movement/presence only, no classification |

62% of users attempting facial ID beyond 70 ft (21m) with a 4MP camera report failure — resolution and lens determine range, not AI.

---

## What to Present as Roadmap (Not Demo-Live)

- Multi-camera Re-ID (OSNet)
- Criminal DB face matching (→ NCRB CrPI API)
- VLM-based explainable alerts
- Feed-spoofing / replay-attack detection
- **Weapon detection via a trained object model.** The live demo only has the
  `AIM` *posture* heuristic (`src/posture.py` Rule 5) — a held two-handed
  forward grip. Do not call it "gun detection". A custom-trained YOLO weapon
  class is the real capability.
- Per-endpoint auth (camera intake, dashboard, tunnel link are all currently open)

---

## Project Documentation

All reference material lives in `docs/`:
- `IBVAP_Project_Context_TechStack_1.md` — full project context, architecture decisions, hardware cost math
- `IBVAP_Technical_Specifications_1.md` — submission-ready numbers with [HARD]/[TUNABLE]/[MISSING] tags
- `IBVAP_References.md` — 22 numbered citations (R1–R22) for every hard number

## Agent Workflows

`.agent/workflows/` contains named design-pass workflows (adapt, animate, audit, bolder, clarify, colorize, critique, delight, extract, harden, normalize, onboard, optimize, polish, quieter, simplify, teach-impeccable). `.agent/skills/frontend-design.md` governs all frontend UI work.
