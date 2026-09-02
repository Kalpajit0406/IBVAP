# IBVAP — Technical Specifications Sheet
## The actual numbers to put in your SIH submission (PPT, prototype doc, Q&A)

Everything below is what to *state* as your system requirements — pulled from all the research/modelling done so far, reduced to defensible, submission-ready numbers. Each has a status tag:
**[HARD]** = don't change this, it's grounded in verified data · **[TUNABLE]** = your team's engineering choice, defend it if asked · **[STATE AS RANGE]** = don't give a single number, give the range with the reason

---

## 1. COMPUTE / GPU REQUIREMENTS

| Spec | Value | Status |
|---|---|---|
| **Minimum GPU (development/demo)** | NVIDIA RTX 3050 Laptop, 6GB VRAM, CUDA 12.6 | [HARD] — this is what you're actually building on |
| **Minimum VRAM for YOLO26n inference** | 4GB | [HARD] |
| **Recommended VRAM for full pipeline (detection+tracking+ANPR+face+ReID)** | 6GB+ | [HARD] |
| **Production deployment GPU** | NVIDIA RTX 4000 SFF Ada Generation, 20GB, 70W | [TUNABLE] — chosen for single-slot form factor + no datacenter EULA restriction |
| **Cameras per production GPU** | 45–60 (state as **50, engineering estimate**) | [STATE AS RANGE] — not vendor-published, say so if asked |
| **Inference precision** | FP16 (TensorRT) for deployment; FP32 acceptable for dev | [HARD] |
| **CUDA version** | 12.6 (matches RTX 3050 driver 616.x) | [HARD] |
| **Framework** | PyTorch 2.x + Ultralytics ≥8.3.0 | [HARD] |
| **Python version** | 3.11.x (NOT 3.13/3.14 — PyTorch unsupported) | [HARD] |
| **CPU sizing (production)** | ~7 cameras/core, assumes DeepStream zero-copy pipeline | [TUNABLE] — state the assumption explicitly |
| **Power draw per production GPU** | 70W (SFF Ada) — runs on standard single-phase circuit | [HARD] |

**If a judge asks "what's your minimum hardware requirement" — say this:**
> "Development and demo run on a single consumer GPU with 6GB+ VRAM — we built and tested on an RTX 3050 laptop. Production deployment uses workstation-class GPUs (RTX 4000 Ada) chosen for single-slot form factor, low power draw, and NVIDIA's datacenter licensing terms, which restrict GeForce cards in 24/7 unattended installations."

---

## 2. MODEL SPECIFICATIONS

| Spec | Value | Status |
|---|---|---|
| **Detection model** | YOLO26n (nano) | [TUNABLE] — confirm this is your team's final decision, not YOLOv10 |
| **Accuracy (COCO mAP 50-95)** | 40.9 | [HARD] — Ultralytics official benchmark |
| **Inference latency (T4 TensorRT)** | 1.7 ms | [HARD] |
| **Inference latency (CPU ONNX)** | 38.9 ms | [HARD] |
| **Parameters** | 2.4M | [HARD] |
| **FLOPs** | 5.4B | [HARD] |
| **Training base** | COCO pretrained (80 classes); fine-tune only if adding border-specific classes | [TUNABLE] |
| **Tracker** | ByteTrack (recommended) or DeepSORT | [STATE YOUR CHOICE] — resolve before submission, see tradeoffs below |
| **Tracker — ByteTrack** | 80.3 MOTA (MOT17), 171 FPS, lower compute | [HARD] |
| **Tracker — DeepSORT** | Better occlusion recovery via appearance embedding, heavier compute | [HARD] |
| **ANPR** | EasyOCR (primary) or PaddleOCR (fallback) | [TUNABLE] |
| **Face detection** | RetinaFace — detection only, no live matching in demo | [HARD — ethical/legal necessity, not a preference] |
| **Multi-camera Re-ID** | OSNet — **architecture/target only**; demo uses simplified heuristic | [STATE AS ROADMAP] — do not claim this is live |
| **License** | AGPL-3.0 (YOLO26, all Ultralytics models) | [HARD] — acknowledge, don't hide |

**Resolve before submission:** ByteTrack vs DeepSORT must be the same across every slide and the actual code. Recommendation: **ByteTrack**, because it maximizes cameras-per-GPU (your cost argument depends on this) and is the default in Ultralytics' own pipeline. Use DeepSORT only if your Re-ID roadmap story needs the shared appearance-embedding lineage more than it needs the throughput.

---

## 3. CAMERA / VIDEO INPUT SPECIFICATIONS

| Spec | Value | Status |
|---|---|---|
| **Ingest protocol** | RTSP / ONVIF (Profile S, G, T) | [HARD] — matches existing SSB fleet |
| **AI inference stream** | Substream: 720p (1280×720) @ 15–25 fps | [HARD] — never the primary/recording stream |
| **Effective inference rate** | 8–10 fps (frame-skip every 2nd–3rd frame) | [TUNABLE] |
| **Tracker update rate** | Full substream rate (15–25 fps) | [HARD] |
| **Recording/forensic stream** | 1080p–4K @ 25–30 fps, H.264/H.265 (untouched by AI) | [HARD] |
| **Thermal stream (where present)** | 640×512 or 384×288, 8–14 µm band, NETD ≤40mK, ~25 fps | [HARD] |
| **Bandwidth per camera (AI substream)** | 512 Kbps – 1.5 Mbps (state **1.5 Mbps** as planning figure) | [STATE AS RANGE] |
| **Bandwidth for 50 cameras (one GPU node)** | ~75 Mbps sustained | [HARD, derived] |

### Effective detection range — the three-tier model (state this explicitly)

| Tier | Camera type | Range | AI capability |
|---|---|---|---|
| **Tier 1 — Full analytics** | Fixed bullet/dome, 2–4MP | 0–80m (30–80m IR range) | Detection + ANPR + face detection + behaviour, full confidence |
| **Tier 2 — Detection only** | Long-range PTZ, 30–45x zoom | 80–300m (150–300m IR) | Person/vehicle detection reliable; ANPR/face degrade sharply beyond ~150–200m |
| **Tier 3 — Thermal blob** | Bi-spectrum thermal | 1.5–3km (human), 4–8km (vehicle) | Movement/presence alerting only — no classification detail at this range |

**Why this matters for your pitch:** 62% of users attempting facial ID beyond 70 feet (~21m) with a 4MP camera report failure — resolution and lens matter more than any AI improvement past a certain distance. State the tiering as a *design decision*, not a limitation you were caught not knowing about.

---

## 4. NETWORK & BACKHAUL

| Spec | Value | Status |
|---|---|---|
| **Per-camera AI substream** | 1.5 Mbps | [HARD] |
| **50-camera node (one GPU)** | ~75 Mbps | [derived] |
| **200-camera node (4 GPUs)** | ~300 Mbps | [derived] |
| **400-camera deployment (2× 4-GPU nodes)** | ~600 Mbps aggregate, split across 2 sites | [derived] |
| **Switch spec** | 10GbE aggregation (24-port for 4-GPU node, 48-port for 8-GPU) | [TUNABLE] |
| **NIC** | Dual-port 25GbE per node | [TUNABLE] |
| **Backhaul dependency** | Assumes existing fibre/microwave at CIBMS-sector or ICP sites | [STATE AS ASSUMPTION] — new OFC laying is a separate, large cost not included in your model |

---

## 5. STORAGE

| Spec | Value | Status |
|---|---|---|
| **Boot storage** | 2× 960GB NVMe, RAID-1 | [TUNABLE] |
| **Evidence/hash-chain buffer** | 4TB NVMe per GPU-equivalent, scale with node size | [TUNABLE] |
| **Evidence retention policy** | Not yet defined — **needs a number before submission** | [MISSING — see below] |
| **Offline store-and-forward buffer** | Local SQLite queue; size = (alert rate × avg event size × expected outage duration) | [TUNABLE] — pick a concrete number, e.g. "72 hours of buffered alerts" |

**Gap to close:** you don't currently have a stated evidence retention period (30 days? 90 days? matches BSA 2023 requirements?). Pick one — "90 days local, then archived to central command" is a reasonable, defensible default.

---

## 6. SOFTWARE STACK VERSIONS (for reproducibility / technical documentation slide)

```
Python           3.11.9
PyTorch          2.x  (cu126 build)
Ultralytics      ≥8.3.0
OpenCV           (bundled with Ultralytics — do not pin separately)
FastAPI          ≥0.115.0
Uvicorn          ≥0.30.0
PostgreSQL       15+ with PostGIS extension
EasyOCR          ≥1.7.2
InsightFace      ≥0.7.3
onnxruntime-gpu  ≥1.19.0
React            18.x
Docker           24.x+ (containerized deployment)
```

---

## 7. RISK SCORING ENGINE — PARAMETERS TO ACTUALLY DEFINE

You've named this feature repeatedly but never fixed numbers. For the demo/PPT, state these concretely:

| Parameter | Suggested value | Status |
|---|---|---|
| Risk score scale | 0–100 | [TUNABLE — already used in your diagram] |
| Critical alert threshold | ≥70 | [TUNABLE — already used in your diagram] |
| Score inputs | Zone sensitivity (weight 40%) + time-of-day (weight 20%) + behavior pattern (weight 40%) | [TUNABLE — pick weights, defend the logic] |
| Target false-positive rate | State a number, e.g. "<15% at launch, tuned down via active learning" | [MISSING — pick one] |

A judge asking "how is the risk score calculated" and getting "it's a weighted engine" with no numbers is a weak answer. Getting specific weights — even if simple and provisional — is a strong one.

---

## 8. WHAT NOT TO STATE AS FIXED (say "designed for" instead)

These are architecture/roadmap items — don't present them as live capabilities in the demo:

- Multi-camera Re-ID (OSNet) — demo uses simplified heuristic
- Criminal DB face matching — architected to call NCRB CrPI via API, no live matching in demo
- VLM-based explainable alerts — mentioned as future enhancement only if you haven't built it
- Feed-spoofing/replay-attack detection — mention as designed, not demonstrated, unless built

---

## 9. ONE-PAGE SUMMARY TABLE (for the actual slide)

| Category | Spec |
|---|---|
| Detection model | YOLO26n, 40.9 mAP, 1.7ms TensorRT |
| Tracker | ByteTrack, 80.3 MOTA |
| Dev GPU | RTX 3050, 6GB VRAM |
| Production GPU | RTX 4000 SFF Ada, 20GB, 70W |
| Cameras/GPU | ~50 (engineering estimate) |
| AI input stream | 720p substream, 15–25fps, RTSP/ONVIF |
| Effective inference rate | 8–10fps with frame-skip |
| Detection range (full analytics) | 0–80m |
| Detection range (detection-only) | 80–300m |
| Detection range (thermal presence) | 1.5–8km |
| Risk threshold | 70/100 |
| Evidence integrity | SHA-256 hash-chain |
| Software license | AGPL-3.0 (acknowledged) |
| Deployment | Docker, hardware-agnostic |

---

*Every [HARD] number here has a citation in your References doc. Every [TUNABLE] and [MISSING] item is a decision your team needs to make once, consistently, before the deck goes final — inconsistency across slides is the single most common thing that erodes judge confidence in an otherwise strong technical pitch.*
