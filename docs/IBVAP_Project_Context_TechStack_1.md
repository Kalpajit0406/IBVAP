# IBVAP — Full Project Context & Tech Stack Reference
**PS 26187 — AI-Based Intelligent Video Analytics Platform for Border Surveillance**
Ministry of Home Affairs / Sashastra Seema Bal (SSB) · Theme: Blockchain & Cybersecurity

---

## 1. Project Basics

- **Team:** 6 people, "some" CV/ML experience (tutorials/coursework, not production)
- **Timeline:** 1–2 weeks build window
- **Core premise:** Software-only AI analytics layer on top of *existing* CCTV (no proprietary FRS/ANPR hardware needed)
- **Why this PS was chosen over alternatives considered:** highest live-demo impact, lowest technical-invention risk (all core CV components are mature/off-the-shelf), strong cost-effectiveness story vs. proprietary hardware

---

## 2. ⚠️ OPEN DECISION — resolve before finalizing

**Detection/tracking model mismatch across your own slides:**
- Tech Stack slide says: **YOLOv8n + ByteTrack**
- Architecture diagram (teammate's) says: **YOLOv10 + DeepSORT**

My original recommendation was YOLOv8n specifically because it has far more tutorials/community support for a team at your stated experience level — safer under time pressure. YOLOv10 is newer/potentially more accurate; DeepSORT is a reasonable alternative to ByteTrack but slightly heavier. **Pick one and make every slide/doc consistent** — a judge cross-referencing two slides that name different models will notice immediately.

---

## 3. Final Tech Stack (pending the decision above)

| Layer | Tools | Notes |
|---|---|---|
| **Video Ingestion** | OpenCV + FFmpeg (RTSP/ONVIF) | Pull the camera's **secondary/AI substream** (720p, 15–25fps) — never the primary 4K/1080p recording stream, for cost/compute reasons (see §5) |
| **Detection & Tracking** | YOLOv8n + ByteTrack **OR** YOLOv10 + DeepSORT *(resolve above)* | Frame-skip to ~8–10fps effective inference; run tracker at fuller rate (cheap) |
| **ANPR** | EasyOCR / PaddleOCR on cropped vehicle regions | Not full-frame — only triggered on vehicle detection |
| **Face Detection** | RetinaFace | **Detection only** for MVP demo — no live matching against a real criminal DB (see §7) |
| **Multi-Cam Tracking** | OSNet Re-ID (architecture/target) | **Stretch feature** — for the actual demo, use a simplified heuristic (timestamp + rough visual similarity), not a full Re-ID model. Be honest about this if asked. |
| **GIS Layer** | Leaflet (frontend) + PostGIS (backend) | Call this a **"GIS Digital Map,"** not a "Digital Twin" — a twin implies live 3D/continuous sync you're not building |
| **Risk/Threat Scoring** | Custom rule-weighted engine (zone + time + behavior → 0–100 score) | Threshold ~70 triggers "Critical Alert"; this is just arithmetic on signals you already have — cheap, high value |
| **Backend / Alerting** | FastAPI + WebSocket + PostgreSQL/PostGIS | Event log, real-time push |
| **Dashboard** | React + Tailwind CSS | Live map, alert feed, event history |
| **Security & Integrity** | SHA-256 hashing + hash-chain ledger (blockchain evidence) + Camera Tamper/Health Monitor | This is your **entire "Blockchain & Cybersecurity" theme payload** — don't let it get thin |
| **Edge Resilience** | Local store-and-forward queue (SQLite buffer) | Buffers alerts when network is down, syncs on reconnect — genuinely good live-demo moment (kill network, show it still logs) |
| **Deployment** | NVIDIA Jetson Orin Nano (small posts) / Orin NX (junction posts) + Docker | See exact cost math in §5 |

---

## 4. Architecture Flow (two-track pipeline, as built in the final slide)

**Track A — Detection & Risk:**
```
Existing CCTV Cameras (RTSP/ONVIF)
   ├─→ Camera Tamper/Health Monitor [NEW]
   └─→ Edge Store-and-Forward (offline) [NEW]
        ↓
Video Ingestion & Frame Sampling (Day/Night + Thermal Modes)
        ↓
AI Video Analysis (YOLO + Tracker)
        ↓
  ┌──────────┬──────────┬──────────┬──────────┐
Person    Vehicle    Threat/     Virtual
Detected  Detected   Behaviour   Fence [NEW]
(CrPI     (ANPR +    (Weapon +   (Geofence +
face      Make/      Crowd       Zone Breach)
match)    Color ID)  Pattern)
  └──────────┴──────────┴──────────┘
        ↓
Predictive Pathing & GIS (Cross-Camera Tracking)
        ↓
Event & Risk Engine (Risk Score + Spatial Context)
        ↓
  ┌─────────────┬──────────────┐
Low/Normal    High Risk/Anomaly
(Continue     (Real-Time Alert Gen)
Monitoring)         ↓
              Command Dashboard (GIS View)
                    ↓ [escalates on High-Risk Alert]
```

**Track B — Verification & Evidence:**
```
Human Verify (diamond)
  ├─→ Dismiss (False Alarm) → Log & Continue → [feeds Active Learning retrain loop back to AI Video Analysis]
  └─→ Confirm (True Threat) → Intercept Dispatch (send location to police)
                                    ↓
                          Event + Face Evidence (DB)
                                    ↓
                      SHA-256 Hash + Blockchain Ledger [NEW]
                                    ↓
                            Secure Audit Log (DB)
```

Also annotated (not literal wired loops, kept as labels for clarity at this diagram density):
- **Active Learning:** false-alarm dismissals feed back to retrain the detection model
- **Hardware Actuation:** High-Risk Alert can trigger PTZ camera auto-aim/tracking

---

## 5. Hardware & Cost-Optimization Math (for Feasibility slide / Q&A)

**Core lever:** run AI inference on the camera's low-res AI substream, never the 4K/1080p primary stream. Combine with frame-skipping (process every 2nd–3rd frame, ~8–10fps effective) while tracking runs fuller-rate.

| Device | Native full-rate streams | Effective streams (with frame-skip) | Unit cost | Cost/stream |
|---|---|---|---|---|
| Jetson Orin Nano Super 8GB | 4–6 | 10–14 | ~₹21,000–25,000 | ~₹1,700–2,100 |
| Jetson Orin NX 16GB | 16–18 | 35–45 | ~₹50,000–66,000 | ~₹1,150–1,400 |

**Takeaway line for judges:** *cheapest device ≠ cheapest deployment* — NX is more cost-efficient per camera once a post has 5+ cameras.

**Tiering:**
- Small/remote BOP (2–4 cams): Orin Nano
- Medium/junction post (5–10 cams): Orin NX
- Regional command hub (cross-BOP aggregation, Re-ID, heavier reasoning): Orin AGX or small on-prem GPU server

**Comparison point:** proprietary FRS/ANPR hardware runs into lakhs per unit; this approach is ~₹1,150–2,100 in shared compute per camera stream.

**Caveat to state explicitly:** these throughput numbers assume YOLOv8n/INT8 via TensorRT. If your hackathon demo runs unoptimized PyTorch (likely, given the timeline), say so — don't let a judge catch a gap between demo FPS and claimed production numbers.

---

## 6. Existing real-world context to cite (for credibility)

- **CIBMS** (Comprehensive Integrated Border Management System) — BSF's existing multi-sensor smart-fence system, ~₹1 crore/km; IBVAP is positioned as the cheaper **software-only analytics layer** for SSB's less-instrumented posts, not a CIBMS competitor
- **NCRB's CrPI** (Crime and Criminal Profiling Identification, launched June 2026) — India's actual national biometric-matching platform, built on the Criminal Procedure (Identification) Act, 2022. **IBVAP's face-match module should be positioned as integrating with CrPI via API, not maintaining an independent criminal database.**

---

## 7. Face Recognition / Criminal DB — Handle Carefully

- Technically easy (embeddings + FAISS/cosine similarity + threshold), but **there is no legally usable public criminal-face dataset** — don't build or claim a real one
- For demo: mock watchlist using consented team photos or a public research dataset (e.g., LFW), clearly labeled as simulated
- **Always frame a match as a lead for human verification, not automated action** — this is both the ethically correct design and the strongest answer to a judge's ethics question
- Real deployment = forward query to NCRB CrPI, don't reinvent the database

---

## 8. Differentiators Already Locked In

1. Blockchain-anchored tamper-evident evidence ledger (SHA-256 hash-chain) — direct answer to the "Blockchain" half of your theme
2. Camera tamper/feed-health monitoring — direct answer to the "Cybersecurity" half
3. Edge-first, bandwidth-aware architecture (metadata upstream, not raw video)
4. AI Risk/Threat Scoring (reduces alert fatigue narrative)
5. GIS-based geofenced border map
6. Active-learning feedback loop (false-alarm → retrain)

**Discussed but not yet built into slides — available as stretch/answer-if-asked:**
- VLM-based explainable alerts (zero-shot anomaly detection with natural-language reasoning, e.g. LAVAD/VERA-style) — bigger "wow" than rule-based alerts, but needs GPU budget/prompt-tuning time
- Feed-spoofing/replay-attack detection (a real border-camera attack vector)
- Local-language (Hindi/regional) alert text
- Weather robustness demo (fog/rain test clip)

---

## 9. Files Delivered So Far

- `IBVAP_Technical_Approach_Slide_v3.pptx` — icon-chip style tech stack + simpler pipeline
- `IBVAP_Technical_Approach_Slide_v4.pptx` (or in-progress) — full two-track flowchart matching teammate's diagram, with all missing nodes added (this is the most current/complete version)

---

## 10. What to send back for verification

When you build the tech stack with other tools/models, come back with specifics I can sanity-check:
- Final YOLO version + tracker choice (resolve §2)
- Actual code/config for anything touching the risk-scoring thresholds, GIS zone definitions, or the hash-chain implementation
- Any new libraries/services you're adding that aren't in §3, so I can flag cost, licensing, or feasibility issues before they're baked into slides
