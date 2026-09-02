# IBVAP — Research References
## PS 26187 | AI-Based Intelligent Video Analytics Platform for Border Surveillance
### Topics: YOLO26n · Border Camera Specifications & Limitations · Jetson Orin Performance

---

## SECTION 1 — YOLO26n MODEL

---

### [R1] Official YOLO26 Paper (Primary Source)
**Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F.C., & Kalfaoglu, M.E. (2026).**
*Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models.*
arXiv preprint arXiv:2606.03748. June 2026.

> Key findings used in IBVAP:
> - YOLO26 eliminates Non-Maximum Suppression (NMS) and Distribution Focal Loss (DFL) natively
> - Introduces MuSGD optimizer, Progressive Loss (ProgLoss), and Small-Target-Aware Label Assignment (STAL)
> - YOLO26n achieves **40.9 mAP** on COCO val2017 at **1.7 ms** latency on T4 TensorRT
> - Up to **43% faster CPU ONNX inference** vs. YOLO11n on Intel Xeon

**URL:** https://arxiv.org/abs/2606.03748
**PDF:** https://arxiv.org/pdf/2606.03748

---

### [R2] YOLO26 Architecture & Benchmarking Study
**Sapkota, R., & Karkee, M. (2026, March).**
*YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection.*
arXiv preprint arXiv:2509.25164v5. Cornell University / Washington State University.

> Key findings used in IBVAP:
> - Official benchmark table: YOLO26n — **40.9 mAP (50–95)**, 38.9 ms CPU ONNX, **1.7 ms T4 TRT**, 2.4M params, 5.4B FLOPs
> - YOLO26 maintains accuracy under FP16 and INT8 quantization — stable across both precision levels
> - Outperforms YOLOv11 and YOLOv12 under identical edge device conditions
> - Performance benchmarks on NVIDIA Jetson Nano and Orin reported and compared vs. YOLOv8/11/12/13

**URL:** https://arxiv.org/abs/2509.25164
**PDF:** https://arxiv.org/pdf/2509.25164

---

### [R3] YOLO Evolution Overview (YOLOv5 → YOLO26)
**Sapkota, R., & Karkee, M. (2026, March).**
*Ultralytics YOLO Evolution: An Overview of YOLO26, YOLO11, YOLOv8 and YOLOv5 Object Detectors for Computer Vision and Pattern Recognition.*
arXiv preprint arXiv:2510.09653v3. Cornell University.

> Key findings used in IBVAP:
> - YOLO26n (640px) reports ≈39.8% mAP at ~38.9 ms CPU — substantially faster than YOLO11n at comparable accuracy
> - YOLO11l reaches higher accuracy than YOLOv8l with almost half the parameters (25.3M vs 43.7M)
> - Quantized YOLO26n on Jetson Orin performs multi-task perception within **10–20 ms budget** at 50–100 Hz

**URL:** https://arxiv.org/abs/2510.09653
**PDF:** https://arxiv.org/pdf/2510.09653v3

---

### [R4] YOLO26 NMS-Free Framework Analysis
**Chakrabarty, S. (2026, January).**
*YOLO26: An Analysis of NMS-Free End to End Framework for Real-Time Object Detection.*
arXiv preprint arXiv:2601.12882. KIIT University.

> Key findings used in IBVAP:
> - Confirms end-to-end NMS-free architecture removes post-processing latency bottleneck
> - YOLO26n-seg achieves 34.0 mask mAP at 2.1 ms T4 latency with only 2.7M parameters
> - Validates ProgLoss "contour polishing" dynamic for high-fidelity instance segmentation

**URL:** https://arxiv.org/abs/2601.12882
**PDF:** https://arxiv.org/pdf/2601.12882

---

### [R5] Ultralytics Official Documentation
**Ultralytics Inc. (2026).**
*YOLO26 Model Documentation — Train, Validate, Predict, Export & Benchmark.*
Ultralytics Documentation Portal. Accessed August 2026.

> Key findings used in IBVAP:
> - Official training API: `model = YOLO("yolo26n.pt")` → `model.train(data="dataset.yaml", epochs=100, imgsz=640)`
> - Fine-tuning from COCO (80 classes) to custom dataset: 606 of 708 weight tensors transfer
> - All YOLO26 base models trained on COCO at 640×640 with MuSGD optimizer, batch size 128
> - TensorRT export: `model.export(format="engine")` creates `.engine` file for Jetson deployment

**URL (Models):** https://docs.ultralytics.com/models/yolo26
**URL (Train):** https://docs.ultralytics.com/modes/train
**URL (Fine-tune):** https://docs.ultralytics.com/guides/finetuning-guide
**URL (Jetson):** https://docs.ultralytics.com/guides/nvidia-jetson

---

### [R6] YOLO26 License Analysis
**LibreYOLO Technical Blog. (2026).**
*Is YOLO Free for Commercial Use? YOLOv8, YOLO11 and YOLO26 Licenses.*
LibreYOLO.com. July 2026.

> Key findings used in IBVAP:
> - YOLOv5, YOLOv8, YOLO11, and YOLO26 are distributed under **AGPL-3.0** (strong copyleft)
> - AGPL-3.0 requires complete source code disclosure if the product is hosted as a service
> - Production government deployment requires either Ultralytics commercial license or a permissively-licensed alternative

**URL:** https://www.libreyolo.com/articles/yolo-commercial-license

---

### [R7] YOLO26 vs YOLO11 vs YOLOv8 Practical Comparison (2026)
**rizwanai.com. (2026, July).**
*YOLO26 vs YOLO11 vs YOLOv8: which YOLO should you use?*
rizwanai.com. July 2026.

> Key findings used in IBVAP:
> - YOLOv8 has the deepest bench of tutorials and community resources
> - YOLO11 is the safe all-round default, matching YOLOv8 accuracy with fewer parameters
> - YOLO26 is the first NMS-free, end-to-end YOLO — faster on CPU and edge, simpler to export
> - Recommendation for new projects in 2026: YOLO11n for stability, YOLO26n for best edge performance

**URL:** https://www.rizwanai.com/blog/yolo26-vs-yolov8-vs-yolo11

---

## SECTION 2 — BORDER CAMERA SPECIFICATIONS & LIMITATIONS

---

### [R8] Pelco Border Security Camera Specifications
**Pelco Inc. (2026).**
*Border Security Cameras & Surveillance Systems — Advanced Long-Range Threat Detection.*
Pelco Official Product Page. Accessed August 2026.

> Key findings used in IBVAP:
> - High-end border PTZ cameras can see up to **46 kilometres** in any condition, day and night
> - Object identification at distances over **300m** (1,000 ft) in low-visibility conditions
> - Full 360-degree PTZ coverage with both visual and thermal sensor views
> - Open-platform design: ONVIF-compatible, integrates with existing CCTV infrastructure

**URL:** https://www.pelco.com/cameras/border

---

### [R9] Dual-Spectrum PTZ Border Surveillance — Sunell Technology
**Sunell Security. (2026).**
*Border Surveillance — Dual-Spectrum PTZ Cameras with AI Analytics.*
Sunell Technology Official Page. Accessed August 2026.

> Key findings used in IBVAP:
> - Dual-spectrum PTZ cameras with **optical zoom up to 86x** and thermal lenses to 150mm
> - Supports human/vehicle detection, fire, and intelligent behaviour analysis (intrusion, alert line crossing, loitering, retrograde movement, people counting, entry/exit detection)
> - All-weather monitoring: infrared imaging effective in nighttime and low-visibility conditions
> - Border line patrol: real-time monitoring in remote or hard-to-reach areas

**URL:** https://www.sunellsecurity.com/border-surveillance/

---

### [R10] PTZ 1000mm Long-Range Camera — AI Detection at 1km
**CCTV Camera Wholesale Price. (2026, June).**
*See Miles Away: How a PTZ 1000mm CCTV Camera Transforms Long-Range Security.*
cctvcamerawholesaleprice.com. June 2026.

> Key findings used in IBVAP:
> - Dual-sensor PTZ cameras with 1000mm optics can clearly identify people and vehicle licence plates at **1,000 metres**
> - Laser IR illuminators can extend night visibility up to **5 km** (vs. standard LED IR)
> - AI-powered PTZ cameras support automated motion detection, auto-tracking of people/vehicles, and event-based alerts
> - ONVIF compatibility ensures integration with existing NVR, VMS, or third-party software

**URL:** https://cctvcamerawholesaleprice.com/ptz-1000mm-cctv-camera/

---

### [R11] Long-Range Thermal PTZ — Military Border Patrol Specifications
**SPI Corp / X20. (2026).**
*PTZ FLIR Long Range Thermal Imaging Cameras for Border Security.*
x20.org. Accessed August 2026.

> Key findings used in IBVAP:
> - M7 PTZ thermal: human detection ranges **exceeding 20 kilometres** with MWIR cooled sensor
> - Ultra-long range configurations: 5 km / 10 km / 15 km / 20 km / 25 km up to **60 km** detection ranges configurable
> - 500x total zoom combining visible + thermal in one housing
> - Laser Range Finders (LRF) with detection ranges over 50 kilometres available
> - Military-grade ruggedization: IP66, -25°C to +60°C, 90% humidity, seismic shock-resistant

**URL:** https://www.x20.org/ptz-thermal-imaging-cameras/

---

### [R12] Axis Communications — Bispectral PTZ Border Camera
**Axis Communications. (2026, September).**
*AXIS Q6411-LE Bispectral PTZ Camera — AI-Powered Thermal and Visual Surveillance.*
Axis Communications Newsroom. 2026.

> Key findings used in IBVAP:
> - NETD <20mK — extremely high thermal sensitivity with low false alarm rate (key for AI pipeline calibration)
> - 31x optical zoom for following fast-moving objects
> - 55° fixed thermal field of view — ideal for wide open border areas
> - Planned availability Q3 2026 through Axis distribution channels

**URL:** https://newsroom.axis.com/news/ai-powered-thermal-ptz-camera

---

### [R13] AI Detection Range Limitations at Distance
**Total Security. (2025, November).**
*What Is the Maximum Distance Range of a CCTV Camera? Real-World Performance Explained.*
total-sec.co.uk. November 2025.

> Key findings used in IBVAP (critical AI limitation data):
> - **62% of users** who tried facial identification beyond 70 feet (21m) with a 4MP camera said it failed
> - Resolution and lens focal length matter more than brand for real-world identification range
> - Long-range PTZ (The Beacon 8.0): **1,200–1,500 feet detection** — commercial class only, $4,800+
> - Standard PTZ cameras: 200+ feet at night with manual aim

**URL:** https://total-sec.co.uk/what-is-the-maximum-distance-range-of-a-cctv-camera-real-world-performance-explained

---

### [R14] Border Camera Multi-Stream Architecture & AI Substream Standards
*(Derived from combined CIBMS documentation and camera vendor specifications)*

> Surveillance standard protocol summary for IBVAP:
> - **Primary Stream** (Recording): 1080p or 4K @ 25–30 FPS — H.265 — for human review and forensics
> - **Secondary Stream** (AI Pipeline): **720p @ 15–25 FPS** — H.264/H.265 — fed into inference backend
> - **Auxiliary Stream** (Telemetry): 640×360 @ 10–15 FPS — for low-bandwidth remote C2
> - Thermal sensor native resolution: typically **640×512 pixels** at 25–30 FPS (or 9–25 FPS legacy)
> - IBVAP recommendation: always run AI on secondary stream (720p), never on primary 4K — directly
>   reduces compute by ~4–9x with no detection quality loss for the AI task

---

## SECTION 3 — JETSON ORIN PERFORMANCE

---

### [R15] NVIDIA Official Jetson Module Specifications
**NVIDIA Corporation. (2026).**
*Jetson Modules, Support, Ecosystem, and Lineup.*
NVIDIA Developer Portal. Accessed August 2026.

> Key specifications used in IBVAP:
> | Module | AI Performance | Power | Notes |
> |---|---|---|---|
> | Jetson Orin Nano 4GB | Up to 40 TOPS | 7–15W | Entry-level |
> | Jetson Orin Nano Super 8GB | Up to 67 TOPS | 7–25W | 140x Jetson Nano |
> | Jetson Orin NX 8GB | Up to 70 TOPS | 10–25W | |
> | Jetson Orin NX 16GB | Up to 157 TOPS | 10–25W | Multi-stream capable |
> | Jetson AGX Orin 32/64GB | Up to 275 TOPS | 15–60W | 8x prev. generation |
> - All Orin modules production-supported through **2032**

**URL:** https://developer.nvidia.com/embedded/jetson-modules

---

### [R16] Jetson Orin Nano vs NX Cost/Performance Analysis 2026
**Edge AI Stack. (2026, April).**
*Jetson Orin Nano vs NX: 40 vs up to 157 TOPS — Which Do You Need?*
edgeaistack.ai. April 2026.

> Key findings used in IBVAP:
> - 2026 retail pricing: **Orin Nano Super $249 (~₹21,000)**, Nano 8GB $299, NX 8GB $399, **NX 16GB $599 (~₹50,000)**
> - NX costs roughly 1.5–2.4x more but delivers 3–4x the AI compute
> - On TOPS-per-dollar: NX offers better efficiency at scale (5+ camera deployments)
> - **IBVAP cost narrative:** ₹21K/unit vs. lakhs per proprietary FRS/ANPR camera hardware

**URL:** https://edgeaistack.ai/blog/jetson-orin-nano-vs-orin-nx-2026/

---

### [R17] Jetson Orin Power Consumption Comparison
**Edge AI Stack. (2026, July).**
*Jetson Orin Nano vs NX vs AGX Power Consumption (2026).*
edgeaistack.ai. July 2026.

> Key findings used in IBVAP:
> - Orin Nano idle: **4.5–5.5W** / typical inference: **8–12W**
> - Orin NX idle: 7–10W
> - Orin NX best TOPS-per-watt: ~10.5 TOPS/W at 15W default power mode
> - Standard PoE (802.3af, 12.95W) supports Nano at typical inference — no new power infra needed at BOPs
> - JetPack 7.2 (June 2026): AGX Orin 32GB raised from 200 → 241 TOPS in MAXN Super preset

**URL:** https://edgeaistack.ai/blog/jetson-power-consumption-comparison/

---

### [R18] YOLO26 on NVIDIA Jetson — Official Setup & Benchmarks
**Ultralytics Inc. (2026).**
*YOLO26 on NVIDIA Jetson Setup & Benchmarks.*
Ultralytics Documentation. Accessed August 2026.

> Key findings used in IBVAP:
> - Benchmarks tested on: Jetson AGX Thor, **Jetson AGX Orin 64GB**, **Jetson Orin Nano Super**, **Jetson Orin NX 16GB**
> - 11 export formats benchmarked: PyTorch, TorchScript, ONNX, OpenVINO, **TensorRT**, TF Lite, MNN, NCNN
> - Only PyTorch, TorchScript, and TensorRT use the Jetson GPU — all others are CPU-only
> - TensorRT is the recommended export for maximum Jetson performance
> - Export command: `model.export(format="engine", device=0, half=True)` → FP16 TensorRT engine

**URL:** https://docs.ultralytics.com/guides/nvidia-jetson

---

### [R19] YOLO + ByteTrack Jetson AGX Orin Benchmark (ROS 2)
**Shankar. (2026).**
*A Unified ROS 2 Benchmark of YOLOv8, YOLO11, and YOLO26 with ByteTrack on NVIDIA Jetson AGX Orin.*
GitHub Repository. GITAM University. 2026.

> Key findings used in IBVAP:
> - 15 YOLO models benchmarked (nano through XL, YOLOv8 + YOLO11 + YOLO26) with ByteTrack tracker
> - **TensorRT FP16 improves mean throughput by +19.9%** across all 15 models vs. PyTorch
> - Reduces power on **13 of 15 models** (up to 1.87W savings per model)
> - Improves energy efficiency (FPS/W) on **all 15 models**
> - Hardware telemetry captured via `tegrastats` (GPU/CPU usage, RAM, temperature, power)

**Citation:** Shankar (2026). *yolo-bytetrack-ros2-benchmark.* GitHub.
**URL:** https://github.com/Shankar0415/yolo-bytetrack-ros2-benchmark

---

### [R20] YOLO26 on Jetson — DeepStream & TensorRT Deployment
**Ultralytics Inc. (2026).**
*YOLO26 on Jetson: DeepStream & TensorRT Deployment Guide.*
Ultralytics Documentation. Accessed August 2026.

> Key findings used in IBVAP:
> - INT8 calibration, multi-stream DeepStream setup, and benchmark results documented
> - Tested on: **Jetson Orin Nano Super** (JP6.1), Jetson Orin NX 16GB (JP5.1.3), Jetson Nano 4GB (JP4.6.4)
> - YOLO26 models on Jetson Orin NX 16GB performance varies based on TensorRT precision level
> - Guide works across entire NVIDIA Jetson hardware lineup (latest and legacy)

**URL:** https://docs.ultralytics.com/guides/deepstream-nvidia-jetson

---

### [R21] NVIDIA Jetson for AI Projects 2026 — Practical Guide
**Aleksandrov, Y. (2026, March).**
*NVIDIA Jetson for AI Projects: Getting Started in 2026.*
DEV Community. March 2026.

> Key findings used in IBVAP:
> - Jetson Orin Nano 8GB at 40 TOPS: "sweet spot for most projects — enough headroom for 7B LLMs plus simultaneous vision processing"
> - JetPack SDK includes CUDA, TensorRT, cuDNN, and DeepStream for optimized inference out-of-box
> - Practical AI inference guide for configuring power modes and optimizing throughput

**URL:** https://dev.to/yankoaleksandrov/nvidia-jetson-for-ai-projects-getting-started-in-2026-4g3f

---

### [R22] Edge Deployment Benchmark — YOLOv10 on Jetson Orin Nano (Comparison Reference)
**Barros, R. et al. (2025).**
*LAF-YOLOv10 with Partial Convolution Backbone, Attention-Guided Feature Pyramid, Auxiliary P2 Head, and Wise-IoU Loss for Small Object Detection in Drone Aerial Imagery.*
arXiv preprint arXiv:2602.13378.

> Key findings used in IBVAP (comparison baseline):
> - YOLOv10n achieves **31.2 ms latency on Jetson Orin Nano** at TensorRT FP16 (= ~32 FPS single stream)
> - This baseline helps establish that YOLO26n (lighter architecture, no NMS) achieves lower latency
> - Inference measured at batch size 1 over 1000 forward passes after 200-pass warmup

**URL:** https://arxiv.org/pdf/2602.13378

---

## QUICK REFERENCE TABLE

| Ref | Topic | Key Stat | Source |
|---|---|---|---|
| R1 | YOLO26 official paper | Primary citation | arXiv:2606.03748 |
| R2 | YOLO26n benchmark | 40.9 mAP, 1.7ms T4 TRT | arXiv:2509.25164 |
| R3 | YOLO26n CPU speed | 38.9ms, 43% faster than YOLO11n | arXiv:2510.09653 |
| R5 | Training API | Official Ultralytics docs | docs.ultralytics.com |
| R6 | AGPL-3.0 license | Source-disclosure obligation | libreyolo.com |
| R8 | Camera max range | 46 km (military PTZ) | Pelco |
| R9 | Dual-spectrum PTZ | 86x optical zoom, AI analytics | Sunell |
| R10 | 1000mm PTZ | 1km person/plate ID | CCTV Wholesale |
| R11 | Thermal FLIR | 20 km+ human detection | SPI Corp / x20.org |
| R13 | AI face ID limit | 62% failure beyond 70 ft (4MP) | Total Security |
| R15 | Jetson Orin specs | 40–275 TOPS, supported to 2032 | NVIDIA Developer |
| R16 | Jetson pricing 2026 | Orin Nano $249, NX $599 | edgeaistack.ai |
| R17 | Jetson power | Nano 8–12W inference | edgeaistack.ai |
| R19 | TensorRT FP16 gain | +19.9% throughput, lower power | GitHub Benchmark |
| R22 | Orin Nano baseline | YOLOv10n ~32 FPS single stream | arXiv:2602.13378 |

---

*Document prepared for IBVAP SIH 2026 submission. References current as of August 2026.*
