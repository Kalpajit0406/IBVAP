# IBVAP — Startup Guide (new machine, from zero)

This walks a **complete beginner** from a fresh Windows PC to a running IBVAP
demo with a phone streaming into it. Budget **30–45 minutes**, most of it
downloads.

> Already know your way around Python + CUDA? The 6-line version is at the very
> bottom under **"Fast path"**.

---

## 0. What you need first

| | requirement | why |
|---|---|---|
| **GPU** | NVIDIA, CUDA-capable, **≥ 6 GB VRAM** (RTX 3050 or better) | the detector runs on the GPU. CPU-only works but ~10× slower and no TensorRT — fine only for a quick look. |
| **OS** | Windows 10 / 11 (primary target) | Linux works too — see §10. |
| **Disk** | ~6 GB free | PyTorch (~2.5 GB) + models + deps. |
| **Network** | internet for the first run | one-time downloads of PyTorch and the YOLO weights. |
| **Time** | 30–45 min | ~25 min is `pip` downloading. |

Check your GPU is seen by Windows — open **PowerShell** (Start → type
"PowerShell") and run:

```powershell
nvidia-smi
```

* Prints a table with your GPU name and a "Driver Version" → **good, continue.**
* `nvidia-smi is not recognized` → your NVIDIA driver is missing or old.
  Download **"Game Ready" or "Studio" driver** for your card from
  <https://www.nvidia.com/download/index.aspx>, install, **reboot**, try again.
  Any driver from the last ~2 years supports what we need.

---

## 1. Install the three system tools

### 1a. Python **3.11** (not 3.12, 3.13, or 3.14)

The pinned versions of PyTorch and Ultralytics only support 3.11.

1. Go to <https://www.python.org/downloads/windows/> → find the latest
   **"Python 3.11.x"** → download **"Windows installer (64-bit)"**.
2. Run it. On the first screen **tick "Add python.exe to PATH"**, then
   "Install Now".
3. New PowerShell window, check:

   ```powershell
   python --version
   ```

   Must say `Python 3.11.x`. If it says 3.12/3.13 (you had another Python), use
   `py -3.11` everywhere this guide says `python`.

### 1b. Git

1. <https://git-scm.com/download/win> → download → install with all defaults.
2. Check:

   ```powershell
   git --version
   ```

### 1c. (optional, for a faster model) nothing yet

TensorRT is installed later in §4c and is optional. Skip for now.

---

## 2. Download the code

```powershell
cd $HOME\Documents
git clone https://github.com/Kalpajit0406/IBVAP.git
cd IBVAP
```

You're now in the project folder. Everything below runs from here.

---

## 3. Set up the Python environment

### 3a. Create an isolated environment (recommended)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. **Every new terminal** you use for
this project needs that `Activate.ps1` line again.

> If PowerShell blocks the activate script ("running scripts is disabled"),
> run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` then retry.

### 3b. Install PyTorch **with CUDA** — do this FIRST, on its own

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

~2.5 GB, a few minutes. This special index URL is the **CUDA 12.6 build**.
A plain `pip install torch` gives you a CPU-only build and the GPU won't be
used — if you did that by mistake, `pip uninstall torch torchvision` and run
the line above.

### 3c. Install everything else

```powershell
pip install -r requirements.txt
```

### 3d. Confirm the GPU is visible to PyTorch

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')"
```

Expected:

```
CUDA available: True
NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

If it says `False` / `CPU ONLY`, see **Troubleshooting §9** before continuing —
the demo will still run but only on CPU.

---

## 4. One-time setup files

A fresh clone deliberately does **not** contain model weights, the TLS
certificate, or test videos (they're large or machine-specific). You create
them now.

### 4a. TLS certificate — lets phones use their camera on your WiFi

```powershell
python gen_cert.py
```

Creates `cert.pem` + `key.pem`, valid for your PC's **current** LAN IP
addresses. Browsers block `getUserMedia` (camera) on plain `http://`, so the
phone link must be `https://` — this cert provides it.
**Re-run this whenever your PC joins a different WiFi** (its IP changes).

### 4b. Test videos — a self-contained demo without real cameras

`make_test_videos.py` turns a folder of images into 4 short 720p/24fps clips.
Pick **one** option:

**Option A — the dataset the project was tuned on** (needs a free Kaggle account):

```powershell
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('constantinwerner/human-detection-dataset'))"
```

Copy the path it prints, then (note the sub-folder and the quotes):

```powershell
python make_test_videos.py --src "C:\Users\<you>\.cache\kagglehub\datasets\constantinwerner\human-detection-dataset\versions\5\human detection dataset"
```

**Option B — any folder of photos you already have:**

```powershell
python make_test_videos.py --src "C:\Users\<you>\Pictures"
```

**Option C — skip test videos entirely.** You'll drive the demo with real
phones (§6) or your own video files: `python feed_test.py --src "C:\my\clips" --cams 4`.

Either way you end up with a `test_videos\` folder.

### 4c. TensorRT engine — optional, ~2× faster inference

The pipeline runs fine on `yolo26n.pt` (PyTorch). For the faster path:

```powershell
pip install "tensorrt-cu12==10.13.3.9" onnx onnxslim
python export_engine.py
```

Takes ~8 minutes and produces `yolo26n.engine` **for this GPU only**.
`config.yaml` already points at `yolo26n.engine`; if the file isn't there the
detector automatically falls back to `yolo26n.pt`, so this step is safe to skip
and add later.

> **Must be `10.13.3.9`.** TensorRT 11.x removed an API the exporter uses and
> the build will fail.

---

## 5. First run

```powershell
python run_demo.py --cams 4
```

The first run also downloads `yolo26n.pt` and `yolo26n-pose.pt` (~16 MB) — this
is normal, once.

This one command starts:

1. the **server** (dashboard on `http://localhost:8090`, phone intake on
   `https://<lan-ip>:8443`),
2. a **replay** of the 4 test clips as if they were 4 cameras,
3. your **browser** at the dashboard.

You should see a 2×2 video mosaic with green/blue boxes on people and
vehicles, a **CONNECTED DEVICES** panel on the right, and an **ALERT LOG**.

Press **Ctrl+C** in the terminal to stop everything.

**No test videos?** Run the server alone and use real phones:

```powershell
python run_demo.py --no-feed
```

---

## 6. Stream from a phone

### 6a. Phone on the **same WiFi** as the PC

1. Find the PC's IP: `ipconfig` → look for **IPv4 Address** under your active
   adapter, e.g. `192.168.1.42`. (The server also prints it on startup.)
2. On the phone's browser open: `https://192.168.1.42:8443/cam/0`
3. A security warning appears (self-signed cert). Tap
   **Advanced → Proceed / Continue** — once per phone.
4. Tap **Allow** for the camera. It's now streaming; watch it appear on the
   dashboard.
5. Second phone → `/cam/1`, third → `/cam/2`, etc.

**If the phone can't load the page:**
* First server run pops a **Windows Firewall** dialog — click **Allow access**
  (Private networks). If you dismissed it: Windows Security → Firewall → Allow
  an app → add `python.exe`.
* Guest / public WiFi often has "client isolation" that blocks phone-to-PC
  traffic. Use a home/hotspot network, or the tunnel below.

### 6b. Phone on **mobile data or a different network** — via ngrok

One-time:

1. Free account at <https://dashboard.ngrok.com> → copy your **authtoken**.
2. Install ngrok: <https://ngrok.com/download> (unzip `ngrok.exe` somewhere on
   PATH), then:

   ```powershell
   ngrok config add-authtoken <your-token>
   ```

Every demo:

```powershell
python server.py
python tunnel.py        # in a second terminal
```

`tunnel.py` prints a public `https://<something>.ngrok-free.dev/cam/0` link and
a QR code — open or scan it on any phone, anywhere. That URL is **stable** for
your account (same every run). Or do it in one go:
`python run_demo.py --no-feed --tunnel`.

### 6c. Phone via **your own domain**

`stream.yourdomain.com/cam/0` — see **`docs/CUSTOM_DOMAIN.md`** (cloudflared
named tunnel).

---

## 6½. Connect real CCTV cameras

You'll usually be given **NVR access** (one IP + login, several channels), not
per-camera IPs. Every NVR/DVR/IP-camera speaks **RTSP** — that's all IBVAP
needs. Use each camera's **sub-stream** (low-res), not the main recording
stream.

1. **Find them** (optional):

   ```powershell
   pip install onvif-zeep
   python discover_cameras.py --user admin --pass <pw>
   ```

   Prints the cameras on the LAN and a ready-to-paste `streams:` block.

2. **Test each URL** before trusting it:

   ```powershell
   python rtsp_probe.py "rtsp://admin:pass@10.0.0.50:554/Streaming/Channels/102"
   ```

   Tells you if it opens, its resolution/fps, and whether it's a good analytics
   stream. (`vlc "<url>"` also works as a quick check.)

3. **Add them to `config.yaml`** — replace the `- { id: 0, ... url: "ws" ... }`
   phone rows (or add alongside them):

   ```yaml
   streams:
     - { id: 0, name: "Main Gate", url: "rtsp://admin:pass@10.0.0.50:554/Streaming/Channels/102", zone_sensitivity: 0.9 }
     - { id: 1, name: "Corridor",  url: "rtsp://admin:pass@10.0.0.50:554/Streaming/Channels/202", zone_sensitivity: 0.6 }
   ```

   `config.yaml` already lists the RTSP URL patterns for Hikvision, Dahua, Axis,
   Uniview, etc. in comments.

4. **Run** `python server.py` — each camera opens on its own thread,
   auto-reconnects, and shows on the dashboard as `kind: rtsp` with its live fps
   and reconnect count. Kick a frozen feed with a `POST /api/reconnect/<id>`.

Full guide (how CCTV is wired, vendor URL tables, what to ask college IT for,
VLAN/firewall notes, troubleshooting): **`docs/CCTV_INTEGRATION.md`**.

---

## 7. Verify the install

```powershell
python diagnose.py             # end-to-end health check  → "ALL CHECKS PASSED"
python tests/test_posture.py   # pose/weapon geometry     → "12/12 passed"
python benchmark.py --cams 4   # throughput comparison    (needs test_videos)
```

---

## 8. Where the settings live

Everything is in **`config.yaml`**:

| key | meaning |
|---|---|
| `model.weights` | `yolo26n.engine` (default) or `yolo26n.pt` |
| `model.detect_fps` / `stream_fps` | 8 fps YOLO detection, 24 fps track display |
| `model.max_batch` | frames per GPU pass (also the engine's fixed batch — re-export if changed) |
| `model.motion.*` | motion-gate sensitivity (lower = more frames reach the GPU) |
| `pose.aim_*` | two-handed weapon-posture thresholds |
| `risk.threshold_high` / `threshold_critical` | alert cutoffs (0–100) |
| `streams:` | RTSP / webcam / file sources for **local mode** (`python main.py`) |

Local mode without the web UI (OpenCV window, uses `streams:` from the config):

```powershell
python main.py
python main.py --verify-chain      # check the evidence hash chain
```

`CLAUDE.md` has the full architecture, the measured numbers, and why each piece
is built the way it is.

---

## 9. Troubleshooting

| symptom | cause & fix |
|---|---|
| `torch.cuda.is_available()` → **False** | CPU-only torch got installed. `pip uninstall -y torch torchvision`, then re-run the **§3b** line with the `--index-url .../cu126`. |
| `nvidia-smi` not recognized | NVIDIA driver missing. Install from nvidia.com, reboot. |
| `python --version` shows 3.12 / 3.13 | Install Python **3.11**; recreate the venv with `py -3.11 -m venv .venv`. |
| `ModuleNotFoundError` for fastapi / cv2 / ultralytics | The venv isn't active. Run `.\.venv\Scripts\Activate.ps1` (prompt shows `(.venv)`). |
| `Activate.ps1 cannot be loaded ... running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer **Y**, retry. |
| `make_test_videos.py`: *"Source images not found"* | The default path is Kaggle's cache and doesn't exist on your PC. Pass `--src "<folder of images>"` (see §4b). |
| `export_engine.py` fails with a `BuilderFlag` / API error | You have TensorRT 11.x. `pip install "tensorrt-cu12==10.13.3.9" onnx onnxslim` and retry. |
| Server: *"Port 8090 (or 8443) is already answering"* | Another server instance is running, or (rarely) Steam is on 8080. Close the other one: `Get-Process python | Stop-Process -Force`. |
| Phone: *"getUserMedia unavailable"* / camera won't turn on | The page must be **HTTPS**. On the LAN: run `python gen_cert.py`, use the `:8443` link. Off-LAN: use `python tunnel.py`. |
| Phone loads nothing at `https://192.168.x.x:8443` | Same WiFi? Windows Firewall — allow `python.exe` on Private networks. Guest WiFi with client isolation blocks it — use `tunnel.py`. |
| Certificate warning won't go away / mentions a different IP | Your PC's IP changed since `gen_cert.py`. Re-run it, restart the server. |
| ngrok: *"authtoken"* / *"authentication failed"* | `ngrok config add-authtoken <token>` — token from dashboard.ngrok.com. |
| Feed from a phone is several seconds behind | Weak phone uplink. The **latency badge** on that tile shows it; `camera.html` already drops frames to recover. Prefer same-WiFi, close other bandwidth-heavy apps. |
| `run_demo.py` prints *"A process exited"* and stops | Usually the replay can't find `test_videos\`. Build them (§4b) or start with `--no-feed`. |
| First `python server.py` is slow to say "ready" | It's downloading `yolo26n.pt` / `yolo26n-pose.pt` and warming up the GPU. One-time, ~20–40 s. |

---

## 10. Linux

Same flow, with:

* activate the venv with `source .venv/bin/activate`
* forward-slash paths
* `nvidia-smi` from the distro's NVIDIA driver package; CUDA toolkit not needed
  (PyTorch bundles its own runtime)
* the Windows-console UTF-8 shims in the scripts are simply no-ops

---

## 11. What a fresh clone doesn't ship (and how to regenerate)

| missing | command |
|---|---|
| `yolo26n.pt`, `yolo26n-pose.pt` | downloaded automatically on first `server.py` run |
| `yolo26n.engine`, `yolo26n.onnx` | `python export_engine.py` (per-GPU) |
| `test_videos\` | `python make_test_videos.py --src <images>` |
| `cert.pem`, `key.pem` | `python gen_cert.py` |
| `data\` (events DB, hash chain) | created at runtime |

---

## Fast path (for the impatient)

```powershell
git clone https://github.com/Kalpajit0406/IBVAP.git && cd IBVAP
py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python gen_cert.py
python make_test_videos.py --src "C:\some\folder\of\images"
python run_demo.py --cams 4
```

Optional TensorRT: `pip install "tensorrt-cu12==10.13.3.9" onnx onnxslim && python export_engine.py`.
Phone off-LAN: `ngrok config add-authtoken <token>` once, then `python run_demo.py --no-feed --tunnel`.
