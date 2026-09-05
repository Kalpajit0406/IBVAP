# Connecting real CCTV cameras to IBVAP

For the college demo you'll be handed access to 5–10 existing cameras. This is
everything you need to get them into the pipeline.

---

## 1. How CCTV is wired, in practice

| setup | what you get | how you reach it |
|---|---|---|
| **IP cameras + NVR** (today's standard) | The cameras are PoE, plugged into an **NVR** (or a PoE switch behind one). The NVR records and re-exposes every camera as a numbered **channel**. | Usually **one NVR IP + one login**, N channels. |
| **IP cameras, no NVR** | Each camera has its own IP on the LAN. | One IP + login **per camera**. |
| **Analog cameras + DVR** (older) | Coax (BNC) cameras into a **DVR** that digitises them. | Same as an NVR — one DVR IP, N channels. |

In **all three** cases the thing you talk to (camera, NVR, or DVR) speaks
**RTSP**. That's the only protocol IBVAP needs.

Other things a device may also offer — you don't need them, but they exist:
ONVIF (a discovery/control standard — useful for *finding* cameras and their
RTSP URLs), HTTP MJPEG/snapshot (old cameras), HLS/WebRTC (some VMS), and
vendor SDKs (Hikvision/Dahua — proprietary, avoid).

---

## 2. Main stream vs sub-stream — **use the sub-stream**

Every IP-camera / NVR channel publishes at least two encodings:

| | resolution | fps | bitrate | for |
|---|---|---|---|---|
| **main** | 1080p – 4K | 15–30 | 4–8 Mbps | recording |
| **sub** | D1 / 640×360 / 720p | 10–15 | 0.3–1 Mbps | **analytics, previews** |

Detection runs at 640 px internally and at demo range (people a few metres from
the camera) the sub-stream loses nothing that matters. It also cuts CPU decode
cost ~4× — which is what lets one laptop carry 8–10 streams. IBVAP's
`config.yaml` `cctv.decode_fps` further caps decoding at 15 fps.

**Always point `url:` at the sub-stream** (see the table below for how).

---

## 3. RTSP URL patterns by vendor

`rtsp://<user>:<pass>@<ip>:<port>/<path>` — port is **554** unless told
otherwise. For an **NVR**, the channel number is encoded in the path.

| vendor | main stream | sub stream |
|---|---|---|
| **Hikvision** (& most OEM/rebrands) | `/Streaming/Channels/101` | `/Streaming/Channels/102` |
| Hikvision NVR, channel *n* | `/Streaming/Channels/<n>01` | `/Streaming/Channels/<n>02` |
| **Dahua / CP Plus / Amcrest** | `/cam/realmonitor?channel=1&subtype=0` | `/cam/realmonitor?channel=1&subtype=1` |
| Dahua NVR, channel *n* | `channel=<n>&subtype=0` | `channel=<n>&subtype=1` |
| **Axis** | `/axis-media/media.amp` | `/axis-media/media.amp?resolution=640x360` |
| **Uniview (UNV)** | `/media/video1` | `/media/video2` |
| **Bosch** | `/rtsp_tunnel` | `/rtsp_tunnel?inst=2` |
| **Hanwha / Samsung** | `/profile1/media.smp` | `/profile2/media.smp` |
| **Vivotek** | `/live.sdp` | `/live2.sdp` |
| **generic ONVIF** | run `discover_cameras.py` — paths vary | same |

So four Hikvision-NVR channels, sub-stream:

```
rtsp://admin:pass@10.0.0.50:554/Streaming/Channels/102
rtsp://admin:pass@10.0.0.50:554/Streaming/Channels/202
rtsp://admin:pass@10.0.0.50:554/Streaming/Channels/302
rtsp://admin:pass@10.0.0.50:554/Streaming/Channels/402
```

**Credentials in the URL:** if the password contains `@ : / ?` you must
percent-encode it (`@`→`%40`, `:`→`%3A`, `/`→`%2F`). Or set a demo password
without those characters.

---

## 4. What to ask the college IT / lab staff for

1. The **NVR (or camera) IP address(es)** and which subnet/VLAN they're on.
2. A **username + password** — ideally a **read-only / "viewer" account**, not
   the admin one.
3. Confirmation that **RTSP is enabled** on the device (some ship with it off —
   it's a checkbox in the NVR's Network → Advanced settings).
4. The **RTSP port** if it's not 554.
5. How many **channels** and, if possible, a **channel → location** list
   ("ch3 = main gate") so you can set `zone_sensitivity` meaningfully.
6. Whether your demo laptop will be **on the same network** as the cameras, or
   needs a port on a switch / a VLAN allowance / a static IP. Cameras are very
   often on an **isolated CCTV VLAN** with no route from the general LAN.
7. If remote: a way in (VPN, or the NVR's ports forwarded). Do **not** rely on
   the NVR's P2P cloud — it won't give you clean RTSP.

---

## 5. Get them into IBVAP

### a. Find the cameras (optional)

```bash
python discover_cameras.py --user admin --pass <pw>
```

WS-Discovery on the LAN; with credentials + `pip install onvif-zeep` it also
prints each camera's real RTSP URL and a paste-ready `streams:` block.

### b. Test each URL before you trust it

```bash
python rtsp_probe.py "rtsp://admin:pass@10.0.0.50:554/Streaming/Channels/102"
```

Reports open time, resolution, real fps, jitter, and a verdict (it warns if you
gave it a main stream, or if fps/jitter are bad). A quick sanity check is also
`vlc "<url>"`.

### c. Put them in `config.yaml`

```yaml
streams:
  - { id: 0, name: "Main Gate", url: "rtsp://admin:pass@10.0.0.50:554/Streaming/Channels/102", zone_sensitivity: 0.9 }
  - { id: 1, name: "Corridor",  url: "rtsp://admin:pass@10.0.0.50:554/Streaming/Channels/202", zone_sensitivity: 0.6 }
  - { id: 2, name: "Parking",   url: "rtsp://admin:pass@10.0.0.50:554/Streaming/Channels/302", zone_sensitivity: 0.7 }
  - { id: 3, name: "Perimeter", url: "rtsp://admin:pass@10.0.0.50:554/Streaming/Channels/402", zone_sensitivity: 0.9 }

cctv:
  transport: tcp        # keep TCP — UDP RTSP tears on any loss
  decode_fps: 15
  reconnect_delay: 3.0
  stall_timeout: 8.0
```

`id` values are free integers; keep them distinct from any phone slots. A row
with `url: ws` (or no `url`) stays a phone slot — CCTV and phones run together.

### d. Run

```bash
python server.py            # or:  python run_demo.py --no-feed
```

Each camera opens on its own decode thread, auto-reconnects, and appears on the
dashboard (`http://localhost:8090/monitor`) as `kind: rtsp` with its live fps
and reconnect count. A frozen feed can be kicked with
`POST /api/reconnect/<id>`.

---

## 6. How IBVAP handles a real camera (`src/rtsp_capture.py`)

- **One decode thread per camera.** A frozen or unplugged camera can never
  stall the shared muxer — `.read()` just returns the newest decoded frame or
  nothing.
- **RTSP forced over TCP** (`OPENCV_FFMPEG_CAPTURE_OPTIONS`) with a 5 s socket
  timeout, so a dead host fails fast instead of hanging.
- **`grab()` every loop, `retrieve()` (decode) only at `decode_fps`** — the
  cheap part keeps the socket drained (no latency build-up), the expensive part
  runs only as often as needed.
- **Auto-reconnect with backoff** + a **stall watchdog** that reopens a stream
  that goes quiet without erroring.
- **Every frame normalised** to 1280×720 so the detector sees one size
  regardless of camera.
- **Credentials redacted** in every log line and in `/status` / `/devices`.

---

## 7. Troubleshooting

| symptom | fix |
|---|---|
| `rtsp_probe.py` → "FAILED / cannot open" | `ping` the IP first. Wrong subnet/VLAN is the #1 cause — cameras are usually isolated. Then check port, path, credentials, and that RTSP is enabled on the NVR. |
| Opens, then drops every few seconds | Network loss on a UDP stream — you're not on TCP. Keep `transport: tcp`. Move the laptop onto the CCTV switch. |
| Green smears / blocky artifacts | Same — packet loss on UDP, or a main stream saturating the link. Use TCP + the sub-stream. |
| Very high latency (several seconds) | You pointed at the **main** stream. Switch to the sub-stream; drop `decode_fps`. |
| Only the NVR shows in `discover_cameras.py`, not each camera | Normal for NVR setups. Use the channel URL patterns in §3. |
| Auth works in VLC but not here | Password has `@`/`:`/`/` — percent-encode it in the URL. |
| 8–10 cams pegging the CPU | Sub-streams + `decode_fps: 10`. If still hot, drop some cameras or lower `image_size` in `config.yaml` (needs a TensorRT re-export). GPU (NVDEC) decode is the real fix but needs a custom OpenCV/FFmpeg build — out of scope for the demo. |
| Camera is H.265/HEVC and won't decode | Rare with `opencv-python` (its FFmpeg has HEVC). If it happens, set the camera's sub-stream codec to H.264 in its web UI. |
