"""
diagnose.py — end-to-end pipeline check. Run while server.py is running.

    python diagnose.py

Checks, in order:
  1. Server reachable            (GET /status)
  2. Camera WebSocket accepts    (WS /ws/camera/99 + push a synthetic frame)
  3. Detector produced a frame   (/status buffered_bgr)
  4. MJPEG endpoint emits JPEGs  (GET /stream/99, read one part)
"""
import asyncio
import ssl
import sys
import time
import urllib.request

import cv2
import numpy as np
import websockets

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

BASE = "https://127.0.0.1:8443"
WS   = "wss://127.0.0.1:8443"
CAM  = 99

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

OK, BAD = "[ OK ]", "[FAIL]"


def http_get(path: str, timeout: float = 5.0):
    req = urllib.request.Request(BASE + path)
    return urllib.request.urlopen(req, timeout=timeout, context=CTX)


def make_test_frame(i: int) -> bytes:
    """640x480 frame with a person-shaped white block and frame counter."""
    img = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.rectangle(img, (260, 120), (380, 420), (200, 200, 200), -1)   # body
    cv2.circle(img, (320, 90), 45, (200, 200, 200), -1)               # head
    cv2.putText(img, f"TEST FRAME {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


async def main() -> int:
    failures = 0

    # ── 1. Server reachable ──────────────────────────────────────────────────
    print("\n1. Server reachable")
    try:
        body = http_get("/status").read().decode()
        print(f"   {OK} /status → {body[:200]}")
    except Exception as e:
        print(f"   {BAD} cannot reach {BASE}/status → {e}")
        print("        Start the server first:  python server.py")
        return 1

    # ── 2. Camera WebSocket ──────────────────────────────────────────────────
    print("\n2. Camera WebSocket intake")
    try:
        async with websockets.connect(f"{WS}/ws/camera/{CAM}", ssl=CTX,
                                      max_size=5_000_000) as ws:
            for i in range(30):
                await ws.send(make_test_frame(i))
                await asyncio.sleep(0.05)
            print(f"   {OK} pushed 30 synthetic frames to CAM-{CAM}")
            await asyncio.sleep(1.5)   # let the detector chew on them

            # ── 3. Detector output ───────────────────────────────────────────
            print("\n3. Detector produced annotated frames")
            import json
            st = json.loads(http_get("/status").read().decode())
            buf = st.get("buffered_bgr", {})            # {cam_id: True} once a frame is annotated
            if str(CAM) in buf:
                print(f"   {OK} CAM-{CAM} has an annotated frame buffered")
                print(f"        device={st.get('device')}  "
                      f"fps={st.get('fps')}  infer={st.get('inference_ms')} ms")
                print(f"        meta={st.get('meta', {}).get(str(CAM))}")
            else:
                print(f"   {BAD} no annotated frame for CAM-{CAM}")
                print(f"        buffered_bgr keys: {list(buf)}")
                failures += 1

            # ── 4. MJPEG endpoint ────────────────────────────────────────────
            print("\n4. MJPEG endpoint emits JPEG parts")
            try:
                resp = http_get(f"/stream/{CAM}", timeout=8.0)
                ctype = resp.headers.get("Content-Type", "")
                print(f"        Content-Type: {ctype}")
                if "multipart/x-mixed-replace" not in ctype:
                    print(f"   {BAD} wrong content type")
                    failures += 1
                else:
                    # Keep pushing so the stream has something to emit
                    async def keep_pushing():
                        for i in range(40):
                            await ws.send(make_test_frame(1000 + i))
                            await asyncio.sleep(0.05)
                    pusher = asyncio.create_task(keep_pushing())

                    chunk = await asyncio.get_running_loop().run_in_executor(
                        None, resp.read, 65536)
                    await pusher

                    if b"--frame" in chunk and b"\xff\xd8" in chunk:
                        jpeg_start = chunk.find(b"\xff\xd8")
                        print(f"   {OK} received {len(chunk)} bytes, "
                              f"JPEG SOI at offset {jpeg_start}")
                    else:
                        print(f"   {BAD} no JPEG data in stream "
                              f"(got {len(chunk)} bytes)")
                        print(f"        head: {chunk[:120]!r}")
                        failures += 1
                resp.close()
            except Exception as e:
                print(f"   {BAD} MJPEG read failed → {e}")
                failures += 1

    except Exception as e:
        print(f"   {BAD} camera WebSocket failed → {e}")
        failures += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    if failures == 0:
        print("ALL CHECKS PASSED — pipeline is healthy.")
        print(f"Open the dashboard:  {BASE}/monitor")
        print(f"A CAM-{CAM} tile with a white figure should be visible.")
    else:
        print(f"{failures} CHECK(S) FAILED — see above.")
    print("─" * 60)
    return failures


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
