"""
rtsp_probe.py — check one camera / NVR URL before you trust it in a demo.

    python rtsp_probe.py "rtsp://user:pass@10.0.0.50:554/Streaming/Channels/102"
    python rtsp_probe.py "rtsp://..." --seconds 10 --transport udp --save frame.jpg

Reports: whether it opens, how long that took, resolution / fps / codec, the
real measured frame interval, decode time, and a verdict on whether it's good
for analytics (sub-stream ≤ ~1280 wide is ideal).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def redact(u: str) -> str:
    import re
    return re.sub(r"://([^:/@]+):([^@/]+)@", r"://\1:***@", u)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="rtsp:// / http:// / webcam index / file path")
    ap.add_argument("--seconds", type=float, default=8.0, help="how long to sample")
    ap.add_argument("--transport", choices=["tcp", "udp"], default="tcp")
    ap.add_argument("--save", metavar="PATH", help="write one decoded frame to PATH")
    args = ap.parse_args()

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{args.transport}|stimeout;5000000|max_delay;500000")

    import cv2

    src = int(args.url) if args.url.isdigit() else args.url
    api = cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_ANY

    print(f"\nURL       : {redact(str(args.url))}")
    print(f"transport : {args.transport}")
    print("opening   : ", end="", flush=True)

    t0 = time.perf_counter()
    cap = cv2.VideoCapture(src, api)
    open_s = time.perf_counter() - t0

    if not cap.isOpened():
        print(f"FAILED after {open_s:.1f}s")
        print("\n  Not reachable. Check, in order:")
        print("   • can you ping the IP from this PC?")
        print("   • right port (554 is default) and path? (docs/CCTV_INTEGRATION.md)")
        print("   • username / password correct and URL-encoded if they contain @ : /")
        print("   • is RTSP enabled on the camera/NVR? some ship with it off")
        print("   • try --transport udp, or a VLC test: vlc <url>")
        return 1
    print(f"ok in {open_s:.1f}s")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps_reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    fourcc = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ") or "?"
    print(f"resolution: {w}x{h}")
    print(f"fps (hdr) : {fps_reported:.1f}")
    print(f"codec     : {fourcc}")

    print(f"\nsampling {args.seconds:.0f}s ...")
    intervals, decode_ms = [], []
    last = None
    frames = 0
    saved = False
    end = time.perf_counter() + args.seconds
    while time.perf_counter() < end:
        d0 = time.perf_counter()
        ok, frame = cap.read()
        dt = (time.perf_counter() - d0) * 1000.0
        if not ok or frame is None:
            print("  read() returned nothing — stream dropped mid-sample")
            break
        frames += 1
        decode_ms.append(dt)
        now = time.perf_counter()
        if last is not None:
            intervals.append(now - last)
        last = now
        if args.save and not saved:
            cv2.imwrite(args.save, frame)
            saved = True
            print(f"  saved a frame → {args.save}")
    cap.release()

    if not intervals:
        print("  got no usable frames.")
        return 1

    import statistics
    mean_iv = statistics.mean(intervals)
    real_fps = 1.0 / mean_iv if mean_iv else 0.0
    jitter = statistics.pstdev(intervals) * 1000.0
    print(f"\n  frames sampled : {frames}")
    print(f"  real fps       : {real_fps:.1f}  (interval {mean_iv*1000:.0f} ms ± {jitter:.0f} ms)")
    print(f"  decode time    : {statistics.mean(decode_ms):.1f} ms/frame avg, "
          f"{max(decode_ms):.0f} ms worst")

    print("\n  verdict:")
    big = w > 1400 or h > 900
    if big:
        print(f"   ⚠ {w}x{h} is a MAIN stream. Use the sub-stream for analytics")
        print("     (Hikvision: …/Channels/x02, Dahua: …&subtype=1). Lower CPU,")
        print("     same detection quality at demo range.")
    else:
        print(f"   ✓ {w}x{h} is a good analytics resolution.")
    if real_fps < 6:
        print(f"   ⚠ only {real_fps:.1f} fps — tracking will be choppy. Raise the")
        print("     camera's sub-stream fps if you can.")
    if jitter > 120:
        print(f"   ⚠ high jitter ({jitter:.0f} ms) — network is lossy or the NVR is")
        print("     busy. Prefer --transport tcp; move the PC closer to the switch.")
    if not big and real_fps >= 6 and jitter <= 120:
        print("   ✓ solid. Add it to config.yaml streams:.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
