"""
run_demo.py — one command to launch the whole demo.

    python run_demo.py                 # server + 4 replayed cameras
    python run_demo.py --cams 6        # 6 cameras
    python run_demo.py --no-feed       # server only (real phones)
    python run_demo.py --src E:\\clips  # replay your own folder
    python run_demo.py --no-feed --tunnel   # + public URL for phones off the LAN

Ctrl+C stops everything.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

HERE = Path(__file__).parent
MONITOR = "http://localhost:8090/monitor"

# Windows consoles default to cp1252, which raises UnicodeEncodeError the first
# time any child prints a non-ASCII character. Force UTF-8 for every child.
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def wait_for_server(timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://localhost:8090/status", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, default=4)
    ap.add_argument("--fps", type=float, default=None,
                    help="override source frame rate (default: native)")
    ap.add_argument("--src", type=Path, default=None,
                    help="folder of footage to replay (default: test_videos)")
    ap.add_argument("--no-feed", action="store_true",
                    help="skip replay; stream from real phones instead")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--tunnel", action="store_true",
                    help="also open a public URL (tunnel.py) so phones off the "
                         "LAN — on mobile data or another network — can stream")
    args = ap.parse_args()

    src = args.src or (HERE / "test_videos")
    if not args.no_feed and not src.exists():
        print(f"No footage at {src}")
        print("Build the test clips first:  python make_test_videos.py")
        return 1

    procs: list[subprocess.Popen] = []
    try:
        print("Starting IBVAP server...")
        procs.append(subprocess.Popen([sys.executable, "-u", "server.py"],
                                      cwd=HERE, env=CHILD_ENV))

        if not wait_for_server():
            print("Server did not come up in time - check its output above.")
            return 1
        print(f"Server ready: {MONITOR}")

        if not args.no_feed:
            cmd = [sys.executable, "-u", "feed_test.py",
                   "--cams", str(args.cams), "--src", str(src)]
            if args.fps:
                cmd += ["--fps", str(args.fps)]
            print(f"Replaying {args.cams} camera(s) from {src.name}...")
            procs.append(subprocess.Popen(cmd, cwd=HERE, env=CHILD_ENV))

        tunnel_proc = None
        if args.tunnel:
            tcmd = [sys.executable, "-u", "tunnel.py"]
            # If a cloudflared named tunnel is configured, use the custom domain.
            if os.environ.get("IBVAP_CF_TUNNEL"):
                tcmd += ["--named", os.environ["IBVAP_CF_TUNNEL"]]
                if os.environ.get("IBVAP_CF_HOSTNAME"):
                    tcmd += ["--hostname", os.environ["IBVAP_CF_HOSTNAME"]]
                print(f"Opening custom-domain tunnel ({os.environ.get('IBVAP_CF_HOSTNAME', os.environ['IBVAP_CF_TUNNEL'])})...")
            else:
                print("Opening public tunnel (phones on any network)...")
            tunnel_proc = subprocess.Popen(tcmd, cwd=HERE, env=CHILD_ENV)
            procs.append(tunnel_proc)

        if not args.no_browser:
            webbrowser.open(MONITOR)

        print("\nRunning. Press Ctrl+C to stop.\n")
        while True:
            for p in procs:
                if p.poll() is None:
                    continue
                if p is tunnel_proc:
                    # A failed/closed tunnel must not take the demo down — the
                    # LAN still works. Warn once, then stop watching it.
                    print("Tunnel exited (LAN still up). Re-run "
                          "'python tunnel.py' to retry a public link.")
                    procs.remove(p)
                    tunnel_proc = None
                    break
                print(f"A process exited (code {p.returncode}) - shutting down.")
                return p.returncode or 1
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
        return 0
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    sys.exit(main())
