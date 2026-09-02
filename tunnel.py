"""
tunnel.py -- expose the IBVAP server on the public internet so a phone that is
NOT on the same WiFi (mobile data, a different network) can still open the
camera link and stream.

    python tunnel.py                 # auto-pick cloudflared, else ngrok (random/ngrok-fixed URL)
    python tunnel.py --backend ngrok
    python tunnel.py --port 8090     # which local port to expose (default: HTTP 8090)
    python tunnel.py --named ibvap --hostname stream.mathswithsd.in
                                    # YOUR domain via a cloudflared named tunnel
                                    # (one-time setup: docs/CUSTOM_DOMAIN.md)

Why the HTTP port, not 8443: the tunnel terminates TLS itself with a real,
publicly-trusted certificate, so the phone sees a normal https:// URL and
getUserMedia works with NO "your connection is not private" warning. The
self-signed cert on :8443 is only for same-LAN use.

    cloudflared  -- no account, no signup. Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
                    then: cloudflared --version
    ngrok        -- needs a free authtoken once:
                    https://dashboard.ngrok.com/get-started/your-authtoken
                    then: ngrok config add-authtoken <token>

STABLE link: with ngrok you already have one. The free plan assigns each
account ONE permanent static domain (e.g. adjective-adjective-noun.ngrok-free.dev)
and plain `ngrok http 8090` binds to it every run -- same URL as long as the
authtoken is the same. No purchase, no --domain needed.
  --domain only renames it to something memorable (e.g. ibvap.ngrok-free.app);
  claim that name first at https://dashboard.ngrok.com/domains, or set
  $IBVAP_TUNNEL_DOMAIN once.
  (cloudflared's free tunnels are always random; a fixed cloudflared URL needs
  a real domain on a Cloudflare account.)

SECURITY: while this runs, the dashboard and the camera intake are reachable by
anyone with the URL — there is no authentication. Use it for a demo, watch who
you share the link with, and Ctrl+C to close the tunnel when done.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

HTTP_PORT = 8090


def _qr(url: str) -> None:
    try:
        import qrcode
        q = qrcode.QRCode(border=1)
        q.add_data(url)
        q.make(fit=True)
        q.print_ascii(out=sys.stdout, invert=True)
    except Exception:
        print("  (install 'qrcode' for a scannable code:  pip install qrcode)")


def _panel(public: str, cam_path: str = "cam") -> None:
    cam0 = f"{public}/{cam_path}/0"
    print("\n" + "=" * 64)
    print("  IBVAP is now reachable from anywhere:\n")
    print(f"    Monitor      {public}/monitor")
    print(f"    Phone 1      {cam0}")
    print(f"    Phone 2      {public}/{cam_path}/1     (increment per phone)")
    print("=" * 64)
    print("  Scan on the first phone (any network — WiFi or mobile data):\n")
    _qr(cam0)
    print("=" * 64)
    if "ngrok-free" in public:
        print("  This is your account's permanent ngrok domain — same URL every run.")
    elif "trycloudflare.com" not in public:
        print("  Custom domain — this URL is permanent.")
    print("  This link is PUBLIC and unauthenticated. Ctrl+C to close it.\n")


# ── cloudflared: named tunnel on a custom domain ─────────────────────────────
def _run_cloudflared_named(name: str, hostname: str | None, config: str | None) -> int:
    """`cloudflared tunnel run <name>` — uses the domain + ingress rules set up
    once with `cloudflared tunnel create/route`. See docs/CUSTOM_DOMAIN.md."""
    exe = shutil.which("cloudflared")
    cmd = [exe, "tunnel", "--no-autoupdate"]
    if config:
        cmd += ["--config", config]
    cmd += ["run", name]
    print(f"  cloudflared tunnel run {name}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    shown = False
    try:
        for line in proc.stdout:                       # type: ignore[union-attr]
            line = line.rstrip()
            low = line.lower()
            if not shown and ("registered tunnel connection" in low
                              or "connection registered" in low
                              or "started tunnel" in low):
                shown = True
                if hostname:
                    _panel(f"https://{hostname}", cam_path="cam")
                else:
                    print("  Tunnel up. Public hostname is whatever "
                          "`cloudflared tunnel route dns` was pointed at.\n")
            elif ("err" in low and "error" not in low) or "failed" in low \
                    or "error=" in low:
                print(f"  cloudflared: {line}")
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


# ── cloudflared: quick tunnel (random *.trycloudflare.com) ───────────────────
def _run_cloudflared(port: int) -> int:
    exe = shutil.which("cloudflared")
    proc = subprocess.Popen(
        [exe, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    url_re = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    public = None
    try:
        for line in proc.stdout:                       # type: ignore[union-attr]
            line = line.rstrip()
            if not public:
                m = url_re.search(line)
                if m:
                    public = m.group(0)
                    _panel(public)
            elif "ERR" in line or "error" in line.lower():
                print(f"  cloudflared: {line}")
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


# ── ngrok ────────────────────────────────────────────────────────────────────
def _run_ngrok(port: int, domain: str | None = None) -> int:
    exe = shutil.which("ngrok")
    cmd = [exe, "http", str(port), "--log", "stdout", "--log-format", "logfmt"]
    if domain:
        # ngrok's free plan grants one reserved *.ngrok-free.app subdomain.
        # Binding to it makes the public URL identical on every run.
        cmd += ["--url", domain]
        print(f"  binding to reserved domain: {domain}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    def _poll_api() -> str | None:
        try:
            raw = urllib.request.urlopen(
                "http://127.0.0.1:4040/api/tunnels", timeout=1).read()
            for t in json.loads(raw).get("tunnels", []):
                if t.get("public_url", "").startswith("https://"):
                    return t["public_url"]
        except Exception:
            pass
        return None

    public = None
    deadline = time.time() + 20
    try:
        while proc.poll() is None:
            # ngrok prints auth errors to its log — surface them and stop.
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                low = line.lower()
                if "authentication failed" in low or "err_ngrok_4018" in low \
                        or "authtoken" in low and "err" in low:
                    print("\n  ngrok needs a one-time free authtoken:")
                    print("    1. sign up:  https://dashboard.ngrok.com/signup")
                    print("    2. copy it:  https://dashboard.ngrok.com/get-started/your-authtoken")
                    print("    3. run:      ngrok config add-authtoken <token>")
                    print("    then re-run: python tunnel.py\n")
                    proc.terminate()
                    return 1
                if domain and ("err_ngrok_3200" in low or "not found" in low
                               or "is not authorized" in low or "err_ngrok_334" in low):
                    print(f"\n  ngrok won't bind '{domain}'. Claim it first at")
                    print("    https://dashboard.ngrok.com/domains  (free plan: 1 domain)")
                    print("  or drop --domain to get a random URL.\n")
                    proc.terminate()
                    return 1
            if not public:
                public = _poll_api()
                if public:
                    _panel(public)
            if not public and time.time() > deadline:
                print("  ngrok started but no public URL after 20s — check "
                      "http://127.0.0.1:4040 for its status.")
                deadline = float("inf")
            time.sleep(0.3)
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


BACKENDS = {"cloudflared": _run_cloudflared, "ngrok": _run_ngrok}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=list(BACKENDS),
                    help="force a backend (default: cloudflared if present, else ngrok)")
    ap.add_argument("--port", type=int, default=HTTP_PORT,
                    help=f"local port to expose (default {HTTP_PORT}, the HTTP dashboard port)")
    ap.add_argument("--domain", default=os.environ.get("IBVAP_TUNNEL_DOMAIN"),
                    help="ngrok reserved subdomain for a URL that never changes "
                         "(claim a free one at dashboard.ngrok.com/domains). "
                         "Also read from $IBVAP_TUNNEL_DOMAIN.")
    ap.add_argument("--named", default=os.environ.get("IBVAP_CF_TUNNEL"),
                    help="run a pre-created cloudflared NAMED tunnel by name "
                         "(your own domain, e.g. stream.mathswithsd.in). "
                         "One-time setup: docs/CUSTOM_DOMAIN.md. Also $IBVAP_CF_TUNNEL.")
    ap.add_argument("--hostname", default=os.environ.get("IBVAP_CF_HOSTNAME"),
                    help="the hostname the named tunnel serves, for the printed "
                         "links (e.g. stream.mathswithsd.in). Also $IBVAP_CF_HOSTNAME.")
    ap.add_argument("--cf-config", help="path to a cloudflared config.yml "
                    "(default: cloudflared's own ~/.cloudflared/config.yml)")
    args = ap.parse_args()

    # Is the server actually up?
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{args.port}/status", timeout=1)
    except Exception:
        print(f"Nothing is answering on http://127.0.0.1:{args.port} — "
              f"start the server first:  python server.py")
        return 1

    # Custom-domain path: a pre-created cloudflared named tunnel.
    if args.named:
        if not shutil.which("cloudflared"):
            print("cloudflared is not on PATH — install it and run the one-time "
                  "setup in docs/CUSTOM_DOMAIN.md first.")
            return 1
        try:
            return _run_cloudflared_named(args.named, args.hostname, args.cf_config)
        except KeyboardInterrupt:
            return 0

    backend = args.backend
    if not backend:
        backend = "cloudflared" if shutil.which("cloudflared") else \
                  "ngrok" if shutil.which("ngrok") else None
    if not backend:
        print("Neither 'cloudflared' nor 'ngrok' is on PATH.\n"
              "  cloudflared (no signup):  "
              "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/\n"
              "  ngrok (free authtoken):   https://ngrok.com/download")
        return 1

    if args.domain and backend != "ngrok":
        print(f"  (--domain is ngrok-only; ignoring it for {backend})")
        args.domain = None

    print(f"Starting {backend} tunnel to http://localhost:{args.port} ...")
    try:
        if backend == "ngrok":
            return _run_ngrok(args.port, args.domain)
        return BACKENDS[backend](args.port)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
