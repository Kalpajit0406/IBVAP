"""
discover_cameras.py — find ONVIF cameras / NVRs on the local network and print
a ready-to-paste config.yaml `streams:` block.

    python discover_cameras.py                       # list devices on the LAN
    python discover_cameras.py --user admin --pass secret   # + resolve RTSP URLs

WS-Discovery (the listing step) is pure stdlib. Resolving the actual RTSP URL
for each camera needs the credentials and the `onvif-zeep` package:
    pip install onvif-zeep

Notes:
  * Some networks block multicast between VLANs — run this on the same subnet
    as the cameras, or ask college IT for the URLs directly.
  * An NVR usually answers for itself, not each camera. If you only see the
    NVR, get its channel URLs from docs/CCTV_INTEGRATION.md (the 101/102… or
    channel=N&subtype=1 patterns).
"""
from __future__ import annotations

import argparse
import re
import socket
import sys
import time
import uuid

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

WSD_ADDR, WSD_PORT = "239.255.255.250", 3702

_PROBE = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
    'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
    'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
    'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
    '<e:Header>'
    '<w:MessageID>uuid:{mid}</w:MessageID>'
    '<w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
    '<w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>'
    '</e:Header>'
    '<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>'
    '</e:Envelope>'
)


def discover(timeout: float = 4.0) -> list[dict]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.settimeout(timeout)
    sock.bind(("", 0))

    msg = _PROBE.format(mid=uuid.uuid4()).encode()
    try:
        sock.sendto(msg, (WSD_ADDR, WSD_PORT))
    except OSError as e:
        print(f"Could not send the discovery probe: {e}")
        return []

    seen: dict[str, dict] = {}
    end = time.time() + timeout
    while time.time() < end:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            break
        xml = data.decode("utf-8", "replace")
        xaddrs = re.findall(r"<[^>]*XAddrs>\s*([^<]+?)\s*<", xml)
        xaddr = xaddrs[0].split()[0] if xaddrs and xaddrs[0].split() else ""
        scopes = " ".join(re.findall(r"<[^>]*Scopes>\s*([^<]+?)\s*<", xml))
        ip = addr[0]
        if ip in seen:
            continue

        def scope(key: str) -> str:
            m = re.search(rf"onvif://www\.onvif\.org/{key}/([^\s]+)", scopes)
            return m.group(1).replace("%20", " ") if m else ""

        seen[ip] = {
            "ip": ip,
            "xaddr": xaddr,
            "name": scope("name") or scope("hardware") or "camera",
            "hardware": scope("hardware"),
            "location": scope("location"),
        }
    sock.close()
    return list(seen.values())


def resolve_rtsp(dev: dict, user: str, pw: str) -> list[dict]:
    """Return [{'name','width','height','url'}, ...] for each media profile."""
    try:
        from onvif import ONVIFCamera
    except ImportError:
        return []
    m = re.search(r"https?://([^:/]+)(?::(\d+))?", dev["xaddr"] or f"http://{dev['ip']}")
    host = m.group(1) if m else dev["ip"]
    port = int(m.group(2)) if m and m.group(2) else 80
    try:
        cam = ONVIFCamera(host, port, user, pw)
        media = cam.create_media_service()
        out = []
        for p in media.GetProfiles():
            try:
                res = p.VideoEncoderConfiguration.Resolution
                w, h = int(res.Width), int(res.Height)
            except Exception:
                w = h = 0
            uri = media.GetStreamUri({
                "StreamSetup": {"Stream": "RTP-Unicast",
                                "Transport": {"Protocol": "RTSP"}},
                "ProfileToken": p.token,
            }).Uri
            # inject credentials into the returned URL
            uri = re.sub(r"://", f"://{user}:{pw}@", uri, count=1)
            out.append({"name": p.Name, "width": w, "height": h, "url": uri})
        return out
    except Exception as e:
        print(f"  {dev['ip']}: ONVIF query failed ({e})")
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=4.0)
    ap.add_argument("--user", default="")
    ap.add_argument("--pass", dest="pw", default="")
    args = ap.parse_args()

    print(f"\nProbing the LAN for ONVIF devices ({args.timeout:.0f}s) ...\n")
    devs = discover(args.timeout)
    if not devs:
        print("No ONVIF devices answered. Either there are none on this subnet,")
        print("multicast is filtered, or the cameras have ONVIF disabled.")
        print("Get the RTSP URLs from college IT / docs/CCTV_INTEGRATION.md.")
        return 1

    print(f"Found {len(devs)} device(s):\n")
    yaml_rows = []
    for i, d in enumerate(devs):
        print(f"  [{i}] {d['ip']:<15} {d['name']}"
              + (f"  ({d['hardware']})" if d['hardware'] else "")
              + (f"  @ {d['location']}" if d['location'] else ""))
        print(f"      onvif: {d['xaddr'] or '(none advertised)'}")

        profiles = resolve_rtsp(d, args.user, args.pw) if args.user else []
        if profiles:
            profiles.sort(key=lambda p: (p["width"] or 9999) * (p["height"] or 9999))
            for p in profiles:
                tag = "  <- sub-stream, use this" if p is profiles[0] and len(profiles) > 1 else ""
                print(f"      {p['width']}x{p['height']:<5} {redact(p['url'])}{tag}")
            best = profiles[0]
            yaml_rows.append((i, d['name'], best['url'], best['width']))
        else:
            yaml_rows.append((i, d['name'], f"rtsp://{args.user or 'user'}:"
                              f"{args.pw or 'pass'}@{d['ip']}:554/  # fill in the path",
                              0))
        print()

    if not args.user:
        print("Tip: re-run with  --user admin --pass <pw>  (and `pip install "
              "onvif-zeep`) to fetch the exact RTSP URLs.\n")

    print("── paste into config.yaml ──────────────────────────────────────")
    print("streams:")
    for cid, name, url, w in yaml_rows:
        sens = 0.9 if any(k in name.lower() for k in ("gate", "perim", "entr")) else 0.6
        print(f'  - {{ id: {cid}, name: "{name[:20]}", '
              f'url: "{url}", zone_sensitivity: {sens} }}')
    print("────────────────────────────────────────────────────────────────\n")
    return 0


def redact(u: str) -> str:
    return re.sub(r"://([^:/@]+):([^@/]+)@", r"://\1:***@", u)


if __name__ == "__main__":
    sys.exit(main())
