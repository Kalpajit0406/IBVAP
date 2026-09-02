# Serving IBVAP on your own domain — `stream.mathswithsd.in/cam/0`

Goal: phones on any network open `https://stream.mathswithsd.in/cam/0`,
`…/cam/1`, … and the dashboard at `https://stream.mathswithsd.in/monitor`.
Fixed URL, real certificate, no app install.

## Why not just Netlify

Netlify hosts static files. It cannot proxy a subdomain to a server running on
your laptop behind home NAT, and it cannot carry the camera WebSocket. Your
Netlify site for `mathswithsd.in` keeps working unchanged — this only adds a
`stream` subdomain that points at a tunnel.

## Why Cloudflare Tunnel

It is the only free option that does **custom domain + WebSocket + no exposed
home IP**. ngrok custom domains need a paid plan; cloudflared *quick* tunnels
are always random. The one catch: the domain's DNS must be on Cloudflare. Your
registrar keeps the domain; only the nameservers change. The Netlify site is
re-pointed with the same records, so it stays up.

---

## One-time setup (~20 min, mostly waiting for DNS)

### 1. Move `mathswithsd.in` DNS to Cloudflare

1. Create a free account at <https://dash.cloudflare.com>, **Add a site** →
   `mathswithsd.in`. Cloudflare scans and imports the existing records.
2. **Check the import**: the site records that make `mathswithsd.in` /
   `www.mathswithsd.in` resolve to Netlify must be present. In your Netlify
   dashboard, *Domain management → check the values Netlify expects* — usually
   either an `A` record for the apex to Netlify's load balancer, or a `CNAME`
   for `www` to `<your-site>.netlify.app`. Re-add anything Cloudflare missed.
   Set those site records to **DNS only** (grey cloud) so Netlify's own TLS is
   unaffected.
3. At your **registrar** (where `mathswithsd.in` is registered), replace the
   nameservers with the two Cloudflare gave you. Propagation is minutes to a
   few hours. Cloudflare emails you when the zone is active.
4. Confirm the site still loads: `https://mathswithsd.in` should be unchanged.

### 2. Install cloudflared

<https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/>
Then check: `cloudflared --version`

### 3. Create the tunnel and route the subdomain

```bash
cloudflared tunnel login                       # browser: pick the mathswithsd.in zone
cloudflared tunnel create ibvap                # prints a UUID + a credentials .json path
cloudflared tunnel route dns ibvap stream.mathswithsd.in
```

The last command creates a proxied `CNAME stream → <uuid>.cfargotunnel.com` in
Cloudflare DNS automatically.

### 4. Write the cloudflared config

Copy `cloudflared.example.yml` from the repo root to
`%USERPROFILE%\.cloudflared\config.yml` and fill in the tunnel UUID and the
credentials-file path from step 3:

```yaml
tunnel: ibvap
credentials-file: C:\Users\<you>\.cloudflared\<uuid>.json
ingress:
  - hostname: stream.mathswithsd.in
    service: http://localhost:8090
  - service: http_status:404
```

---

## Every demo

```bash
python server.py
python tunnel.py --named ibvap --hostname stream.mathswithsd.in
```

or, in one process:

```bash
setx IBVAP_CF_TUNNEL   ibvap                      # once
setx IBVAP_CF_HOSTNAME stream.mathswithsd.in      # once  (open a new shell after)
python run_demo.py --no-feed --tunnel
```

Phones then open:

| | |
|---|---|
| Dashboard | `https://stream.mathswithsd.in/monitor` |
| Phone 1 | `https://stream.mathswithsd.in/cam/0` |
| Phone 2 | `https://stream.mathswithsd.in/cam/1` |

`/cam/N` and `/camera/N` both work (`server.py` serves both paths).

---

## Security

The tunnel exposes the dashboard and camera intake **with no authentication** —
anyone with the URL can watch the feeds or push frames. It is a demo tool. Stop
`cloudflared` (Ctrl+C) when you are done; the subdomain then returns nothing.
Per-endpoint auth is on the roadmap in CLAUDE.md.
