"""
Generate a self-signed SSL cert for local HTTPS.
Run once:  python gen_cert.py
Then restart the server — it auto-detects cert.pem / key.pem.

On phones: navigate to https://10.x.x.x:8443/camera/0
           tap Advanced → Proceed (once per device)
"""

import datetime
import ipaddress
import socket
import sys

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
except ImportError:
    print("Installing cryptography…")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cryptography", "-q"])
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

# Detect all local IPs automatically
local_ips = []
try:
    hostname = socket.gethostname()
    for info in socket.getaddrinfo(hostname, None):
        ip = info[4][0]
        if ":" not in ip and not ip.startswith("127."):
            local_ips.append(ip)
except Exception:
    pass
local_ips = list(dict.fromkeys(local_ips))  # deduplicate

print(f"Generating cert for IPs: {local_ips}")

key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

san_entries = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]
for ip in local_ips:
    try:
        san_entries.append(x509.IPAddress(ipaddress.IPv4Address(ip)))
    except Exception:
        pass

subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ibvap-local")])
cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.datetime.utcnow())
    .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
    .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
    .sign(key, hashes.SHA256())
)

with open("cert.pem", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

with open("key.pem", "wb") as f:
    f.write(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))

print("cert.pem and key.pem created.")
print(f"\nRestart server, then on phones open:")
for ip in local_ips:
    print(f"  https://{ip}:8443/camera/0")
print("  (tap Advanced → Proceed when Chrome warns about the cert)")
