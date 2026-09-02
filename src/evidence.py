from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class EvidenceChain:
    """
    Append-only SHA-256 hash chain stored as newline-delimited JSON.
    Each record includes the hash of the previous record, making the log
    tamper-evident: any modification breaks every subsequent hash.
    """

    GENESIS = "0" * 64

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash = self._load_tip()

    def _load_tip(self) -> str:
        if not self._path.exists():
            return self.GENESIS
        lines = self._path.read_bytes().splitlines()
        if not lines:
            return self.GENESIS
        return json.loads(lines[-1]).get("hash", self.GENESIS)

    def append(self, event: dict) -> str:
        """Append an event and return its SHA-256 hash."""
        record = {
            "timestamp": time.time(),
            "prev_hash": self._prev_hash,
            "event": event,
        }
        payload = json.dumps(record, sort_keys=True).encode()
        record["hash"] = hashlib.sha256(payload).hexdigest()

        with self._path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        self._prev_hash = record["hash"]
        return record["hash"]

    def verify(self) -> tuple[bool, int]:
        """
        Walk the chain from genesis. Returns (ok, first_bad_line_number).
        ok=True means the chain is intact.
        """
        if not self._path.exists():
            return True, -1

        prev = self.GENESIS
        for lineno, raw in enumerate(self._path.read_bytes().splitlines(), start=1):
            rec = json.loads(raw)
            if rec.get("prev_hash") != prev:
                return False, lineno
            check = {k: v for k, v in rec.items() if k != "hash"}
            expected = hashlib.sha256(
                json.dumps(check, sort_keys=True).encode()
            ).hexdigest()
            if rec.get("hash") != expected:
                return False, lineno
            prev = rec["hash"]
        return True, -1
