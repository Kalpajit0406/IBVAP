from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path


class EventStore:
    """
    SQLite-backed store-and-forward event queue.

    All writes are serialised through a lock so the store is safe to call
    from multiple threads. When network connectivity is restored a sync
    routine can drain unsynced rows.
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS events (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        ts        REAL    NOT NULL,
        cam_id    INTEGER NOT NULL,
        level     TEXT    NOT NULL,
        score     REAL    NOT NULL,
        persons   INTEGER NOT NULL DEFAULT 0,
        vehicles  INTEGER NOT NULL DEFAULT 0,
        details   TEXT    NOT NULL DEFAULT '{}',
        ev_hash   TEXT,
        synced    INTEGER NOT NULL DEFAULT 0
    )
    """

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(self.DDL)
            self._conn.commit()

    def log(
        self,
        cam_id: int,
        level: str,
        score: float,
        persons: int = 0,
        vehicles: int = 0,
        details: dict | None = None,
        ev_hash: str | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (ts, cam_id, level, score, persons, vehicles, details, ev_hash)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    time.time(), cam_id, level, score, persons, vehicles,
                    json.dumps(details or {}), ev_hash,
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def unsynced(self) -> list[sqlite3.Row]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            return self._conn.execute(
                "SELECT * FROM events WHERE synced=0 ORDER BY id"
            ).fetchall()

    def mark_synced(self, event_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE events SET synced=1 WHERE id=?", (event_id,))
            self._conn.commit()

    def recent(self, limit: int = 20) -> list[sqlite3.Row]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            return self._conn.execute(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
