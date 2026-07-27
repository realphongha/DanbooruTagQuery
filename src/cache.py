"""
SQLite cache for Danbooru tag metadata (category, wiki body).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path("data/tag_cache.db")


class TagCache:
    """Persistent cache for tag info fetched from the Danbooru API."""

    def __init__(self, db: str | Path = DB_PATH):
        self.db = Path(db)
        self.db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS tag_cache (
                name TEXT PRIMARY KEY,
                category INTEGER,
                wiki_body TEXT,
                fetched_at REAL NOT NULL
            )"""
        )
        self._conn.commit()

    def get(self, name: str) -> tuple[int | None, str | None] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT category, wiki_body FROM tag_cache WHERE name = ?",
                (name,),
            )
            row = cur.fetchone()
            return row if row else None

    def set(self, name: str, category: int | None, wiki_body: str | None):
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO tag_cache (name, category, wiki_body, fetched_at)
                   VALUES (?, ?, ?, ?)""",
                (name, category, wiki_body, time.time()),
            )
            self._conn.commit()

    def bulk_set(self, items: list[tuple[str, int | None, str | None]]):
        """Insert/replace multiple rows in a single transaction."""
        with self._lock:
            now = time.time()
            self._conn.executemany(
                """INSERT OR REPLACE INTO tag_cache (name, category, wiki_body, fetched_at)
                   VALUES (?, ?, ?, ?)""",
                [(name, cat, body, now) for name, cat, body in items],
            )
            self._conn.commit()

    def clear(self):
        with self._lock:
            self._conn.execute("DELETE FROM tag_cache")
            self._conn.commit()

    def size(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM tag_cache")
            return cur.fetchone()[0]

    def close(self):
        self._conn.close()
