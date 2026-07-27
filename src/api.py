"""
Danbooru API client using curl_cffi to bypass Cloudflare.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from curl_cffi import requests
from dotenv import load_dotenv

load_dotenv()

_USER = os.environ.get("DANBOORU_USER")
_KEY = os.environ.get("DANBOORU_API_KEY")
_AUTH = (_USER, _KEY) if _USER and _KEY else None

BASE = "https://danbooru.donmai.us"

# ── rate limiter (per-process, thread-safe) ─────────────────────────────────

class _RateLimiter:
    """Simple token bucket: max ``calls`` per ``window`` seconds."""

    def __init__(self, calls: float = 2.0, window: float = 1.0):
        self._calls = calls
        self._window = window
        self._lock = threading.Lock()
        self._tokens = calls
        self._last = time.monotonic()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._calls, self._tokens + elapsed * (self._calls / self._window))
            self._last = now
            if self._tokens < 1:
                sleep = (1 - self._tokens) * (self._window / self._calls)
                time.sleep(sleep)
                self._tokens = 0
                self._last = time.monotonic()
            else:
                self._tokens -= 1

_RATE_LIMITER = _RateLimiter(calls=2.0)  # 2 requests/second


CATEGORY_MAP = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
}
_CAT_TO_ID = {v: k for k, v in CATEGORY_MAP.items()}


def configure(user: str | None = None, key: str | None = None):
    """Override the API credentials used for subsequent requests."""
    global _AUTH
    if user and key:
        _AUTH = (user, key)
    else:
        load_dotenv()
        u = os.environ.get("DANBOORU_USER")
        k = os.environ.get("DANBOORU_API_KEY")
        _AUTH = (u, k) if u and k else None


def get_auth_info() -> tuple[str | None, str | None]:
    """Return the current (user, key) without exposing the full key."""
    if _AUTH is None:
        return None, None
    k = _AUTH[1]
    masked = (k[:4] + "…" + k[-2:]) if k and len(k) > 8 else "…"
    return _AUTH[0], masked


def get(path: str, params: dict[str, Any] | None = None) -> list[dict] | dict:
    """GET a Danbooru JSON endpoint (rate-limited)."""
    _RATE_LIMITER.acquire()
    url = f"{BASE}{path}"
    resp = requests.get(url, params=params, auth=_AUTH, impersonate="chrome")
    resp.raise_for_status()
    return resp.json()


def wiki(title: str) -> dict | None:
    """Fetch a wiki page by title."""
    data = get("/wiki_pages.json", {"search[title]": title})
    return data[0] if data else None


def wiki_body(title: str) -> str | None:
    """Get the body text of a wiki page."""
    page = wiki(title)
    return page["body"] if page else None


def posts(tags: str, limit: int = 20, page: int = 1) -> list[dict]:
    """Search posts by tag string."""
    return get("/posts.json", {"tags": tags, "limit": limit, "page": page})  # type: ignore[return-value]


def tags(search: str, limit: int = 20) -> list[dict]:
    """Search tags by name (fuzzy)."""
    return get("/tags.json", {"search[name_matches]": search, "limit": limit})  # type: ignore[return-value]


def tag_info(name: str) -> dict | None:
    """Get exact tag info by name (id, name, category, post_count, …)."""
    data = get("/tags.json", {"search[name]": name})
    return data[0] if data else None


def tag_category(name: str) -> tuple[int, str] | None:
    """Get (category_id, category_name) for a tag."""
    info = tag_info(name)
    if info is None:
        return None
    cat = info["category"]
    return cat, CATEGORY_MAP.get(cat, "unknown")
