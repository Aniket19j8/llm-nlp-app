"""
Response Cache — Thread-safe LRU cache with TTL
"""

import time
import threading
from collections import OrderedDict
from typing import Any, Optional


class ResponseCache:
    """In-memory LRU cache with time-to-live expiration."""

    def __init__(self, max_size: int = 2000, ttl_seconds: int = 3600):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return round(self._hits / total, 4) if total else 0.0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._store:
                ts, value = self._store[key]
                if time.time() - ts < self._ttl:
                    self._store.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    del self._store[key]
            self._misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.time(), value)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        return {
            "size": self.size,
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }
