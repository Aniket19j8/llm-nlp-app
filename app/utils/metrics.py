"""
Metrics Collector — Request tracking & latency monitoring
"""

import time
import threading
from collections import defaultdict


class MetricsCollector:
    """Thread-safe request metrics collector."""

    def __init__(self):
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._requests: list[dict] = []
        self._endpoint_counts: dict[str, int] = defaultdict(int)
        self._endpoint_latencies: dict[str, list[float]] = defaultdict(list)
        self._total = 0

    @property
    def uptime(self) -> float:
        return round(time.time() - self._start_time, 2)

    @property
    def total_requests(self) -> int:
        return self._total

    @property
    def avg_latency(self) -> float:
        with self._lock:
            all_lat = [
                lat for lats in self._endpoint_latencies.values() for lat in lats
            ]
            return round(sum(all_lat) / len(all_lat), 2) if all_lat else 0.0

    def record_request(self, path: str, latency_ms: float) -> None:
        with self._lock:
            self._total += 1
            self._endpoint_counts[path] += 1
            self._endpoint_latencies[path].append(latency_ms)
            # Keep only last 10_000 per endpoint
            if len(self._endpoint_latencies[path]) > 10_000:
                self._endpoint_latencies[path] = self._endpoint_latencies[path][-5_000:]

    def snapshot(self) -> dict:
        with self._lock:
            endpoints = {}
            for path, count in self._endpoint_counts.items():
                lats = self._endpoint_latencies[path]
                endpoints[path] = {
                    "count": count,
                    "avg_ms": round(sum(lats) / len(lats), 2) if lats else 0,
                    "p95_ms": round(
                        sorted(lats)[int(len(lats) * 0.95)] if lats else 0, 2
                    ),
                    "max_ms": round(max(lats), 2) if lats else 0,
                }
            return {
                "uptime_seconds": self.uptime,
                "total_requests": self._total,
                "avg_latency_ms": self.avg_latency,
                "endpoints": endpoints,
            }
