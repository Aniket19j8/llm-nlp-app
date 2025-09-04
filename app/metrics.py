from prometheus_client import Counter, Histogram


REQUESTS = Counter("app_requests_total", "Total API requests", ["endpoint"])
LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5)
)