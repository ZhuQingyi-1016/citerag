import json
import logging
import time
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

logger = logging.getLogger("citerag.observability")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


REQUEST_COUNT = Counter(
    "citerag_requests_total",
    "Total number of handled requests",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "citerag_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint", "method"],
)

UPLOAD_COUNT = Counter(
    "citerag_upload_requests_total",
    "Total number of upload requests",
    ["status"],
)

UPLOAD_LATENCY = Histogram(
    "citerag_upload_latency_seconds",
    "Upload latency in seconds",
)

INDEX_COUNT = Counter(
    "citerag_index_requests_total",
    "Total number of index requests",
    ["status", "chunk_method"],
)

INDEX_LATENCY = Histogram(
    "citerag_index_latency_seconds",
    "Index latency in seconds",
    ["chunk_method"],
)

ASK_COUNT = Counter(
    "citerag_ask_requests_total",
    "Total number of ask requests",
    ["status", "retrieval_mode"],
)

ASK_LATENCY = Histogram(
    "citerag_ask_latency_seconds",
    "Ask latency in seconds",
    ["retrieval_mode"],
)

RETRIEVAL_HITS = Histogram(
    "citerag_retrieval_hits_count",
    "Number of hits returned by retrieval",
    ["retrieval_mode"],
    buckets=(0, 1, 2, 3, 5, 10),
)

CITATION_COUNT = Histogram(
    "citerag_citations_count",
    "Number of citations returned in answers",
    ["retrieval_mode"],
    buckets=(0, 1, 2, 3, 5, 10),
)


def now_perf() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def log_event(event: str, **kwargs: Any) -> None:
    payload = {"event": event, **kwargs}
    logger.info(json.dumps(payload, ensure_ascii=False))


def metrics_response() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST