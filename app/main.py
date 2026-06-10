import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from app.core.observability import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    log_event,
    metrics_response,
)
from app.core.tracing import setup_tracing
from app.dependencies import get_index_service, get_retrieval_service
from app.routers import health, index, qa, upload


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_svc = get_index_service()
    bm25_result = index_svc.rebuild_bm25_from_sqlite()
    print("[startup] rebuild_bm25_from_sqlite =", bm25_result, flush=True)

    api_key = os.getenv("AIHUBMIX_API_KEY")
    if api_key:
        retrieval_svc = get_retrieval_service()
        vector_result = retrieval_svc.rebuild_vectors_from_sqlite()
        print("[startup] rebuild_vectors_from_sqlite =", vector_result, flush=True)
    else:
        print("[startup] skip vector rebuild: missing AIHUBMIX_API_KEY", flush=True)

    yield


app = FastAPI(title="CiteRAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_tracing()
FastAPIInstrumentor.instrument_app(app)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    endpoint = request.url.path
    method = request.method
    status = str(response.status_code)

    REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)

    log_event(
        "http_request",
        endpoint=endpoint,
        method=method,
        status_code=response.status_code,
        duration_ms=round(duration * 1000, 2),
    )

    return response


@app.get("/metrics")
def metrics():
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


app.include_router(health.router)
app.include_router(upload.router)
app.include_router(index.router)
app.include_router(qa.router)
