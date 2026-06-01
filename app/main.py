"""
main.py — FastAPI entrypoint for Store Intelligence API.  Port 8000.

Endpoints:
    POST   /events/ingest
    GET    /stores/{store_id}/metrics
    GET    /stores/{store_id}/funnel
    GET    /stores/{store_id}/heatmap
    GET    /stores/{store_id}/anomalies
    GET    /health
    *      /cameras/*   (see cameras.py)

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .database  import create_tables, check_db
from .ingestion import ingest_events
from .metrics   import compute_metrics, compute_funnel, compute_heatmap
from .anomalies import compute_anomalies
from .health    import compute_health
from .scheduler import start_scheduler
from .cameras   import router as cameras_router
from .models    import IngestRequest, IngestResponse

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("api")


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: DB init + scheduler start
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    logger.info("DB tables ready.")

    logger.info("Starting background scheduler…")
    start_scheduler()          # polls /cameras every 10 s, dispatches to /cameras/detect/camera

    logger.info("Store Intelligence API started on port 8000.")
    yield
    logger.info("Store Intelligence API shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Store Intelligence API",
    version     = "1.0.0",
    description = "Apex Retail offline analytics — CCTV events → live store metrics.",
    lifespan    = lifespan,
)

# Mount camera routes at /cameras
app.include_router(cameras_router)

# CORS — allow dashboard on any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request logging middleware
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-ID") or str(uuid.uuid4())
    request.state.trace_id = trace_id
    t0 = time.perf_counter()

    response: Response = await call_next(request)

    latency_ms  = int((time.perf_counter() - t0) * 1000)
    path_parts  = request.url.path.split("/")
    store_id    = (
        path_parts[2]
        if len(path_parts) >= 3 and path_parts[1] == "stores"
        else None
    )

    logger.info(
        "Request",
        extra={
            "trace_id":    trace_id,
            "method":      request.method,
            "endpoint":    request.url.path,
            "store_id":    store_id,
            "status_code": response.status_code,
            "latency_ms":  latency_ms,
        },
    )

    response.headers["X-Trace-ID"] = trace_id
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_db(trace_id: str):
    if not check_db():
        raise HTTPException(
            status_code=503,
            detail={
                "error":    "SERVICE_UNAVAILABLE",
                "message":  "Database is currently unavailable. Retry after a moment.",
                "trace_id": trace_id,
            },
        )


def _get_trace(request: Request) -> str:
    return getattr(request.state, "trace_id", str(uuid.uuid4()))


# ─────────────────────────────────────────────────────────────────────────────
# POST /events/ingest
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/events/ingest",
    response_model=IngestResponse,
    summary="Ingest a batch of detection events (idempotent by event_id)",
)
def post_ingest(payload: IngestRequest, request: Request):
    trace_id = _get_trace(request)
    _require_db(trace_id)

    if not payload.events:
        return IngestResponse(accepted=0, rejected=0, duplicate=0)

    t0     = time.perf_counter()
    result = ingest_events(payload, trace_id)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "Ingest",
        extra={
            "trace_id":    trace_id,
            "endpoint":    "/events/ingest",
            "store_id":    list({e.store_id for e in payload.events}),
            "event_count": len(payload.events),
            "accepted":    result.accepted,
            "rejected":    result.rejected,
            "duplicate":   result.duplicate,
            "latency_ms":  latency_ms,
            "status_code": 207 if result.rejected else 200,
        },
    )

    status_code = 207 if result.rejected > 0 else 200
    return JSONResponse(content=result.model_dump(), status_code=status_code)


# ─────────────────────────────────────────────────────────────────────────────
# GET /stores/{store_id}/metrics
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/stores/{store_id}/metrics",
    summary="Real-time store metrics: visitors, conversion rate, dwell, queue",
)
def get_metrics(store_id: str, request: Request):
    trace_id = _get_trace(request)
    _require_db(trace_id)
    try:
        return compute_metrics(store_id)
    except Exception as exc:
        logger.exception("metrics failed store=%s", store_id, extra={"trace_id": trace_id})
        raise HTTPException(
            status_code=500,
            detail={"error": "INTERNAL_ERROR", "trace_id": trace_id},
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# GET /stores/{store_id}/funnel
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/stores/{store_id}/funnel",
    summary="Conversion funnel: Entry → Zone Visit → Billing Queue → Purchase",
)
def get_funnel(store_id: str, request: Request):
    trace_id = _get_trace(request)
    _require_db(trace_id)
    try:
        return compute_funnel(store_id)
    except Exception as exc:
        logger.exception("funnel failed store=%s", store_id, extra={"trace_id": trace_id})
        raise HTTPException(
            status_code=500,
            detail={"error": "INTERNAL_ERROR", "trace_id": trace_id},
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# GET /stores/{store_id}/heatmap
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/stores/{store_id}/heatmap",
    summary="Zone visit frequency and dwell, normalised 0–100 for grid rendering",
)
def get_heatmap(store_id: str, request: Request):
    trace_id = _get_trace(request)
    _require_db(trace_id)
    try:
        return compute_heatmap(store_id)
    except Exception as exc:
        logger.exception("heatmap failed store=%s", store_id, extra={"trace_id": trace_id})
        raise HTTPException(
            status_code=500,
            detail={"error": "INTERNAL_ERROR", "trace_id": trace_id},
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# GET /stores/{store_id}/anomalies
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/stores/{store_id}/anomalies",
    summary="Active anomalies: queue spike, conversion drop, dead zone",
)
def get_anomalies(store_id: str, request: Request):
    trace_id = _get_trace(request)
    _require_db(trace_id)
    try:
        return compute_anomalies(store_id)
    except Exception as exc:
        logger.exception("anomalies failed store=%s", store_id, extra={"trace_id": trace_id})
        raise HTTPException(
            status_code=500,
            detail={"error": "INTERNAL_ERROR", "trace_id": trace_id},
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    summary="Service health: DB status, last event timestamp per store, STALE_FEED warning",
)
def get_health(request: Request):
    try:
        result      = compute_health()
        status_code = 200 if result.db_ok else 503
        return JSONResponse(content=result.model_dump(), status_code=status_code)
    except Exception as exc:
        logger.exception("health check failed")
        return JSONResponse(
            status_code=503,
            content={
                "status":     "degraded",
                "db_ok":      False,
                "stores":     [],
                "checked_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "error":      str(exc),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Global exception handler — never leak stack traces to the client
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    trace_id = _get_trace(request)
    logger.exception(
        "Unhandled exception",
        extra={"trace_id": trace_id, "path": request.url.path},
    )
    return JSONResponse(
        status_code=500,
        content={
            "error":    "INTERNAL_ERROR",
            "message":  "An unexpected error occurred.",
            "trace_id": trace_id,
        },
    )