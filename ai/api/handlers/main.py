"""
main.py — FastAPI entry point, Lambda-compatible via Mangum.

Module run: uv run python -m ai.api.handlers.main
Lambda:     handler = Mangum(app)

Environment variables required (no defaults):
  DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
  LOG_LEVEL (optional, defaults to INFO)
"""
import time
import uuid

from fastapi import FastAPI, Request, Response
from mangum import Mangum

from ai.api.routers.models_router import router as models_router
from ai.utils.log import get_logger

log = get_logger(__name__)

app = FastAPI(title="AIOI AI API", version="0.3.0")
app.include_router(models_router)


@app.middleware("http")
async def request_log_middleware(request: Request, call_next) -> Response:
    """Log one structured line per request. Never logs body content."""
    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    response = await call_next(request)
    latency = int((time.time() - t0) * 1000)
    log.info(
        "http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=latency,
    )
    return response


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.3.0"}


# Lambda handler — used by Mangum in Phase 5
handler = Mangum(app, lifespan="off")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai.api.handlers.main:app", host="0.0.0.0", port=8000, reload=True)