"""
main.py — add lifespan for SentenceTransformer and register rag_router.
Add these changes to the existing Phase 3 main.py.
"""
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ai.pipelines.rag.retrieval_router import set_embedder
import time
import uuid

from fastapi import FastAPI, Request, Response
from mangum import Mangum

from ai.api.routers.models_router import router as models_router
from ai.utils.log import get_logger


load_dotenv()

log = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model once at startup; release on shutdown."""
    embed_mode     = os.getenv("EMBED_MODE", "local")
    embed_quantize = os.getenv("EMBED_QUANTIZE", "false").lower() == "true"
    model_name     = os.getenv("EMBED_MODEL_LOCAL", "all-MiniLM-L6-v2")

    if embed_mode == "local":
        from sentence_transformers import SentenceTransformer
        if embed_quantize:
            try:
                app.state.embedder = SentenceTransformer(
                    model_name,
                    backend="onnx",
                    model_kwargs={"file_name": "model_quantized.onnx"},
                )
            except Exception:
                app.state.embedder = SentenceTransformer(model_name)
        else:
            app.state.embedder = SentenceTransformer(model_name)            
    else:
        # Bedrock mode — embedding happens in embed_and_load; at query time
        # we use Titan via boto3 directly. A lightweight wrapper can be set here.
        app.state.embedder = None
    set_embedder(app.state.embedder)  # make available to RAG router

    yield  # app runs here

    # Cleanup (model will be GC'd; nothing explicit needed for sentence-transformers)
    app.state.embedder = None

app = FastAPI(title="AIOI AI API", version="0.4.0", lifespan=lifespan)

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


app.include_router(models_router)

# Add after existing router includes:
from ai.api.routers.rag_router import router as rag_router
app.include_router(rag_router)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.3.0"}


# Lambda handler — used by Mangum in Phase 5
handler = Mangum(app, lifespan="off")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai.api.handlers.main:app", host="0.0.0.0", port=8000, reload=True)
