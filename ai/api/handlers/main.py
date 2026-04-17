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

load_dotenv()


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

# Add after existing router includes:
from ai.api.routers.rag_router import router as rag_router
app.include_router(rag_router)
