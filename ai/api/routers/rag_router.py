from fastapi import APIRouter
from pydantic import BaseModel
from ai.pipelines.rag.retrieval_router import classify_query
from ai.pipelines.rag.rag_pipeline import retrieve, generate_answer
import os
from dotenv import load_dotenv

load_dotenv()
router = APIRouter(prefix="/rag", tags=["rag"])

class QueryRequest(BaseModel):
    query: str
    customer_id: str | None = None
    mode: str = "local"   # "local" | "bedrock"

class QueryResponse(BaseModel):
    answer: str
    strategy: str
    sources: list[dict]

# TODO: add /rag/debug endpoint behind DEBUG_MODE env flag.
# Returns: routing decision + retrieved chunks, no LLM generation.
# Disable in production. See CROSS_PHASE.md §9.4.

@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    from sentence_transformers import SentenceTransformer
    import psycopg2

    # NOTE: move SentenceTransformer load to FastAPI lifespan — see CROSS_PHASE.md §5
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode(req.query).tolist()

    strategy = classify_query(req.query)
    if req.customer_id:
        strategy["customer_id"] = req.customer_id

    conn = psycopg2.connect(...)  # use get_conn() from db helpers
    chunks = retrieve(q_emb, strategy, conn)
    answer = generate_answer(req.query, chunks, mode=req.mode)

    return QueryResponse(answer=answer, strategy=strategy["strategy"], sources=chunks)
