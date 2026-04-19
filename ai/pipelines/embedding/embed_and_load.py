"""
embed_and_load.py — embeds all chunks and writes to pgvector document_chunks table.

Supports local (sentence-transformers) and Bedrock (Titan Embeddings V2) backends.
HNSW index is created after bulk load — not before (see CROSS_PHASE.md §4).

Module run:
  uv run python -m ai.pipelines.embedding.embed_and_load
  uv run python -m ai.pipelines.embedding.embed_and_load --dry-run

Environment variables (all in .env):
  EMBED_MODE         local | bedrock          (default: local)
  EMBED_BATCH_SIZE   int                      (default: 64)
  EMBED_QUANTIZE     true | false             (default: false)
  EMBED_MODEL_LOCAL  sentence-transformers ID (default: all-MiniLM-L6-v2)
  AWS_DEFAULT_REGION                          (required for bedrock mode)
  BEDROCK_EMBEDDING_MODEL                     (required for bedrock mode)
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Import shared DB helper
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from db.load_json import get_conn  # noqa: E402

from .chunk_faq import chunk_faq_records
from .chunk_router import route

log = logging.getLogger(__name__)


# ── Embedder factory ────────────────────────────────────────────────────────

def get_embedder(mode: str):
    """Return a callable embed(texts: list[str]) -> list[list[float]]."""
    if mode == "local":
        model_name = os.getenv("EMBED_MODEL_LOCAL", "all-MiniLM-L6-v2")
        quantize    = os.getenv("EMBED_QUANTIZE", "false").lower() == "true"
        batch_size  = int(os.getenv("EMBED_BATCH_SIZE", "64"))

        if quantize:
            # int8 ONNX quantization — requires: uv add sentence-transformers[onnx] optimum[onnxruntime]
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(
                    model_name,
                    backend="onnx",
                    model_kwargs={"file_name": "model_quantized.onnx"},
                )
                log.info("embed_mode", extra={"mode": "local-onnx-int8", "model": model_name})
            except Exception as exc:
                log.warning(
                    "onnx_quantize_failed",
                    extra={"error": str(exc), "fallback": "float32"},
                )
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name)
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            log.info("embed_mode", extra={"mode": "local-float32", "model": model_name})

        def embed(texts: list[str]) -> list[list[float]]:
            return model.encode(texts, batch_size=batch_size, show_progress_bar=True).tolist()

        return embed

    elif mode == "bedrock":
        import boto3, json

        region   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        model_id = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
        client   = boto3.client("bedrock-runtime", region_name=region)
        log.info("embed_mode", extra={"mode": "bedrock", "model": model_id})

        def embed(texts: list[str]) -> list[list[float]]:
            results = []
            for text in texts:
                body = json.dumps({"inputText": text})
                resp = client.invoke_model(
                    modelId=model_id,
                    body=body,
                    contentType="application/json",
                )
                results.append(json.loads(resp["body"].read())["embedding"])
            return results

        return embed

    else:
        raise ValueError(f"Unknown EMBED_MODE: '{mode}'. Choose: local | bedrock")


# ── DB loader ───────────────────────────────────────────────────────────────

def load_chunks(chunks: list[dict], embeddings: list[list[float]], conn) -> int:
    """Insert chunks into document_chunks. Returns rows inserted."""
    from psycopg2.extras import execute_values

    rows = []
    for chunk, emb in zip(chunks, embeddings):
        rows.append((
            chunk["chunk_id"],
            chunk["source_type"],
            chunk["doc_type"],
            chunk.get("policy_number"),
            chunk.get("customer_id"),
            chunk.get("state"),
            chunk.get("page_number"),
            chunk.get("section"),
            chunk.get("chunk_index"),
            chunk.get("token_count"),
            chunk["chunk_text"],
            emb,
        ))

    sql = """
        INSERT INTO document_chunks
          (chunk_id, source_type, doc_type, policy_number, customer_id, state,
           page_number, section, chunk_index, token_count, chunk_text, embedding)
        VALUES %s
        ON CONFLICT (chunk_id) DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(
            cur, sql, rows,
            template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)",
        )
    conn.commit()
    return len(rows)

# ── Customer ID backfill ─────────────────────────────────────────────────────

def backfill_customer_ids(conn) -> int:
    """
    Populate customer_id on document_chunks rows where it is NULL
    by joining through the policies table on policy_number.
    Safe to run multiple times (only updates NULL rows).
    Returns number of rows updated.
    """
    sql = """
        UPDATE document_chunks dc
        SET customer_id = p.customer_id
        FROM policies p
        WHERE dc.policy_number = p.policy_number
          AND dc.customer_id IS NULL
          AND dc.policy_number IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        updated = cur.rowcount
    conn.commit()
    log.info("customer_id_backfilled", extra={"rows_updated": updated})
    return updated
# ── Main ────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    mode = os.getenv("EMBED_MODE", "local")
    embed = get_embedder(mode)

    conn = None if dry_run else get_conn()
    total_inserted = 0

    # ── FAQ chunks ──────────────────────────────────────────────────────────
    faq_path = Path("faqs/faq_corpus.json")
    if not faq_path.exists():
        log.warning("faq_corpus_missing", extra={"path": str(faq_path)})
    else:
        faq_chunks = chunk_faq_records(faq_path)
        if dry_run:
            print(f"[dry-run] FAQ chunks: {len(faq_chunks)}")
        else:
            faq_texts = [c["chunk_text"] for c in faq_chunks]
            faq_embs  = embed(faq_texts)
            n = load_chunks(faq_chunks, faq_embs, conn)
            total_inserted += n
            log.info("faq_loaded", extra={"rows": n})

    # ── PDF chunks ──────────────────────────────────────────────────────────
    docs_dir = Path("documents")
    if not docs_dir.exists():
        log.warning("documents_dir_missing", extra={"path": str(docs_dir)})
    else:
        pdf_files = list(docs_dir.glob("*.pdf"))
        log.info("pdf_files_found", extra={"count": len(pdf_files)})
        skipped = 0
        for pdf in pdf_files:
            try:
                chunks = route(pdf)
            except NotImplementedError:
                skipped += 1
                continue
            except Exception as exc:
                log.error("chunk_failed", extra={"file": pdf.name, "error": str(exc)})
                continue

            if dry_run:
                print(f"[dry-run] {pdf.name}: {len(chunks)} chunks")
            else:
                texts = [c["chunk_text"] for c in chunks]
                embs  = embed(texts)
                n = load_chunks(chunks, embs, conn)
                total_inserted += n

        if skipped:
            log.info("chunkers_not_implemented", extra={"skipped": skipped})
    # ── Backfill customer_id from policies table ─────────────────────────────
    if not dry_run and conn:
        backfilled = backfill_customer_ids(conn)
        log.info("backfill_complete", extra={"rows": backfilled})

    # ── HNSW index — create AFTER bulk load ─────────────────────────────────
    if not dry_run and conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                ON document_chunks USING hnsw (embedding vector_cosine_ops)
            """)
        conn.commit()
        log.info("hnsw_index_created")
        conn.close()

    if not dry_run:
        log.info("embed_load_complete", extra={"total_inserted": total_inserted})
    else:
        print("[dry-run] complete — no DB writes performed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunks and load into pgvector.")
    parser.add_argument("--dry-run", action="store_true", help="Count chunks without writing to DB")
    args = parser.parse_args()

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    main(dry_run=args.dry_run)