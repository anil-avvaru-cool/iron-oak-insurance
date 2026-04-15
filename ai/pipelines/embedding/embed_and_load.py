"""
embed_and_load.py — embeds chunks and writes to pgvector.
Supports both local (sentence-transformers) and Bedrock (Titan) embedding.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

load_dotenv()

def get_embedder(mode: str = "local"):
    if mode == "local":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        def embed(texts: list[str]) -> list[list[float]]:
            return model.encode(texts, show_progress_bar=True).tolist()
        return embed
    elif mode == "bedrock":
        import boto3, json
        client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_DEFAULT_REGION","us-east-1"))
        model_id = os.getenv("BEDROCK_EMBEDDING_MODEL","amazon.titan-embed-text-v2:0")
        def embed(texts: list[str]) -> list[list[float]]:
            results = []
            for t in texts:
                body = json.dumps({"inputText": t})
                resp = client.invoke_model(modelId=model_id, body=body, contentType="application/json")
                results.append(json.loads(resp["body"].read())["embedding"])
            return results
        return embed
    else:
        raise ValueError(f"Unknown embed mode: {mode}")

def load_chunks(chunks: list[dict], embeddings: list[list[float]], conn):
    rows = []
    for chunk, emb in zip(chunks, embeddings):
        rows.append((
            chunk["chunk_id"], chunk["source_type"], chunk["doc_type"],
            chunk.get("policy_number"), chunk.get("customer_id"), chunk.get("state"),
            chunk.get("page_number"), chunk.get("section"), chunk.get("chunk_index"),
            chunk.get("token_count"), chunk["chunk_text"], emb,
        ))
    sql = """
        INSERT INTO document_chunks
          (chunk_id, source_type, doc_type, policy_number, customer_id, state,
           page_number, section, chunk_index, token_count, chunk_text, embedding)
        VALUES %s
        ON CONFLICT (chunk_id) DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)")
    conn.commit()

def main(mode: str = "local"):
    from .chunk_faq import chunk_faq_records
    from .chunk_router import route

    embed = get_embedder(mode)
    conn  = psycopg2.connect(
        host=os.getenv("DB_HOST","localhost"), port=os.getenv("DB_PORT",5432),
        dbname=os.getenv("DB_NAME","aioi"), user=os.getenv("DB_USER","aioi"),
        password=os.getenv("DB_PASSWORD","aioi_local"),
    )

    # FAQ chunks
    faq_chunks = chunk_faq_records(Path("faqs/faq_corpus.json"))
    faq_texts  = [c["chunk_text"] for c in faq_chunks]
    faq_embs   = embed(faq_texts)
    load_chunks(faq_chunks, faq_embs, conn)
    print(f"Loaded {len(faq_chunks)} FAQ chunks")

    # PDF chunks (skips stubs gracefully)
    for pdf in Path("documents").glob("*.pdf"):
        try:
            chunks = route(pdf)
            texts  = [c["chunk_text"] for c in chunks]
            embs   = embed(texts)
            load_chunks(chunks, embs, conn)
        except NotImplementedError:
            pass  # PDF chunkers are stubs until implemented

    # Create HNSW index after bulk load — not before (see CROSS_PHASE.md §4)
    with conn.cursor() as cur:
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON document_chunks USING hnsw (embedding vector_cosine_ops)
        """)
    conn.commit()
    print("HNSW index created.")
    conn.close()

if __name__ == "__main__":
    import sys
    main(mode=sys.argv[1] if len(sys.argv) > 1 else "local")