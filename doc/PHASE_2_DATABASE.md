# Phase 2 — Database Layer

**Git tag:** `v0.2.0`  
**Deliverable:** `docker compose up -d && uv run python db/load_json.py` — full dataset in Postgres, queryable.

**Meetup demo:** Load data live, run state distribution queries, show fraud-flagged vs. clean claims, demonstrate a JSONB query for PIP fields.

---

## Table of Contents

1. [Docker Compose](#1-docker-compose)
2. [Database Schema](#2-database-schema)
3. [Bulk Loader](#3-bulk-loader)
4. [Verification & Git Tag](#4-verification--git-tag)

---

## 1. Docker Compose

**`docker-compose.yml`**
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: aioi
      POSTGRES_USER: aioi
      POSTGRES_PASSWORD: aioi_local
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/schema.sql:/docker-entrypoint-initdb.d/01_schema.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U aioi"]
      interval: 5s
      timeout: 5s
      retries: 10

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    profiles:
      - phase4   # only started with: docker compose --profile phase4 up

volumes:
  pgdata:
  ollama_models:
```

> **Note:** Ollama is behind a `phase4` profile — `docker compose up -d` in Phases 2 and 3 starts only Postgres. Phase 4 starts it with `docker compose --profile phase4 up -d`.

---

## 2. Database Schema

**`db/schema.sql`**
```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Customers ──────────────────────────────────────────────────────────────
CREATE TABLE customers (
    customer_id     VARCHAR(20)  PRIMARY KEY,
    first_name      VARCHAR(100) NOT NULL,
    last_name       VARCHAR(100) NOT NULL,
    state           CHAR(2)      NOT NULL,
    zip             CHAR(5),
    email           VARCHAR(255),
    dob             DATE,
    credit_score    SMALLINT,
    created_at      TIMESTAMPTZ  DEFAULT NOW(),
    source          VARCHAR(50)  NOT NULL DEFAULT 'synthetic-v1'
);

-- ── Policies ───────────────────────────────────────────────────────────────
CREATE TABLE policies (
    policy_number   VARCHAR(20)  PRIMARY KEY,
    customer_id     VARCHAR(20)  NOT NULL REFERENCES customers(customer_id),
    state           CHAR(2)      NOT NULL,
    effective_date  DATE         NOT NULL,
    expiry_date     DATE         NOT NULL,
    status          VARCHAR(30)  NOT NULL,
    coverages       JSONB        NOT NULL,   -- state-specific coverage details
    vehicle         JSONB        NOT NULL,   -- make, model, year, vin
    premium_annual  NUMERIC(10,2),
    drive_score     NUMERIC(5,2),
    agent_id        VARCHAR(20),
    source          VARCHAR(50)  NOT NULL DEFAULT 'synthetic-v1'
);

-- ── Claims ─────────────────────────────────────────────────────────────────
CREATE TABLE claims (
    claim_id           VARCHAR(20)  PRIMARY KEY,
    policy_number      VARCHAR(20)  NOT NULL REFERENCES policies(policy_number),
    customer_id        VARCHAR(20)  NOT NULL REFERENCES customers(customer_id),
    state              CHAR(2)      NOT NULL,
    incident_date      DATE,
    filed_date         DATE,
    claim_type         VARCHAR(50),
    status             VARCHAR(30),
    claim_amount       NUMERIC(12,2),
    settlement_amount  NUMERIC(12,2),
    adjuster_notes     TEXT,
    incident_narrative TEXT,
    is_fraud           BOOLEAN      NOT NULL DEFAULT FALSE,
    fraud_signals      TEXT[],
    source             VARCHAR(50)  NOT NULL DEFAULT 'synthetic-v1'
);

-- ── Telematics ─────────────────────────────────────────────────────────────
CREATE TABLE telematics (
    trip_id                VARCHAR(30)  PRIMARY KEY,
    policy_number          VARCHAR(20)  NOT NULL REFERENCES policies(policy_number),
    customer_id            VARCHAR(20)  NOT NULL REFERENCES customers(customer_id),
    trip_date              TIMESTAMPTZ,
    distance_miles         NUMERIC(8,2),
    duration_minutes       NUMERIC(8,2),
    hard_brakes            SMALLINT,
    rapid_accelerations    SMALLINT,
    speeding_events        SMALLINT,
    night_driving_pct      NUMERIC(5,4),
    drive_score            NUMERIC(5,2),
    source                 VARCHAR(50)  NOT NULL DEFAULT 'synthetic-v1'
);

-- ── Document Embeddings (pgvector) ─────────────────────────────────────────
CREATE TABLE document_chunks (
    chunk_id        VARCHAR(100) PRIMARY KEY,
    source_type     VARCHAR(30)  NOT NULL,  -- 'policy_document' | 'faq'
    doc_type        VARCHAR(30)  NOT NULL,  -- 'declaration_page' | 'claim_letter' | 'renewal_notice' | 'faq'
    policy_number   VARCHAR(20),
    customer_id     VARCHAR(20),
    state           CHAR(2),
    page_number     SMALLINT,
    section         VARCHAR(100),
    chunk_index     SMALLINT,
    token_count     SMALLINT,
    chunk_text      TEXT         NOT NULL,
    embedding       vector(384),            -- 384-dim for all-MiniLM-L6-v2; 1024 for Titan V2
    created_at      TIMESTAMPTZ  DEFAULT NOW()
);

-- ── Indexes ────────────────────────────────────────────────────────────────
CREATE INDEX idx_policies_customer   ON policies(customer_id);
CREATE INDEX idx_policies_state      ON policies(state);
CREATE INDEX idx_policies_status     ON policies(status);

CREATE INDEX idx_claims_policy       ON claims(policy_number);
CREATE INDEX idx_claims_customer     ON claims(customer_id);
CREATE INDEX idx_claims_state        ON claims(state);
CREATE INDEX idx_claims_fraud        ON claims(is_fraud);
CREATE INDEX idx_claims_status       ON claims(status);

CREATE INDEX idx_telematics_policy   ON telematics(policy_number);
CREATE INDEX idx_telematics_customer ON telematics(customer_id);

CREATE INDEX idx_chunks_source_type  ON document_chunks(source_type);
CREATE INDEX idx_chunks_policy       ON document_chunks(policy_number);
CREATE INDEX idx_chunks_customer     ON document_chunks(customer_id);
CREATE INDEX idx_chunks_state        ON document_chunks(state);

-- pgvector HNSW index — added after embeddings are loaded (Phase 4)
-- CREATE INDEX idx_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);
```

> **Flag — embedding dimensions:** The schema declares `vector(384)` for `all-MiniLM-L6-v2`. Titan Embeddings V2 outputs 1024 dimensions. If you anticipate switching to Titan in Phase 5 without re-embedding, use `vector(1024)` from the start and accept slightly larger index size locally. Recommendation: keep `vector(384)` for local; re-ingest with Titan at Phase 5. See [CROSS_PHASE.md](./CROSS_PHASE.md) §2 for migration options.

---

## 3. Bulk Loader

**`db/load_json.py`**
```python
"""
Bulk loader — reads generated JSON files and inserts into Postgres.
Usage: uv run python db/load_json.py
       uv run python db/load_json.py --truncate   # wipe and reload
"""
import json, argparse, os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

load_dotenv()

def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "aioi"),
        user=os.getenv("DB_USER", "aioi"),
        password=os.getenv("DB_PASSWORD", "aioi_local"),
    )

def load_table(conn, table: str, records: list[dict], columns: list[str]):
    rows = [[r.get(c) for c in columns] for r in records]
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s ON CONFLICT DO NOTHING"
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    print(f"  {table}: {len(rows):,} rows loaded")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truncate", action="store_true")
    args = parser.parse_args()

    data = Path("data")
    conn = get_conn()

    if args.truncate:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE telematics, claims, policies, customers CASCADE")
        conn.commit()
        print("Tables truncated.")

    customers  = json.loads((data / "customers.json").read_text())
    policies   = json.loads((data / "policies.json").read_text())
    claims     = json.loads((data / "claims.json").read_text())
    telematics = json.loads((data / "telematics.json").read_text())

    import json as _json
    # Serialize JSONB fields
    for p in policies:
        p["coverages"] = _json.dumps(p["coverages"])
        p["vehicle"]   = _json.dumps(p["vehicle"])

    load_table(conn, "customers",  customers,  ["customer_id","first_name","last_name","state","zip","email","dob","credit_score","created_at","source"])
    load_table(conn, "policies",   policies,   ["policy_number","customer_id","state","effective_date","expiry_date","status","coverages","vehicle","premium_annual","drive_score","agent_id","source"])
    load_table(conn, "claims",     claims,     ["claim_id","policy_number","customer_id","state","incident_date","filed_date","claim_type","status","claim_amount","settlement_amount","adjuster_notes","incident_narrative","is_fraud","fraud_signals","source"])
    load_table(conn, "telematics", telematics, ["trip_id","policy_number","customer_id","trip_date","distance_miles","duration_minutes","hard_brakes","rapid_accelerations","speeding_events","night_driving_pct","drive_score","source"])

    conn.close()
    print("\n✓ Load complete.")

if __name__ == "__main__":
    main()
```

---

## 4. Verification & Git Tag

### Verification

```powershell
# Start Postgres only
docker compose up -d postgres

# Wait for healthy, then load, truncate mostly
#uv run python db/load_json.py
uv run python db/load_json.py --truncate

# Spot-check queries
docker exec -it iron-oak-insurance-postgres-1 psql -U aioi -d aioi -c `
  "SELECT state, COUNT(*) FROM policies GROUP BY state ORDER BY count DESC LIMIT 10;"

docker exec -it iron-oak-insurance-postgres-1 psql -U aioi -d aioi -c `
  "SELECT is_fraud, COUNT(*) FROM claims GROUP BY is_fraud;"

docker exec -it iron-oak-insurance-postgres-1 psql -U aioi -d aioi -c `
  "SELECT p.state, p.coverages->'pip'->'required' AS pip_required, COUNT(*) `
   FROM policies p GROUP BY 1,2 ORDER BY 1;"

# Maintenance: To clean up if required.
docker compose down -v
```

### Phase Gate Checklist

- [ ] `docker compose up -d` succeeds
- [ ] All 4 tables populated with expected row counts
- [ ] FK integrity holds (no orphaned claims or policies)
- [ ] JSONB query for PIP fields returns results
- [ ] `document_chunks` table exists and is empty (embeddings load in Phase 4)

### Git Tag

```bash
git add -A
git commit -m "Phase 2: database layer — schema, docker-compose, bulk loader"
git tag v0.2.0
```

---

*Previous: [PHASE_1_DATA_GEN.md](./PHASE_1_DATA_GEN.md) · Next: [PHASE_3_ML.md](./PHASE_3_ML.md)*
