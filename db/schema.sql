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
    reported_passengers  SMALLINT,
    num_witnesses        SMALLINT,
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

CREATE TABLE iso_claim_history (
    id               SERIAL PRIMARY KEY,
    customer_id      VARCHAR(20) REFERENCES customers(customer_id),
    vin              CHAR(17),
    prior_carrier    VARCHAR(50),
    prior_claim_date DATE,
    prior_claim_type VARCHAR(50),
    prior_claim_amount NUMERIC(12,2),
    role             VARCHAR(20),
    fraud_indicator  BOOLEAN DEFAULT FALSE,
    source           VARCHAR(50) DEFAULT 'synthetic-iso-v1'
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

CREATE INDEX idx_iso_customer ON iso_claim_history(customer_id);
CREATE INDEX idx_iso_vin      ON iso_claim_history(vin);

CREATE INDEX idx_chunks_source_type  ON document_chunks(source_type);
CREATE INDEX idx_chunks_policy       ON document_chunks(policy_number);
CREATE INDEX idx_chunks_customer     ON document_chunks(customer_id);
CREATE INDEX idx_chunks_state        ON document_chunks(state);

-- pgvector HNSW index — added after embeddings are loaded (Phase 4)
-- CREATE INDEX idx_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);