# Phase 1 — Repository Bootstrap, Toolchain & Data Generation

**Git tag:** `v0.1.0`  
**Deliverable:** `uv run python data-gen/generators/run_all.py` produces a complete local dataset.

**Meetup demo:** Run `run_all.py` live, inspect JSON records, open a generated PDF, show how state rules differ between TX and MI.

---

## Table of Contents

1. [Repository Bootstrap](#1-repository-bootstrap)
2. [Toolchain & Dependency Management](#2-toolchain--dependency-management)
3. [Config Files](#3-config-files)
4. [JSON Schemas](#4-json-schemas)
5. [Generator Contracts](#5-generator-contracts)
6. [run_all.py — Entry Point](#6-run_allpy--entry-point)
7. [Verification & Git Tag](#7-verification--git-tag)

---

## 1. Repository Bootstrap

### 1.1 Create the Repository

```bash
# Linux / macOS
mkdir iron-oak-insurance && cd iron-oak-insurance
git init
git branch -M main
```

```powershell
# Windows
mkdir iron-oak-insurance; cd iron-oak-insurance
git init
git branch -M main
```

### 1.2 Full Folder Structure

Create all directories up front so imports never fail and Git tags are clean.

```bash
# Linux / macOS
mkdir -p data-gen/generators data-gen/schemas data-gen/config \
         ai/agents/claims_agent ai/agents/fraud_agent ai/agents/policy_advisor_agent \
         ai/pipelines/ingestion ai/pipelines/embedding ai/pipelines/rag \
         ai/models/fraud_detection ai/models/risk_scoring ai/models/churn_prediction \
         ai/api/handlers ai/api/routers \
         db infra/cdk tests/unit tests/integration \
         data documents faqs
```

```powershell
# Windows
$dirs = @(
  "data-gen\generators","data-gen\schemas","data-gen\config",
  "ai\agents\claims_agent","ai\agents\fraud_agent","ai\agents\policy_advisor_agent",
  "ai\pipelines\ingestion","ai\pipelines\embedding","ai\pipelines\rag",
  "ai\models\fraud_detection","ai\models\risk_scoring","ai\models\churn_prediction",
  "ai\api\handlers","ai\api\routers",
  "db","infra\cdk","tests\unit","tests\integration",
  "data","documents","faqs"
)
$dirs | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ }
```

### 1.3 Root Files

**`.gitignore`**
```gitignore
# Generated data — never committed
data/
documents/
faqs/

# Environment
.env
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# uv
.venv/
uv.lock        # commit this if sharing exact reproducible builds

# AWS CDK
infra/cdk/cdk.out/
infra/cdk/node_modules/

# Models cached locally
.cache/
*.pt
*.onnx
```

**`.env.example`** — committed, no values
```dotenv
# AWS credentials — required for Phase 5 only
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1

# Bedrock
BEDROCK_MODEL_ID_SONNET=anthropic.claude-sonnet-4-6
BEDROCK_MODEL_ID_HAIKU=anthropic.claude-haiku-4-5-20251001
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=aioi
DB_USER=aioi
DB_PASSWORD=aioi_local

# API
API_HOST=0.0.0.0
API_PORT=8000

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=all-minilm:l6-v2
```

> **Flag:** `requirements.txt` at the repo root is a generated export artifact (see Section 2.4), not the source of truth. The source of truth is `pyproject.toml`.

---

## 2. Toolchain & Dependency Management

### 2.1 Install uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify: `uv --version`

### 2.2 Initialize the Project

```bash
uv init --no-workspace
uv python pin 3.13   #  Use latest version
```

This creates `pyproject.toml`. Edit it:

**`pyproject.toml`**
```toml
[project]
name = "iron-oak-insurance"
version = "0.1.0"
description = "AIOI AI Strategy & Data Platform"
requires-python = ">=3.11"
dependencies = []   # populated per phase below

[dependency-groups]
dev = [
  "pytest>=8.0",
  "pytest-asyncio>=0.23",
  "httpx>=0.27",       # FastAPI test client
  "ruff>=0.4",         # linter + formatter
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 2.3 Dependencies by Phase

Add dependencies as each phase is built — do not front-load all packages.

**Phase 1 — Data Generation**
```bash
uv add faker python-dateutil reportlab Pillow tqdm
```

**Phase 2 — Database**
```bash
uv add psycopg2-binary sqlalchemy alembic python-dotenv
```

**Phase 3 — ML Models**
```bash
uv add xgboost scikit-learn pandas numpy fastapi uvicorn mangum
# mangum = Lambda adapter for FastAPI
```

**Phase 4 — RAG Pipeline**
```bash
uv add sentence-transformers pgvector PyMuPDF langchain-community \
       langchain-postgres openai   # openai client used as Ollama-compatible interface
```

**Phase 5 — Bedrock Agents**
```bash
uv add boto3 aws-cdk-lib constructs
```

### 2.4 Generate requirements.txt for Docker

After each phase, regenerate so Docker builds stay in sync:

```bash
uv export --no-dev --format requirements-txt > requirements.txt
```

Commit `requirements.txt` alongside `pyproject.toml`. Docker's `pip install -r requirements.txt` uses this file; local development uses `uv sync`.

### 2.5 Virtual Environment

```bash
uv sync          # creates .venv, installs all declared deps
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows
```

### 2.6 Linting & Formatting

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
```

Add to `pyproject.toml`:
```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I"]   # pycodestyle, pyflakes, isort
```

---

## 3. Config Files

### `data-gen/config/states.json` — structure (excerpt)

```json
{
  "TX": {
    "weight": 9,
    "no_fault": false,
    "min_liability": {"bodily_injury_per_person": 30000, "bodily_injury_per_accident": 60000, "property_damage": 25000},
    "uninsured_motorist_required": false,
    "total_loss_threshold": 1.00,
    "pip_required": false,
    "pip_limit": null,
    "claims_ack_days": 15,
    "claims_settle_days": 30
  },
  "MI": {
    "weight": 4,
    "no_fault": true,
    "min_liability": {"bodily_injury_per_person": 50000, "bodily_injury_per_accident": 100000, "property_damage": 10000},
    "uninsured_motorist_required": false,
    "total_loss_threshold": 0.75,
    "pip_required": true,
    "pip_limit": 500000,
    "claims_ack_days": 10,
    "claims_settle_days": 30
  },
  "PA": {
    "weight": 5,
    "no_fault": true,
    "min_liability": {"bodily_injury_per_person": 15000, "bodily_injury_per_accident": 30000, "property_damage": 5000},
    "uninsured_motorist_required": false,
    "total_loss_threshold": 0.75,
    "pip_required": true,
    "pip_limit": 5000,
    "claims_ack_days": 15,
    "claims_settle_days": 30
  }
}
```

> Suggested weight scale: 1–10, where CA/TX/FL get 9–10, mid-size states (PA, OH, IL) get 5–7, and low-population states (WY, ND, VT) get 1. This maps roughly to US vehicle registration distribution.  
> All 50 states + DC must be present. The `weight` field is read by `customer_gen.py` for population distribution.

### `data-gen/config/coverage_rules.json` — structure

```json
{
  "coverage_types": ["liability", "collision", "comprehensive", "pip", "uninsured_motorist", "gap", "roadside"],
  "deductible_options": [250, 500, 1000, 2500],
  "liability_limits": ["15/30/5", "25/50/10", "50/100/25", "100/300/50", "250/500/100"],
  "drive_score_discount_tiers": [
    {"min": 90, "discount_pct": 0.15},
    {"min": 75, "discount_pct": 0.08},
    {"min": 60, "discount_pct": 0.03},
    {"min": 0,  "discount_pct": 0.00}
  ]
}
```

---

## 4. JSON Schemas

All schemas live in `data-gen/schemas/`. Every generator validates output against its schema before writing.

### `customer.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["customer_id","first_name","last_name","state","zip","email","dob","created_at","source"],
  "properties": {
    "customer_id":  {"type": "string", "pattern": "^CUST-[0-9]{4,6}$"},
    "first_name":   {"type": "string"},
    "last_name":    {"type": "string"},
    "state":        {"type": "string", "minLength": 2, "maxLength": 2},
    "zip":          {"type": "string", "pattern": "^[0-9]{5}$"},
    "email":        {"type": "string", "format": "email"},
    "dob":          {"type": "string", "format": "date"},
    "credit_score": {"type": "integer", "minimum": 300, "maximum": 850},
    "created_at":   {"type": "string", "format": "date-time"},
    "source":       {"type": "string", "const": "synthetic-v1"}
  }
}
```

### `policy.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["policy_number","customer_id","state","effective_date","expiry_date","status","coverages","premium_annual","source"],
  "properties": {
    "policy_number":   {"type": "string", "pattern": "^[A-Z]{2}-[0-9]{5}$"},
    "customer_id":     {"type": "string"},
    "state":           {"type": "string"},
    "effective_date":  {"type": "string", "format": "date"},
    "expiry_date":     {"type": "string", "format": "date"},
    "status":          {"type": "string", "enum": ["active","lapsed","cancelled","pending_renewal"]},
    "coverages":       {"type": "object"},
    "vehicle":         {"type": "object"},
    "premium_annual":  {"type": "number", "minimum": 0},
    "drive_score":     {"type": ["number","null"], "minimum": 0, "maximum": 100},
    "agent_id":        {"type": "string"},
    "source":          {"type": "string", "const": "synthetic-v1"}
  }
}
```

### `claim.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["claim_id","policy_number","customer_id","state","incident_date","filed_date","claim_type","status","is_fraud","source"],
  "properties": {
    "claim_id":        {"type": "string", "pattern": "^CLM-[0-9]{4,6}$"},
    "policy_number":   {"type": "string"},
    "customer_id":     {"type": "string"},
    "state":           {"type": "string"},
    "incident_date":   {"type": "string", "format": "date"},
    "filed_date":      {"type": "string", "format": "date"},
    "claim_type":      {"type": "string", "enum": ["collision","comprehensive","liability","pip","uninsured_motorist"]},
    "status":          {"type": "string", "enum": ["open","under_review","approved","denied","settled"]},
    "claim_amount":    {"type": "number", "minimum": 0},
    "settlement_amount": {"type": ["number","null"]},
    "adjuster_notes":  {"type": "string"},
    "incident_narrative": {"type": "string"},
    "is_fraud":        {"type": "boolean"},
    "fraud_signals":   {"type": "array", "items": {"type": "string"}},
    "source":          {"type": "string", "const": "synthetic-v1"}
  }
}
```

### `telematics.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["trip_id","policy_number","customer_id","trip_date","distance_miles","duration_minutes","drive_score","source"],
  "properties": {
    "trip_id":           {"type": "string", "pattern": "^TRIP-[0-9]{6,8}$"},
    "policy_number":     {"type": "string"},
    "customer_id":       {"type": "string"},
    "trip_date":         {"type": "string", "format": "date-time"},
    "distance_miles":    {"type": "number", "minimum": 0},
    "duration_minutes":  {"type": "number", "minimum": 0},
    "hard_brakes":       {"type": "integer", "minimum": 0},
    "rapid_accelerations": {"type": "integer", "minimum": 0},
    "speeding_events":   {"type": "integer", "minimum": 0},
    "night_driving_pct": {"type": "number", "minimum": 0, "maximum": 1},
    "drive_score":       {"type": "number", "minimum": 0, "maximum": 100},
    "source":            {"type": "string", "const": "synthetic-v1"}
  }
}
```

### `faq.schema.json`

> The full `faq.schema.json` (with `additionalProperties: false`, all property descriptions, and five worked examples) is the canonical version defined in Phase 4's data-gen section. A summary schema suitable for Phase 1 bootstrap is below — replace it with the full version when implementing Phase 4.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["faq_id","category","subcategory","question","answer","applicable_states","tags","source","version"],
  "properties": {
    "faq_id":             {"type": "string", "pattern": "^faq-[a-z0-9-]+-[0-9]{3}$"},
    "category":           {"type": "string", "enum": ["coverage_concepts","state_rules","claims_process","costs_discounts","policy_management"]},
    "subcategory":        {"type": "string"},
    "question":           {"type": "string"},
    "answer":             {"type": "string"},
    "applicable_states":  {"type": "array", "items": {"type": "string"}},
    "tags":               {"type": "array", "items": {"type": "string"}},
    "source":             {"type": "string", "const": "synthetic-faq-v1"},
    "version":            {"type": "string"}
  }
}
```

---

## 5. Generator Contracts

Each generator follows the same interface so `run_all.py` treats them uniformly:

```python
# Pattern for every generator
def generate(count: int, config: dict, states_data: dict) -> list[dict]:
    """Return a list of validated records."""
    ...

def main(count: int, output_path: Path, config: dict, states_data: dict) -> None:
    records = generate(count, config, states_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"[{output_path.name}] wrote {len(records):,} records → {output_path}")
```

**Key generator notes:**

- `customer_gen.py` — uses `Faker` with locale `en_US`. State assignment uses weighted random choice from `states.json` population weights. IDs: `CUST-{n:05d}`.
- `policy_gen.py` — one policy per customer minimum; 15% of customers get a second policy. Reads state rules to set mandatory coverages (e.g., PIP for MI/PA). Policy number: `{STATE_CODE}-{n:05d}`.
- `claim_gen.py` — generates 1–3 claims per policy at a configurable rate (default: 30% of policies have at least one claim). Injects fraud signals at 3–5% rate: `["claim_delta_high", "frequency_spike", "telematics_anomaly", "rapid_refiling"]`. Adjuster notes and narratives generated via `Faker` sentence templates with claim-type-specific vocabulary.
- `telematics_gen.py` — generates trips per policy proportional to policy age. Drive Score computed from component events: `100 - (hard_brakes * 2) - (rapid_accel * 1.5) - (speeding_events * 3) - (night_driving_pct * 10)`, clamped to `[0, 100]`.
- `faq_gen.py` — generates FAQ records in two passes. First pass: coverage concept FAQs seeded from `coverage_rules.json`, applicable to all states (`applicable_states: ["ALL"]`). Second pass: state-specific FAQs generated programmatically from `states.json` — one no-fault explainer per no-fault state, one total loss threshold entry per state. FAQ IDs: `faq-{subcategory}-{n:03d}`. All records tagged `source: synthetic-faq-v1`. Output: `faqs/faq_corpus.json` (gitignored). The full five-category taxonomy is expanded in Phase 4.
- `document_gen.py` — uses `reportlab` to produce PDFs. Three document types: `decl_{POLICY_NUMBER}.pdf`, `claim_letter_{CLAIM_ID}.pdf`, `renewal_{POLICY_NUMBER}.pdf`. Filename convention is load-bearing — `chunk_router.py` in Phase 4 uses it for document type detection without ML classification.

---

## 6. `run_all.py` — Entry Point

```python
"""
run_all.py — single entry point for all data generation.

Usage:
  uv run python data-gen/generators/run_all.py
  uv run python data-gen/generators/run_all.py --customers 500 --fraud-rate 0.05
"""
import argparse, json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers",    type=int,   default=1000)
    parser.add_argument("--fraud-rate",   type=float, default=0.04)
    parser.add_argument("--trips-per-policy", type=int, default=50)
    parser.add_argument("--pdf-docs",     type=int,   default=1000,
                        help="Total PDFs to generate (declaration + claim + renewal combined)")
    parser.add_argument("--state-focus",  type=str,   default=None,
                        help="Comma-separated state codes to oversample, e.g. TX,PA")
    args = parser.parse_args()

    # Load configs
    config_dir = Path("data-gen/config")
    states_data    = json.loads((config_dir / "states.json").read_text())
    coverage_rules = json.loads((config_dir / "coverage_rules.json").read_text())
    config = {"fraud_rate": args.fraud_rate, "coverage_rules": coverage_rules}

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Generation order matters — policies depend on customers, claims on policies
    from customer_gen   import main as gen_customers
    from policy_gen     import main as gen_policies
    from claim_gen      import main as gen_claims
    from telematics_gen import main as gen_telematics
    from document_gen   import main as gen_documents

    gen_customers(args.customers,                           data_dir / "customers.json",   config, states_data)
    gen_policies(args.customers,                            data_dir / "policies.json",    config, states_data)
    gen_claims(args.customers,                              data_dir / "claims.json",      config, states_data)
    gen_telematics(args.customers * args.trips_per_policy,  data_dir / "telematics.json",  config, states_data)
    gen_documents(args.pdf_docs,                            Path("documents"),             config, states_data)

    print("\n✓ All data generated.")
    print(f"  customers:  {data_dir}/customers.json")
    print(f"  policies:   {data_dir}/policies.json")
    print(f"  claims:     {data_dir}/claims.json")
    print(f"  telematics: {data_dir}/telematics.json")
    print(f"  documents:  documents/ ({args.pdf_docs} PDFs)")

if __name__ == "__main__":
    main()
```

---

## 7. Verification & Git Tag

### Verification

```bash
uv run python data-gen/generators/run_all.py --customers 100 --pdf-docs 50
# Expected: data/*.json + documents/*.pdf created

# Spot-check fraud rate
python -c "import json; d=json.load(open('data/claims.json')); fraud=[c for c in d if c['is_fraud']]; print(f'{len(fraud)}/{len(d)} fraud ({len(fraud)/len(d):.1%})')"
```

### Phase Gate Checklist

- [ ] All generators produce valid JSON
- [ ] Fraud rate is within 3–5%
- [ ] All 50 states + DC present in `customers.json`
- [ ] PDFs generated with correct filename convention (`decl_`, `claim_letter_`, `renewal_`)
- [ ] `uv run ruff check .` passes with no errors

### Git Tag

```bash
git add -A
git commit -m "Phase 1: data generation — all generators + schemas + config"
git tag v0.1.0
```

---

*Next: [PHASE_2_DATABASE.md](./PHASE_2_DATABASE.md)*
