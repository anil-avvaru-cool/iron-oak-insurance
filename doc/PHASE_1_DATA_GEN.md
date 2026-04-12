# Phase 1 — Repository Bootstrap, Toolchain & Data Generation

**Git tag:** `v0.1.0`  
**Deliverable:** `uv run python data-gen/generators/run_all.py` produces a complete local dataset across all five layers.

**Meetup demo:** Run `run_all.py` live, inspect JSON records, open a generated PDF, show how state rules differ between TX and MI, demonstrate `verify_all.py` catching a data quality issue.

---

## Table of Contents

1. [Repository Bootstrap](#1-repository-bootstrap)
2. [Toolchain & Dependency Management](#2-toolchain--dependency-management)
3. [Config Files](#3-config-files)
4. [JSON Schemas](#4-json-schemas)
5. [Shared Validator](#5-shared-validator)
6. [Generator Implementations](#6-generator-implementations)
   - 6.1 [customer_gen.py](#61-customer_genpy)
   - 6.2 [policy_gen.py](#62-policy_genpy)
   - 6.3 [claim_gen.py](#63-claim_genpy)
   - 6.4 [telematics_gen.py](#64-telematics_genpy)
   - 6.5 [document_gen.py](#65-document_genpy)
   - 6.6 [faq_gen.py (Phase 4 stub)](#66-faq_genpy-phase-4-stub)
7. [run_all.py — Entry Point](#7-run_allpy--entry-point)
8. [Data Verification](#8-data-verification)
   - 8.1 [verify_customers.py](#81-verify_customerspy)
   - 8.2 [verify_policies.py](#82-verify_policiespy)
   - 8.3 [verify_claims.py](#83-verify_claimspy)
   - 8.4 [verify_telematics.py](#84-verify_telematicspy)
   - 8.5 [verify_documents.py](#85-verify_documentspy)
   - 8.6 [verify_all.py](#86-verify_allpy)
9. [Verification & Git Tag](#9-verification--git-tag)

---

## Design Decisions This Phase

| Decision | Rationale |
|---|---|
| No default env values | `EnvironmentError` raised immediately on missing config — no silent failures in production |
| `validate_records()` called inside every generator | Drift between generator logic and schema is caught at generation time, not at DB load time |
| `drive_score=null` for non-telematics customers | Not a missing value — a valid business state. Telematics generator explicitly skips null-score policies |
| Verification as separate Python files | Each verifier runs standalone (`python verify_customers.py`) or via `verify_all.py`; no test framework required |
| Filename convention load-bearing | `decl_`, `claim_letter_`, `renewal_` prefixes used by `chunk_router.py` in Phase 4 — do not rename |
| All 7 coverage keys always present | `required` always emitted (never omitted for optional coverages) — required by Phase 3 JSONB queries |
| Coverage elections depend on vehicle age | Older vehicles less likely to carry collision/comprehensive — realistic policyholder behavior |

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

# Environment — no values committed
.env

# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.venv/

# uv lockfile — commit this for reproducible builds
# uv.lock

# AWS CDK
infra/cdk/cdk.out/
infra/cdk/node_modules/

# Cached models
.cache/
*.pt
*.onnx
ai/models/**/*.json
ai/models/**/fairness_reports/
```

**`.env.example`** — committed, no values
```dotenv
# AWS credentials — required for Phase 5 only
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1

# Bedrock model IDs
BEDROCK_MODEL_ID_SONNET=anthropic.claude-sonnet-4-6
BEDROCK_MODEL_ID_HAIKU=anthropic.claude-haiku-4-5-20251001
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

# Database — no defaults; all required
DB_HOST=
DB_PORT=
DB_NAME=
DB_USER=
DB_PASSWORD=

# API
API_HOST=0.0.0.0
API_PORT=8000

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=all-minilm:l6-v2

# Logging
LOG_LEVEL=INFO
```

> **No default values policy:** `.env.example` has intentionally blank values. Any code that reads env vars must call `_require_env("VAR_NAME")` — a helper that raises `EnvironmentError` immediately rather than silently using a wrong value. This prevents the failure mode where a misconfigured deploy connects to the wrong database unnoticed.

---

## 2. Toolchain & Dependency Management

### 2.1 Install uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify: `uv --version`

### 2.2 Initialize the Project

```bash
uv init --no-workspace
uv python pin 3.13
```

**`pyproject.toml`**
```toml
[project]
name = "iron-oak-insurance"
version = "0.1.0"
description = "AIOI AI Strategy & Data Platform"
requires-python = ">=3.11"
dependencies = []

[dependency-groups]
dev = [
  "pytest>=8.0",
  "pytest-asyncio>=0.23",
  "httpx>=0.27",
  "ruff>=0.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I"]
```

### 2.3 Dependencies by Phase

**Phase 1 — Data Generation**
```bash
uv add faker python-dateutil reportlab Pillow tqdm jsonschema
```

**Phase 2 — Database**
```bash
uv add psycopg2-binary sqlalchemy alembic python-dotenv
```

**Phase 3 — ML Models**
```bash
uv add xgboost scikit-learn pandas numpy fastapi uvicorn mangum
```

**Phase 4 — RAG Pipeline**
```bash
uv add sentence-transformers pgvector PyMuPDF langchain-community \
       langchain-postgres openai
```

**Phase 5 — Bedrock Agents**
```bash
uv add boto3 aws-cdk-lib constructs
```

### 2.4 Generate requirements.txt for Docker

```bash
uv export --no-dev --format requirements-txt > requirements.txt
```

> Never edit `requirements.txt` by hand. It is a generated artifact from `pyproject.toml`.

### 2.5 Virtual Environment

```bash
uv sync
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows
```

---

## 3. Config Files

### `data-gen/config/states.json`

Complete configuration for all 50 states + DC. Each entry includes:

| Field | Description |
|---|---|
| `weight` | Population weight 1–10 for customer state assignment |
| `no_fault` | Boolean — drives PIP requirement and claim behavior |
| `min_liability` | State-mandated minimum liability limits |
| `uninsured_motorist_required` | Boolean — drives UM coverage requirement |
| `total_loss_threshold` | Fraction of ACV at which vehicle is declared total loss |
| `pip_required` | Boolean — if true, every policy in this state has pip.required=true |
| `pip_limit` | PIP benefit limit in dollars (null for non-PIP states) |
| `claims_ack_days` | State-mandated claim acknowledgment window |
| `claims_settle_days` | State-mandated settlement window |

No-fault states (pip_required=true): **DE, FL, HI, KS, KY, MA, MI, MN, NJ, ND, NY, PA, UT**

UM-required states: **CT, IL, KS, MA, ME, MD, MN, MO, NC, NE, NJ, NY, OR, RI, SC, SD, VA, VT, WA, WI, WV, DC**

### `data-gen/config/coverage_rules.json`

Key fields:

| Field | Description |
|---|---|
| `coverage_types` | Canonical list of 7 coverage types — every policy has all 7 keys |
| `deductible_options` | [250, 500, 1000, 2500] |
| `vehicle_makes_models` | 20 realistic make/model combinations with population weighting |
| `telematics_enrollment_rate` | Default 0.62 — ~62% of policies are telematics-enrolled |
| `multi_policy_rate` | Default 0.15 — ~15% of customers have a second policy |
| `base_premium_ranges` | Low/medium/high risk premium ranges for realistic premium distribution |

---

## 4. JSON Schemas

All schemas live in `data-gen/schemas/`. Every generator validates output against its schema before writing to disk.

### Schema design rules

- `additionalProperties: false` on all top-level objects and nested sub-schemas — prevents field drift
- `required` field always emitted on coverage objects (never omitted for optional coverages)
- `email` and `drive_score` allow `null` — these are valid business states, not missing data
- VIN length fixed at exactly 17 characters
- All `source` fields use `"const": "synthetic-v1"` for governance tagging

### `customer.schema.json` — key fields

| Field | Type | Notes |
|---|---|---|
| `customer_id` | string | Pattern `^CUST-[0-9]{5,6}$` |
| `email` | string\|null | ~3% of customers have no email |
| `dob` | date | Adults 18–85 |
| `credit_score` | integer\|null | 300–850 |

### `policy.schema.json` — sub-schema design

**`vehicle` sub-schema:** `additionalProperties: false` with `year` range 1990–2026 and VIN exactly 17 characters.

**`coverages` sub-schema:** All 7 coverage keys enumerated in `propertyNames.enum`. Each coverage object has:
- `included` (boolean, required)
- `required` (boolean, required — **never omitted**)
- `deductible` (integer|null)
- `limit` (string|null)
- `pip_limit` (integer|null)

> **Flag for Phase 3:** The JSONB query `coverages->'pip'->'required'` in schema verification depends on `required` being present on every pip object. Generator enforces this: `required` is always emitted as `false` for optional coverages.

### `claim.schema.json` — key constraint

`filed_date >= incident_date` is enforced in the generator and verified by `verify_claims.py`. The schema validates types; the verifier validates the date ordering constraint.

### `telematics.schema.json` — enrollment model

Only policies with `drive_score != null` generate trip records. Non-telematics customers have `drive_score=null` in their policy record and zero rows in telematics.json. This is not a data gap — it is the business model.

---

## 5. Shared Validator

**`data-gen/generators/validate.py`**

```python
"""
validate.py — shared schema validator for all AIOI generators.

Usage:
    from validate import validate_records
    validate_records(records, "customer.schema.json")

Raises ValueError with record index and JSON path on first violation.
"""
import json
from pathlib import Path
import jsonschema

_schema_cache: dict = {}
_SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


def validate_records(records: list[dict], schema_name: str) -> None:
    if schema_name not in _schema_cache:
        schema_path = _SCHEMA_DIR / schema_name
        _schema_cache[schema_name] = json.loads(schema_path.read_text())
    schema = _schema_cache[schema_name]
    for i, record in enumerate(records):
        try:
            jsonschema.validate(instance=record, schema=schema)
        except jsonschema.ValidationError as e:
            raise ValueError(
                f"Record {i} failed {schema_name} validation at "
                f"'{'/'.join(str(p) for p in e.absolute_path)}': {e.message}"
            ) from e


def validate_record(record: dict, schema_name: str) -> None:
    validate_records([record], schema_name)
```

Schema path is resolved relative to the validators file location — works regardless of which directory you run from.

---

## 6. Generator Implementations

All generators follow the same interface contract:

```python
def generate(count: int, config: dict, states_data: dict) -> list[dict]:
    """Generate records, validate against schema, return list."""
    ...

def main(count: int, output_path: Path, config: dict, states_data: dict) -> None:
    """Write records to output_path as JSON."""
    ...
```

Generators call `validate_records()` before returning — schema violations are caught before the file is written.

### 6.1 `customer_gen.py`

**Key design decisions:**

| Decision | Implementation |
|---|---|
| State assignment | `random.choices()` with population weights from `states.json` |
| Credit score distribution | `random.gauss(700, 95)` clipped to [300, 850] → realistic right-skewed distribution |
| DOB distribution | Weighted age buckets: 18–24 (8%), 25–34 (20%), 35–44 (22%), 45–54 (20%), 55–64 (17%), 65–74 (10%), 75–85 (3%) |
| Null email rate | ~3% of customers have `email=null` (prefer phone — realistic) |
| ZIP codes | State-specific prefix ranges (not random 5-digit) |
| Agent territories | 20 agents with weighted assignment (geographic clustering) |

**Enrollment model note:** Customer record does not contain a `drive_score` — that lives on the policy. The enrollment decision is made in `policy_gen.py` per policy, so a customer with two policies could have one enrolled and one not.

### 6.2 `policy_gen.py`

**Coverage object construction rules** (enforced in code, not just schema):

| State condition | Generator behavior |
|---|---|
| `pip_required: true` (13 no-fault states) | `pip.included=true`, `pip.required=true`, `pip.pip_limit` set from `states.json` |
| `uninsured_motorist_required: true` (19 states + DC) | `uninsured_motorist.included=true`, `uninsured_motorist.required=true` |
| Collision/comprehensive | `included` based on vehicle age — older vehicles less likely (realistic) |
| GAP coverage | Only generated for vehicles ≤ 3 years old (makes financial sense) |
| `required` field | **Always emitted** on every coverage object, even as `false` |

**Non-telematics customers:**

`drive_score=null` is set for ~38% of policies (100% minus `telematics_enrollment_rate`). This is a deliberate business state:
- Non-telematics policies do not receive drive score discounts
- `telematics_gen.py` explicitly skips policies with `drive_score=null`
- Phase 3's feature engineering uses `COALESCE(drive_score, 50)` for non-enrolled policies

**Premium calculation factors:**
- Base: $1,200
- Vehicle age factor: newer vehicles cost more to insure (replacement value)
- Credit factor: 680 baseline; 300-score customer pays ~35% more
- Telematics discount: 3–15% based on drive score tier
- Coverage breadth: each optional coverage adds ~8% to base
- State factor: FL (1.35×), MI (1.42×), NY (1.38×), ID (0.85×), VT (0.82×)
- ±10% random jitter for realism

### 6.3 `claim_gen.py`

**Fraud injection design:**

Fraud signals are injected as consistent combinations — not random individual signals. This matters for model quality: real fraud patterns involve co-occurring signals.

| Fraud combo | What it models |
|---|---|
| `claim_delta_high` + `recent_policy_reinstatement` | Policy reinstated then immediately claiming |
| `telematics_anomaly` + `incident_location_mismatch` | GPS data contradicts reported accident location |
| `frequency_spike` + `rapid_refiling` | Same customer filing multiple claims in short period |
| `staged_accident_pattern` + `no_police_report` + `multiple_claimants` | Organized ring pattern |

**Claim type assignment:**

Claims are matched to the policy's active coverages. No PIP claim is generated for a policy without PIP coverage. No-fault state policies have PIP weighted 4× higher in type selection.

**Filing lag:** Non-fraud claims use a decreasing-weight distribution (most claims filed in 0–10 days). Fraud claims often file same-day or next-day (urgency signal).

### 6.4 `telematics_gen.py`

**Non-telematics customer handling (explicit):**

```
enrolled_policies = [p for p in policies if p.get("drive_score") is not None]
non_enrolled_count = len(policies) - len(enrolled_policies)
print(f"{non_enrolled_count} non-enrolled — skipping")
```

This is printed during generation and verified by `verify_telematics.py`. It is expected output, not a warning.

**Drive score formula:**

```
score = 100
       - hard_brakes * 2.0 * norm
       - rapid_accelerations * 1.5 * norm
       - speeding_events * 3.0 * norm
       - night_driving_pct * 10.0
```

Where `norm = min(10.0 / distance_miles, 2.0)` normalizes events to per-10-miles — long highway trips are not unfairly penalized for absolute event counts.

**Trip volume model:** Trip count scales with policy age (days since effective date), capped at 2× baseline. New policies have fewer recorded trips. This makes the telematics dataset temporally realistic.

**Hour distribution:** Peaks at 7–9 AM and 5–7 PM (commute pattern). Night hours (10 PM–5 AM) have low probability — when a night-hour trip does occur, `night_driving_pct` is boosted.

### 6.5 `document_gen.py`

**Document type split:** 45% declarations, 35% claim letters, 20% renewal notices.

**Declaration page structure** (critical for Phase 4 RAG accuracy):
- Named insured block → 1 retrievable unit
- Vehicle details → 1 retrievable unit  
- Coverage table → **one row per coverage type, limit and deductible kept together in the same row**
- State-specific notice

This structure is why `chunk_declaration.py` uses section-aware chunking rather than fixed-size. Splitting a coverage row across chunks loses the limit/deductible association.

**Renewal notice structure:**
- Premium change table (table-zone → table-aware chunking in Phase 4)
- Prose explanation section (prose-zone → paragraph chunking in Phase 4)

**Non-enrolled customer handling:** Renewal notices display "Not enrolled in telematics program" in the Drive Score field rather than leaving it blank.

### 6.6 `faq_gen.py` (Phase 4 stub)

Phase 1 creates a stub so `run_all.py` runs without errors. The stub writes an empty `faqs/faq_corpus.json`. Full implementation is in `PHASE_4_RAG.md`.

---

## 7. `run_all.py` — Entry Point

```bash
# Linux / macOS — run from repo root
uv run python data-gen/generators/run_all.py

# With custom parameters
uv run python data-gen/generators/run_all.py --customers 500 --fraud-rate 0.05 --trips-target 25000 --pdf-docs 200

# Skip PDFs for faster dev iteration
uv run python data-gen/generators/run_all.py --customers 5000 --no-pdfs
```

```powershell
# Windows
uv run python data-gen\generators\run_all.py

uv run python data-gen\generators\run_all.py `
    --customers 500 `
    --fraud-rate 0.05 `
    --trips-target 25000 `
    --pdf-docs 200

# Skip PDFs
uv run python data-gen\generators\run_all.py --customers 100 --no-pdfs
```

**CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--customers` | 1000 | Number of customer records |
| `--fraud-rate` | 0.04 | Claim fraud injection rate (0.0–0.15) |
| `--trips-target` | 50000 | Target total trips across enrolled policies |
| `--pdf-docs` | 500 | Total PDFs to generate |
| `--no-pdfs` | false | Skip PDF generation (faster for dev) |
| `--output-dir` | `.` | Root for data/, documents/, faqs/ |

**Generation order is fixed:**

```
1. customers.json     ← standalone
2. policies.json      ← reads customers.json
3. claims.json        ← reads policies.json
4. telematics.json    ← reads policies.json; skips non-enrolled
5. PDFs               ← reads customers, policies, claims
6. FAQs               ← Phase 4 stub (no-op)
```

---

## 8. Data Verification

Each generator has a dedicated verification script. All scripts:
- Are standalone (`python verify_*.py`) — no test framework required
- Accept `--path` to override the default file location
- Return exit code 0 (pass) or 1 (fail)
- Print ✓/✗/⚠ per check with detail
- Run together via `verify_all.py`

### 8.1 `verify_customers.py`

```bash
uv run python data-gen/generators/verify_customers.py
uv run python data-gen/generators/verify_customers.py --path data/customers.json
```

| Check | Pass condition |
|---|---|
| File exists, valid JSON | Non-empty, parseable |
| Schema validation | All records pass `customer.schema.json` |
| Unique customer_id | No duplicates |
| All 50 states + DC present | Full geographic coverage (warn for small datasets) |
| State distribution | No single state > 25% |
| Credit scores in [300, 850] | No out-of-range values |
| Credit score mean realistic | Mean 600–750 |
| DOBs represent adults 18–85 | No minors, no centenarians |
| Null email rate 0–10% | ~3% expected |
| created_at all in the past | No future timestamps |

### 8.2 `verify_policies.py`

```bash
uv run python data-gen/generators/verify_policies.py
uv run python data-gen/generators/verify_policies.py --path data/policies.json --customers data/customers.json
```

| Check | Pass condition |
|---|---|
| Schema validation | All records pass `policy.schema.json` |
| Unique policy_number | No duplicates |
| All 7 coverage types present | Every policy has all 7 coverage keys |
| `required` always emitted | Never missing from any coverage object |
| No-fault states: pip.required=true | All 13 no-fault state policies |
| VINs exactly 17 characters | No truncated or padded VINs |
| drive_score null or in [0, 100] | Valid for both enrolled and non-enrolled |
| Telematics enrollment rate 45–80% | Expected ~62% |
| effective_date < expiry_date | Date order correct |
| Premium in [$300, $8000] | Realistic range (warn-only) |
| Multi-policy rate 8–25% | Expected ~15% |
| Referential integrity | All customer_ids exist in customers.json |

### 8.3 `verify_claims.py`

```bash
uv run python data-gen/generators/verify_claims.py
uv run python data-gen/generators/verify_claims.py --path data/claims.json --policies data/policies.json
```

| Check | Pass condition |
|---|---|
| Schema validation | All records pass `claim.schema.json` |
| Unique claim_id | No duplicates |
| Fraud rate in [2%, 8%] | Expected ~4% |
| Fraud claims have signals | At least one fraud_signal per is_fraud=true |
| Non-fraud claims: no signals | fraud_signals empty for is_fraud=false |
| filed_date >= incident_date | Date ordering correct |
| No future incident_date | All dates in the past |
| settlement_amount <= claim_amount | No over-settlements |
| Denied claims: no settlement | Denied → null settlement |
| Referential integrity | All policy_numbers exist in policies.json |

### 8.4 `verify_telematics.py`

```bash
uv run python data-gen/generators/verify_telematics.py
uv run python data-gen/generators/verify_telematics.py --path data/telematics.json --policies data/policies.json
```

| Check | Pass condition |
|---|---|
| Schema validation | All records pass `telematics.schema.json` |
| Unique trip_id | No duplicates |
| drive_score in [0, 100] | Per-trip score in range |
| night_driving_pct in [0, 1] | Valid fraction |
| All event counts >= 0 | No negative events |
| distance_miles and duration > 0 | No zero-distance trips |
| No future trip_dates | All in the past |
| Non-enrolled policies have no trips | drive_score=null → 0 trip rows |
| Per-policy avg score ≈ policy drive_score | Within ±25 pts (warn-only) |
| Trips per enrolled policy 1–1000 | Reasonable volume |

### 8.5 `verify_documents.py`

```bash
uv run python data-gen/generators/verify_documents.py
uv run python data-gen/generators/verify_documents.py --dir documents/
```

| Check | Pass condition |
|---|---|
| Directory exists | Non-empty |
| Filename convention | All files start with `decl_`, `claim_letter_`, or `renewal_` |
| No zero-byte files | All PDFs have content |
| Valid PDF headers | `%PDF-` magic bytes (sample) |
| decl_ filenames match policies | Policy number exists in policies.json (sample) |
| claim_letter_ filenames match claims | Claim ID exists in claims.json (sample) |

### 8.6 `verify_all.py`

Runs all verifiers in sequence and prints a summary.

```bash
# Linux / macOS
uv run python data-gen/generators/verify_all.py

# Skip PDF verification (faster)
uv run python data-gen/generators/verify_all.py --skip-pdfs

# Custom paths
uv run python data-gen/generators/verify_all.py \
    --data-dir data \
    --docs-dir documents
```

```powershell
# Windows
uv run python data-gen\generators\verify_all.py
uv run python data-gen\generators\verify_all.py --skip-pdfs
```

Example output:
```
=======================================================
  Phase 1 Verification Summary
=======================================================
  ✓  customers
  ✓  policies
  ✓  claims
  ✓  telematics
  ✓  documents

  Overall: ALL PASS ✓
```

---

## 9. Verification & Git Tag

### 9.1 Full build and verify sequence

```bash
# Linux / macOS — from repo root

# Install Phase 1 dependencies
uv add faker python-dateutil reportlab Pillow tqdm jsonschema
uv sync

# Generate (1000 customers, default settings)
uv run python data-gen/generators/run_all.py

# Verify all
uv run python data-gen/generators/verify_all.py

# Quick spot-checks
python -c "
import json
c = json.load(open('data/customers.json'))
p = json.load(open('data/policies.json'))
claims = json.load(open('data/claims.json'))
t = json.load(open('data/telematics.json'))
fraud = [x for x in claims if x['is_fraud']]
enrolled = [x for x in p if x['drive_score'] is not None]
non_enrolled = [x for x in p if x['drive_score'] is None]
print(f'Customers: {len(c):,}')
print(f'Policies: {len(p):,} ({len(enrolled):,} enrolled, {len(non_enrolled):,} non-enrolled)')
print(f'Claims: {len(claims):,} ({len(fraud):,} fraud = {len(fraud)/len(claims):.1%})')
print(f'Telematics: {len(t):,} trips')
"
```

```powershell
# Windows
uv run python data-gen\generators\run_all.py
uv run python data-gen\generators\verify_all.py
```

### 9.2 Phase Gate Checklist

- [ ] All generators produce valid JSON without errors
- [ ] `verify_all.py` reports ALL PASS
- [ ] Fraud rate within 3–5% (verified by `verify_claims.py`)
- [ ] All 50 states + DC present in `customers.json`
- [ ] PDFs generated with correct filename convention (`decl_`, `claim_letter_`, `renewal_`)
- [ ] All policy records pass `policy.schema.json` (vehicle + coverages sub-schemas)
- [ ] All 7 coverage keys present on every policy record
- [ ] `required` field always emitted on coverage objects (never omitted)
- [ ] PIP coverage marked `required: true` for all 13 no-fault state policies
- [ ] All VINs exactly 17 characters
- [ ] Non-enrolled policies (drive_score=null) have zero rows in telematics.json
- [ ] `uv run ruff check .` passes with no errors

### 9.3 Troubleshooting Common Failures

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: faker` | Phase 1 deps not installed | `uv add faker python-dateutil reportlab Pillow tqdm jsonschema` |
| `FileNotFoundError: data/customers.json` | policy_gen/claim_gen run before customer_gen | Always use `run_all.py` — it enforces order |
| `jsonschema.ValidationError: 'required' is a required property` | Coverage object missing `required` field | Check `policy_gen._build_coverages()` — all coverages must emit `required` |
| `verify_policies: PIP violations` | No-fault state policies missing pip.required=true | Check `_NO_FAULT_STATES` set and `_build_coverages()` logic |
| `verify_telematics: trips for non-enrolled policies` | drive_score check incorrect | Confirm `p.get("drive_score") is not None` (not just truthiness) |
| `verify_telematics: 0 enrolled policies` | All policies have drive_score=null | Check `telematics_enrollment_rate` in coverage_rules.json |
| PDF generation fails with `ImportError` | ReportLab not installed | `uv add reportlab Pillow` |

### 9.4 Git Tag

```bash
git add -A
git commit -m "Phase 1: data generation — all generators, schemas, verification scripts"
git tag v0.1.0
```

---

*Next: [PHASE_2_DATABASE.md](./PHASE_2_DATABASE.md)*
