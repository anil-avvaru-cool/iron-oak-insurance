# Avvaru Iron Oak Insurance — AI Strategy & Platform Guide

**Version 2.0 · Q2 2026 · Harrisburg, PA**
**Vehicle Insurance · Personal & Commercial · All 50 States + DC**
*Confidential*

---

| $500M | 420K | $55.5M | 14 mo. |
|---|---|---|---|
| Target Carrier Book Size | Policy Base at Scale | 3-Year Net Benefit | Payback Period |

> **Bottom Line:** A phased AI deployment beginning with fraud and risk models — which run at zero infrastructure cost — establishes the data foundation for higher-value LLM capabilities. Total platform cost at full deployment is ~$755/month against a conservative $4.2M+ annual benefit. Every month of delay compounds loss exposure.

---

## Table of Contents

1. [Company & Project Overview](#1-company--project-overview)
2. [Five-Capability AI Platform](#2-five-capability-ai-platform)
3. [ROI Analysis](#3-roi-analysis)
4. [Deployment Roadmap](#4-deployment-roadmap)
5. [Meetup Demo Guide](#5-meetup-demo-guide)
6. [Repository Structure & Phase Guide](#6-repository-structure--phase-guide)
7. [Data Platform & Synthetic Data](#7-data-platform--synthetic-data)
8. [AI Models — Phase 3](#8-ai-models--phase-3)
9. [RAG Pipeline — Phase 4](#9-rag-pipeline--phase-4)
10. [AWS Bedrock & Phase 5](#10-aws-bedrock--phase-5)
11. [Governance & Operational Readiness](#11-governance--operational-readiness)
12. [Phase Gate Checklists](#12-phase-gate-checklists)

---

## 1. Company & Project Overview

### 1.1 About AIOI

Avvaru Iron Oak Insurance (AIOI) is a fictitious mid-Atlantic regional vehicle insurance carrier used as the foundation for AI development, meetups, and production-grade AWS deployments. The name reflects core values: durability (*iron*), longevity (*oak*), and adaptability.

AIOI covers personal and commercial vehicle insurance across all 50 US states + DC, with full modeling of state-specific regulatory requirements including no-fault rules, PIP mandates, uninsured motorist requirements, and total loss thresholds.

| Attribute | Detail |
|---|---|
| Founded | 2000 · Harrisburg, PA |
| Focus | Vehicle Insurance — Personal & Commercial |
| Coverage | All 50 US States + DC |
| Repository | iron-oak-insurance (single public repo) |
| Purpose | AI/ML showcase, meetups, production-grade AWS deployments |

### 1.2 What "Book Size" Means

In insurance, a carrier's **book** refers to their total portfolio of active policies. Book size is measured in **annual written premium** — the total amount all policyholders pay per year.

> **"$500M book"** = $500 million in annual written premium. At an average personal auto premium of ~$1,190/year, this represents approximately 420,000 policies — a mid-sized regional carrier. Not Progressive, not a local shop. Right in the middle.

All ROI projections in this document are modeled against a $500M carrier book. The demo dataset (5,000 customers, ~$6M premium) demonstrates the mechanics; the projections show the business case at real carrier scale.

| Scale | Customers | Annual Premium | Claims/Year | Fraud Exposure | Purpose |
|---|---|---|---|---|---|
| Demo dataset | 5,000 | ~$6M | ~400 | ~$210K | Live demo — shows mechanics |
| Target carrier | 420,000 | $500M | 85,000 | $24.4M | ROI projections |

### 1.3 Strategic Imperative

The insurance industry is at an inflection point. Carriers that deploy AI across fraud detection, claims processing, risk modeling, and customer service today will operate at a structurally lower combined ratio within three years.

- Fraud losses account for 10–15% of all claim payouts industry-wide. Without AI-based detection, carriers rely on reactive SIU referrals catching only a fraction of coordinated fraud rings.
- Telematics data is rapidly becoming a competitive moat. UBI pricing models consistently outperform on loss ratios by 6–8 percentage points.
- Customer acquisition costs have risen 35% over five years. Reducing churn by 1 point on a $500M book recovers $5M without a single new policy sold.
- Regulators in 22 states now require documented AI governance for pricing models. Early movers build compliance infrastructure once; late movers retrofit it under scrutiny.

---

## 2. Five-Capability AI Platform

### 2.1 Capability Architecture

The platform is structured as five additive capability layers. Each layer is independently deployable and delivers measurable value on its own, while providing the data infrastructure that makes subsequent layers more accurate.

| Capability | Technique | Primary Metric | Infra Cost | Deploy Priority |
|---|---|---|---|---|
| Fraud Detection | XGBoost Classifier | Fraud loss recovery rate | $0/mo | 1 — Deploy immediately |
| Risk Scoring & Churn | XGBoost + Telematics | Loss ratio improvement | $0/mo | 1 — Deploy with fraud |
| RAG Policy & FAQ Q&A | pgvector + Claude Haiku | FNOL handle time | ~$59/mo | 2 — Low cost, high impact |
| Fraud Agent | Claude Sonnet + Bedrock | SIU referral quality | ~$162/mo | 3 — Deploy if budget allows |
| Oak Assist (FNOL) | Claude Sonnet + Bedrock | Cost per claim intake | ~$470/mo | 4 — Optimize with routing |

### 2.2 Why the Platform Approach Matters

Each capability layer consumes and enriches a shared data platform. This is not five isolated tools — it is a compounding platform.

- Telematics data fed into the risk model improves fraud detection accuracy
- Fraud signals inform underwriting risk scores
- Customer churn signals trigger proactive retention through Oak Assist
- All five capabilities share the same Postgres + pgvector instance

| Data Layer | Source | Feeds Into | Est. Size |
|---|---|---|---|
| Customer demographics & behavior | Policy systems, CRM | Churn model, CLV scoring | ~50 MB |
| Policy coverages & premiums | Core policy admin | Risk scoring, underwriting | ~80 MB |
| Claims history & adjuster notes | Claims management | Fraud detection, NLP triage | ~40 MB |
| Telematics (drive score, events) | Telematics / OBD | UBI pricing, fraud signals | ~500 MB |
| Policy documents (PDFs) | Document management | RAG Q&A, Oak Assist | ~1.5 GB |

---

## 3. ROI Analysis

### 3.1 Modeling Assumptions

> All projections modeled against a $500M annual written premium book: 420,000 policies, 85,000 claims/year, $8,200 average claim cost, 3.5% fraud rate, 500 FNOL intakes/day. Benefits are conservatively estimated at 50–70% of industry benchmarks.

### 3.2 Fraud Detection ROI — The Foundation Case

This single capability justifies the entire platform investment. It runs at zero marginal infrastructure cost on AWS Lambda.

| Metric | Without AI | With AI | Annual Impact |
|---|---|---|---|
| Fraud detection rate | 12% | 38% | +26 percentage points |
| Fraud loss recovered | $2.9M | $9.3M | +$6.4M |
| SIU referral accuracy | ~30% | ~72% | Fewer false positives |
| Infrastructure cost | — | $0 (Lambda) | No add-on cost |

> **Net Year 1 benefit: +$6.4M fraud recovery at zero marginal infrastructure cost. This single capability pays back the entire platform engineering investment (~$80K) in under 60 days.**

### 3.3 Three-Year Financial Model

| Capability | Yr 1 | Yr 2 | Yr 3 | 3-Yr Total |
|---|---|---|---|---|
| Fraud Detection | $3.2M | $5.8M | $6.4M | $15.4M |
| Risk Scoring & Churn | $6.2M | $11.9M | $14.3M | $32.4M |
| RAG Policy Q&A | $650K | $1.1M | $1.3M | $3.1M |
| Fraud Agent | $1.0M | $1.8M | $2.1M | $4.9M |
| Oak Assist / FNOL | $480K | $820K | $960K | $2.3M |
| **TOTAL BENEFITS** | **$11.5M** | **$21.4M** | **$25.1M** | **$58.1M** |
| TOTAL INFRA COST | ($0.5M) | ($0.9M) | ($1.2M) | ($2.6M) |
| **NET BENEFIT** | **$11.0M** | **$20.5M** | **$23.9M** | **$55.5M** |

| $55.5M | $2.6M | 2,035% | ~14 mo. |
|---|---|---|---|
| 3-Year Net Benefit | 3-Year Total Infra Cost | 3-Year Gross ROI | Payback Period |

### 3.4 Infrastructure Cost Detail

| Component | Monthly Cost | Annual Cost | Notes |
|---|---|---|---|
| Base infra (RDS + Lambda + API GW) | $62 | $744 | Always-on; shared across all use cases |
| RAG Q&A — Claude Haiku | $59 | $708 | 500 queries/day; prompt caching reduces cost |
| Fraud Agent — Claude Sonnet | $162 | $1,944 | Batch non-urgent reviews for 50% saving |
| Oak Assist — Sonnet + Haiku routing | $470–660 | $5,640–7,920 | Haiku routing cuts 30–50% vs Sonnet-only |
| **TOTAL (optimized)** | **~$753–943** | **~$9,036–11,316** | Fully loaded, all 5 capabilities |

---

## 4. Deployment Roadmap

### 4.1 Phased Rollout

Each phase delivers independent business value while laying the technical foundation for the next. Phases 1 and 2 can run in parallel by separate teams.

| Phase | Capability | Timeline | Investment | Value Gate |
|---|---|---|---|---|
| Phase 1–2 | Data platform & DB layer | Months 1–2 | ~$50K eng | Foundation for all AI |
| Phase 3 | Fraud, Risk & Churn ML | Months 2–3 | ~$30K eng + $0 infra | Fraud recovery >$2M Y1 |
| Phase 4 | RAG Policy Q&A + FAQ | Months 3–5 | ~$40K eng + $59/mo | Call deflection >25% |
| Phase 5a | Fraud Agent (Sonnet) | Months 5–7 | ~$30K eng + $162/mo | SIU hours −50% |
| Phase 5b | Oak Assist FNOL | Months 7–10 | ~$50K eng + $470/mo | FNOL cost −40% |

### 4.2 Minimum Viable Deployment

Phases 1–3 plus RAG Q&A delivers the majority of the 3-year value at a fraction of the full cost. Recommended as the initial authorization.

| MVD Metric | Value |
|---|---|
| Capabilities deployed | Fraud detection, risk scoring, churn prediction, policy Q&A |
| Monthly infrastructure cost | ~$121/month |
| Year 1 projected benefit | $9.85M (fraud + churn + Q&A) |
| Payback period | < 90 days |

---

## 5. Meetup Demo Guide

> **Key principle — Two-track presentation.** The demo dataset (5K customers) demonstrates mechanics. The ROI numbers are industry benchmarks for a $500M carrier book. These are separate tracks — never try to derive carrier-scale ROI from the demo dataset.

### 5.1 Pre-Meetup Setup (One-Time, Done Before the Room Fills)

Run all of this before the meetup. Never run data generation or model training live — it adds wait time with zero demo value.

**Generate the demo dataset**
```bash
uv run python data_gen/generators/run_all.py --customers 5000 --fraud-rate 0.04
```
This produces ~1,400 policies, ~400 claims, ~31K telematics trips. Realistic ratios, fast generation.

**Load to Postgres**
```bash
docker compose up -d
uv run python db/load_json.py
```

**Train all three models**
```bash
uv run python -m ai.models.fraud_detection.model
uv run python -m ai.models.risk_scoring.model
uv run python -m ai.models.churn_prediction.model
```
Takes ~30 seconds on 5K customers. Models are saved to disk — stay loaded for the demo.

**Embed FAQs and start the API**
```bash
uv run python data_gen/generators/faq_gen.py
uv run python -m ai.pipelines.embedding.embed_and_load local
uv run python -m ai.api.handlers.main
```

### 5.2 20-Minute Talk Structure

| Segment | Time | What To Do | What To Say |
|---|---|---|---|
| Opening hook | 2 min | Run the fraud distribution SQL query live | "What does $24M in fraud exposure look like in a database?" |
| Platform overview | 3 min | Show the 5-layer capability table | "Five layers. The first two run for free. The whole thing costs under $800/month." |
| Live demo — fraud scoring | 4 min | Hit the fraud score API with a high-risk claim | "fraud_score: 0.91 — that's XGBoost. Now watch what Claude does with it." |
| Live demo — RAG routing | 4 min | Run FAQ query then policy-specific query | "Same question surface. Completely different retrieval path. This is what prevents hallucination." |
| ROI story | 4 min | Show the $500M projections slide | "At carrier scale, this single model pays back the platform in under 60 days." |
| What's next | 2 min | Show the deployment roadmap | "$121/month gets you 4 capabilities and $9.85M in Year 1 value." |
| Buffer / Q&A | 1 min | — | — |

### 5.3 Live Demo Commands

**Step 1 — Opening hook (data already loaded)**
```sql
SELECT is_fraud, COUNT(*), ROUND(AVG(claim_amount),0) AS avg_amount
FROM claims
GROUP BY is_fraud;
```
Show the numbers, then bridge: *"Our demo has ~400 claims. Let me show you what this looks like at carrier scale in a moment."*

**Step 2 — Fraud model scoring**
```bash
curl -s -X POST http://localhost:8000/models/fraud/score \
  -H "Content-Type: application/json" \
  -d '{"claims": [{"claim_id":"CLM-00001","claim_amount":12000,
       "days_to_file":1,"fraud_signal_count":3,"avg_drive_score":28,
       "state":"TX","claim_type":"collision","customer_claim_count":4,
       "claim_to_premium_ratio":3.2,"hard_brakes_90d":45,
       "vehicle_make":"Toyota","zip_prefix":"750"}]}' | python -m json.tool
```
Point at `fraud_score: 0.91` and `is_fraud_predicted: true`. *"Filed same day, 3 fraud signals, drive score 28, claim-to-premium ratio 3.2x — the model sees all of it at once."*

**Step 3 — RAG routing (two queries back to back)**
```bash
# Routes to FAQ corpus
curl -X POST http://localhost:8000/rag/query \
  -d '{"query":"What does PIP mean?"}'

# Routes to customer policy document
curl -X POST http://localhost:8000/rag/query \
  -d '{"query":"What is the deductible on policy TX-00142?",
       "customer_id":"CUST-08821"}'
```
Show `strategy: "faq"` vs `strategy: "policy_document"` in the response. *"Same endpoint — completely different retrieval path based on query signals."*

**Step 4 — Bridge to carrier-scale ROI**
```sql
-- Visual bridge: demo scale vs carrier scale
SELECT 'demo dataset' AS scale, COUNT(*) AS customers FROM customers
UNION ALL
SELECT 'carrier scale (projected)', 420000;
```
*"Same system, same code, same architecture — just 84x more customers. At that scale, the fraud model alone recovers $6.4M in Year 1."*

### 5.4 The ROI Story (4 minutes, no code)

Deliver these numbers from a slide — do not derive them from the demo dataset.

| The Math | Numbers |
|---|---|
| Annual fraud exposure (carrier scale) | 85K claims × $8,200 avg × 3.5% = $24.4M |
| Detection without AI | 12% catch rate → $2.9M recovered |
| Detection with XGBoost | 38% catch rate → $9.3M recovered |
| Net new recovery, Year 1 | +$6.4M |
| Infrastructure cost for this | $0/month (AWS Lambda) |
| Platform engineering investment | ~$80K (Phases 1–3) |
| Payback period | < 60 days |

> **The punchline:** *"This single model — which runs for free on Lambda — pays back the entire platform investment in under 60 days. Everything else is upside on top of a cost already covered."*

---

## 6. Repository Structure & Phase Guide

### 6.1 Repository Layout

```
iron-oak-insurance/
├── data_gen/
│   ├── generators/          # customer, policy, claim, telematics, doc, faq
│   ├── schemas/             # JSON schemas for all data layers
│   └── config/              # states.json, coverage_rules.json
├── ai/
│   ├── agents/              # Bedrock agents: claims, fraud, policy_advisor
│   ├── pipelines/           # ingestion, embedding, rag
│   ├── models/              # fraud_detection, risk_scoring, churn_prediction
│   └── api/                 # Lambda handlers, FastAPI routers
├── db/                      # schema.sql, load_json.py
├── infra/cdk/               # AWS CDK stacks
├── tests/                   # unit, integration
├── data/                    # gitignored — generated locally
├── documents/               # gitignored — generated locally
├── faqs/                    # gitignored — generated locally
└── docker-compose.yml       # Postgres + pgvector + Ollama
```

### 6.2 Phase Summary

| Phase | Capability | Key Deliverable | Git Tag |
|---|---|---|---|
| 1 | Data generation | All generators + PDFs via `run_all.py` | `v0.1.0` |
| 2 | Database | Postgres + pgvector loaded, queryable | `v0.2.0` |
| 3 | AI & ML models | Fraud, risk, churn via FastAPI | `v0.3.0` |
| 4 | RAG pipeline | Policy doc + FAQ Q&A with retrieval routing | `v0.4.0` |
| 5 | Agents + deploy | Bedrock agents + AWS CDK | `v1.0.0` |

### 6.3 Hardware Requirements by Tier

| Tier | Hardware | Phases Supported |
|---|---|---|
| 1 — Full local | Apple Silicon 16 GB+ or NVIDIA GPU 8 GB+ | All phases locally |
| 2 — Partial local | 8 GB RAM, no GPU | Phases 1–3 fully · Phase 4 with Llama 3.2 3B |
| 3 — Follow along | < 8 GB RAM | Phases 1–3 · Phase 4 via AWS Bedrock API |

---

## 7. Data Platform & Synthetic Data

### 7.1 Data Strategy

All records are generated as JSON locally on demand. The generators are the asset — not the generated data. `data/`, `documents/`, and `faqs/` are gitignored. Anyone cloning the repo runs `run_all.py` to produce their own copy tuned to their parameters.

| Layer | Content | AI Use Cases | Est. Size |
|---|---|---|---|
| Customers | Demographics, renewal/lapse patterns | Churn prediction, CLV | ~50 MB |
| Policies | State-specific coverages, linked to customers | Pricing models, underwriting | ~80 MB |
| Claims | Adjuster notes, narratives, 3–5% fraud injected | Fraud detection, NLP triage | ~40 MB |
| Telematics | 50K trips, Iron Oak Drive Score (0–100) | UBI pricing, risk segmentation | ~500 MB |
| Documents | Declaration pages, claim letters, renewal notices (PDFs) | RAG pipelines, document Q&A | ~800 MB–2 GB |
| FAQs | Coverage concepts, state rules, claims process (JSON + MD) | RAG general Q&A, Oak Assist fallback | ~5–10 MB |

### 7.2 Demo Scale vs. Carrier Scale

> **Never generate 400K customers to match the ROI numbers.** The ROI projections are industry benchmarks — not derived from the demo dataset. Run 5K customers for demos. Present carrier-scale numbers from the ROI slide.

| `run_all.py` Flag | Value | Result |
|---|---|---|
| `--customers` | 5000 | ~1,400 policies, ~400 claims, ~31K trips — demo scale |
| `--fraud-rate` | 0.04 | 4% fraud injection — matches ROI model assumptions |
| `--no-pdfs` | (flag) | Skip PDF generation for faster dev iteration |
| `--pdf-docs` | 200 | Small PDF run for Phase 4 RAG demo |

---

## 8. AI Models — Phase 3

### 8.1 Model Summary

| Model | Type | Target Variable | Key Features |
|---|---|---|---|
| Fraud Detection | XGBoost Classifier | `is_fraud` (binary) | `days_to_file`, `fraud_signal_count`, `avg_drive_score`, `claim_to_premium_ratio` |
| Risk Scoring | XGBoost Regressor | `premium_annual` (proxy) | `drive_score`, `credit_score`, `vehicle_year`, `avg_drive_score_12m` |
| Churn Prediction | XGBoost Classifier | lapsed/cancelled (binary) | `drive_score_delta` (3m − 12m), `avg_premium`, `total_claims` |

### 8.2 The `drive_score_delta` Feature

The churn model's most distinctive feature is `drive_score_delta` — the 3-month average drive score minus the 12-month average. A negative delta means driving behavior is deteriorating. Combined with premium level and claims history, this is the strongest predictor of non-renewal.

### 8.3 Fairness Audit

Every model training run automatically calls `fairness_audit.py`. It performs disparate impact analysis sliced by state, ZIP prefix, and vehicle make. Any slice whose model-predicted positive rate deviates more than 2x from the overall rate — without a matching deviation in the labeled rate — is flagged before production deployment.

---

## 9. RAG Pipeline — Phase 4

### 9.1 Two-Source Retrieval

The RAG pipeline stores two types of content in pgvector, distinguished by `source_type` metadata. Retrieval routing is determined by query signals — not by the user selecting a source.

| `source_type` | Contains | When Retrieved |
|---|---|---|
| `policy_document` | Declaration pages, claim letters, renewal notices | Query contains policy number, customer ID, or "my policy" |
| `faq` | Coverage concepts, state rules, claims process, costs, policy management | Query contains concept signals: "what is", "how does", "what happens" |

### 9.2 Critical Retrieval Design

> The routing logic prevents the most dangerous failure mode: a customer asking *"what is my deductible?"* receiving a generic FAQ about how deductibles work instead of their actual policy figure. Policy number and customer ID are used as **pre-filters before similarity ranking** — not as tie-breakers.

### 9.3 PDF Chunking Strategy

Insurance PDFs are not uniform. A naive fixed-size chunker splits coverage rows mid-record and produces confident wrong answers. Chunking is per document type.

| Document Type | Strategy | Chunk Size | Overlap |
|---|---|---|---|
| Declaration page — coverage table | Section/row aware — limit+deductible kept together | 200–400 tokens | None |
| Declaration page — other sections | Section boundary | 200–350 tokens | None |
| Claim letter | Paragraph | 250–350 tokens | 50 tokens |
| Renewal notice — table zone | Row aware | 200–400 tokens | None |
| Renewal notice — prose zone | Paragraph | 250–350 tokens | 50 tokens |
| FAQ | One per Q&A pair | 100–200 tokens | None |

---

## 10. AWS Bedrock & Phase 5

### 10.1 Local to AWS Component Swap

Same code, minimal changes. Environment variables and mode flags control which backend is used. No rewrites required.

| Component | Local (Phases 1–4) | AWS (Phase 5) | Change Required |
|---|---|---|---|
| PostgreSQL | Docker (`pgvector/pgvector:pg16`) | RDS PostgreSQL 16 + pgvector | Update `DB_HOST` in `.env` |
| Vector store | pgvector on Docker | pgvector on RDS (same extension) | Same queries, new host |
| Embedding | `sentence-transformers` local | Titan Embeddings V2 via Bedrock | Set `mode=bedrock` in `embed_and_load.py` |
| Generation | Ollama (Llama 3.1 8B) | Claude Haiku / Sonnet via Bedrock | Set `mode=bedrock` in `rag_pipeline.py` |
| FastAPI | `uvicorn` local | Lambda + API Gateway via `mangum` | `handler = Mangum(app)` already in place |

### 10.2 Model ID Reference

| Use Case | Local Model | AWS Model | Model ID |
|---|---|---|---|
| Fraud / Risk / Churn | XGBoost (local) | XGBoost on Lambda | N/A (file-based) |
| Embeddings | `all-MiniLM-L6-v2` | Titan Embeddings V2 | `amazon.titan-embed-text-v2:0` |
| RAG Q&A (high volume) | Llama 3.1 8B (Ollama) | Claude Haiku 4.5 | `anthropic.claude-haiku-4-5-20251001` |
| Agents (complex reasoning) | Llama 3.1 8B (Ollama) | Claude Sonnet 4.6 | `anthropic.claude-sonnet-4-6` |

> **Critical:** Verify all model ID strings in `.env` match current Bedrock IDs before deploying Phase 5. Using deprecated model strings instead of `claude-sonnet-4-6` can double inference costs. Audit every reference — `.env`, CDK stacks, and hardcoded strings in agent files.

### 10.3 Cost Optimization Levers

| Lever | Applies To | Mechanism | Est. Annual Saving |
|---|---|---|---|
| Intelligent Prompt Routing | Oak Assist | Auto-route simple queries to Haiku; escalate complex to Sonnet | $2,800–5,400 |
| Bedrock Batch API (50% off) | Fraud Agent | Batch non-urgent overnight fraud reviews | ~$970 |
| Prompt Caching on RAG Context | All RAG use cases | Repeated policy document context tokens cost 10% on cache hit | ~$420–840 |
| pgvector vs OpenSearch Serverless | Vector store | pgvector on existing RDS vs $350/mo minimum floor | $4,200/year |
| ARM/Graviton Lambda | All Lambda functions | ~20% compute cost reduction vs x86 | ~$180 |
| Stop RDS in dev/test | Base infra | Compute pauses; storage charges continue | ~$320 (dev env) |

---

## 11. Governance & Operational Readiness

### 11.1 Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Model drift (fraud/risk) | Medium | High | Quarterly retraining; CloudWatch alarm on p95 `fraud_score` drift > 0.15 |
| Embedding dimension migration (local → AWS) | Low | Medium | Re-ingest with Titan at Phase 5; keep local/cloud tables separate |
| Bedrock model ID deprecation | Low | Medium | Pin model IDs in CDK; audit before each Phase 5 deploy |
| LLM hallucination on policy Q&A | Medium | High | RAG — answers grounded in source docs only; grounding checks via Guardrails |
| OpenSearch cost overrun | Low | Medium | Use pgvector on RDS until sustained traffic justifies switch |
| Prompt injection | Medium | High | Sanitize free-text inputs; post-retrieval filter; Bedrock Guardrails topic denial |

### 11.2 Regulatory & Compliance

- **Adverse action notices:** XGBoost feature importance satisfies explainability for pricing models; Claude Sonnet Fraud Agent narrative satisfies it for SIU referrals.
- **State-specific rule compliance:** The data platform models all 50 state regulatory variations. FAQ corpus is seeded directly from `states.json` — no manual state-by-state content maintenance.
- **Data governance:** All synthetic records tagged `source: synthetic-v1`. Production data requires a PII tagging and masking layer — 2-week engineering addition to the ingestion pipeline.
- **AI bias audits:** Demographic distribution checks built into the data platform. Fairness audit runs automatically after every model retrain.

### 11.3 Logging Contract

All log entries emitted as structured JSON to stdout (CloudWatch Logs compatible). Minimum fields per request:

```json
{
  "timestamp":        "ISO-8601",
  "request_id":       "UUID",
  "endpoint":         "/rag/query | /models/fraud/score | agent_invocation",
  "strategy":         "policy_document | faq | both",
  "policy_number":    "TX-00142 | null",
  "customer_id":      "CUST-8821 | null",
  "model_id":         "claude-haiku-4-5-20251001 | xgboost-fraud-v1",
  "latency_ms":       1240,
  "input_tokens":     1500,
  "output_tokens":    320,
  "chunks_retrieved": 5,
  "fraud_score":      0.82,
  "error":            null
}
```

> **PII warning:** Never log `chunk_text`, `adjuster_notes`, `incident_narrative`, or any free-text field. Log identifiers only. Enable Bedrock Guardrails PII redaction on agent outputs before writing to any log sink.

---

## 12. Phase Gate Checklists

### Phase 1 — Data Generation

- [ ] All generators produce valid JSON without errors
- [ ] `verify_all.py` reports ALL PASS
- [ ] Fraud rate within 3–5%
- [ ] All 50 states + DC present in `customers.json`
- [ ] PDFs generated with correct filename convention (`decl_`, `claim_letter_`, `renewal_`)
- [ ] All 7 coverage keys present on every policy record
- [ ] `required` field always emitted on coverage objects
- [ ] Non-enrolled policies (`drive_score=null`) have zero rows in `telematics.json`

### Phase 2 — Database

- [ ] `docker compose up -d` succeeds
- [ ] All 4 tables populated, FK integrity holds
- [ ] JSONB query for PIP fields returns results
- [ ] `document_chunks` table exists and is empty

### Phase 3 — ML Models

- [ ] All 3 models train without errors; ROC-AUC > 0.85 for fraud and churn
- [ ] Risk model writes `risk_model.json` and `risk_model.bounds.json`
- [ ] Fairness audit runs for all 3 models; JSON reports written
- [ ] All three API endpoints return scored results with `request_id` and `latency_ms`
- [ ] Missing env var causes `EnvironmentError` — no silent defaults

### Phase 4 — RAG Pipeline

- [ ] FAQ chunks loaded into pgvector (`SELECT COUNT(*) WHERE source_type='faq'` > 0)
- [ ] Policy query routes to `policy_document`
- [ ] FAQ query routes to `faq`
- [ ] HNSW index present on `document_chunks.embedding`
- [ ] Prompt injection test strings rejected
- [ ] Structured JSON logging active on all endpoints

### Phase 5 — Bedrock & AWS

- [ ] `cdk synth` produces no errors
- [ ] Lambda cold start < 10s
- [ ] Bedrock model IDs validated (no legacy strings)
- [ ] Bedrock Guardrails ARN attached in CDK stack
- [ ] Guardrails topic denial tested with off-topic queries
- [ ] PII not present in CloudWatch Logs
- [ ] Intelligent prompt routing tested: simple query → Haiku, FNOL → Sonnet

---

*Avvaru Iron Oak Insurance is a fictitious company created for AI development, meetups, and production-grade AWS showcase projects.*

*All benefit estimates are based on industry benchmarks and internal modeling assumptions. Actual results will vary.*