# Avvaru Iron Oak Insurance (AIOI) — Implementation Plan

**Strategy Version:** 1.4  
**Plan Version:** 1.1  
**Date:** March 2026  
**Scope:** All phases — Data Generation through Bedrock Agents & AWS Deployment

---

## How to Use This Plan

Each phase is a self-contained file. Work through them in order — every phase delivers independent business value and lays the foundation for the next. Phases 1 and 2 can be started in parallel by separate teams.

Check out any phase via its Git tag: `git checkout v0.1.0`, `v0.2.0`, etc.

---

## Phase Files

| File | Contents | Git Tag |
|------|----------|---------|
| [PHASE_1_DATA_GEN.md](./PHASE_1_DATA_GEN.md) | Repo bootstrap, toolchain, config files, JSON schemas, all generators, `run_all.py` | `v0.1.0` |
| [PHASE_2_DATABASE.md](./PHASE_2_DATABASE.md) | Docker Compose, `db/schema.sql`, bulk loader, verification queries | `v0.2.0` |
| [PHASE_3_ML.md](./PHASE_3_ML.md) | Feature engineering, fraud/risk/churn XGBoost models, FastAPI endpoints | `v0.3.0` |
| [PHASE_4_RAG.md](./PHASE_4_RAG.md) | FAQ generator, PDF chunkers, embedding pipeline, retrieval router, RAG Q&A endpoint | `v0.4.0` |
| [PHASE_5_BEDROCK.md](./PHASE_5_BEDROCK.md) | AWS prerequisites, Bedrock agents, CDK stacks, deploy commands, cost optimization | `v1.0.0` |
| [CROSS_PHASE.md](./CROSS_PHASE.md) | Toolchain decisions, embedding dimension flags, phase gate checklist, operational readiness TODOs | — |

---

## Phase Summary

| Phase | Capability | Key Deliverable | Tag |
|-------|------------|-----------------|-----|
| 1 | Data generation | All generators + PDFs via `run_all.py` | `v0.1.0` |
| 2 | Database | Postgres + pgvector loaded, queryable | `v0.2.0` |
| 3 | AI & ML models | Fraud, risk, churn via FastAPI | `v0.3.0` |
| 4 | RAG pipeline | Policy doc + FAQ Q&A with retrieval routing | `v0.4.0` |
| 5 | Agents + deploy | Bedrock agents + AWS CDK | `v1.0.0` |

---

## Hardware Prerequisites

| Tier | Hardware | Phases Supported |
|------|----------|-----------------|
| 1 — Full local | Apple Silicon 16 GB+ or NVIDIA GPU 8 GB+ | All phases locally |
| 2 — Partial local | 8 GB RAM, no GPU | Phases 1–3 fully · Phase 4 with Llama 3.2 3B |
| 3 — Follow along | < 8 GB RAM | Phases 1–3 · Phase 4 via AWS Bedrock API |

**Minimum:** 8 GB RAM · 10 GB free disk · Docker  
**Recommended:** 16 GB RAM · Apple Silicon or NVIDIA GPU  
**Phase 4+:** Ollama required (auto-pulled by docker-compose)  
**AWS path:** No local GPU needed — Bedrock handles inference

---

## Minimum Viable Deployment

To get business value with the lowest initial commitment, complete Phases 1–3 plus the FAQ Q&A portion of Phase 4:

| MVD Metric | Value |
|------------|-------|
| Capabilities deployed | Fraud detection, risk scoring, churn prediction, policy Q&A |
| Monthly infrastructure cost | ~$121/month |
| Year 1 projected benefit | $9.85M (fraud + churn + Q&A) |
| Payback period | < 90 days |

---

*Avvaru Iron Oak Insurance is a fictitious company created for AI development, meetups, and production-grade AWS showcase projects.*
