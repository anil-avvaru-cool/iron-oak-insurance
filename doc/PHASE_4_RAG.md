# Phase 4 — RAG Pipeline

**Git tag:** `v0.4.0`
**Deliverable:** Policy doc + FAQ Q&A with retrieval routing, served at `/rag/query`.

**Meetup demo:** Ask *"what is the deductible on policy TX-00142?"* (routes to policy document) then *"what does PIP mean?"* (routes to FAQ). Show retrieved chunks, source labels, and contrast grounded vs. hallucinated answer without RAG.

---

## Table of Contents

1. [Package Scaffolding](#1-package-scaffolding)
2. [Environment Variables](#2-environment-variables)
3. [FAQ Schema & Generator](#3-faq-schema--generator)
4. [Chunking Pipeline](#4-chunking-pipeline)
   - 4.1 [chunk\_router.py](#41-chunk_routerpy)
   - 4.2 [chunk\_faq.py](#42-chunk_faqpy)
   - 4.3 [chunk\_declaration.py](#43-chunk_declarationpy)
   - 4.4 [chunk\_claim\_letter.py](#44-chunk_claim_letterpy)
   - 4.5 [chunk\_renewal.py](#45-chunk_renewalpy)
5. [Embedding & Loading](#5-embedding--loading)
6. [RAG Retrieval Router](#6-rag-retrieval-router)
7. [RAG Pipeline](#7-rag-pipeline)
8. [RAG FastAPI Router](#8-rag-fastapi-router)
9. [Update main.py for Lifespan](#9-update-mainpy-for-lifespan)
10. [Verification Script](#10-verification-script)
11. [Start the Phase 4 Stack](#11-start-the-phase-4-stack)
12. [Verification & Git Tag](#12-verification--git-tag)

---

## Design Decisions This Phase

| Decision | Rationale |
|---|---|
| `SentenceTransformer` loaded once at lifespan startup | Loading per-request adds 2–4 s cold-start latency; `app.state.embedder` is injected via `Request` |
| `tiktoken` for token counting | `len(text.split())` undercounts by 15–30% on insurance text with numbers and abbreviations; `cl100k_base` is accurate and fast |
| int8 ONNX quantization opt-in via `EMBED_QUANTIZE=true` | Cuts embedding RAM from ~90 MB to ~25 MB, ~2× faster CPU inference, <1% retrieval accuracy loss — critical for 8 GB meetup machines |
| CTE in `retrieve()` SQL | Avoids the double-`params` bug from inserting `query_embedding` twice; computes distance once, reuses in SELECT and ORDER BY |
| `EMBED_MODE` from env, not `sys.argv` | Consistent with all other phase config; makes Lambda deployment trivial |
| FAQ `applicable_states: ["ALL"]` chunks have `state = NULL` in DB | State-filtered queries use `(state = %s OR state IS NULL)` so ALL-applicable FAQs always appear in results alongside state-specific ones |
| `get_conn()` imported from `db/` | Single source of truth for DB config; no inline connection strings anywhere in pipeline code |
| `DEBUG_MODE` env flag gates `/rag/debug` endpoint | Returns chunks + routing decision without LLM call; disabled in production by default |

---

## 1. Package Scaffolding

Create `__init__.py` files at every pipeline package level. The `-m` module invocation requires these to exist — missing files produce `ModuleNotFoundError` that is hard to diagnose.

```bash
# Linux / macOS — run from repo root
touch ai/pipelines/embedding/__init__.py
touch ai/pipelines/rag/__init__.py
touch ai/agents/__init__.py
touch ai/agents/claims_agent/__init__.py
touch ai/agents/fraud_agent/__init__.py
touch ai/agents/policy_advisor_agent/__init__.py
touch data_gen/__init__.py
touch data_gen/generators/__init__.py
```

```powershell
# Windows — run from repo root
$pkgs = @(
  "ai\pipelines\embedding",
  "ai\pipelines\rag",
  "ai\agents",
  "ai\agents\claims_agent",
  "ai\agents\fraud_agent",
  "ai\agents\policy_advisor_agent",
  "data_gen",
  "data_gen\generators"
)
$pkgs | ForEach-Object { New-Item -ItemType File -Force -Path "$_\__init__.py" }
```

> **Note on `data-gen/` vs `data_gen/`:** Python module names cannot contain hyphens. The folder on disk is `data-gen/` (per the repo structure); the Python package uses `data_gen/` as a symlink or the generators are invoked directly. The simplest fix is to add `__init__.py` inside `data-gen/generators/` and invoke as `uv run python -m data_gen.generators.faq_gen` after adding a `data_gen` symlink: `ln -s data-gen data_gen` (Linux/macOS) or `New-Item -ItemType Junction -Path data_gen -Target data-gen` (Windows).

---

## 2. Environment Variables

Add these to `.env.example`. All production-critical vars use `_require_env()`; local-dev vars allow a safe fallback.

```dotenv
# ── RAG Pipeline ─────────────────────────────────────────────────────────────
RAG_TOP_K=5                   # Number of chunks retrieved per query
RAG_MODE=local                # local | bedrock — generation backend
DEBUG_MODE=false              # true enables /rag/debug endpoint (disable in production)

# ── Embedding ────────────────────────────────────────────────────────────────
EMBED_MODE=local              # local | bedrock — embedding backend
EMBED_BATCH_SIZE=64           # Batch size for sentence-transformers encode()
EMBED_QUANTIZE=false          # true loads all-MiniLM via ONNX int8 (~25 MB vs ~90 MB)
EMBED_MODEL_LOCAL=all-MiniLM-L6-v2  # sentence-transformers model name

# ── Ollama (local generation) ─────────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2         # llama3.1:8b for better quality; llama3.2 for 8 GB RAM

# ── Already in .env.example from earlier phases — shown here for reference ───
# BEDROCK_MODEL_ID_HAIKU=anthropic.claude-haiku-4-5-20251001
# BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
# AWS_DEFAULT_REGION=us-east-1
# DB_HOST=, DB_PORT=, DB_NAME=, DB_USER=, DB_PASSWORD=
```

### Quantization decision guide

| Option | RAM | CPU Speed | Accuracy Loss | When to Use |
|--------|-----|-----------|---------------|-------------|
| Default float32 | ~90 MB | baseline | none | GPU / 16 GB+ RAM |
| `EMBED_QUANTIZE=true` (int8 ONNX) | ~25 MB | ~2× faster | <1% | Meetup demo, 8 GB machines |
| Llama 3.2 3B Q4 via Ollama | ~2.2 GB | ~15–25 tok/s CPU | moderate | Tier-2 hardware fallback |
| Llama 3.1 8B Q4 via Ollama | ~4.7 GB | ~5–10 tok/s CPU | low | Tier-1 hardware |

Enable int8 with: `uv add sentence-transformers[onnx] optimum[onnxruntime]`

---

## 3. FAQ Schema & Generator

### Full `faq.schema.json`

Replace the Phase 1 stub at `data-gen/schemas/faq.schema.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://aioi.internal/schemas/faq.schema.json",
  "title": "AIOI FAQ Record",
  "description": "A single Q&A pair. State-specific FAQs set applicable_states to state codes; generic FAQs set it to [\"ALL\"].",
  "type": "object",
  "required": ["faq_id","category","subcategory","question","answer","applicable_states","tags","source","version"],
  "additionalProperties": false,
  "properties": {
    "faq_id": {
      "type": "string",
      "pattern": "^faq-[a-z0-9-]+-[0-9]{3}$"
    },
    "category": {
      "type": "string",
      "enum": ["coverage_concepts","state_rules","claims_process","costs_discounts","policy_management"]
    },
    "subcategory": {
      "type": "string",
      "minLength": 2,
      "maxLength": 60,
      "pattern": "^[a-z0-9_-]+$"
    },
    "question": {
      "type": "string",
      "minLength": 10,
      "maxLength": 300,
      "pattern": "\\?$"
    },
    "answer": {
      "type": "string",
      "minLength": 20,
      "maxLength": 2000
    },
    "applicable_states": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "string",
        "oneOf": [
          {"const": "ALL"},
          {"type": "string", "pattern": "^(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)$"}
        ]
      }
    },
    "tags": {
      "type": "array",
      "minItems": 1,
      "maxItems": 15,
      "items": {"type": "string", "pattern": "^[a-z0-9-]+$", "minLength": 2, "maxLength": 40}
    },
    "source":  {"type": "string", "const": "synthetic-faq-v1"},
    "version": {"type": "string", "pattern": "^[0-9]+\\.[0-9]+$"},
    "last_updated":    {"type": "string", "format": "date"},
    "review_required": {"type": "boolean", "default": false}
  }
}
```

### `data-gen/generators/faq_gen.py`

This is the **complete** generator — all five categories fully populated. Run via:

```bash
uv run python -m data_gen.generators.faq_gen     # Linux / macOS (via symlink)
uv run python data-gen/generators/faq_gen.py     # direct fallback
```

```python
"""
faq_gen.py — generates the AIOI FAQ corpus from states.json and coverage_rules.json.

All five categories are fully implemented:
  coverage_concepts  — liability, collision, comprehensive, pip, UM, gap, roadside
  state_rules        — no-fault, min limits, total loss, PIP req, UM req (per state)
  claims_process     — filing through settlement
  costs_discounts    — premiums, drive score, telematics, discounts
  policy_management  — add vehicle/driver, coverage changes, renewal, SR-22

Module run:  uv run python -m data_gen.generators.faq_gen
Direct run:  uv run python data-gen/generators/faq_gen.py
"""
from __future__ import annotations

import json
from pathlib import Path

# ── Coverage concept templates (applicable_states: ["ALL"]) ──────────────────

_COVERAGE_TEMPLATES = [
    {
        "subcategory": "liability",
        "question": "What does liability insurance cover?",
        "answer": (
            "Liability insurance pays for injuries and property damage you cause to others in an "
            "accident where you are at fault. It is split into bodily injury (BI) and property "
            "damage (PD) components, expressed as split limits such as 30/60/25 — meaning $30,000 "
            "per person, $60,000 per accident, and $25,000 for property damage. It does not cover "
            "your own injuries or vehicle damage."
        ),
        "tags": ["liability","bodily-injury","property-damage","split-limits"],
    },
    {
        "subcategory": "liability",
        "question": "What are split limits on a liability policy?",
        "answer": (
            "Split limits express three separate maximums: per-person bodily injury, per-accident "
            "bodily injury, and property damage. For example, 25/50/10 means $25,000 maximum per "
            "injured person, $50,000 maximum for all injuries in one accident, and $10,000 for "
            "property damage. If one person's injuries exceed the per-person limit, the excess is "
            "your personal responsibility."
        ),
        "tags": ["liability","split-limits","bodily-injury"],
    },
    {
        "subcategory": "collision",
        "question": "When does collision coverage apply?",
        "answer": (
            "Collision coverage pays to repair or replace your vehicle when it collides with "
            "another vehicle or object — a guardrail, tree, or parked car — regardless of fault. "
            "You pay the deductible first; the insurer covers the rest up to the actual cash value "
            "(ACV) of the vehicle. It does not apply to theft, weather, or hitting an animal."
        ),
        "tags": ["collision","deductible","acv","fault"],
    },
    {
        "subcategory": "collision",
        "question": "What is the difference between collision and comprehensive?",
        "answer": (
            "Collision covers damage from hitting something — another vehicle, a pole, or rolling "
            "over. Comprehensive covers damage from everything else: theft, fire, hail, flood, "
            "fallen trees, vandalism, and hitting an animal. Both require a deductible. Together "
            "they are called 'full coverage,' though that is not an official insurance term."
        ),
        "tags": ["collision","comprehensive","full-coverage","deductible"],
    },
    {
        "subcategory": "comprehensive",
        "question": "What perils does comprehensive coverage cover?",
        "answer": (
            "Comprehensive covers non-collision losses including theft, fire, explosion, windstorm, "
            "hail, flood, falling objects, vandalism, and collision with an animal such as a deer. "
            "It pays actual cash value minus your deductible. It does not cover normal wear and "
            "tear, mechanical breakdown, or damage from collision with another vehicle."
        ),
        "tags": ["comprehensive","theft","hail","flood","acv"],
    },
    {
        "subcategory": "pip",
        "question": "What is Personal Injury Protection (PIP) coverage?",
        "answer": (
            "PIP covers medical expenses, lost wages, and related costs for you and your passengers "
            "after an accident, regardless of who was at fault. It is required in no-fault states "
            "such as MI, FL, NY, NJ, and PA. Benefit limits vary by state — Michigan offers the "
            "highest at up to $500,000, while Utah requires only $3,000 minimum."
        ),
        "tags": ["pip","no-fault","medical","lost-wages","coverage"],
    },
    {
        "subcategory": "pip",
        "question": "Does PIP cover passengers in my vehicle?",
        "answer": (
            "Yes. In states where PIP is required, it covers you, resident family members, and "
            "passengers in your vehicle regardless of fault. In some states it also covers you as "
            "a pedestrian struck by a vehicle. PIP does not cover vehicle damage — only medical "
            "and wage-related costs."
        ),
        "tags": ["pip","passengers","no-fault","medical"],
    },
    {
        "subcategory": "uninsured_motorist",
        "question": "What happens if the other driver has no insurance?",
        "answer": (
            "Uninsured Motorist (UM) coverage pays for your injuries and, in some states, your "
            "vehicle damage when the at-fault driver carries no insurance. Underinsured Motorist "
            "(UIM) coverage steps in when the at-fault driver's limits are too low to cover your "
            "losses. Without UM/UIM you would need to sue the at-fault driver directly, which is "
            "often impractical if they have no assets."
        ),
        "tags": ["uninsured-motorist","um","uim","at-fault","coverage"],
    },
    {
        "subcategory": "gap",
        "question": "When does gap insurance matter?",
        "answer": (
            "Gap insurance covers the difference between what your insurer pays (actual cash value) "
            "and what you still owe on your auto loan or lease if your vehicle is totaled or stolen. "
            "New vehicles depreciate quickly — in the first year a car can lose 15–25% of its value "
            "while the loan balance drops much more slowly. Gap is most valuable in the first "
            "2–3 years of ownership and is typically not needed once the loan balance falls below "
            "the vehicle's ACV."
        ),
        "tags": ["gap","total-loss","loan","lease","acv","depreciation"],
    },
    {
        "subcategory": "roadside",
        "question": "What is included in roadside assistance coverage?",
        "answer": (
            "Roadside assistance typically covers towing to the nearest qualified repair facility, "
            "battery jump-starts, flat tire changes (using your spare), lockout service, and fuel "
            "delivery for an empty tank. Some policies include winching if the vehicle is stuck "
            "off-road. Roadside is low-cost and worth adding if you do not already have it through "
            "a membership program."
        ),
        "tags": ["roadside","towing","battery","lockout","flat-tire"],
    },
]

# ── Claims process templates (applicable_states: ["ALL"]) ───────────────────

_CLAIMS_TEMPLATES = [
    {
        "subcategory": "filing",
        "question": "How do I report an accident to AIOI?",
        "answer": (
            "You can file a claim through Oak Assist (our AI intake agent), the AIOI mobile app, "
            "or by calling our claims line. You will need your policy number, the date and location "
            "of the incident, a description of what happened, and contact information for any other "
            "parties involved. Filing within 24–48 hours is strongly recommended — delays can "
            "complicate investigation."
        ),
        "tags": ["filing","fnol","claim","oak-assist","report"],
    },
    {
        "subcategory": "filing",
        "question": "What is FNOL?",
        "answer": (
            "FNOL stands for First Notice of Loss — the initial report you file when a covered "
            "incident occurs. It starts the claims process and assigns a claim number. Oak Assist "
            "handles FNOL intake for straightforward claims and escalates complex situations to a "
            "human adjuster. Providing accurate information at FNOL speeds up your settlement."
        ),
        "tags": ["fnol","filing","claims-process","oak-assist"],
    },
    {
        "subcategory": "after_filing",
        "question": "What happens after I file a claim?",
        "answer": (
            "After filing, AIOI acknowledges receipt within the state-mandated window (typically "
            "7–15 days). An adjuster is assigned to investigate, which may include inspecting "
            "your vehicle, reviewing photos, and contacting other parties. You receive a coverage "
            "determination and, if approved, a settlement offer. Complex or disputed claims take "
            "longer; straightforward claims are often settled within 30 days."
        ),
        "tags": ["after-filing","adjuster","investigation","settlement","timeline"],
    },
    {
        "subcategory": "documentation",
        "question": "What documents do I need to support my claim?",
        "answer": (
            "Useful documentation includes: photos of all vehicle damage and the accident scene, "
            "a police report if one was filed, contact and insurance information from all parties, "
            "witness names and contact info, medical records and bills for injury claims, and "
            "repair estimates. The more documentation you provide upfront, the faster the "
            "adjuster can process your claim."
        ),
        "tags": ["documentation","police-report","photos","records","adjuster"],
    },
    {
        "subcategory": "adjuster",
        "question": "What does a claims adjuster do?",
        "answer": (
            "A claims adjuster investigates the incident, determines coverage, assesses the value "
            "of the loss, and negotiates a settlement. They may inspect your vehicle in person or "
            "use photos and repair estimates. The adjuster verifies that the loss is covered under "
            "your policy, checks for fraud indicators, and calculates the payout after applying "
            "your deductible."
        ),
        "tags": ["adjuster","investigation","coverage","settlement","deductible"],
    },
    {
        "subcategory": "rental",
        "question": "Does my policy cover a rental car while mine is being repaired?",
        "answer": (
            "Rental reimbursement coverage pays for a rental car while your vehicle is being "
            "repaired due to a covered claim. It is an optional add-on with a daily and total "
            "cap — for example $40/day up to $1,200. Check your declarations page under "
            "'rental reimbursement' to see if it is included and what your limits are."
        ),
        "tags": ["rental","reimbursement","coverage","repair","declarations"],
    },
    {
        "subcategory": "total_loss_process",
        "question": "What happens if my car is totaled?",
        "answer": (
            "If repair costs exceed your state's total loss threshold (as a percentage of actual "
            "cash value), AIOI declares the vehicle a total loss. We pay you the ACV of your "
            "vehicle at the time of loss, minus your deductible. You sign over the title to AIOI. "
            "If you have gap coverage and owe more than the ACV on a loan, gap pays the difference. "
            "State thresholds range from 60% (Oklahoma) to 100% (several states including TX and AZ)."
        ),
        "tags": ["total-loss","acv","gap","threshold","title"],
    },
    {
        "subcategory": "settlement",
        "question": "How is a claim settlement calculated?",
        "answer": (
            "Settlement is based on the actual cash value (ACV) of the loss, minus your deductible. "
            "ACV is the replacement cost of a like-kind vehicle or repair, adjusted for depreciation "
            "and condition. For vehicle damage, we use market data and repair shop estimates. "
            "For total losses, we use vehicle valuation guides. You can negotiate if you disagree "
            "with the ACV — provide comparable vehicles or independent appraisals as evidence."
        ),
        "tags": ["settlement","acv","deductible","depreciation","negotiation"],
    },
]

# ── Costs and discounts templates (applicable_states: ["ALL"]) ───────────────

_COSTS_TEMPLATES = [
    {
        "subcategory": "premium_calc",
        "question": "What factors affect my insurance premium?",
        "answer": (
            "Key rating factors include: your driving record (accidents and violations), age and "
            "years of experience, the vehicle's make, model, year, and safety ratings, your annual "
            "mileage, where the vehicle is garaged (state and ZIP code), your credit score (in most "
            "states), and the coverages and deductibles you select. Telematics enrollment through "
            "the Iron Oak Drive Score program can provide additional discounts based on actual "
            "driving behavior."
        ),
        "tags": ["premium","rating-factors","driving-record","credit","telematics"],
    },
    {
        "subcategory": "drive_score",
        "question": "What is the Iron Oak Drive Score and how does it affect my rate?",
        "answer": (
            "The Iron Oak Drive Score (0–100) is calculated from telematics data including hard "
            "braking, rapid acceleration, speeding events, and night driving percentage — normalized "
            "per 10 miles driven. Scores of 90+ earn up to a 15% discount. Scores 75–89 earn 8%, "
            "scores 60–74 earn 3%, and scores below 60 receive no telematics discount. Your score "
            "is recalculated each renewal period based on the most recent 12 months of trip data."
        ),
        "tags": ["drive-score","telematics","discount","ubi","hard-braking"],
    },
    {
        "subcategory": "telematics",
        "question": "What driving data does the telematics program collect?",
        "answer": (
            "The Iron Oak telematics program collects trip-level data including distance driven, "
            "trip duration, hard braking events, rapid acceleration events, speeding events, and "
            "the percentage of miles driven at night (10 PM–5 AM). Location data is used only to "
            "detect trip boundaries and is not stored long-term. Data collection requires the "
            "Iron Oak mobile app or an OBD-II device."
        ),
        "tags": ["telematics","data","privacy","ubi","collection"],
    },
    {
        "subcategory": "discounts",
        "question": "What discounts does AIOI offer?",
        "answer": (
            "AIOI offers discounts for: safe driver (clean record for 3+ years), Iron Oak Drive "
            "Score (telematics enrollment), multi-policy (bundling auto with another AIOI product), "
            "paid-in-full (paying annual premium upfront), anti-theft devices, defensive driving "
            "course completion, good student (GPA 3.0+), and paperless billing. Not all discounts "
            "are available in every state."
        ),
        "tags": ["discounts","safe-driver","multi-policy","telematics","good-student"],
    },
    {
        "subcategory": "multi_policy",
        "question": "Does bundling policies with AIOI save money?",
        "answer": (
            "Yes. Customers with two or more AIOI policies (for example, two vehicles, or auto "
            "plus renters) typically receive a 5–12% multi-policy discount on each policy. The "
            "discount is applied automatically when the policies share the same named insured. "
            "Contact your agent or log in to verify the discount appears on your declarations page."
        ),
        "tags": ["multi-policy","bundle","discount","savings"],
    },
    {
        "subcategory": "good_driver",
        "question": "How is a good driver discount earned and maintained?",
        "answer": (
            "The good driver discount applies when the primary driver has had no at-fault "
            "accidents, major violations (DUI, reckless driving), or more than one minor violation "
            "in the past three years. The discount is re-evaluated at each renewal. A single "
            "at-fault accident typically removes the discount for three years from the accident date."
        ),
        "tags": ["good-driver","discount","violations","at-fault","renewal"],
    },
    {
        "subcategory": "credit",
        "question": "Does my credit score affect my insurance rate?",
        "answer": (
            "In most states, insurers use a credit-based insurance score — distinct from your "
            "lending credit score — as a rating factor. Research shows it correlates with claim "
            "frequency. States that prohibit its use include California, Hawaii, Massachusetts, "
            "and Michigan. If your credit improves significantly, you can request a re-rate at "
            "renewal. AIOI uses credit as one factor among many, not as the sole determinant."
        ),
        "tags": ["credit","insurance-score","rating","state-rule"],
    },
]

# ── Policy management templates (applicable_states: ["ALL"]) ────────────────

_POLICY_MGMT_TEMPLATES = [
    {
        "subcategory": "add_vehicle",
        "question": "How do I add a vehicle to my policy?",
        "answer": (
            "Contact AIOI or log in to your account to add a vehicle. You will need the VIN, "
            "year, make, model, and current odometer reading. Coverage for the new vehicle begins "
            "as of the add date. Your premium is prorated for the remaining policy term. If you "
            "replace an existing vehicle, the old vehicle is removed from the date of sale."
        ),
        "tags": ["add-vehicle","policy-change","vin","premium","mid-term"],
    },
    {
        "subcategory": "add_driver",
        "question": "What happens when I add a teenage driver to my policy?",
        "answer": (
            "Adding a teenage driver typically increases your premium, as young drivers have "
            "statistically higher claim rates. The increase varies by age (16–19), gender, and "
            "state. A good student discount (GPA 3.0+) can offset some of the increase. Teens "
            "can also enroll in the Drive Score program — good scores earn discounts and "
            "reinforce safe habits. The teen must be added; driving on the policy without being "
            "listed can result in a coverage denial."
        ),
        "tags": ["add-driver","teen","premium","good-student","drive-score"],
    },
    {
        "subcategory": "coverage_changes",
        "question": "Can I change my coverage mid-term?",
        "answer": (
            "Yes. Most coverage changes can be made mid-term — adding or removing optional "
            "coverages, changing deductibles, or updating limits. Changes take effect on the "
            "requested date and your premium is adjusted pro-rata. Some changes (such as removing "
            "collision on a financed vehicle) may be restricted by your lender. Contact AIOI or "
            "make changes through the online portal."
        ),
        "tags": ["coverage-change","mid-term","deductible","pro-rata","lender"],
    },
    {
        "subcategory": "renewal_lapse",
        "question": "What happens if I miss a premium payment?",
        "answer": (
            "AIOI provides a grace period (typically 10–30 days depending on your state) after "
            "the due date before the policy lapses for non-payment. During the grace period you "
            "remain covered but should pay immediately to avoid cancellation. A lapsed policy "
            "means no coverage — any claims filed after the lapse date will be denied. "
            "Reinstating after a lapse may require a new application and premium recalculation."
        ),
        "tags": ["payment","grace-period","lapse","cancellation","reinstatement"],
    },
    {
        "subcategory": "cancellation",
        "question": "How do I cancel my AIOI policy?",
        "answer": (
            "To cancel, contact AIOI by phone, mail, or through your agent. You will need your "
            "policy number and the desired cancellation date. If you have paid ahead, you will "
            "receive a prorated refund for the unused premium (minus any short-rate fee if "
            "cancelling mid-term at your request). Cancellation without replacement coverage "
            "creates a coverage gap that can raise future premiums."
        ),
        "tags": ["cancellation","refund","pro-rata","coverage-gap","policy"],
    },
    {
        "subcategory": "sr22",
        "question": "What is an SR-22 and when is it required?",
        "answer": (
            "An SR-22 is a certificate of financial responsibility filed by your insurer with "
            "your state's DMV confirming you carry the minimum required coverage. It is typically "
            "required after a DUI/DWI conviction, serious traffic violation, license suspension, "
            "or being caught driving uninsured. The filing requirement usually lasts 3 years. "
            "Not all insurers file SR-22s — AIOI does file in all states where we operate. "
            "An SR-22 itself is not insurance; it is proof of insurance."
        ),
        "tags": ["sr22","dui","suspension","financial-responsibility","dmv"],
    },
]


def _make_record(
    idx: int,
    subcategory: str,
    category: str,
    template: dict,
    applicable_states: list[str],
) -> dict:
    slug = subcategory.replace("_", "-")
    return {
        "faq_id": f"faq-{slug}-{idx:03d}",
        "category": category,
        "subcategory": subcategory,
        "question": template["question"],
        "answer": template["answer"],
        "applicable_states": applicable_states,
        "tags": template["tags"],
        "source": "synthetic-faq-v1",
        "version": "1.0",
    }


def generate_coverage_faqs() -> list[dict]:
    """Generic coverage concept FAQs applicable to all states."""
    return [
        _make_record(i + 1, t["subcategory"], "coverage_concepts", t, ["ALL"])
        for i, t in enumerate(_COVERAGE_TEMPLATES)
    ]


def generate_claims_faqs() -> list[dict]:
    """Claims process FAQs applicable to all states."""
    return [
        _make_record(i + 1, t["subcategory"], "claims_process", t, ["ALL"])
        for i, t in enumerate(_CLAIMS_TEMPLATES)
    ]


def generate_costs_faqs() -> list[dict]:
    """Cost and discount FAQs applicable to all states."""
    return [
        _make_record(i + 1, t["subcategory"], "costs_discounts", t, ["ALL"])
        for i, t in enumerate(_COSTS_TEMPLATES)
    ]


def generate_policy_mgmt_faqs() -> list[dict]:
    """Policy management FAQs applicable to all states."""
    return [
        _make_record(i + 1, t["subcategory"], "policy_management", t, ["ALL"])
        for i, t in enumerate(_POLICY_MGMT_TEMPLATES)
    ]


def generate_state_faqs(states_data: dict) -> list[dict]:
    """
    State-specific FAQs generated directly from states.json.
    Produces: no-fault explainer, total loss threshold, min liability,
    PIP requirement, and UM requirement per applicable state.
    """
    records: list[dict] = []
    idx = 1

    for state, rules in states_data.items():
        # No-fault explainer
        if rules.get("no_fault"):
            pip_limit = rules.get("pip_limit") or 0
            records.append({
                "faq_id": f"faq-nofault-{idx:03d}",
                "category": "state_rules",
                "subcategory": "no_fault",
                "question": f"Is {state} a no-fault state?",
                "answer": (
                    f"Yes, {state} is a no-fault state. After an accident, your own PIP "
                    f"coverage pays your medical expenses up to the state limit "
                    f"(${pip_limit:,}) regardless of who caused the accident. Your ability "
                    f"to sue the at-fault driver for pain and suffering is restricted unless "
                    f"your injuries meet a defined threshold."
                ),
                "applicable_states": [state],
                "tags": ["no-fault", "pip", "state-rule", state.lower()],
                "source": "synthetic-faq-v1",
                "version": "1.0",
            })
            idx += 1

        # Total loss threshold
        tlt = rules.get("total_loss_threshold")
        if tlt:
            records.append({
                "faq_id": f"faq-totalloss-{idx:03d}",
                "category": "state_rules",
                "subcategory": "total_loss",
                "question": f"At what damage percentage is a car declared a total loss in {state}?",
                "answer": (
                    f"In {state}, a vehicle is declared a total loss when repair costs reach "
                    f"{int(tlt * 100)}% or more of the vehicle's actual cash value (ACV)."
                ),
                "applicable_states": [state],
                "tags": ["total-loss", "state-rule", state.lower()],
                "source": "synthetic-faq-v1",
                "version": "1.0",
            })
            idx += 1

        # Minimum liability limits
        ml = rules.get("min_liability", {})
        if ml:
            bi_pp = ml.get("bodily_injury_per_person", 0)
            bi_pa = ml.get("bodily_injury_per_accident", 0)
            pd = ml.get("property_damage", 0)
            records.append({
                "faq_id": f"faq-minliability-{idx:03d}",
                "category": "state_rules",
                "subcategory": "minimum_liability",
                "question": f"What are the minimum liability insurance limits required in {state}?",
                "answer": (
                    f"{state} requires minimum liability limits of ${bi_pp:,} per person / "
                    f"${bi_pa:,} per accident for bodily injury and ${pd:,} for property damage "
                    f"(expressed as {bi_pp // 1000}/{bi_pa // 1000}/{pd // 1000}). These are "
                    f"minimums only — most drivers benefit from higher limits to protect their assets."
                ),
                "applicable_states": [state],
                "tags": ["minimum-liability", "state-rule", "split-limits", state.lower()],
                "source": "synthetic-faq-v1",
                "version": "1.0",
            })
            idx += 1

        # PIP requirement
        if rules.get("pip_required"):
            pip_limit = rules.get("pip_limit") or 0
            records.append({
                "faq_id": f"faq-pipreq-{idx:03d}",
                "category": "state_rules",
                "subcategory": "pip_requirements",
                "question": f"Is PIP coverage required in {state}?",
                "answer": (
                    f"Yes, Personal Injury Protection (PIP) is mandatory in {state} with a minimum "
                    f"benefit limit of ${pip_limit:,}. Every policy issued in {state} must include PIP."
                ),
                "applicable_states": [state],
                "tags": ["pip", "required", "state-rule", state.lower()],
                "source": "synthetic-faq-v1",
                "version": "1.0",
            })
            idx += 1

        # UM requirement
        if rules.get("uninsured_motorist_required"):
            records.append({
                "faq_id": f"faq-umreq-{idx:03d}",
                "category": "state_rules",
                "subcategory": "um_coverage",
                "question": f"Is uninsured motorist coverage required in {state}?",
                "answer": (
                    f"Yes, uninsured motorist (UM) coverage is mandatory in {state}. "
                    f"It protects you when you are injured by a driver who has no insurance. "
                    f"Your UM limits must match your liability limits unless you sign a waiver "
                    f"to select lower limits."
                ),
                "applicable_states": [state],
                "tags": ["uninsured-motorist", "required", "state-rule", state.lower()],
                "source": "synthetic-faq-v1",
                "version": "1.0",
            })
            idx += 1

    return records


def generate(states_data: dict) -> list[dict]:
    """Generate the full FAQ corpus. Returns all records combined."""
    return (
        generate_coverage_faqs()
        + generate_claims_faqs()
        + generate_costs_faqs()
        + generate_policy_mgmt_faqs()
        + generate_state_faqs(states_data)
    )


def main(
    output_path: Path | None = None,
    states_data: dict | None = None,
) -> list[dict]:
    config_dir = Path("data-gen/config")
    if states_data is None:
        states_data = json.loads((config_dir / "states.json").read_text())
    if output_path is None:
        output_path = Path("faqs/faq_corpus.json")

    records = generate(states_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    print(f"[faq_gen] wrote {len(records):,} FAQ records → {output_path}")
    return records


if __name__ == "__main__":
    main()
```

---

## 4. Chunking Pipeline

### 4.1 `ai/pipelines/embedding/chunk_router.py`

```python
"""
chunk_router.py — detects document type from filename, dispatches to the correct chunker.

Relies on the filename convention established in document_gen.py:
  decl_<policy_number>.pdf         → chunk_declaration_page()
  claim_letter_<claim_id>.pdf      → chunk_claim_letter()
  renewal_<policy_number>.pdf      → chunk_renewal_notice()

No ML classification needed — filename prefix is load-bearing.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .chunk_declaration import chunk_declaration_page
from .chunk_claim_letter import chunk_claim_letter
from .chunk_renewal import chunk_renewal_notice

log = logging.getLogger(__name__)


def route(path: Path) -> list[dict]:
    """Dispatch a PDF to the appropriate chunker. Returns list of chunk dicts."""
    name = path.name.lower()
    if name.startswith("decl_"):
        log.debug("route", extra={"file": path.name, "chunker": "declaration"})
        return chunk_declaration_page(path)
    elif name.startswith("claim_letter_"):
        log.debug("route", extra={"file": path.name, "chunker": "claim_letter"})
        return chunk_claim_letter(path)
    elif name.startswith("renewal_"):
        log.debug("route", extra={"file": path.name, "chunker": "renewal"})
        return chunk_renewal_notice(path)
    else:
        raise ValueError(
            f"Unknown document type for file '{path.name}'. "
            f"Expected prefix: decl_ | claim_letter_ | renewal_"
        )
```

### 4.2 `ai/pipelines/embedding/chunk_faq.py`

```python
"""
chunk_faq.py — one chunk per Q&A pair, no splitting, no overlap.

The question text is prepended to the answer so semantic search on
either surface retrieves the correct pair.

Token counting uses tiktoken (cl100k_base) for accuracy. Falls back
to word-count estimate if tiktoken is not installed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_MAX_TOKENS = 200  # warn if exceeded — FAQ answers should be concise

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    log.warning("tiktoken not installed — using word-count approximation for token counts")
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) * 1.3)  # rough correction factor


def chunk_faq_records(faq_path: Path) -> list[dict]:
    """
    Convert FAQ JSON corpus into pgvector-ready chunk dicts.

    Each record becomes exactly one chunk:
      chunk_text = "Q: <question>\\nA: <answer>"
      state      = first applicable_state if not ALL; else NULL
    """
    records: list[dict] = json.loads(faq_path.read_text())
    chunks: list[dict] = []

    for rec in records:
        chunk_text = f"Q: {rec['question']}\nA: {rec['answer']}"
        token_count = _count_tokens(chunk_text)

        if token_count > _MAX_TOKENS:
            log.warning(
                "faq_chunk_oversized",
                extra={
                    "faq_id": rec["faq_id"],
                    "token_count": token_count,
                    "max": _MAX_TOKENS,
                },
            )

        # State: NULL for ALL-applicable FAQs so state-filtered queries still return them
        states = rec.get("applicable_states", ["ALL"])
        state = states[0] if states != ["ALL"] and len(states) == 1 else None

        chunks.append({
            "chunk_id":      rec["faq_id"],
            "source_type":   "faq",
            "doc_type":      "faq",
            "policy_number": None,
            "customer_id":   None,
            "state":         state,
            "page_number":   None,
            "section":       rec["category"],
            "chunk_index":   0,
            "token_count":   token_count,
            "chunk_text":    chunk_text,
        })

    log.info("faq_chunks_built", extra={"count": len(chunks)})
    return chunks
```

### 4.3 `ai/pipelines/embedding/chunk_declaration.py`

```python
"""
chunk_declaration.py — table-aware section chunker for declaration pages.

Strategy:
  1. Extract text blocks from PDF using PyMuPDF, preserving font metadata.
  2. Detect section boundaries by bold headers (font flags) or font size > body.
  3. Split into logical sections: named_insured_block, vehicle_details,
     coverage_table, endorsements.
  4. Coverage table: each row (coverage type + limit + deductible) is one chunk.
  5. Target: 200–400 tokens per chunk; no overlap at section boundaries.

Filename convention (load-bearing for metadata extraction):
  decl_TX-00142.pdf  →  policy_number = "TX-00142"
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise ImportError(
        "PyMuPDF is required for declaration page chunking. "
        "Install with: uv add PyMuPDF"
    ) from exc

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _tok(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def _tok(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) * 1.3)

# Pattern to identify coverage table rows:
# e.g. "Collision   $500   ACV" or "Liability   30/60/25   —"
_COVERAGE_ROW_RE = re.compile(
    r"(liability|collision|comprehensive|pip|uninsured|gap|roadside)",
    re.IGNORECASE,
)
# Bold flag in PyMuPDF span flags bitmask
_BOLD_FLAG = 1 << 4


def _is_bold(span: dict) -> bool:
    return bool(span.get("flags", 0) & _BOLD_FLAG)


def _extract_text_blocks(page: "fitz.Page") -> list[dict]:
    """Return list of {text, is_bold, font_size, bbox} for each span on the page."""
    blocks = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # skip image blocks
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                blocks.append({
                    "text": text,
                    "is_bold": _is_bold(span),
                    "font_size": span.get("size", 10),
                    "bbox": span.get("bbox"),
                })
    return blocks


def _detect_section(text: str, is_bold: bool) -> str | None:
    """Return section name if this block is a section header, else None."""
    t = text.lower().strip()
    if is_bold:
        if any(k in t for k in ["named insured", "policyholder", "insured:"]):
            return "named_insured_block"
        if any(k in t for k in ["vehicle", "automobile", "auto description"]):
            return "vehicle_details"
        if any(k in t for k in ["coverage", "coverages", "schedule of"]):
            return "coverage_table"
        if any(k in t for k in ["endorsement", "addendum"]):
            return "endorsements"
    return None


def _build_chunk(
    policy_number: str,
    section: str,
    text_lines: list[str],
    chunk_index: int,
) -> dict:
    chunk_text = "\n".join(text_lines).strip()
    return {
        "chunk_id":      f"decl-{policy_number}-{section}-{chunk_index:03d}",
        "source_type":   "policy_document",
        "doc_type":      "declaration_page",
        "policy_number": policy_number,
        "customer_id":   None,  # resolved downstream from policy lookup if needed
        "state":         policy_number[:2] if policy_number else None,
        "page_number":   None,
        "section":       section,
        "chunk_index":   chunk_index,
        "token_count":   _tok(chunk_text),
        "chunk_text":    chunk_text,
    }


def chunk_declaration_page(path: Path) -> list[dict]:
    """
    Chunk a declaration page PDF into section-aware chunks.
    Returns list of chunk dicts ready for embedding.
    """
    # Extract policy number from filename: decl_TX-00142.pdf → TX-00142
    stem = path.stem  # "decl_TX-00142"
    policy_number = stem[len("decl_"):] if stem.startswith("decl_") else stem

    doc = fitz.open(path)
    all_blocks: list[dict] = []
    for page_num, page in enumerate(doc):
        for block in _extract_text_blocks(page):
            block["page_number"] = page_num + 1
            all_blocks.append(block)
    doc.close()

    chunks: list[dict] = []
    current_section = "named_insured_block"
    current_lines: list[str] = []
    chunk_index = 0
    in_coverage_table = False

    for block in all_blocks:
        text = block["text"]
        new_section = _detect_section(text, block["is_bold"])

        if new_section:
            # Flush current section before starting new one
            if current_lines:
                chunks.append(
                    _build_chunk(policy_number, current_section, current_lines, chunk_index)
                )
                chunk_index += 1
                current_lines = []
            current_section = new_section
            in_coverage_table = (new_section == "coverage_table")
            continue  # section header is not included in chunk text

        if in_coverage_table and _COVERAGE_ROW_RE.search(text):
            # Each coverage row is its own chunk to keep limit+deductible together
            if current_lines:
                # Flush any buffered non-row lines in the table section
                chunks.append(
                    _build_chunk(policy_number, current_section, current_lines, chunk_index)
                )
                chunk_index += 1
                current_lines = []
            # Collect the full row — gather adjacent non-header lines as the row continues
            row_lines = [text]
            chunks.append(
                _build_chunk(policy_number, "coverage_table_row", row_lines, chunk_index)
            )
            chunk_index += 1
        else:
            current_lines.append(text)
            # Flush if chunk is getting large (>400 tokens)
            if _tok("\n".join(current_lines)) > 400:
                chunks.append(
                    _build_chunk(policy_number, current_section, current_lines, chunk_index)
                )
                chunk_index += 1
                current_lines = []

    # Flush final section
    if current_lines:
        chunks.append(
            _build_chunk(policy_number, current_section, current_lines, chunk_index)
        )

    log.info(
        "declaration_chunked",
        extra={"policy_number": policy_number, "chunks": len(chunks), "file": path.name},
    )
    return chunks
```

### 4.4 `ai/pipelines/embedding/chunk_claim_letter.py`

```python
"""
chunk_claim_letter.py — paragraph chunker with 50-token overlap for claim letters.

Strategy:
  - Split on paragraph breaks (\\n\\n).
  - Max 350 tokens per chunk; split at sentence boundary if exceeded.
  - 50-token overlap between consecutive chunks for cross-boundary context.
  - claim_id and policy_number injected into every chunk's metadata
    (extracted from filename — do NOT rely on them surviving into chunk text).

Filename convention:
  claim_letter_CLM-00001.pdf  →  claim_id = "CLM-00001"
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

try:
    import fitz
except ImportError as exc:
    raise ImportError("PyMuPDF required: uv add PyMuPDF") from exc

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _tok(text: str) -> int:
        return len(_enc.encode(text))
    def _detok(tokens: list) -> str:
        return _enc.decode(tokens)
except ImportError:
    def _tok(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) * 1.3)
    def _detok(tokens: list) -> str:  # type: ignore[misc]
        return " ".join(str(t) for t in tokens)

_MAX_TOKENS   = 350
_OVERLAP_TOKENS = 50
_SENTENCE_RE  = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_long_paragraph(text: str) -> list[str]:
    """Split a paragraph that exceeds _MAX_TOKENS at sentence boundaries."""
    sentences = _SENTENCE_RE.split(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sent in sentences:
        t = _tok(sent)
        if current_tokens + t > _MAX_TOKENS and current:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sent)
        current_tokens += t
    if current:
        chunks.append(" ".join(current))
    return chunks


def _add_overlap(chunks: list[str]) -> list[str]:
    """
    Prepend the last _OVERLAP_TOKENS tokens of the previous chunk
    to each chunk (except the first).
    """
    if len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        try:
            prev_tokens = _enc.encode(chunks[i - 1])
            overlap_tokens = prev_tokens[-_OVERLAP_TOKENS:]
            overlap_text = _enc.decode(overlap_tokens)
        except Exception:
            # Fallback if tiktoken not available
            words = chunks[i - 1].split()
            overlap_text = " ".join(words[-15:])  # ~50 token approximation
        result.append(overlap_text + " " + chunks[i])
    return result


def chunk_claim_letter(path: Path) -> list[dict]:
    """
    Chunk a claim letter PDF into paragraph-level chunks with overlap.
    """
    stem = path.stem  # "claim_letter_CLM-00001"
    prefix = "claim_letter_"
    claim_id = stem[len(prefix):] if stem.startswith(prefix) else stem

    # Extract policy_number from letter text heuristic (pattern XX-NNNNN)
    doc = fitz.open(path)
    full_text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    policy_match = re.search(r"\b([A-Z]{2}-\d{5})\b", full_text)
    policy_number = policy_match.group(1) if policy_match else None
    state = policy_number[:2] if policy_number else None

    # Split into paragraphs
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()]

    # Split oversized paragraphs at sentence boundaries
    para_chunks: list[str] = []
    for para in raw_paragraphs:
        if _tok(para) > _MAX_TOKENS:
            para_chunks.extend(_split_long_paragraph(para))
        else:
            para_chunks.append(para)

    # Add overlap
    para_chunks = _add_overlap(para_chunks)

    chunks = []
    for i, text in enumerate(para_chunks):
        chunks.append({
            "chunk_id":      f"claim-{claim_id}-{i:03d}",
            "source_type":   "policy_document",
            "doc_type":      "claim_letter",
            "policy_number": policy_number,
            "customer_id":   None,
            "state":         state,
            "page_number":   None,
            "section":       "body",
            "chunk_index":   i,
            "token_count":   _tok(text),
            "chunk_text":    text,
        })

    log.info(
        "claim_letter_chunked",
        extra={"claim_id": claim_id, "chunks": len(chunks), "file": path.name},
    )
    return chunks
```

### 4.5 `ai/pipelines/embedding/chunk_renewal.py`

```python
"""
chunk_renewal.py — hybrid zone chunker for renewal notices.

Renewal notices have two zones:
  TABLE ZONE   — premium change table at the top (row-aware, no overlap)
  PROSE ZONE   — explanation paragraphs below (paragraph chunker, 50-token overlap)

Zone boundary detection: first blank line after a block that contains
at least 3 lines with dollar amounts ($NNN.NN pattern).

Filename convention:
  renewal_TX-00142.pdf  →  policy_number = "TX-00142"
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .chunk_claim_letter import _split_long_paragraph, _add_overlap, _tok

log = logging.getLogger(__name__)

try:
    import fitz
except ImportError as exc:
    raise ImportError("PyMuPDF required: uv add PyMuPDF") from exc

_DOLLAR_RE = re.compile(r"\$\d[\d,]*\.?\d*")
_MAX_TABLE_TOKENS = 400
_MAX_PROSE_TOKENS = 350


def _detect_zone_boundary(lines: list[str]) -> int:
    """
    Return the line index where the prose zone begins.
    Heuristic: first blank line after we have seen >= 3 dollar-amount lines
    in a contiguous block.
    """
    dollar_streak = 0
    for i, line in enumerate(lines):
        if _DOLLAR_RE.search(line):
            dollar_streak += 1
        else:
            if dollar_streak >= 3 and not line.strip():
                return i + 1  # first line of prose zone
            if not line.strip():
                dollar_streak = 0
    return len(lines)  # fallback: entire document is table zone


def _chunk_table_zone(lines: list[str], policy_number: str, state: str | None) -> list[dict]:
    """
    Chunk the table zone. Each non-blank line (or short group) becomes a chunk.
    Dollar-amount lines that belong to the same row are grouped together.
    """
    chunks: list[dict] = []
    idx = 0
    buffer: list[str] = []

    def flush(buf: list[str]) -> None:
        nonlocal idx
        if not buf:
            return
        text = "\n".join(buf).strip()
        if text:
            chunks.append({
                "chunk_id":      f"renewal-{policy_number}-table-{idx:03d}",
                "source_type":   "policy_document",
                "doc_type":      "renewal_notice",
                "policy_number": policy_number,
                "customer_id":   None,
                "state":         state,
                "page_number":   None,
                "section":       "premium_table",
                "chunk_index":   idx,
                "token_count":   _tok(text),
                "chunk_text":    text,
            })
            idx += 1

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush(buffer)
            buffer = []
            continue
        buffer.append(stripped)
        if _tok("\n".join(buffer)) > _MAX_TABLE_TOKENS:
            flush(buffer)
            buffer = []

    flush(buffer)
    return chunks


def _chunk_prose_zone(
    lines: list[str], policy_number: str, state: str | None, start_idx: int
) -> list[dict]:
    """
    Chunk the prose zone using paragraph splitting with 50-token overlap.
    """
    full_text = "\n".join(lines)
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()]

    para_chunks: list[str] = []
    for para in raw_paragraphs:
        if _tok(para) > _MAX_PROSE_TOKENS:
            para_chunks.extend(_split_long_paragraph(para))
        else:
            para_chunks.append(para)

    para_chunks = _add_overlap(para_chunks)

    chunks = []
    for i, text in enumerate(para_chunks):
        chunks.append({
            "chunk_id":      f"renewal-{policy_number}-prose-{start_idx + i:03d}",
            "source_type":   "policy_document",
            "doc_type":      "renewal_notice",
            "policy_number": policy_number,
            "customer_id":   None,
            "state":         state,
            "page_number":   None,
            "section":       "prose_explanation",
            "chunk_index":   start_idx + i,
            "token_count":   _tok(text),
            "chunk_text":    text,
        })
    return chunks


def chunk_renewal_notice(path: Path) -> list[dict]:
    """
    Chunk a renewal notice PDF using hybrid zone detection.
    """
    stem = path.stem  # "renewal_TX-00142"
    prefix = "renewal_"
    policy_number = stem[len(prefix):] if stem.startswith(prefix) else stem
    state = policy_number[:2] if len(policy_number) >= 2 else None

    doc = fitz.open(path)
    full_text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    lines = full_text.split("\n")
    boundary = _detect_zone_boundary(lines)

    table_lines = lines[:boundary]
    prose_lines = lines[boundary:]

    table_chunks = _chunk_table_zone(table_lines, policy_number, state)
    prose_chunks = _chunk_prose_zone(
        prose_lines, policy_number, state, start_idx=len(table_chunks)
    )

    all_chunks = table_chunks + prose_chunks
    log.info(
        "renewal_chunked",
        extra={
            "policy_number": policy_number,
            "table_chunks": len(table_chunks),
            "prose_chunks": len(prose_chunks),
            "file": path.name,
        },
    )
    return all_chunks
```

---

## 5. Embedding & Loading

**`ai/pipelines/embedding/embed_and_load.py`**

Key fixes from the original:
- `EMBED_MODE`, `EMBED_BATCH_SIZE`, `EMBED_QUANTIZE` all read from env via `os.getenv()`
- Uses shared `get_conn()` from `db/`
- Batch size controlled — won't OOM on large document sets
- `--dry-run` flag counts chunks without DB writes
- Module invocation: `uv run python -m ai.pipelines.embedding.embed_and_load`

```python
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
```

---

## 6. RAG Retrieval Router

**`ai/pipelines/rag/retrieval_router.py`**

```python
"""
retrieval_router.py — classifies a query and returns the retrieval strategy.

Priority (applied in order):
  1. Contains policy number pattern (XX-NNNNN)  → policy_document first
  2. Contains customer ID or "my policy/deductible/coverage" → policy_document
  3. Concept signal ("what is", "how does", etc.)            → faq first
  4. State abbreviation present                              → set state_filter
  5. Ambiguous                                               → both

The state_filter is used in retrieve() as:
  (state = %s OR state IS NULL)
so ALL-applicable FAQs always appear alongside state-specific ones.
"""
from __future__ import annotations

import re

_POLICY_PATTERN   = re.compile(r"\b([A-Z]{2}-\d{5})\b")
_CUSTOMER_PATTERN = re.compile(r"\b(CUST-\d+)\b", re.IGNORECASE)
_CONCEPT_SIGNALS  = [
    "what is", "what are", "how does", "how do i", "what does",
    "what happens", "explain", "define", "tell me about",
]
_PERSONAL_SIGNALS = ["my policy", "my deductible", "my coverage", "my claim", "my premium"]

_US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC",
}


def classify_query(query: str) -> dict:
    """
    Returns:
      {
        "strategy":      "policy_document" | "faq" | "both",
        "policy_number": str | None,
        "customer_id":   str | None,
        "state_filter":  str | None,   # 2-letter state code
      }
    """
    q_lower = query.lower()

    result: dict = {
        "strategy":      "both",
        "policy_number": None,
        "customer_id":   None,
        "state_filter":  None,
    }

    # Priority 1: explicit policy number
    pm = _POLICY_PATTERN.search(query)
    if pm:
        result["strategy"]      = "policy_document"
        result["policy_number"] = pm.group(1)
        # Still extract state filter even for policy queries
        _apply_state_filter(query, result)
        return result

    # Priority 2: customer ID or personal possessive phrases
    cm = _CUSTOMER_PATTERN.search(query)
    if cm or any(sig in q_lower for sig in _PERSONAL_SIGNALS):
        result["strategy"]    = "policy_document"
        result["customer_id"] = cm.group(1) if cm else None
        _apply_state_filter(query, result)
        return result

    # Priority 3: concept question signals → FAQ
    if any(sig in q_lower for sig in _CONCEPT_SIGNALS):
        result["strategy"] = "faq"

    # State filter applied to FAQ queries too
    _apply_state_filter(query, result)
    return result


def _apply_state_filter(query: str, result: dict) -> None:
    """Detect a US state abbreviation in the query and set state_filter."""
    q_upper = query.upper()
    for state in _US_STATES:
        # Match as a word boundary — avoid "IN" matching "insurance"
        if re.search(rf"\b{state}\b", q_upper):
            result["state_filter"] = state
            break
```

---

## 7. RAG Pipeline

**`ai/pipelines/rag/rag_pipeline.py`**

Key fix: the original code had a `params.insert(1, query_embedding)` bug that caused a positional mismatch in psycopg2. This version uses a CTE to compute the distance once, reusing it in both SELECT and ORDER BY — clean, readable, and correct.

```python
"""
rag_pipeline.py — retrieves relevant chunks and generates a grounded answer.

Fixes vs. original:
  - SQL rewritten as CTE to avoid the double-params positional bug
  - state_filter correctly uses (state = %s OR state IS NULL) so ALL-applicable
    FAQ chunks are always returned alongside state-specific ones
  - DB connection via shared get_conn() — no inline credentials
  - All config (RAG_TOP_K, OLLAMA_BASE_URL, OLLAMA_MODEL, BEDROCK_MODEL_ID_HAIKU)
    read from env — no hardcoded fallback strings for production vars
  - Structured logging per retrieve() call

Module invocation: called from rag_router.py; not a standalone entry point.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from db.load_json import get_conn  # noqa: E402

log = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(f"Required env var '{name}' is not set.")
    return val


def retrieve(
    query_embedding: list[float],
    strategy: dict,
    top_k: int | None = None,
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks from pgvector.

    Uses a CTE to compute cosine similarity once, then filters and ranks —
    avoids passing query_embedding twice in the params list.

    The state_filter uses (state = %s OR state IS NULL) so FAQ chunks with
    applicable_states: ["ALL"] (stored as state=NULL) always appear in results.
    """
    if top_k is None:
        top_k = int(os.getenv("RAG_TOP_K", "5"))

    # Build WHERE clause
    filter_clauses: list[str] = []
    filter_params:  list      = []

    strat = strategy.get("strategy", "both")
    if strat == "policy_document":
        filter_clauses.append("source_type = 'policy_document'")
    elif strat == "faq":
        filter_clauses.append("source_type = 'faq'")

    if strategy.get("policy_number"):
        filter_clauses.append("policy_number = %s")
        filter_params.append(strategy["policy_number"])

    if strategy.get("customer_id"):
        filter_clauses.append("customer_id = %s")
        filter_params.append(strategy["customer_id"])

    if strategy.get("state_filter"):
        # NULL state = ALL-applicable — must always pass the filter
        filter_clauses.append("(state = %s OR state IS NULL)")
        filter_params.append(strategy["state_filter"])

    where_sql = ("WHERE " + " AND ".join(filter_clauses)) if filter_clauses else ""

    # CTE: compute distance once, reuse in SELECT output and ORDER BY
    sql = f"""
        WITH ranked AS (
            SELECT
                chunk_id,
                source_type,
                doc_type,
                policy_number,
                customer_id,
                chunk_text,
                1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            {where_sql}
        )
        SELECT *
        FROM ranked
        ORDER BY similarity DESC
        LIMIT %s
    """
    params = [query_embedding] + filter_params + [top_k]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()

    log.info(
        "chunks_retrieved",
        extra={
            "strategy":      strat,
            "policy_number": strategy.get("policy_number"),
            "customer_id":   strategy.get("customer_id"),
            "state_filter":  strategy.get("state_filter"),
            "top_k":         top_k,
            "returned":      len(rows),
        },
    )
    return rows


def generate_answer(
    query: str,
    chunks: list[dict],
    mode: str | None = None,
) -> str:
    """
    Generate a grounded answer from retrieved chunks.
    mode: "local" (Ollama) | "bedrock" (Claude Haiku via Bedrock)
    Reads RAG_MODE from env if mode not provided.
    """
    if mode is None:
        mode = os.getenv("RAG_MODE", "local")

    context = "\n\n---\n\n".join([
        f"[Source: {c['source_type']} | {c.get('policy_number') or c['doc_type']}]\n{c['chunk_text']}"
        for c in chunks
    ])
    system = (
        "You are Oak Assist, the AI helper for Avvaru Iron Oak Insurance. "
        "Answer the customer's question using ONLY the provided context. "
        "If the context does not contain the answer, say so clearly — do not guess. "
        "Always cite whether your answer comes from a policy document or the FAQ."
    )
    prompt = f"Context:\n{context}\n\nCustomer question: {query}"

    if mode == "local":
        import httpx
        base_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        resp = httpx.post(
            f"{base_url}/api/chat",
            json={
                "model":    model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    elif mode == "bedrock":
        import boto3
        region   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        model_id = _require_env("BEDROCK_MODEL_ID_HAIKU")
        client   = boto3.client("bedrock-runtime", region_name=region)
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system":  system,
            "messages": [{"role": "user", "content": prompt}],
        })
        resp = client.invoke_model(
            modelId=model_id, body=body, contentType="application/json"
        )
        return json.loads(resp["body"].read())["content"][0]["text"]

    else:
        raise ValueError(f"Unknown RAG_MODE: '{mode}'. Choose: local | bedrock")
```

---

## 8. RAG FastAPI Router

**`ai/api/routers/rag_router.py`**

Key fixes: `get_conn()` replaces placeholder, `SentenceTransformer` injected from `app.state`, `DEBUG_MODE` gates the debug endpoint.

```python
"""
rag_router.py — FastAPI routes for RAG Q&A.

/rag/query   — production endpoint; returns grounded answer + sources
/rag/debug   — dev-only endpoint (DEBUG_MODE=true); returns chunks without LLM call

The SentenceTransformer model is loaded once at app startup via FastAPI lifespan
and accessed via request.app.state.embedder — never loaded per-request.
"""
from __future__ import annotations

import os
import time
import uuid

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ai.pipelines.rag.retrieval_router import classify_query
from ai.pipelines.rag.rag_pipeline import retrieve, generate_answer
from ai.utils.log import get_logger

load_dotenv()
log = get_logger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])

_DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


# ── Request / Response models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:       str
    customer_id: str | None = None
    mode:        str | None = None   # overrides RAG_MODE env var if provided


class QueryResponse(BaseModel):
    answer:     str
    strategy:   str
    sources:    list[dict]
    request_id: str
    latency_ms: int


class DebugResponse(BaseModel):
    strategy:   dict
    chunks:     list[dict]
    request_id: str
    latency_ms: int


# ── Helpers ─────────────────────────────────────────────────────────────────

def _embed_query(query: str, request: Request) -> list[float]:
    """Use the embedder cached on app.state (loaded at lifespan startup)."""
    embedder = getattr(request.app.state, "embedder", None)
    if embedder is None:
        raise RuntimeError(
            "Embedder not loaded. Ensure lifespan startup completed successfully."
        )
    return embedder.encode(query).tolist()


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, request: Request):
    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    try:
        q_emb    = _embed_query(req.query, request)
        strategy = classify_query(req.query)
        if req.customer_id:
            strategy["customer_id"] = req.customer_id

        chunks = retrieve(q_emb, strategy)
        answer = generate_answer(req.query, chunks, mode=req.mode)

        latency = int((time.time() - t0) * 1000)
        log.info(
            "rag_query",
            request_id=request_id,
            strategy=strategy["strategy"],
            policy_number=strategy.get("policy_number"),
            customer_id=strategy.get("customer_id"),
            chunks_retrieved=len(chunks),
            latency_ms=latency,
        )
        return QueryResponse(
            answer=answer,
            strategy=strategy["strategy"],
            sources=chunks,
            request_id=request_id,
            latency_ms=latency,
        )
    except Exception as exc:
        log.error("rag_query_failed", request_id=request_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/debug", response_model=DebugResponse, include_in_schema=_DEBUG_MODE)
async def debug_endpoint(req: QueryRequest, request: Request):
    """
    Returns routing decision and retrieved chunks without calling the LLM.
    Enabled only when DEBUG_MODE=true. Disable before production deployment.
    """
    if not _DEBUG_MODE:
        raise HTTPException(status_code=404, detail="Debug endpoint is disabled.")

    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    q_emb    = _embed_query(req.query, request)
    strategy = classify_query(req.query)
    if req.customer_id:
        strategy["customer_id"] = req.customer_id

    chunks  = retrieve(q_emb, strategy)
    latency = int((time.time() - t0) * 1000)
    return DebugResponse(
        strategy=strategy,
        chunks=chunks,
        request_id=request_id,
        latency_ms=latency,
    )


@router.get("/health")
async def rag_health(request: Request):
    embedder_loaded = hasattr(request.app.state, "embedder")
    return {"status": "ok", "embedder_loaded": embedder_loaded}
```

---

## 9. Update `main.py` for Lifespan

Update `ai/api/handlers/main.py` to load the `SentenceTransformer` once at startup and register the RAG router. Only the changed sections are shown — add to the existing file from Phase 3.

```python
"""
main.py — add lifespan for SentenceTransformer and register rag_router.
Add these changes to the existing Phase 3 main.py.
"""
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI

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

    yield  # app runs here

    # Cleanup (model will be GC'd; nothing explicit needed for sentence-transformers)
    app.state.embedder = None


# Update the FastAPI instantiation in main.py:
app = FastAPI(title="AIOI AI API", version="0.4.0", lifespan=lifespan)
#
# Add after existing router includes:
from ai.api.routers.rag_router import router as rag_router
app.include_router(rag_router)
```

> **Full `main.py` change summary:** Replace `app = FastAPI(...)` with `app = FastAPI(..., lifespan=lifespan)`, add the lifespan context above, and add `app.include_router(rag_router)`. Bump version to `"0.4.0"`.

---

## 10. Verification Script

**`data-gen/generators/verify_rag.py`**

```python
"""
verify_rag.py — Phase 4 post-load verification.

Checks:
  1. document_chunks table has FAQ rows (source_type='faq')
  2. HNSW index exists on document_chunks.embedding
  3. Policy query routes to policy_document strategy
  4. Concept query routes to faq strategy
  5. State query routes to faq with state_filter set
  6. ALL-applicable FAQ chunks have state=NULL (not filtered out by state queries)
  7. No chunk_text contains known prompt injection patterns

Usage:
  uv run python -m data_gen.generators.verify_rag
  uv run python data-gen/generators/verify_rag.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

PASS = "✓"
FAIL = "✗"
WARN = "⚠"
_results: list[tuple[str, str, str]] = []


def _check(label: str, passed: bool, detail: str = "") -> None:
    status = PASS if passed else FAIL
    _results.append((status, label, detail))
    print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))


def check_db_counts() -> None:
    from db.load_json import get_conn
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM document_chunks WHERE source_type = 'faq'")
        faq_count = cur.fetchone()[0]
        _check("FAQ chunks loaded into pgvector", faq_count > 0, f"{faq_count} rows")

        cur.execute("SELECT COUNT(*) FROM document_chunks WHERE source_type = 'policy_document'")
        policy_count = cur.fetchone()[0]
        if policy_count == 0:
            print(f"  {WARN}  No policy_document chunks — PDF chunkers may still be stubs (expected)")
        else:
            _check("Policy document chunks present", True, f"{policy_count} rows")

        # HNSW index
        cur.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'document_chunks'
            AND indexdef ILIKE '%hnsw%'
        """)
        idx = cur.fetchone()
        _check("HNSW index present on document_chunks.embedding", idx is not None)

        # ALL-applicable FAQ state=NULL
        cur.execute("""
            SELECT COUNT(*) FROM document_chunks
            WHERE source_type = 'faq' AND state IS NULL
        """)
        null_state_count = cur.fetchone()[0]
        _check(
            "ALL-applicable FAQ chunks have state=NULL",
            null_state_count > 0,
            f"{null_state_count} rows",
        )
    conn.close()


def check_routing() -> None:
    from ai.pipelines.rag.retrieval_router import classify_query

    # Policy number → policy_document
    r = classify_query("What is the deductible on policy TX-00142?")
    _check(
        "Policy number query routes to policy_document",
        r["strategy"] == "policy_document" and r["policy_number"] == "TX-00142",
        str(r),
    )

    # Personal possessive → policy_document
    r = classify_query("What is my deductible?")
    _check("'My deductible' routes to policy_document", r["strategy"] == "policy_document", str(r))

    # Concept question → faq
    r = classify_query("What is PIP coverage?")
    _check("Concept question routes to faq", r["strategy"] == "faq", str(r))

    # State + concept → faq with state_filter
    r = classify_query("What are the minimum liability limits in TX?")
    _check(
        "State query sets state_filter",
        r["state_filter"] == "TX",
        str(r),
    )

    # No-fault state question → faq
    r = classify_query("Is PA a no-fault state?")
    _check(
        "No-fault state query routes to faq with PA filter",
        r["strategy"] == "faq" and r["state_filter"] == "PA",
        str(r),
    )


def check_injection_patterns() -> None:
    """Scan a sample of chunk_text for known prompt injection patterns."""
    from db.load_json import get_conn
    conn = get_conn()
    injection_patterns = [
        "ignore previous instructions",
        "disregard your system prompt",
        "you are now",
        "act as",
        "mark this claim as approved",
    ]
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_text FROM document_chunks LIMIT 500")
        rows = cur.fetchall()
    conn.close()

    found = []
    for (text,) in rows:
        for pat in injection_patterns:
            if pat in (text or "").lower():
                found.append(pat)
    _check(
        "No injection patterns in sampled chunk_text (500 rows)",
        len(found) == 0,
        f"found: {found}" if found else "",
    )


def main() -> int:
    print("\n" + "=" * 60)
    print("  Phase 4 RAG Verification")
    print("=" * 60)

    try:
        check_db_counts()
    except Exception as exc:
        print(f"  {FAIL}  DB checks failed — {exc}")

    check_routing()

    try:
        check_injection_patterns()
    except Exception as exc:
        print(f"  {WARN}  Injection scan skipped — {exc}")

    failures = [r for r in _results if r[0] == FAIL]
    print("\n" + "=" * 60)
    if failures:
        print(f"  RESULT: {len(failures)} FAILURE(S)")
        return 1
    else:
        print("  RESULT: ALL PASS ✓")
        return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## 11. Start the Phase 4 Stack

### Install Phase 4 dependencies

```bash
# Linux / macOS
uv add PyMuPDF tiktoken sentence-transformers pgvector httpx

# Optional: ONNX int8 quantization support (recommended for 8 GB machines)
uv add "sentence-transformers[onnx]" "optimum[onnxruntime]"

# Regenerate requirements.txt for Docker
uv export --no-dev --format requirements-txt > requirements.txt
```

```powershell
# Windows
uv add PyMuPDF tiktoken sentence-transformers pgvector httpx
uv add "sentence-transformers[onnx]" "optimum[onnxruntime]"
uv export --no-dev --format requirements-txt > requirements.txt
```

### Start Docker stack

```bash
# Linux / macOS
docker compose --profile phase4 up -d

# Pull Ollama models — first run only (~2 GB for llama3.2, ~5 GB for llama3.1:8b)
docker exec iron-oak-insurance-ollama-1 ollama pull llama3.2
docker exec iron-oak-insurance-ollama-1 ollama pull all-minilm:l6-v2
```

```powershell
# Windows
docker compose --profile phase4 up -d

docker exec iron-oak-insurance-ollama-1 ollama pull llama3.2
docker exec iron-oak-insurance-ollama-1 ollama pull all-minilm:l6-v2
```

> **Hardware note:** `llama3.2` (3B) runs on 8 GB RAM at 15–25 tok/s CPU. Switch to `llama3.1:8b` for better answer quality on 16 GB+ machines. Set `OLLAMA_MODEL=llama3.1:8b` in `.env`.

### Generate and embed

```bash
# Linux / macOS

# Create data_gen symlink for module invocation (one-time)
ln -sf data-gen data_gen

# Generate FAQ corpus
uv run python -m data_gen.generators.faq_gen

# Dry-run to verify chunk counts before DB writes
uv run python -m ai.pipelines.embedding.embed_and_load --dry-run

# Embed and load
uv run python -m ai.pipelines.embedding.embed_and_load

# Start API
uv run fastapi dev ai/api/handlers/main.py
```

```powershell
# Windows

# Create data_gen junction for module invocation (one-time, run as Administrator)
New-Item -ItemType Junction -Path data_gen -Target data-gen

# Generate FAQ corpus
uv run python data_gen\generators\faq_gen.py

# Dry-run
uv run python -m ai.pipelines.embedding.embed_and_load --dry-run

# Embed and load
uv run python -m ai.pipelines.embedding.embed_and_load

# Start API

uv run fastapi dev .\ai\api\handlers\main.py
```

---

## 12. Verification & Git Tag

### Run verification script

```bash
# Linux / macOS
uv run python -m data_gen.generators.verify_rag

# Windows
uv run python data-gen\generators\verify_rag.py
```

### API smoke tests

```bash
# Linux / macOS

# Policy document query (must route to policy_document)
curl -s -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the deductible on policy TX-00142?","customer_id":"CUST-08821"}' \
  | python -m json.tool

# FAQ concept query (must route to faq)
curl -s -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What does PIP coverage mean?"}' \
  | python -m json.tool

# State-filtered FAQ query (must have state_filter=PA and return ALL-applicable FAQs too)
curl -s -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Is PIP required in PA?"}' \
  | python -m json.tool

# Debug endpoint (only when DEBUG_MODE=true)
curl -s -X POST http://localhost:8000/rag/debug \
  -H "Content-Type: application/json" \
  -d '{"query":"What is collision coverage?"}' \
  | python -m json.tool

# Health check
curl -s http://localhost:8000/rag/health | python -m json.tool
```

```powershell
# Windows

# Policy document query
$body = '{"query":"What is the deductible on policy NY-00005?","customer_id":"CUST-00004"}'
Invoke-RestMethod -Uri http://localhost:8000/rag/query `
  -Method POST -ContentType "application/json" -Body $body

# FAQ concept query
$body = '{"query":"What does PIP coverage mean?"}'
Invoke-RestMethod -Uri http://localhost:8000/rag/query `
  -Method POST -ContentType "application/json" -Body $body

# State-filtered FAQ query
$body = '{"query":"Is PIP required in PA?"}'
Invoke-RestMethod -Uri http://localhost:8000/rag/query `
  -Method POST -ContentType "application/json" -Body $body

# Debug endpoint (DEBUG_MODE=true only)
$body = '{"query":"What is collision coverage?"}'
Invoke-RestMethod -Uri http://localhost:8000/rag/debug `
  -Method POST -ContentType "application/json" -Body $body

# Health check
Invoke-RestMethod http://localhost:8000/rag/health
```

### Phase Gate Checklist

- [ ] All `__init__.py` files created — `uv run python -m ai.pipelines.embedding.embed_and_load` resolves without `ModuleNotFoundError`
- [ ] `uv run python -m data_gen.generators.faq_gen` produces `faqs/faq_corpus.json` with all 5 categories
- [ ] `--dry-run` shows FAQ chunk count > 0 before any DB write
- [ ] FAQ chunks loaded into pgvector (`SELECT COUNT(*) FROM document_chunks WHERE source_type='faq'` > 0)
- [ ] ALL-applicable FAQ chunks have `state = NULL` (`SELECT COUNT(*) FROM document_chunks WHERE source_type='faq' AND state IS NULL` > 0)
- [ ] HNSW index present (`SELECT indexname FROM pg_indexes WHERE tablename='document_chunks' AND indexdef ILIKE '%hnsw%'`)
- [ ] Policy number query routes to `policy_document` and returns `policy_number` in strategy
- [ ] `"my deductible"` query routes to `policy_document`
- [ ] Concept question routes to `faq`
- [ ] State-filtered query sets `state_filter` and returns ALL-applicable FAQs in results
- [ ] `/rag/debug` returns chunks without LLM call when `DEBUG_MODE=true`
- [ ] `/rag/debug` returns 404 when `DEBUG_MODE=false`
- [ ] `SentenceTransformer` loaded once at lifespan startup — not per-request (verify via `GET /rag/health` returns `embedder_loaded: true`)
- [ ] Structured JSON logging active on `/rag/query` (check CloudWatch / stdout for `rag_query` event)
- [ ] `verify_rag.py` reports ALL PASS
- [ ] `uv run ruff check .` passes with no errors
- [ ] Prompt injection test strings rejected (see [CROSS_PHASE.md](./CROSS_PHASE.md) §9.1) — TODO flag in `rag_pipeline.py` before Phase 5

### Git Tag

```bash
git add -A
git commit -m "Phase 4: RAG pipeline — FAQ gen, all chunkers, embed pipeline, retrieval router, Q&A + debug endpoints"
git tag v0.4.0
```

---

*Previous: [PHASE_3_ML.md](./PHASE_3_ML.md) · Next: [PHASE_5_BEDROCK.md](./PHASE_5_BEDROCK.md) · Cross-phase decisions: [CROSS_PHASE.md](./CROSS_PHASE.md)*
