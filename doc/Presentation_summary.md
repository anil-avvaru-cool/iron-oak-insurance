# AIOI Meetup — 20-Minute Technical Talk

Here's a tight agenda covering the platform architecture, one live demo path, and one ROI story that will land with a mixed technical/business audience.

---

## Opening Hook (2 min)

**"What does ~$210K in fraud exposure look like in a database?"**

Pull up a single claims query live:

```sql
SELECT is_fraud, COUNT(*), ROUND(AVG(claim_amount),0) AS avg_amount
FROM claims
GROUP BY is_fraud;
```

Show the numbers. Then say: *"We're going to build the system that catches this — and I'll show you the economics at the end."*

---

## The 5-Layer Platform (3 min)

Draw or show the stack — one sentence per layer:

| Layer | What It Does | Cost |
|---|---|---|
| XGBoost Fraud Model | Scores every claim on arrival | $0/mo (Lambda) |
| XGBoost Risk + Churn | Prices risk, flags retention | $0/mo (Lambda) |
| RAG Policy Q&A | Answers "what's my deductible?" from actual PDFs | ~$59/mo |
| Fraud Agent (Sonnet) | Turns a score into an investigation brief | ~$162/mo |
| Oak Assist FNOL | Handles claim intake end-to-end | ~$470/mo |

**Key message:** *Layers 1 and 2 are essentially free to run. The entire platform at full deployment costs under $800/month.*

---

## Live Demo Path (10 min)

Run these in order — each one builds on the previous.

**Step 1 — Generate the data (1 min)**
```bash
uv run python data_gen/generators/run_all.py --customers 200 --no-pdfs
```
Open `data/claims.json` — show a fraud record side by side with a clean one. Point out `fraud_signals` array.

**Step 2 — Load to Postgres and query (2 min)**
```bash
docker compose up -d
uv run python db/load_json.py
```
```sql
-- Show state distribution
SELECT state, COUNT(*) FROM policies GROUP BY state ORDER BY count DESC LIMIT 5;

-- Show the fraud signals in raw form
SELECT claim_id, fraud_signals FROM claims WHERE is_fraud = true LIMIT 3;
```

**Step 3 — Train the fraud model and score live (3 min)**
```bash
uv run python -m ai.models.fraud_detection.model
```
Show the ROC-AUC. Then hit the API:
```bash
curl -X POST http://localhost:8000/models/fraud/score \
  -H "Content-Type: application/json" \
  -d '{"claims": [{"claim_id":"CLM-00001","claim_amount":12000,
       "days_to_file":1,"fraud_signal_count":3,"avg_drive_score":28,
       "state":"TX","claim_type":"collision","customer_claim_count":4,
       "claim_to_premium_ratio":3.2,"hard_brakes_90d":45,"vehicle_make":"Toyota",
       "zip_prefix":"750"}]}'
```
Point at `fraud_score: 0.91`. *"That's the XGBoost classifier. Now watch what happens when we hand this to Claude."*

**Step 4 — RAG demo, two queries back to back (4 min)**

Run the FAQ query first — it routes to the general knowledge layer:
```bash
curl -X POST http://localhost:8000/rag/query \
  -d '{"query":"What does PIP mean?"}'
```
Show the `strategy: faq` in the response.

Then run a policy-specific query — it routes to the customer's document:
```bash
curl -X POST http://localhost:8000/rag/query \
  -d '{"query":"What is the deductible on policy TX-00142?","customer_id":"CUST-08821"}'
```
Show `strategy: policy_document`. *"Same question surface — completely different retrieval path. This is what prevents hallucination."*

---

## The ROI Story: Fraud Detection (3 min)

Keep this concrete. Use the numbers from the executive brief.

**The baseline:**
- 85,000 claims/year at $8,200 average cost
- Industry fraud rate: ~3.5% — that's **$24.4M in annual fraud exposure**
- Manual SIU detection catches about 12% of it: **$2.9M recovered**

**With the XGBoost model:**
- Detection rate goes from 12% → 38%
- Recovered fraud: **$9.3M/year**
- Infrastructure cost for this capability: **$0/month** (Lambda)
- Net new recovery in Year 1: **+$6.4M**

**The punchline:**

> *"This single model — which runs for free on AWS Lambda — pays back the entire platform investment in under 60 days. Everything else: the risk model, the RAG pipeline, the FNOL agent — that's upside on top of a cost already covered."*

Put the payback math on screen if you have a slide:

| Investment | Return | Payback |
|---|---|---|
| ~$80K engineering (Phases 1–3) | $6.4M Year 1 fraud recovery | < 60 days |

---

## What's Next / Call to Action (2 min)

Three things to leave them with:

1. **Phase 4 is where the LLMs earn their keep** — RAG turns static PDFs into a live Q&A system. The chunking strategy (section-aware for declaration pages, paragraph for claim letters) is the difference between useful retrieval and confident wrong answers.

2. **The minimum viable deployment is 4 capabilities at $121/month** — fraud detection, risk scoring, churn prediction, and policy Q&A. That covers the vast majority of the 3-year value at a fraction of the full cost.

3. **The data platform is the moat** — every capability shares the same Postgres + pgvector instance. Telematics data improves fraud scores. Fraud signals inform risk pricing. Churn signals trigger Oak Assist retention flows. It compounds.

---

## One Slide If You Need It

```
┌─────────────────────────────────────────────────────┐
│  $24.4M fraud exposure  →  $9.3M recovered          │
│  Infrastructure cost:       $0/month                │
│  Payback period:            < 60 days               │
│                                                     │
│  Full platform (5 capabilities): ~$755/month        │
│  3-year net benefit:             $55.5M             │
└─────────────────────────────────────────────────────┘
```

---

**Time check:** Hook (2) + Stack (3) + Demo (10) + ROI (3) + Next (2) = 20 minutes exactly, with zero slide transitions in the demo section.