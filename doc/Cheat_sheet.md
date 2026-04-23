## Important commands
uv run python data_gen/generators/run_all.py --customers 10000 --fraud-rate 0.04
uv run python data_gen/generators/run_all.py --customers 10000 --fraud-rate 0.04 --no-pdfs

docker compose up -d postgres
docker compose up -d ollama

uv run python db/load_json.py --truncate
docker exec iron-oak-insurance-postgres-1 psql -U aioi -d aioi -c "SELECT is_fraud, COUNT(*), ROUND(AVG(claim_amount),0) AS avg_amount FROM claims GROUP BY is_fraud;" > sqlOut.txt

**Train all three models**
```bash
uv run python -m ai.models.fraud_detection.model
uv run python -m ai.models.risk_scoring.model
uv run python -m ai.models.churn_prediction.model
```

# Health check
curl http://localhost:8000/health

# Fraud score
# Positive example
curl -s -X POST http://localhost:8000/models/fraud/score `
  -H "Content-Type: application/json" `
  -d '{
  "claims": [{
    "claim_id": "CLM-99001",
    "claim_amount": 18500,
    "claim_to_premium_ratio": 4.8,
    "days_to_file": 0,
    "customer_claim_count": 5,
    "avg_drive_score": 22,
    "hard_brakes_90d": 67,
    "state": "FL",
    "vehicle_make": "BMW",
    "zip_prefix": "331",
    "claim_type": "collision",
    "fraud_signal_count": 4
  }]
}' | python -m json.tool

# Neutral example:
curl -s -X POST http://localhost:8000/models/fraud/score `
  -H "Content-Type: application/json" `
  -d '{
  "claims": [{
    "claim_id": "CLM-99003",
    "claim_amount": 6200,
    "claim_to_premium_ratio": 1.8,
    "days_to_file": 3,
    "customer_claim_count": 2,
    "avg_drive_score": 54,
    "hard_brakes_90d": 18,
    "state": "TX",
    "vehicle_make": "Ford",
    "zip_prefix": "750",
    "claim_type": "collision",
    "fraud_signal_count": 1
  }]
}' | python -m json.tool

# Negative example
curl -s -X POST http://localhost:8000/models/fraud/score `
  -H "Content-Type: application/json" `
  -d '{
  "claims": [{
    "claim_id": "CLM-99002",
    "claim_amount": 1800,
    "claim_to_premium_ratio": 0.6,
    "days_to_file": 5,
    "customer_claim_count": 1,
    "avg_drive_score": 84,
    "hard_brakes_90d": 3,
    "state": "VT",
    "vehicle_make": "Toyota",
    "zip_prefix": "054",
    "claim_type": "comprehensive",
    "fraud_signal_count": 0
  }]
}' | python -m json.tool


