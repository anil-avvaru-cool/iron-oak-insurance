- For Policy specific question, no chunks retrieved, how to troubleshoot?

Here is the request and response:

 $body = '{"query":"What is the deductible on policy TX-00142?","customer_id":"CUST-08821"}'
(iron-oak-insurance) PS C:\src\iron-oak-insurance> Invoke-RestMethod -Uri http://localhost:8000/rag/query `
  -Method POST -ContentType "application/json" -Body $body

answer     : I don't have access to specific information about Policy TX-00142. The deductibles for policies are typically found in the policy document itself, but I do not have the ability to retrieve or 
             display this information. Could you please refer to your policy document or contact Avvaru Iron Oak Insurance directly for more information?
strategy   : policy_document
sources    : {}
request_id : f4597130
latency_ms : 21064

The strategy is routing correctly to `policy_document`, but `sources` is empty — meaning the retrieval query returned zero chunks. Let's diagnose systematically.

**Step 1 — Check if any policy_document chunks exist at all:**

```sql
SELECT source_type, COUNT(*) FROM document_chunks GROUP BY source_type;
```

Run this in psql:
```powershell
docker exec -it iron-oak-insurance-postgres-1 psql -U aioi -d aioi -c "SELECT source_type, COUNT(*) FROM document_chunks GROUP BY source_type;"
```

**Step 2 — Check if TX-00142 specifically has chunks:**

```powershell
docker exec -it iron-oak-insurance-postgres-1 psql -U aioi -d aioi -c "SELECT chunk_id, source_type, policy_number, customer_id, section FROM document_chunks WHERE policy_number = 'TX-00142' LIMIT 10;"
```

**Step 3 — Check if CUST-08821 has chunks:**

```powershell
docker exec -it iron-oak-insurance-postgres-1 psql -U aioi -d aioi -c "SELECT chunk_id, policy_number, customer_id FROM document_chunks WHERE customer_id = 'CUST-08821' LIMIT 10;"
```

**Step 4 — Check if the embedding column is populated (not null):**

```powershell
docker exec -it iron-oak-insurance-postgres-1 psql -U aioi -d aioi -c "SELECT COUNT(*) as total, COUNT(embedding) as has_embedding FROM document_chunks WHERE source_type = 'policy_document';"
```
