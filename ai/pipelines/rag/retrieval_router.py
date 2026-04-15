"""
retrieval_router.py — determines retrieval strategy from the query string.

Priority rules (applied in order):
  1. Contains policy number pattern → policy_document first, FAQ fallback
  2. Contains customer ID or "my policy" → policy_document first
  3. General concept signal ("what is","how does","what happens") → faq first
  4. State keyword present → filter FAQ by applicable_states
  5. Ambiguous → search both, rank by similarity, label source in response
"""
import re

POLICY_PATTERN   = re.compile(r"\b[A-Z]{2}-\d{5}\b")
CUSTOMER_PATTERN = re.compile(r"\bCUST-\d+\b", re.IGNORECASE)
CONCEPT_SIGNALS  = ["what is", "how does", "what does", "how do i", "what happens", "explain", "define"]
US_STATES        = {"AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
                    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
                    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
                    "VA","WA","WV","WI","WY","DC"}

def classify_query(query: str) -> dict:
    q = query.lower()
    result = {
        "strategy":      "both",      # policy_document | faq | both
        "state_filter":  None,
        "policy_number": None,
        "customer_id":   None,
    }

    pm = POLICY_PATTERN.search(query)
    if pm:
        result["strategy"]      = "policy_document"
        result["policy_number"] = pm.group()
        return result

    cm = CUSTOMER_PATTERN.search(query)
    if cm or "my policy" in q or "my deductible" in q or "my coverage" in q:
        result["strategy"]    = "policy_document"
        result["customer_id"] = cm.group() if cm else None
        return result

    if any(sig in q for sig in CONCEPT_SIGNALS):
        result["strategy"] = "faq"

    # State filter — check for state abbreviation
    for st in US_STATES:
        if f" {st.lower()} " in f" {q} ":
            result["state_filter"] = st
            break

    return result