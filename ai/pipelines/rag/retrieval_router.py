"""
retrieval_router.py — classifies a query and returns the retrieval strategy.

Routing logic (applied in order):

  1. Structural pre-filters (regex — unambiguous patterns):
       a. Contains policy number pattern (XX-NNNNN)  → policy_document
       b. Contains customer ID (CUST-NNNNN)          → policy_document
       c. Contains personal possessive phrases        → policy_document
            ("my policy", "my deductible", etc.)

  2. Semantic routing (anchor-based cosine similarity):
       - Embed the query using the same all-MiniLM-L6-v2 model used for retrieval.
       - Compare against pre-computed centroids for "faq" and "policy_document" anchors.
       - If cosine similarity gap >= CONFIDENCE_THRESHOLD → pick winner.
       - Otherwise → "both" (ambiguous, search everything).

  Why semantic over keyword:
       "Is PIP required in PA?" has no keyword signal but is semantically
       close to FAQ anchors about coverage concepts and state rules.
       Keyword routing would fall through to "both"; semantic routing
       correctly routes it to "faq".

  State filter:
       Applied as a post-step to all strategies.
       Used in retrieve() as: (state = %s OR state IS NULL)
       so ALL-applicable FAQs always appear alongside state-specific ones.

  Anchor embeddings:
       Computed once at module load and cached as numpy arrays.
       Each centroid is the mean of its anchor embeddings.
       No network calls — uses the local sentence-transformers model.

  Embedder injection:
       Call set_embedder(model) from FastAPI lifespan after the
       SentenceTransformer is loaded, so this module never loads
       its own copy of the model.

       from ai.pipelines.rag.retrieval_router import set_embedder
       set_embedder(app.state.embedder)

       If set_embedder() is never called, the router loads its own
       instance on first use (safe for standalone testing).
"""
from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING
import os
from dotenv import load_dotenv
import logging

import numpy as np

load_dotenv()

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Structural pre-filter patterns
# ---------------------------------------------------------------------------

_POLICY_PATTERN   = re.compile(r"\b([A-Z]{2}-\d{5})\b")
_CUSTOMER_PATTERN = re.compile(r"\b(CUST-\d+)\b", re.IGNORECASE)

_PERSONAL_SIGNALS = [
    "my policy", "my deductible", "my coverage", "my claim",
    "my premium", "my vehicle", "my car", "my renewal",
]

_US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC",
}

# Common English words that are also state codes — require stronger signal to match
_AMBIGUOUS_STATES = {"IN", "OR", "ME", "MA", "OH", "OK", "HI", "ID", "LA", "DE", "AL"}

# ---------------------------------------------------------------------------
# Semantic routing — anchor sentences
# ---------------------------------------------------------------------------

# These represent the *meaning space* of each route.
# More anchors = more stable centroid. Keep them diverse across subcategories.
# Do not overlap with the other route's anchors or the centroid will drift.

_FAQ_ANCHORS: list[str] = [
    # coverage concepts
    "What is Personal Injury Protection coverage?",
    "What does comprehensive insurance cover?",
    "How does collision coverage work?",
    "What is gap insurance and when do I need it?",
    "What is uninsured motorist coverage?",
    "Explain liability insurance limits.",
    "What does roadside assistance include?",
    # state rules
    "Is PIP required in my state?",
    "Which states are no-fault states?",
    "What are the minimum liability limits in Texas?",
    "At what damage percentage is a car totaled in Florida?",
    "Is uninsured motorist coverage mandatory in New York?",
    "Does Pennsylvania require personal injury protection?",
    # claims process
    "How do I file a claim after an accident?",
    "What happens after I report a claim?",
    "What documents do I need to support my claim?",
    "How long does the claims process take?",
    "Does my policy cover a rental car while mine is repaired?",
    # costs and discounts
    "What factors affect my insurance premium?",
    "How does the Iron Oak Drive Score affect my rate?",
    "What discounts does AIOI offer?",
    "How is my premium calculated?",
    "Does my credit score affect my insurance rate?",
    # policy management
    "How do I add a car to my policy?",
    "What happens if I miss a payment?",
    "What is an SR-22 and when is it required?",
    "Can I change my coverage mid-term?",
]

_POLICY_DOC_ANCHORS: list[str] = [
    # deductible and limit lookups — always tied to a specific policy/claim
    "What is the deductible on policy TX-00142?",
    "What are the coverage limits on this specific policy?",
    "Show me the collision deductible for this vehicle.",
    "What is the liability limit on my current policy?",
    "What coverages are included in policy NY-00891?",
    # claim record lookups
    "What is the settlement amount on claim CLM-00123?",
    "What did the adjuster note on claim CLM-00456?",
    "Has this claim been approved or denied?",
    "What is the status of claim CLM-00789?",
    # declaration page fields — specific to one document
    "What is the effective date of policy FL-00234?",
    "What vehicle is listed on this declaration page?",
    "What is the annual premium on this policy?",
    "What is the VIN on the policy?",
    "Who is the named insured on this policy?",
]

# Confidence threshold: if |faq_sim - policy_sim| < threshold → "both"
# Lower = more decisive routing; higher = more "both" fallback.
# 0.08 is a good starting point — tune if you see misroutes in production.
CONFIDENCE_THRESHOLD: float = 0.05

# ---------------------------------------------------------------------------
# Embedder management
# ---------------------------------------------------------------------------

_embedder: "SentenceTransformer | None" = None
_embedder_lock = threading.Lock()

# Pre-computed centroids (set once after embedder is available)
_faq_centroid: np.ndarray | None = None
_policy_centroid: np.ndarray | None = None
_centroids_ready = False


def set_embedder(model: "SentenceTransformer") -> None:
    """
    Inject the shared SentenceTransformer instance from FastAPI lifespan.
    Call this once at startup — do not call per request.

    Example (in main.py lifespan):
        from ai.pipelines.rag.retrieval_router import set_embedder
        set_embedder(app.state.embedder)
    """
    global _embedder
    _embedder = model
    _build_centroids()


def _get_embedder() -> "SentenceTransformer":
    """Return the shared embedder, loading one if set_embedder() was never called."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer
                model_name = os.getenv("EMBED_MODEL_LOCAL")
                if model_name is None:
                    raise RuntimeError("EMBED_MODEL_LOCAL not set in .env")                
                logging.getLogger("sentence_transformers.models.Transformer").setLevel(logging.ERROR)
                _embedder = SentenceTransformer(model_name)
                print(f" Warning! Loaded embedder model '{model_name}' in retrieval_router, check if set_embedder() is being called properly to avoid duplicate loads.")
                _build_centroids()
    return _embedder


def _build_centroids() -> None:
    """Embed anchor sentences and compute per-route centroids. Called once."""
    global _faq_centroid, _policy_centroid, _centroids_ready
    model = _embedder
    if model is None:
        return
    faq_embs    = model.encode(_FAQ_ANCHORS,        normalize_embeddings=True, show_progress_bar=False)
    policy_embs = model.encode(_POLICY_DOC_ANCHORS, normalize_embeddings=True, show_progress_bar=False)
    _faq_centroid    = np.mean(faq_embs,    axis=0)
    _policy_centroid = np.mean(policy_embs, axis=0)
    # Re-normalize centroids to unit length for consistent cosine sim
    _faq_centroid    /= np.linalg.norm(_faq_centroid)
    _policy_centroid /= np.linalg.norm(_policy_centroid)
    _centroids_ready = True


def _semantic_strategy(query: str) -> str:
    """
    Embed query, compare to faq/policy centroids, return strategy string.
    Returns "faq", "policy_document", or "both".
    """
    if not _centroids_ready:
        _get_embedder()  # triggers _build_centroids if needed

    model = _get_embedder()
    q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]

    faq_sim    = float(np.dot(q_emb, _faq_centroid))
    policy_sim = float(np.dot(q_emb, _policy_centroid))
    gap        = faq_sim - policy_sim
    print(f"faq_sim={faq_sim:.4f}  policy_sim={policy_sim:.4f}  gap={gap:.4f}")

    if gap >= CONFIDENCE_THRESHOLD:
        return "faq"
    elif gap <= -CONFIDENCE_THRESHOLD:
        return "policy_document"
    else:
        return "both"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify_query(query: str) -> dict:
    """
    Classify a query and return retrieval routing metadata.

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

    # --- Priority 1: explicit policy number (structural) ---
    pm = _POLICY_PATTERN.search(query)
    if pm:
        result["strategy"]      = "policy_document"
        result["policy_number"] = pm.group(1)
        _apply_state_filter(query, result)
        return result

    # --- Priority 2: customer ID or personal possessives (structural) ---
    cm = _CUSTOMER_PATTERN.search(query)
    if cm or any(sig in q_lower for sig in _PERSONAL_SIGNALS):
        result["strategy"]    = "policy_document"
        result["customer_id"] = cm.group(1) if cm else None
        _apply_state_filter(query, result)
        return result

    # --- Priority 3: semantic routing ---
    result["strategy"] = _semantic_strategy(query)

    # --- State filter (applied to all strategies) ---
    _apply_state_filter(query, result)
    return result


def _apply_state_filter(query: str, result: dict) -> None:
    """
    Detect a US state abbreviation and set state_filter.
    
    Collects ALL matching state codes, then resolves ambiguity:
    - Unambiguous codes (TX, FL, PA, etc.) win immediately.
    - Ambiguous codes (IN, OR, ME, etc.) are only accepted if no
      unambiguous code was found.
    - If multiple unambiguous codes found, picks the last one
      (state codes typically appear at the end of a query).
    """
    q_upper = query.upper()
    
    unambiguous: list[str] = []
    ambiguous: list[str]   = []
    
    for state in _US_STATES:
        if re.search(rf"\b{state}\b", q_upper):
            if state in _AMBIGUOUS_STATES:
                ambiguous.append(state)
            else:
                unambiguous.append(state)
    
    if unambiguous:
        # Multiple unambiguous matches are rare but possible ("TX and FL rates")
        # Pick the last one by position in the original query
        result["state_filter"] = max(
            unambiguous,
            key=lambda s: q_upper.rfind(s)
        )
    elif ambiguous:
        # Only accept an ambiguous match if it appears in a context that
        # looks like a state reference — preceded or followed by a state-like signal
        # For now: accept single ambiguous match, reject if multiple (too noisy)
        if len(ambiguous) == 1:
            result["state_filter"] = ambiguous[0]
        # else: too ambiguous, leave state_filter as None