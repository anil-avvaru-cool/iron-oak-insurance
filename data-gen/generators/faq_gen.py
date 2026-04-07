"""
faq_gen.py — generates FAQ corpus from states.json and coverage_rules.json.

Phase 4 generator. This stub is created in Phase 1 so run_all.py can reference it;
full implementation is in PHASE_4_RAG.md.

Each FAQ is a discrete Q&A pair:
    - State-specific FAQs: applicable_states = [state_code]
    - Generic FAQs: applicable_states = ["ALL"]
"""
from __future__ import annotations

import json
from pathlib import Path


def main(output_path: Path, config: dict, states_data: dict) -> None:
    """Phase 4 stub — full implementation in Phase 4 build."""
    # Minimal placeholder so run_all.py can call this without error
    print(f"[faq_gen] Phase 4 stub — skipping FAQ generation (implement in Phase 4)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        with open(output_path, "w") as f:
            json.dump([], f)


if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "config"
    states_data = json.loads((config_dir / "states.json").read_text())
    coverage_rules = json.loads((config_dir / "coverage_rules.json").read_text())
    main(
        output_path=Path("faqs/faq_corpus.json"),
        config={"coverage_rules": coverage_rules},
        states_data=states_data,
    )
