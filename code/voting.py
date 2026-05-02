"""Decide the canonical product_area + company from retrieved evidence.

Two strategies, picked dynamically:

1. **Score-weighted vote** across all evidence chunks. Robust when many chunks
   from the same product_area appear — they reinforce each other.

2. **Top-1 chunk fallback** when the vote signal is weak. Happens when the top
   chunk's product_area only appears once in the candidate set; in that case
   the weighted vote can be drowned out by 4 mediocre chunks from a different
   area, even though the top hit is the right answer.

Threshold tuned so that "only one strong chunk says travel_support" still wins
over "four medium chunks say general_support".
"""

from __future__ import annotations

import os
from collections import Counter


# If the top-1 chunk's score is more than this multiple of the second-place
# chunk's score, override the weighted vote and use the top-1 product_area.
TOP1_OVERRIDE_RATIO = float(os.environ.get("VOTING_TOP1_RATIO", "1.5"))


def weighted_product_area(evidence_by_step: dict) -> tuple[str, str]:
    """Return (product_area, company).

    Strategy: weighted vote, then check if the top-1 chunk should override
    based on its score vs the rest. Falls back to ("uncategorized", "") only
    when there is literally no evidence.
    """
    pa_votes: Counter = Counter()
    co_votes: Counter = Counter()
    flat: list[dict] = []

    for step_id, pkg in evidence_by_step.items():
        for ev in pkg.get("evidence", []):
            md = ev["metadata"]
            score = float(ev.get("rerank_score") or ev.get("score") or 1.0)
            pa_votes[md["product_area"]] += score
            co_votes[md["company"]] += score
            flat.append({
                "score": score,
                "product_area": md["product_area"],
                "company": md["company"],
            })

    if not flat:
        return ("uncategorized", "")

    flat.sort(key=lambda x: x["score"], reverse=True)
    top = flat[0]
    runner_up_score = flat[1]["score"] if len(flat) > 1 else 0.0

    # Top-1 override: if the top chunk dominates by ratio, use its area + company
    # even if the weighted vote disagrees.
    if runner_up_score == 0.0 or top["score"] >= runner_up_score * TOP1_OVERRIDE_RATIO:
        return (top["product_area"], top["company"])

    return (
        pa_votes.most_common(1)[0][0],
        co_votes.most_common(1)[0][0],
    )
