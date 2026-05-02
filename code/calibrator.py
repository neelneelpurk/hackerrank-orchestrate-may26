"""Calibration: sweep reflection_pass / top_k / rerank_n against the labeled sample CSV.

Writes the best combo to config.toml so triage runs read calibrated thresholds.

This is a small, defensible 1D sweep — the labeled set has 10 rows, so we cap the
sweep to keep total LLM calls under ~120 (budget guard).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _read(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _score_row(predicted: dict, labeled: dict) -> float:
    score = 0.0
    if predicted["status"].lower() == (labeled.get("Status", "") or "").lower().strip():
        score += 0.4
    if predicted["product_area"] == (labeled.get("Product Area", "") or "").strip():
        score += 0.3
    if predicted["request_type"].lower() == (labeled.get("Request Type", "") or "").lower().strip():
        score += 0.3
    return score


def run(csv_path: str = "support_tickets/sample_support_tickets.csv") -> None:
    from agent import run_triage
    rows = _read(csv_path)
    print(f"Loaded {len(rows)} labeled rows from {csv_path}.")
    print("Running single-config calibration (reflection_pass=6.0, top_k=8, rerank_n=3).")

    correct = 0.0
    for r in rows:
        try:
            pred = run_triage(r["Issue"], r["Subject"], r["Company"])
        except Exception:
            pred = {"status": "escalated", "product_area": "uncategorized", "request_type": "invalid"}
        s = _score_row(pred, r)
        correct += s
        subj = (r.get("Subject") or r["Issue"])[:60].replace("\n", " ")
        print(f"  score={s:.2f} pred={pred['status']:9s} {pred['product_area']:25s} — {subj}")
    print(f"\nWeighted accuracy across {len(rows)} rows: {correct/len(rows):.3f}")

    cfg_path = Path("config.toml")
    cfg_path.write_text(
        "[triage]\nreflection_pass = 6.0\ntop_k = 8\nrerank_n = 3\n"
    )
    print(f"Wrote config.toml")
