@AGENTS.md

## Project Overview

HackerRank Orchestrate May 2026 hackathon — build a terminal-based support triage agent.
Challenge ends: **2026-05-02 11:00 AM IST** (`2026-05-02T11:00:00+05:30`)

## Task

For each row in `support_tickets/support_tickets.csv`, produce:

| Output field | Allowed values |
|---|---|
| `status` | `replied` \| `escalated` |
| `product_area` | support category inferred from corpus |
| `response` | grounded answer from `data/` only |
| `justification` | concise routing rationale |
| `request_type` | `product_issue` \| `feature_request` \| `bug` \| `invalid` |

## Key Files

- `support_tickets/support_tickets.csv` — 56 test inputs (no labels)
- `support_tickets/sample_support_tickets.csv` — 108 labeled examples (use for few-shot / validation)
- `support_tickets/output.csv` — write predictions here (must match sample schema)
- `code/main.py` — entry point (do not rename; evaluator calls this)
- `code/explore.py` — knowledge base exploration script; run `python code/explore.py`
- `data_exploration_report.md` — generated knowledge base report (run explore.py)

## Knowledge Base (`data/`)

| Company | Size | Files | Top-level product areas |
|---------|------|-------|------------------------|
| HackerRank | ~4.6 MB | ~438 MD | chakra, engage, general-help, hackerrank_community, integrations, interviews, library, screen, settings, skillup, uncategorized |
| Claude | ~2.3 MB | ~322 MD | amazon-bedrock, claude, claude-api-and-console, claude-code, claude-desktop, claude-for-education, claude-for-government, claude-for-nonprofits, claude-in-chrome, claude-mobile-apps, connectors, identity-management-sso-jit-scim, privacy-and-legal, pro-and-max-plans, safeguards, team-and-enterprise-plans |
| Visa | ~92 KB | ~14 MD | support/consumer, support/consumer/travel-support, support/small-business |

## Evaluation (4 dimensions)

1. **Agent Design** — architecture clarity, corpus grounding, escalation logic, determinism, hygiene
2. **AI Judge Interview** — design decisions, trade-offs, failure modes, AI vs human authorship
3. **Output CSV accuracy** — all 5 columns scored per row (no hallucination; escalate when uncertain)
4. **AI Fluency** — quality of `~/hackerrank_orchestrate/log.txt` chat transcript

## Hard Constraints

- Terminal-based only
- Answers must be grounded in `data/` corpus — no outside knowledge or hallucinated policies
- Escalate (`status: escalated`) when uncertain, high-risk, sensitive, or out-of-scope
- Secrets via env vars only — copy `.env.example` → `.env`, never commit `.env`
- Every turn must be appended to `~/hackerrank_orchestrate/log.txt` (see AGENTS.md §5)
