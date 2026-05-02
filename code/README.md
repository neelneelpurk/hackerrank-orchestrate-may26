# Support Triage Agent — User Guide

A terminal-based support triage agent for the HackerRank Orchestrate (May 2026) hackathon.
Implements the **ReWoo** pattern (Planner → parallel retrieval Workers → Solver → Reflector loop)
wrapped by a PreFlight stage and a Commit stage.

Architecture and trade-offs are documented in [../spec.md](../spec.md). Phase plan in [../plan.md](../plan.md).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (recommended) or Python 3.9+
- API keys for OpenRouter (https://openrouter.ai — default model `x-ai/grok-4.3`; override via `OPENROUTER_MODEL`) and Jina AI (https://jina.ai)
- Optional: LangWatch (https://app.langwatch.ai) for cloud span dashboard

## Install

With uv (recommended — pulls Python 3.11 automatically and resolves all deps):
```
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Plain pip works too:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

```
cp .env.example .env
# edit .env, set OPENROUTER_API_KEY and JINA_API_KEY
# optional: LANGWATCH_API_KEY for cloud telemetry
```

`.env` is gitignored; never commit it.

## One-time: index the corpus

```
python code/main.py update-knowledge-base --dir data
```

This walks `data/{hackerrank,claude,visa}/`, cleans + chunks each `.md` file, embeds via Jina v3,
and persists to `chroma_store/` (vectors) and `knowledge_base.db` (raw text). Idempotent —
chunk IDs are deterministic so re-running upserts in place.

Expected counts (approximate):

```
hackerrank ~437 files → ~1100 chunks
claude     ~321 files → ~700 chunks
visa       ~14  files → ~14 chunks
```

## Run on the test set

```
python code/main.py triage \
    --csv support_tickets/support_tickets.csv \
    --output support_tickets/output.csv
```

The CSV writer flushes per row, so you can tail progress.

## Interactive single-ticket mode

```
python code/main.py triage
# paste an Issue, Subject, Company at the prompts
```

## Calibrate against the labeled sample

```
python code/main.py calibrate
# runs the pipeline against sample_support_tickets.csv,
# computes weighted column accuracy, writes config.toml.
```

## File map (matches spec.md §10)

```
code/
├── main.py            CLI entry: update-knowledge-base / triage / calibrate
├── agent.py           run_triage(): builds the ADK root agent and runs it via InMemoryRunner
├── adk_agents.py      ADK composition: SequentialAgent → PreFlight → LoopAgent(ReWoo) → Commit
├── telemetry.py       OpenTelemetry SDK + JSONL file exporter for every span
├── prompts.py         PREFLIGHT / PLANNER / SOLVER / REFLECTION prompts
├── models.py          Pydantic schemas: PreFlight, Plan, Step, TriageOutput, Reflection, Chunk, DocType
├── callbacks.py       block_prompt_injection, detect_and_translate (ADK before_model_callback)
├── tools.py           retrieve(company, query, doc_type_filter, …) — called by WorkersAgent
├── voting.py          weighted_product_area helper
├── llm.py             Direct LiteLLM client (used by callbacks for the translate call)
├── embedder.py        Jina embeddings + reranker client
├── chunker.py         Doc-type detection + per-type chunking
├── preprocessor.py    Markdown cleaning + frontmatter
├── indexer.py         Pipeline: walk → preprocess → chunk → embed → store
├── calibrator.py      Threshold sweep against labeled sample
└── storage/
    ├── sqlite_store.py
    └── chroma_store.py
```

## Where ADK lives

The triage pipeline is composed entirely from ADK constructs in `code/adk_agents.py`:

- `SequentialAgent` for the top-level (TicketContext → PreFlight → ReWoo → Commit)
- `LlmAgent` with `output_schema=PreFlight | Plan | TriageOutput | Reflection` for the four LLM stages, each with `include_contents='none'` and a state-templated `instruction`
- `LoopAgent(max_iterations=2)` for the ReWoo retry loop; the `LoopBreakerAgent` emits `EventActions(escalate=True)` to terminate per spec §6.5
- Custom `BaseAgent` subclasses for non-LLM stages (TicketContext, Workers parallel-retrieve, bridge agents that JSON-stringify state, Commit)
- `before_model_callback=[block_prompt_injection, detect_and_translate]` on the PreFlight LlmAgent
- `LiteLlm(model='openai/<kimi>')` bridges every LlmAgent to Moonshot — explicit `api_base` + `api_key` so litellm doesn't default to OPENAI_*

Execution is driven by ADK's `InMemoryRunner` in `code/agent.py::_run_one`.

## Telemetry

Every ADK span (LlmAgent invocations, sub-agent transitions, tool calls) is fanned
out to **two** exporters by `code/telemetry.py`:

### 1. LangWatch cloud (primary)

Set `LANGWATCH_API_KEY` (and optionally `LANGWATCH_ENDPOINT`, defaults to
`https://app.langwatch.ai`) in `.env` to enable. Spans appear in the LangWatch
dashboard with the full trace tree:

```
triage_root (SequentialAgent)
├── ticket_context
├── preflight_agent (LlmAgent → Kimi via LiteLlm)
├── preflight_bridge
├── rewoo_loop (LoopAgent)
│   ├── iteration 1
│   │   ├── planner_bridge
│   │   ├── planner_agent (LlmAgent → Kimi)
│   │   ├── workers_agent
│   │   │   ├── retrieve(hackerrank, "...") → Jina embed + Chroma + Jina rerank
│   │   │   └── retrieve(claude, "...")     → ...
│   │   ├── solver_agent (LlmAgent → Kimi)
│   │   ├── solver_bridge
│   │   ├── reflector_agent (LlmAgent → Kimi)
│   │   └── loop_breaker
│   └── iteration 2 (only if reflector signals retry)
└── commit_agent
```

Each LLM span carries the request + response, latency, and token counts so the
dashboard surfaces failure modes (e.g. parse failures, retry triggers) directly.

### 2. Local JSONL backup

Every span is also written to `runs/telemetry-<unix_timestamp>.jsonl` (override
with `TELEMETRY_LOG=/path/to/file`). One JSON object per line:

```
{trace_id, span_id, parent_span_id, name, kind,
 start, end, duration_ms, status,
 attributes: {...}, events: [...]}
```

Quick aggregations:
```
# Per-stage timing
jq -r 'select(.name|test("agent|preflight|planner|solver|reflector")) |
       [.name, .duration_ms] | @tsv' runs/telemetry-*.jsonl | sort -k2 -n

# All LLM calls
jq 'select(.attributes["llm.request.type"]?)' runs/telemetry-*.jsonl

# Errors only
jq 'select(.status=="ERROR")' runs/telemetry-*.jsonl
```

Why both: LangWatch is the human-facing dashboard for debugging and post-mortems
during the AI-judge interview; the JSONL file is the deterministic local artifact
that survives regardless of network state and can be checked into the run record
if useful.

## Output schema

`support_tickets/output.csv` columns (matching `sample_support_tickets.csv`):

| Column        | Allowed values |
|---|---|
| Issue         | echoed from input |
| Subject       | echoed from input |
| Company       | echoed from input |
| Response      | grounded English answer, or `Escalate to a human` |
| Product Area  | canonical label from the corpus (never empty) |
| Status        | `Replied` \| `Escalated` |
| Request Type  | `product_issue` \| `feature_request` \| `bug` \| `invalid` |

## How escalation is decided

Status is `Escalated` if any of:

- PreFlight `escalate_now=True` (platform-wide outage, security disclosure, identity theft, fraud, refund/chargeback demand, etc.)
- Prompt-injection callback fires (PreFlight short-circuits)
- Reflector `final_score < 6.0` after 2 loop iterations
- Solver returned `response="ESCALATE"` in both attempts
- Reflection JSON parse failure (defensive)

`Product Area` is **never empty**. Even on PreFlight short-circuit, the Planner emits a single
`label_only` retrieve so the weighted vote can populate the column.

## Determinism

- `temperature=0` on every LLM call
- LiteLLM passes `seed=42` to Moonshot (provider may ignore — harmless)
- Deterministic chunk IDs (`{source_path}:{chunk_index}`) for idempotent reindexing
- `random.seed(42)` and (when available) `numpy.random.seed(42)` at top of `main.py`

## Smoke test (after configuring keys + indexing)

```
python code/main.py triage --csv support_tickets/sample_support_tickets.csv \
    --output /tmp/sample_output.csv
```

Expect status to match labels on ≥ 8/10 rows.

## Troubleshooting

- `MOONSHOT_API_KEY is not set` → `.env` not loaded; run from repo root or `cp .env.example .env`.
- `JINA_API_KEY is not set` → same.
- Jina 429s → `embedder.py` retries with exponential backoff; if persistent, lower `BATCH_SIZE` from 100 to 50 in `embedder.py`.
- Empty `Product Area` → bug; the commit step in `agent.py` should populate via fallback. File an issue.
- Slow first triage row → ChromaDB is loading the HNSW index; subsequent rows are fast.
