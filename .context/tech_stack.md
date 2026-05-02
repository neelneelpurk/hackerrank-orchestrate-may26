# Tech Stack — HackerRank Orchestrate Support Triage Agent

> Companion to [spec.md](spec.md) — that file is the source of truth for the runtime contract; this one focuses on the libraries, env vars, and integration glue.

## Components

| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.9+ | Developed on 3.9.6; ADK + LiteLLM both work. 3.11 still recommended. |
| Agent framework | Google ADK 1.18+ | `LlmAgent`, `SequentialAgent`, `LoopAgent`, custom `BaseAgent`, `before_model_callback`, `EventActions(escalate=…)`, `adk eval` |
| LLM bridge | **LiteLLM** (`google.adk.models.lite_llm.LiteLlm`) | ADK's official non-Google bridge; everything routed via OpenRouter |
| LLM | xAI **Grok 4.3** via OpenRouter (`x-ai/grok-4.3` default; override with `OPENROUTER_MODEL`) | One key, many models — easy A/B between Grok / Kimi / Claude / GPT-5 by changing one env var |
| Embeddings | Jina `jina-embeddings-v3` | Multilingual (matters because callbacks may pre-translate non-EN tickets, and we don't want a separate index pass for translated queries) |
| Reranker | Jina `jina-reranker-v2-base-multilingual` | Used after Chroma; one rerank pass against the primary query variant |
| Vector DB | ChromaDB | `PersistentClient`, three collections (one per company) |
| Raw document store | SQLite | Python stdlib; canonical chunk text (Chroma stores embedding-time text, which may differ for FAQs) |
| Schema validation | Pydantic v2 | `output_schema=` on every LlmAgent (`PreFlight`, `Plan`, `TriageOutput`, `Reflection`) |
| JSON repair | `json-repair` | Defensive parse around structured-output responses (markdown-fenced JSON, trailing commas, half-quoted strings) |
| Language detection | `langdetect` | Cheap; routes non-English tickets through translation pre-step |
| Telemetry | LangWatch cloud + `JsonlFileSpanExporter` | Both bound to a single `TracerProvider` so every ADK span fans out to both sinks |
| HTTP shim | FastAPI + uvicorn | `code/server.py` exposes `agent.run_triage` over HTTP — used as the LangWatch Scenario target |

---

## Environment Variables

All secrets from environment variables only — never hardcoded. Copy `.env.example` → `.env` (gitignored).

```
OPENROUTER_API_KEY                        # primary auth — every LLM call routes through OpenRouter
OPENROUTER_MODEL=x-ai/grok-4.3            # OpenRouter slug; default Grok 4.3
OPENROUTER_PROVIDER_ORDER=                # optional pin, e.g. "xAI,Together,Fireworks"
JINA_API_KEY                              # Jina embeddings + reranker (one key for both)

# Optional — LangWatch cloud telemetry. JSONL exporter still runs without it.
LANGWATCH_API_KEY
LANGWATCH_ENDPOINT=https://app.langwatch.ai

# Optional — runtime tuning
REFLECTION_PASS_THRESHOLD=5.0             # raise to 6.0 for stricter reply gating
RETRIEVE_TOP_K=50                         # Chroma initial pool per query variant
RETRIEVE_FINAL_N=5                        # evidence chunks fed to Solver per step
```

The legacy `MOONSHOT_API_KEY` / `MOONSHOT_MODEL` env names from earlier Kimi-only configs are still accepted as fallbacks (`adk_agents.py::_llm_model`) so existing `.env` files keep working.

---

## Integration Notes

### LLM — OpenRouter via LiteLLM

ADK's `LlmAgent` accepts any `BaseLlm` instance. `LiteLlm` is the standard adapter for non-Google providers; it speaks LiteLLM's `provider/model` format and forwards `api_key` through to the upstream OpenAI-compatible endpoint.

```python
import os
from google.adk.models.lite_llm import LiteLlm

GROK = LiteLlm(
    model="openrouter/x-ai/grok-4.3",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
    seed=42,
    extra_body={
        # OpenRouter spreads each model across multiple upstream providers; not
        # all support strict JSON-schema response_format. ADK requires that for
        # output_schema=…. Without this hint we get sporadic 400 "model features
        # structured outputs not support" or 404 "no endpoints found".
        "provider": {"require_parameters": True, "allow_fallbacks": True},
    },
)
```

Used as `model=GROK` on every `LlmAgent` (`PreFlight`, `Planner`, `Solver`, `Reflector`). No direct `openai` SDK calls anywhere.

**Why LiteLLM and not the `openai` SDK directly?**
- `LlmAgent` requires a `BaseLlm` — ADK does not consume raw `openai` clients.
- Keeping every LLM call inside ADK gives free `before_model_callback` / `after_model_callback` hooks, `output_schema` enforcement, and `adk eval` trajectory matching.
- LiteLLM standardises retries, timeouts, and error mapping — saves wrapping the SDK ourselves.
- One key, one wrapper, but we can swap upstream model with one env-var change.

### ADK Architecture — ReWoo + bridges

Every LLM step is an `LlmAgent`. The "bridge" agents are tiny custom `BaseAgent`s that JSON-stringify Pydantic outputs into the string fields the next instruction template interpolates via `{state_key}` placeholders. `include_contents='none'` on every `LlmAgent` means each prompt is built deterministically from state, not from accumulated conversation history.

```
ROOT_AGENT (SequentialAgent)
├── ticket_context        BaseAgent — load ticket_text + seed bookkeeping fields
├── preflight_agent       LlmAgent — output_schema=PreFlight
│   └── before_model_callback: [block_prompt_injection, detect_and_translate]
├── preflight_bridge      BaseAgent — JSON-stringify preflight; swap in translated text
│
├── rewoo_loop            LoopAgent, max_iterations=2
│   ├── planner_bridge    BaseAgent — bump iteration; stage retry_hint + previous plan_json
│   ├── planner_agent     LlmAgent — output_schema=Plan
│   ├── workers_agent     custom BaseAgent — dynamic asyncio.gather over plan.steps
│   ├── solver_agent      LlmAgent — output_schema=TriageOutput
│   ├── solver_bridge     BaseAgent — stringify solution; set is_final flag
│   ├── reflector_agent   LlmAgent — output_schema=Reflection (pure scoring, no escalate field)
│   └── loop_breaker      BaseAgent — emits EventActions(escalate=True) when reflection passes
│
└── commit_agent          BaseAgent — assemble final CSV row dict at state['triage_result']
```

- **Workers is a custom `BaseAgent`, not `ParallelAgent`.** ADK's stock `ParallelAgent` requires a fixed `sub_agents` list at construction; our plan is dynamic per ticket (1–4 retrieve steps with per-step companies and filters), so we dispatch retrieves via `asyncio.gather` inside `WorkersAgent._run_async_impl`.
- **`Reflection` has no `escalate` field.** It carries scores only (`grounding`, `completeness`, `safety`, `actionability`, `final_score`, `reason`, `verified_request_type`). Loop control is `LoopBreakerAgent`'s job (uses `EventActions.escalate=True` to mean "exit loop"); human routing is `CommitAgent`'s. Conflating them caused inverted-logic bugs in earlier drafts.
- **State chaining**: every LlmAgent has an `output_key` (`preflight`, `plan`, `solution`, `reflection`). Workers store evidence at `state["evidence_by_step"][step_id]` plus pre-rendered `evidence_blocks` / `evidence_summary` strings.

### Telemetry — dual exporter

[code/telemetry.py::init_telemetry](../code/telemetry.py) builds our own `TracerProvider` (`service.name=hackerrank-orchestrate-triage`), attaches a `BatchSpanProcessor(JsonlFileSpanExporter(runs/telemetry-<ts>.jsonl))`, then publishes it as the global tracer provider. Only after that do we call `langwatch.setup(tracer_provider=<our provider>, …)` — passing the provider in explicitly is what makes LangWatch *add* its OTLP processor instead of replacing ours, so both sinks see the same trace tree.

If `LANGWATCH_API_KEY` isn't set, LangWatch is skipped silently and the JSONL file keeps capturing everything.

### ChromaDB — Persistent Client

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_store")
for company in ("hackerrank", "claude", "visa"):
    client.get_or_create_collection(company, metadata={"hnsw:space": "cosine"})
```

`embedding_function=None` — embeddings are always supplied externally from Jina. We do **not** use `chroma-mcp`; the agent calls Chroma deterministically from inside the `retrieve` function in [code/tools.py](../code/tools.py).

### SQLite — Raw Document Store

```python
import sqlite3
conn = sqlite3.connect("knowledge_base.db")
# Schema: chunks(id, company, source_path, product_area, doc_type,
#                heading_path, content, chunk_index)
```

Chroma metadata stores `sqlite_id` so the retrieve tool refreshes chunk text from SQLite by id at query time. SQLite is the **authoritative** chunk content; Chroma's text may include FAQ question variants used to enrich the embedding signal, which we don't want the Solver to see.

### Jina Embeddings + Reranker

Both endpoints share `JINA_API_KEY`.

- **Embeddings** (`jina-embeddings-v3`): used at index time and at query time.
- **Reranker** (`jina-reranker-v2-base-multilingual`): runs once per retrieve step against the primary query variant; the Planner emits 1–3 semantically-equivalent variants and Workers union the dense candidates before reranking.

[code/embedder.py](../code/embedder.py) wraps both with batching (100 per call), exponential backoff on 429s, and a small in-process query cache for dev re-runs.

### FastAPI shim

[code/server.py](../code/server.py) exposes one endpoint:

```
POST /triage  → { issue, subject, company } → TriageResponse
GET  /healthz
```

It pre-warms the ADK runner + telemetry on startup so the first LangWatch Scenario request doesn't pay warmup latency. Single worker because the cached `InMemoryRunner` lives on the module level and Scenarios are sequential anyway.

---

## Pinned Versions

See `requirements.txt` at the repo root for the locked set. Headline pins:

```
google-adk>=1.18
litellm>=1.50
chromadb>=0.5
pydantic>=2.6
langdetect>=1.0.9
httpx>=0.27          # Jina REST calls
python-dotenv>=1.0   # .env loading in main.py
json-repair          # defensive structured-output parsing
fastapi              # /triage shim
uvicorn              # /triage shim
langwatch            # cloud telemetry (optional — skipped if API key absent)
```

The `adk` CLI (installed by `google-adk`) is used for evaluation only (`adk eval ...`); the runtime triage flow uses ADK's `Runner` directly from `code/main.py`.

---

## Change log vs earlier drafts

| Old | Current |
|---|---|
| Kimi Moonshot via `MOONSHOT_API_KEY` | OpenRouter via `OPENROUTER_API_KEY`; Grok 4.3 default; legacy Moonshot vars accepted as fallback |
| `ParallelAgent` for Workers | Custom `WorkersAgent(BaseAgent)` — plan is dynamic per ticket |
| `Reflection.escalate` field signals routing | Field removed — `LoopBreakerAgent` decides loop exit, `CommitAgent` decides reply/escalate |
| Reflection threshold 6.0 | 5.0 default, env-tunable via `REFLECTION_PASS_THRESHOLD` |
| Single Planner `query` per step | `Step.query_variants` (list, ≤3) — Planner emits multiple paraphrases, Workers union before rerank |
| `top_k=8`, `final_n=3` | `top_k=50` (cheap, gives reranker real choices), `final_n=5` for answer steps; per-source diversification before + after rerank; drop-floor on weak rerank scores |
| No telemetry contract | LangWatch + JSONL dual exporter via shared `TracerProvider` |
| No HTTP entry point | `code/server.py` FastAPI shim for LangWatch Scenarios |
