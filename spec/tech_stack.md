# Tech Stack — HackerRank Orchestrate Support Triage Agent

## Components

| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | Stable; all packages have wheels |
| Agent framework | Google ADK | Custom `BaseAgent` — pipeline orchestration and CLI |
| LLM client | `openai` Python SDK | Used directly with Kimi's `base_url` — no LiteLLM |
| LLM | Kimi Moonshot (OpenAI-compatible) | `moonshot-v1-8k` / `32k` / `128k` |
| Embedding | Jina Embeddings API | `jina-embeddings-v3` — multilingual |
| Reranker | Jina Reranker API | `jina-reranker-v2-base-multilingual` |
| Vector DB | ChromaDB | Persistent client, 3 collections (one per company) |
| Raw document store | SQLite | Python stdlib `sqlite3`, no extra dependency |

---

## Environment Variables

All secrets from environment variables only — never hardcoded. Copy `.env.example` → `.env` (gitignored).

```
MOONSHOT_API_KEY       # Kimi Moonshot LLM
JINA_API_KEY           # Jina embeddings + reranker (shared key)
```

---

## Integration Notes

### Kimi Moonshot — `openai` SDK with custom `base_url`

ADK's only official bridge to non-Google models is LiteLLM, which we are not using. Instead, LLM calls are made directly via the `openai` Python SDK pointing at Kimi's OpenAI-compatible endpoint. This is done inside the custom `TriageAgent` (see architecture below) — ADK's `LlmAgent` is not used for any LLM step.

```python
from openai import OpenAI
import os

llm = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.cn/v1",
)

response = llm.chat.completions.create(
    model="moonshot-v1-8k",   # or moonshot-v1-32k / moonshot-v1-128k
    messages=[...],
    temperature=0,            # deterministic where possible
)
```

### Google ADK Architecture

ADK's `SequentialAgent` runs all sub-agents unconditionally in order — no built-in conditional early-exit. Because the triage pipeline has multiple early-exit branches (invalid short-circuit, pre-retrieval escalation rules, below-threshold out-of-scope), the entire pipeline is a **custom `BaseAgent` subclass** (`TriageAgent`) with full imperative control flow. ADK is used for CLI runner, tool registration, and session management; all LLM calls go through the `openai` SDK directly.

```
TriageAgent (custom BaseAgent)
├── classify_request_type()  (openai SDK call)           — Step 1
├── [escalation rules]       (pure Python regex)          — Step 2
├── [company resolution]     (pure Python text match)     — Step 3
├── retrieve()               (ADK tool → ChromaDB + Jina Embeddings) — Step 4
├── rerank()                 (ADK tool → Jina Reranker)  — Step 5
├── synthesize()             (openai SDK call)            — Step 6
└── reflect()                (openai SDK call)            — Step 7
```

### ChromaDB — Persistent Client

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_store")
hackerrank_col = client.get_or_create_collection("hackerrank")
claude_col     = client.get_or_create_collection("claude")
visa_col       = client.get_or_create_collection("visa")
```

### SQLite — Raw Document Store

```python
import sqlite3

conn = sqlite3.connect("knowledge_base.db")
# Schema: chunks(id, company, source_path, product_area, doc_type, content, chunk_index)
```

Vector DB metadata stores `sqlite_id` to fetch full chunk text after retrieval without re-reading files from disk.

### Jina Embeddings + Reranker

Both use the same `JINA_API_KEY`.

- **Embeddings** (`jina-embeddings-v3`): used at index time (`update-knowledge-base`) and at query time (embed ticket before ChromaDB search)
- **Reranker** (`jina-reranker-v2-base-multilingual`): used after retrieval to rerank topK chunks; top-1 result determines `product_area`
