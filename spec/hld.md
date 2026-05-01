# High-Level Design — HackerRank Orchestrate Support Triage Agent

---

## 1. Overview

A terminal-based support triage agent that processes support tickets across three product ecosystems (HackerRank, Claude, Visa) using only the local support corpus in `data/`. For each ticket it produces five output fields: `status`, `product_area`, `response`, `justification`, `request_type`.

---

## 2. Tech Stack

See [tech_stack.md](tech_stack.md) for the full stack, compatibility flags, and integration notes.

**Summary:** Python 3.11 · Google ADK · Kimi Moonshot via `openai` SDK (no LiteLLM) · Jina Embeddings + Reranker · ChromaDB · SQLite

---

## 3. Storage

### 3.1 Vector Database — ChromaDB (persistent)
```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_store")

hackerrank_col = client.get_or_create_collection("hackerrank")
claude_col     = client.get_or_create_collection("claude")
visa_col       = client.get_or_create_collection("visa")
```
- Each vector record stores the chunk embedding (from Jina) plus metadata:
  `company`, `product_area` (canonical label), `doc_type`, `source_path`, `sqlite_id`
- Used for semantic retrieval at triage time

### 3.2 SQLite — `sqlite3` stdlib (persistent)
```python
import sqlite3
conn = sqlite3.connect("knowledge_base.db")
# Schema: chunks(id, company, source_path, product_area, doc_type, content, chunk_index)
```
- Stores full raw chunk text
- Vector DB metadata holds `sqlite_id` (`chunks.id`) — fetch full text after retrieval without re-reading disk

---

## 4. Terminal Commands

### `update-knowledge-base --dir <path>`
Indexes the corpus into ChromaDB and SQLite.
- Accepts a directory path via flag (defaults to `data/`)
- Preprocesses all `.md` files (chunking strategy — see §5)
- Attaches canonical metadata at index time (see §5.3)
- Idempotent: re-running clears and rebuilds the affected collections
- Embeddings generated via Jina Embeddings API (`JINA_API_KEY`)

### `triage [--csv <input_path> --output <output_path>]`
Runs the triage pipeline.
- **Interactive mode** (no flags): accepts a single ticket interactively from stdin
- **CSV mode** (`--csv`): reads `support_tickets/support_tickets.csv`, writes predictions to `support_tickets/output.csv`
- Shows progress in CSV mode (e.g. `[12/29] Processing...`)

---

## 5. Indexing Pipeline (`update-knowledge-base`)

### 5.1 Chunking strategy
*(To be finalised — discussed separately)*

### 5.2 Embedding
- Jina Embeddings API (`jina-embeddings-v3`)
- Each chunk is embedded and stored in the appropriate company ChromaDB collection

### 5.3 Metadata attached at index time

| Field | Value | How derived |
|---|---|---|
| `company` | `hackerrank` / `claude` / `visa` | Top-level directory name |
| `product_area` | Canonical label (e.g. `screen`, `travel_support`) | Mapped from directory path at index time (label map defined separately) |
| `doc_type` | `how-to` / `faq` / `reference` / `conceptual` / `integration` / `troubleshooting` / `release-notes` / `policy-legal` | Detected from document structure during preprocessing |
| `source_path` | Relative path of source `.md` file | File path |
| `sqlite_id` | Row ID in SQLite `chunks` table | Assigned at insert time |

---

## 6. Triage Pipeline (per ticket)

```
Input ticket (issue, subject, company)
         │
         ▼
 ┌───────────────────────────────┐
 │  STEP 1: Request Type         │  ← RequestTypeAgent (LlmAgent, Kimi)
 │  Classify from issue+subject  │
 │  only (no retrieval)          │
 │  → product_issue / feature_   │
 │    request / bug / invalid    │
 └───────────────┬───────────────┘
                 │
                 ▼
         request_type = invalid?
         ┌── Yes ──────────────────────────────────────────────────┐
         │   Skip retrieval entirely                                │
         │   Return: status=replied, request_type=invalid,          │
         │           response="<polite out-of-scope message>",      │
         │           product_area="", justification="Out of scope"  │
         └─────────────────────────────────────────────────────────┘
                 │ No
                 ▼
 ┌───────────────────────────────┐
 │  STEP 2: Pre-retrieval        │  ← Pure Python rule checks
 │  Escalation Rules             │
 │  Hard-coded pattern checks    │
 │  (before any retrieval)       │
 └───────────────┬───────────────┘
                 │
                 ▼
         Matches escalation pattern?   (see §6.1)
         ┌── Yes ──────────────────────────────────────────────────┐
         │   Return: status=escalated, request_type=<classified>,   │
         │           response="Escalate to a human",                │
         │           justification="<pattern that triggered rule>"  │
         └─────────────────────────────────────────────────────────┘
                 │ No
                 ▼
 ┌───────────────────────────────┐
 │  STEP 3: Company Resolution   │  ← Pure Python (see §6.2)
 └───────────────┬───────────────┘
                 │
                 ▼
 ┌───────────────────────────────┐
 │  STEP 4: Retrieval            │  ← ADK tool: Jina Embed query
 │  Embed query via Jina         │     → ChromaDB query on resolved
 │  Search resolved collection   │       collection(s)
 │  Return topK chunks           │
 └───────────────┬───────────────┘
                 │
                 ▼
         All chunks below similarity threshold?
         ┌── Yes ──────────────────────────────────────────────────┐
         │   request_type = invalid → replied + out-of-scope msg   │
         │   otherwise             → escalated                     │
         └─────────────────────────────────────────────────────────┘
                 │ No
                 ▼
 ┌───────────────────────────────┐
 │  STEP 5: Rerank               │  ← ADK tool: Jina Reranker API
 │  Rerank topK via Jina         │     (jina-reranker-v2-base-multilingual)
 │  product_area = top-1         │
 │  metadata.product_area        │
 └───────────────┬───────────────┘
                 │
                 ▼
 ┌───────────────────────────────┐
 │  STEP 6: Synthesis            │  ← SynthesisAgent (LlmAgent, Kimi)
 │  Combine top 2-3 chunks into  │
 │  one context window           │
 │  Single LLM call →            │
 │  (response, justification)    │
 └───────────────┬───────────────┘
                 │
                 ▼
 ┌───────────────────────────────┐
 │  STEP 7: Self-Reflection      │  ← ReflectionAgent (LlmAgent, Kimi)
 │  Score the generated response │
 │  on 4 dimensions (see §6.3)   │
 │  final_score = weighted avg   │
 └───────────────┬───────────────┘
                 │
                 ▼
         final_score >= 6.0?
         ┌── Yes ──────────────────────────────────────────────────┐
         │   status = replied                                       │
         │   return all 5 fields                                    │
         └─────────────────────────────────────────────────────────┘
         └── No ───────────────────────────────────────────────────┐
             status = escalated                                     │
             response = "Escalate to a human"                      │
             justification = reflection["reason"]                   │
             └───────────────────────────────────────────────────── ┘
```

---

### 6.1 Pre-retrieval Escalation Rules

Hard-coded pattern checks (case-insensitive regex) that trigger escalation before any retrieval:

| Pattern | Examples from test set | Reason |
|---|---|---|
| Platform-wide outage language | "site is down", "none of the pages are accessible", "stopped working completely", "all requests are failing", "X is Down" | Cannot be resolved by support docs; needs live ops |
| Security disclosure | "security vulnerability", "found a bug in your security", "data breach" | Must route to security team |
| Identity theft / fraud report | "my identity has been stolen", "someone stole my card", "fraudulent transaction" | Sensitive; needs human |

Keep this list small and explicit — do not expand heuristically.

---

### 6.2 Company Resolution

```
company field provided (not None / not blank)
    → use that company's ChromaDB collection directly

company field = None or blank
    │
    ├─ Scan issue + subject for company name mentions
    │  (case-insensitive; aliases:
    │   "anthropic", "claude.ai"            → claude
    │   "hackerrank for work", "hacker rank" → hackerrank
    │   "visa card", "visa inc"             → visa)
    │
    ├─ Match found → use that company's collection
    │
    └─ No match found
           │
           ▼
       Embed query via Jina, search ALL 3 collections (same topK each)
           │
           ├─ Top result is clearly dominant
           │  (highest reranker score AND gap to next-company
           │   top result > 0.15)
           │       → infer company from top result metadata
           │       → use that collection's topK results
           │
           ├─ Top results spread across 2+ companies
           │  (score gap ≤ 0.15 between companies)
           │       → escalate: cannot confidently route
           │
           └─ All results below out-of-scope threshold
                   → request_type gate (invalid → reply; else escalate)
```

**Response language:** Always generate the response in English, regardless of the ticket's input language. The corpus is English-only.

---

### 6.3 Self-Reflection Scoring (ReflectionAgent)

Single Kimi LLM call after synthesis. Input: retrieved chunks + generated response. Output: JSON.

**Prompt:**

```
You are evaluating a candidate support response.

Retrieved context:
<top 2-3 chunks>

Generated response:
<response>

Score on each dimension from 1–10. Be strict.
Penalise hard for any claim not traceable to the context above.

1. GROUNDING
   Every factual claim, step, number, and policy in the response
   is directly supported by the retrieved context.
   10 = fully grounded   1 = fabricated steps or hallucinated specifics

2. COMPLETENESS
   The response answers the specific question asked.
   10 = fully addresses what the user asked   1 = generic or off-topic

3. SAFETY
   The response avoids authoritative guidance on sensitive topics
   (billing, legal rights, fraud, account recovery, security)
   that the corpus does not explicitly cover.
   10 = safe or appropriately escalates   1 = unsupported authoritative claim

4. ACTIONABILITY
   The user knows what to do after reading this.
   10 = clear next step   1 = vague, no action

Return JSON only:
{
  "grounding": <1-10>,
  "completeness": <1-10>,
  "safety": <1-10>,
  "actionability": <1-10>,
  "final_score": <weighted float>,
  "reason": "<one sentence on the main weakness>"
}
```

**Scoring weights:**

| Dimension | Weight | Rationale |
|---|---|---|
| Grounding | 0.4 | Hallucination is the primary failure mode; evaluator explicitly checks for it |
| Completeness | 0.3 | Incomplete answers score poorly on `response` accuracy |
| Safety | 0.2 | Sensitive tickets in test set; wrong guidance is worse than escalating |
| Actionability | 0.1 | Helpful but least penalised; a grounded complete answer is implicitly actionable |

`final_score = grounding×0.4 + completeness×0.3 + safety×0.2 + actionability×0.1`

**Escalation threshold:** `final_score < 6.0` → escalate. `reflection["reason"]` becomes the `justification` field.

---

## 7. Output Schema

| Field | Source |
|---|---|
| `status` | `replied` if final_score ≥ 6.0 or short-circuit reply; `escalated` otherwise |
| `product_area` | Metadata of Jina reranker top-1 chunk (canonical label from index) |
| `response` | SynthesisAgent output (or short-circuit message) |
| `justification` | SynthesisAgent output or reflection `reason` on escalation |
| `request_type` | RequestTypeAgent output (issue + subject only) |

---

## 8. Deferred (to be finalised)

- Chunking strategy and chunk size
- Canonical product area label mapping (directory path → label name)
