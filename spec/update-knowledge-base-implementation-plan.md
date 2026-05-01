# update-knowledge-base — Implementation Plan

> Covers the full pipeline for the `update-knowledge-base --dir <path>` command:
> preprocessing → doc type detection → chunking → embedding → SQLite + ChromaDB storage.

---

## 1. Module Structure

```
code/
├── models.py                 — Pydantic models: DocType, Chunk, IndexResult
├── preprocessor.py           — Markdown cleaning (strip frontmatter, images, etc.)
├── chunker.py                — Doc-type-aware splitting into Chunk text segments
├── embedder.py               — Jina Embeddings + Reranker API client
├── indexer.py                — Orchestrates the full pipeline per company
└── storage/
    ├── sqlite_store.py       — Raw chunk text store (sqlite3 stdlib)
    └── chroma_store.py       — ChromaDB PersistentClient wrapper (3 collections)
```

`main.py` calls `indexer.run(data_dir)` when the `update-knowledge-base` command is invoked.

---

## 2. Pydantic Models (`models.py`)

All data models use **Pydantic `BaseModel`** — not dataclasses.

### `DocType`
`str` enum with 8 values:

| Value | Description |
|---|---|
| `how-to` | Step-by-step procedural guides |
| `faq` | Individual question-answer articles |
| `reference` | Index files, settings tables, config lists |
| `conceptual` | Introduction and overview articles |
| `integration` | Third-party integration setup guides |
| `troubleshooting` | Error diagnosis and fixes |
| `release-notes` | Dated product update changelogs |
| `policy-legal` | GDPR, DPA, compliance, terms |

### `Chunk`
```
text:         str            — cleaned chunk content (with heading_path prepended)
company:      str            — "hackerrank" | "claude" | "visa"
product_area: str            — canonical label (e.g. "screen", "travel_support")
doc_type:     DocType
source_path:  str            — relative path from repo root
chunk_index:  int            — 0-based position within the source document
heading_path: str            — "H1 > H2 > H3" breadcrumb (e.g. "Screen > Invite Candidates > Set Expiry Date")
sqlite_id:    Optional[int]  — set to None until after SQLite insert
```

Use `model_validator(mode="after")` to enforce `text` is non-empty and `company` is one of the three valid values.

### `IndexResult`
Summary returned by `indexer.run()`:
```
company:          str
files_processed:  int
chunks_created:   int
duration_seconds: float
```

---

## 3. Preprocessing (`preprocessor.py`)

Applied to each raw `.md` file **before** chunking. Returns cleaned string + extracted frontmatter fields.

### 3.1 Frontmatter extraction (before stripping)

Parse the YAML `---` block at the top. Extract and save:
- `breadcrumbs: list[str]` — used to build `heading_path`
- `title: str` — used as fallback H1 if the document body has a garbled H1 (common in Zendesk exports where the title includes the full TOC)

### 3.2 Cleaning steps (applied in order)

| Step | What | Why |
|---|---|---|
| 1 | Strip YAML frontmatter block (`---…---`) | Not part of the support content |
| 2 | Strip image markdown `![alt](url)` | Images are non-retrievable; alt text in this corpus is typically a filename like `1.png` |
| 3 | Strip horizontal rules (`^-{4,}$`, `^={4,}$`) | Visual separators with no semantic value |
| 4 | Normalise obfuscated email links `[[email protected]](/cdn-cgi/…)` → `[contact support]` | CDN-obfuscated links break text, lose intent |
| 5 | Strip backslash line continuations (`\` at line end) | Zendesk markdown export artifact |
| 6 | Strip HTML comments `<!-- … -->` | Invisible editor annotations |
| 7 | Collapse 3+ consecutive blank lines → 2 | Preserve paragraph breaks without excess whitespace |

**Do NOT strip:** blockquote callouts (`> **Note:**`), tables, code blocks, numbered/bulleted lists, bold text, inline links.

---

## 4. Document Type Detection (`chunker.py` / `indexer.py`)

Detection runs on `(cleaned_content, relative_filepath)`. Rules are evaluated **top-to-bottom; first match wins**.

### Rule 1 — File path (highest priority, O(1) string check)

Check `str(relative_filepath)` before touching content:

| Path contains | → DocType |
|---|---|
| `release-notes` or `release_notes` | `release-notes` |
| `frequently-asked-questions` or `/faqs/` | `faq` |
| `troubleshooting` | `troubleshooting` |
| `privacy-and-legal` or `/policy` | `policy-legal` |
| `getting-started` (and NOT also `integrations`) | `conceptual` |
| `integrations/` AND (`applicant-tracking-systems` OR `single-sign-on` OR `scheduling` OR `productivity`) | `integration` |

**Rationale:** Directory names in this corpus are explicit and deliberate (Zendesk structured export). Path matching requires zero content parsing and has no false positives.

### Rule 2 — Frontmatter breadcrumbs

Breadcrumbs were extracted in preprocessing. If path gave no match:

| Breadcrumbs include | → DocType |
|---|---|
| `"Release Notes"` | `release-notes` |
| `"Frequently Asked Questions"` or `"FAQ"` | `faq` |
| `"Troubleshooting"` | `troubleshooting` |
| `"Integrations"` AND a third-party company name | `integration` |

**Rationale:** Breadcrumbs are structured, machine-generated, and already parsed. More reliable than scanning prose.

### Rule 3 — H1/H2 heading content

Scan first 15 lines of cleaned content:

| Heading contains | → DocType |
|---|---|
| Pattern `<Name> - HackerRank Integration` | `integration` |
| `"introduction to"` or `"what is"` or `"overview"` | `conceptual` |
| `"frequently asked"` | `faq` |
| `"troubleshoot"` or `"error messages"` | `troubleshooting` |

### Rule 4 — Body content signals

Scan full cleaned content:

| Content contains | → DocType |
|---|---|
| 3+ occurrences of `\*\*Step \d+:` (bold step markers) | `how-to` |
| 3+ lines matching `^\d+\.` AND text contains `"prerequisite"` | `how-to` |

### Rule 5 — Fallback

No rule matched → `reference`

---

## 5. Chunking — Per Doc Type (`chunker.py`)

### Universal override (applied before any type-specific logic)

> **If total cleaned word count < 500 → return entire document as a single chunk.**

Captures: all 14 Visa files (avg ~450 words), most individual FAQ files, short conceptual intros, stub articles. No splitting needed or beneficial for these.

---

### 5a. FAQ (`faq`)

**Corpus reality:** FAQ directories contain individual files — one file = one question and its answer. Each filename IS the question (e.g. `7974410169-how-can-i-cancel-a-test-invite.md`). Files are tiny (10–50 lines, 20–150 words).

**Algorithm:**
- Nearly always hits the universal override (< 500 words) → single chunk
- Rare case ≥ 500 words: H2-level split (same as Reference)

**Heading path example:**
`Screen > Frequently Asked Questions > How Can I Cancel a Test Invite?`

**Output chunk example:**
```
[Screen > Frequently Asked Questions > How Can I Cancel a Test Invite?]

How Can I Cancel a Test Invite?

Refer to the Cancel a Test Invite documentation.
```

---

### 5b. How-To Guide (`how-to`)

**Corpus reality:** H1 = document title. H2 sections = major phases (e.g. "Prerequisites", "Inviting candidates to a test", "Set an expiry date"). Within H2 sections, steps may appear as `**Step N: Name**` bold markers OR as plain numbered lists (`1.`, `2.`, `3.`). H3 subsections may appear inside H2s.

**Algorithm:**
1. Split at `^## ` → list of H2 sections
2. Prepend document H1 title as context to every chunk
3. For each H2 section:
   - Word count ≤ 600 → emit as single chunk
   - Word count > 600 → detect step boundaries:
     - Look for `\*\*Step \d+:` (bold step labels) or `^\d+\.` numbered blocks
     - Group ≤ 3 steps per chunk
     - Prepend the H2 heading to each step-group chunk
4. Merge any chunk < 50 words into the following chunk

**Heading path example:**
`Screen > Invite Candidates > Set an Expiry Date`

**Step-group chunk example:**
```
[Screen > Invite Candidates > Inviting Candidates to a Test — Steps 1–3]

Inviting candidates to a test

**Step 1: Access the test**
1. Log in to your HackerRank for Work account.
2. Go to the Tests tab and select the test.

**Step 2: Configure invite settings**
1. Click Invite Candidates.
2. Enter the candidate email addresses...
```

---

### 5c. Integration Guide (`integration`)

**Corpus reality — critical:** Integration guides use `#` H1 for **both the document title AND major section headers**. There are **no `##` H2 sections** in these files. Major sections are: `# Prerequisites`, `# Integrating <Product> with HackerRank`, `# Glossary`, `# Frequently Asked Questions`. Steps within sections use the bold label format `**Step N: Action Name**` followed by numbered lists.

**Algorithm:**
1. Identify all `^# ` H1 lines (not `^## `)
2. First H1 = document title — extract for heading_path, do not emit as a chunk
3. Subsequent H1s = section boundaries — split at each
4. For each section:
   - Word count ≤ 600 → emit as single chunk
   - Word count > 600 → split at `\*\*Step \d+:` bold markers
     - Group **≤ 2 steps per chunk** (integration steps tend to be dense with sub-steps)
     - Prepend section H1 heading to each step-group chunk

**Why not H2-split?** These files have no `##` headings. Applying the H2 splitter would return the entire file as one chunk.

**Heading path example:**
`Integrations > ATS > Greenhouse > Integrating Greenhouse — Steps 1–2`

---

### 5d. Release Notes (`release-notes`)

**Corpus reality:** One file = one calendar period (e.g. `october-2024-release-notes.md`). Within the file: `## **Screen**` = H2 product area, `### Feature Name` = H3 feature. Files can be very large (one HackerRank file is 425 KB). Each H2 section contains multiple H3 features with prose, bullet lists, and tables.

**Algorithm:**
1. Split at `^## ` → list of product-area sections (H2)
2. Each H2 section **including all its nested H3s** = one chunk
3. Prepend document H1 title (the period name) to each chunk
4. Max chunk guard: if an H2 section contains > 3 H3 features, split into H3 sub-groups (H3s 1–2, H3s 3–4, H3 5+), each with the H2 heading prepended

**Why keep H3s with their H2?** H3 = feature, H2 = product area. A ticket about "plagiarism detection" needs to retrieve the chunk that carries both the Screen context (H2) and the feature detail (H3). Splitting them severs that link.

**Heading path example:**
`Release Notes > October 2024 > Screen`

---

### 5e. Troubleshooting (`troubleshooting`)

**Corpus reality:** H2 sections = independent error categories (e.g. `## Login errors`, `## Usage limit warnings and errors`). No cross-references between sections. Claude troubleshooting docs sometimes skip H3, using H4 directly under H2 — keep H4s nested with their H2.

**Algorithm:**
1. Split at `^## ` → one chunk per error category
2. All H3/H4 subsections remain nested within their parent H2 chunk
3. Merge chunks < 50 words into the following chunk

**Heading path example:**
`Claude > Troubleshoot Claude Error Messages > Login Errors`

---

### 5f. Reference (`reference`)

**Corpus reality:** Index files, settings docs, API parameter tables, config option lists. H2 sections contain bullet lists or tables. `index.md` files are pure link catalogs (> 30% of lines are `- [text](url)` format).

**Algorithm:**
1. Split at `^## ` → one chunk per H2 section
2. Tables and bullet lists stay intact within their H2 chunk
3. Index file detection: if > 30% of non-blank lines match `^- \[.+\]\(.+\)` → treat each H2 category block as one chunk; do not sub-split
4. Merge chunks < 50 words into the following chunk

**Heading path example:**
`Settings > Company Level Admin Settings > GDPR Settings`

---

### 5g. Conceptual / Overview (`conceptual`)

**Corpus reality:** Introduction and overview articles. Typically 100–400 words with 2–4 H2 sections (What is X, Key Features, Getting Started). Almost always hits the universal override.

**Algorithm:**
1. Universal override (< 500 words) → single chunk (handles ~95% of these)
2. Word count ≥ 500 → H2-level split
3. Merge chunks < 50 words into following chunk

**Heading path example:**
`Screen > Introduction to HackerRank Screen`

---

### 5h. Policy / Legal (`policy-legal`)

**Corpus reality:** GDPR, DPA, compliance docs. H2 = distinct policy topic, H3 = sub-clause. Policy clauses must never be split.

**Algorithm:**
1. Split at `^## ` → one chunk per policy topic
2. H3 subsections always stay nested with their parent H2 (H3 = sub-clause of the same policy)
3. Do not sub-split even if the chunk exceeds 800 tokens — partial policy text is worse than a large chunk

**Heading path example:**
`Claude > Privacy and Legal > Data Retention`

---

## 6. Heading Path Construction

The heading path is **prepended as the first line of every chunk** in brackets:

```
[Screen > Invite Candidates > Set an Expiry Date]

<chunk content here>
```

**Purpose:**
1. The LLM at synthesis time sees exactly where the content originated
2. Jina embeds this line into the vector — improves retrieval precision because company/product-area context is baked into the embedding

**Construction priority:**
1. Use frontmatter `breadcrumbs` array (present in all HackerRank + Claude files) → `" > ".join(breadcrumbs)`
2. Append current section heading (H2 title, or H1 section title for integration guides)
3. For files without breadcrumbs (some Visa files): `"<Company> > <product_area_label> > <H1 title>"`

**Example outputs:**

| File | Heading path |
|---|---|
| HackerRank How-To | `Screen > Invite Candidates > Set an Expiry Date` |
| Integration guide | `Integrations > ATS > Greenhouse > Prerequisites` |
| Claude troubleshooting | `Claude API and Console > Troubleshooting > Rate Limit Errors` |
| Visa consumer support | `Visa > Travel Support > Exchange Rate Calculator` |
| Release notes chunk | `Release Notes > October 2024 > Screen` |

---

## 7. Product Area Label Mapping

Defined in `indexer.py` as `PRODUCT_AREA_MAP: dict[str, str]`. Resolution uses **longest-prefix match** on the file's relative path. Fallback: immediate parent directory name.

### HackerRank

| Path prefix | Canonical label |
|---|---|
| `screen/` | `screen` |
| `interviews/` | `interviews` |
| `hackerrank_community/` | `community` |
| `integrations/applicant-tracking-systems/` | `integrations_ats` |
| `integrations/single-sign-on-sso` | `integrations_sso` |
| `integrations/scheduling` | `integrations_scheduling` |
| `integrations/productivity` | `integrations_productivity` |
| `integrations/getting-started-with-integrations` | `integrations_overview` |
| `settings/` | `settings` |
| `library/` | `library` |
| `engage/` | `engage` |
| `chakra/` | `chakra` |
| `skillup/` | `skillup` |
| `general-help/` | `general_help` |
| `uncategorized` | `uncategorized` |

### Claude

| Path prefix | Canonical label |
|---|---|
| `claude/account-management` | `account_management` |
| `claude/conversation-management` | `conversation_management` |
| `claude/features-and-capabilities` | `features` |
| `claude/troubleshooting` | `troubleshooting` |
| `claude/usage-and-limits` | `usage_limits` |
| `claude/get-started-with-claude` | `getting_started` |
| `claude/personalization-and-settings` | `personalization` |
| `claude-api-and-console/` | `api_console` |
| `claude-code/` | `claude_code` |
| `claude-desktop/` | `claude_desktop` |
| `claude-mobile-apps/` | `claude_mobile` |
| `claude-in-chrome` | `claude_in_chrome` |
| `team-and-enterprise-plans/` | `enterprise` |
| `pro-and-max-plans/` | `pro_max_plans` |
| `identity-management-sso-jit-scim` | `identity_sso` |
| `privacy-and-legal` | `privacy` |
| `amazon-bedrock/` | `amazon_bedrock` |
| `connectors/` | `connectors` |
| `safeguards` | `safeguards` |
| `claude-for-education` | `education` |
| `claude-for-government` | `government` |
| `claude-for-nonprofits` | `nonprofits` |

### Visa

| Path prefix | Canonical label |
|---|---|
| `support/consumer/travel-support` | `travel_support` |
| `support/consumer` | `consumer_support` |
| `support/small-business` | `small_business` |

---

## 8. SQLite Store (`storage/sqlite_store.py`)

Uses Python stdlib `sqlite3` — no extra dependency.

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    company      TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    product_area TEXT NOT NULL,
    doc_type     TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    content      TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_company      ON chunks(company);
CREATE INDEX IF NOT EXISTS idx_product_area ON chunks(product_area);
```

**Key methods:**
- `insert_chunk(chunk: Chunk) -> int` — inserts and returns the auto-incremented `id` (stored as `sqlite_id` on the Chunk model)
- `get_chunk(sqlite_id: int) -> Chunk` — fetch full text and metadata by ID (used after retrieval to load full context)
- `clear_company(company: str)` — `DELETE FROM chunks WHERE company = ?` (called before re-indexing for idempotency)

---

## 9. ChromaDB Store (`storage/chroma_store.py`)

Uses `chromadb.PersistentClient` directly — **not** the ChromaDB MCP tool.

> **Why not MCP?** The `chroma-mcp` ADK integration is designed for LLM-driven agents that decide _when_ to query ChromaDB via tool calls. Our `TriageAgent` calls ChromaDB deterministically from Python — the MCP layer adds unnecessary overhead and process management complexity.

**Initialisation:**
```
PersistentClient(path="./chroma_store")
Three collections: "hackerrank", "claude", "visa"
Collection metadata: {"hnsw:space": "cosine"}
embedding_function: None  ← embeddings are always provided externally from Jina
```

**Chunk IDs:** `"{source_path}:{chunk_index}"` — deterministic string, enables safe `upsert`.

**Metadata stored per record:** `product_area`, `doc_type`, `source_path`, `sqlite_id`, `heading_path`

**Key methods:**

`add_chunks(company, chunks, embeddings)`:
- Calls `collection.upsert(ids, embeddings, documents, metadatas)`
- Use `upsert` not `add` — safe to re-run without pre-clearing

`query(company, query_embedding, top_k, where)`:
- `company=None` → query all 3 collections and merge results
- `where` dict for optional metadata pre-filter (e.g. `{"doc_type": "faq"}`)
- Returns list of `{text, metadata, distance}` dicts sorted by distance

`clear_company(company)`:
- Delete + recreate collection (for idempotent full re-index)

---

## 10. Jina API Client (`embedder.py`)

Both embedding and reranking use the same `JINA_API_KEY` environment variable.

| Setting | Value |
|---|---|
| Embed model | `jina-embeddings-v3` |
| Rerank model | `jina-reranker-v2-base-multilingual` |
| Embed endpoint | `https://api.jina.ai/v1/embeddings` |
| Rerank endpoint | `https://api.jina.ai/v1/rerank` |
| Batch size | 100 texts per API call |

**Methods:**
- `embed_texts(texts: list[str]) -> list[list[float]]` — splits into batches of 100, calls Jina, returns flat list
- `embed_query(text: str) -> list[float]` — single embedding for a query string at triage time
- `rerank(query: str, documents: list[str], top_n: int) -> list[dict]` — returns top-n `{index, relevance_score, document}` dicts

Raise `EnvironmentError` with a clear message if `JINA_API_KEY` is not set.

---

## 11. Indexer Orchestration (`indexer.py`)

Runs per company in sequence. Full pipeline:

```
For each company in [hackerrank, claude, visa]:
  1. clear_company(company) in SQLite and ChromaDB
  2. Walk data/<company>/ → collect all .md files
  3. For each file:
       a. Read raw content
       b. preprocessor.clean(raw) → (cleaned_content, breadcrumbs, title)
       c. detect_doc_type(cleaned_content, filepath)
       d. resolve_product_area(filepath, company) via PRODUCT_AREA_MAP
       e. chunker.chunk(cleaned_content, doc_type) → list[str] of chunk texts
       f. For each chunk text: build Chunk Pydantic model
  4. Batch embed all chunks for the company:
       embeddings = embedder.embed_texts([c.text for c in all_chunks])
  5. For each (chunk, embedding):
       chunk.sqlite_id = sqlite_store.insert_chunk(chunk)
  6. chroma_store.add_chunks(company, all_chunks, embeddings)
  7. Print IndexResult summary
```

**Batching rationale:** Embed all chunks for one company in a single pass (batches of 100 to Jina) rather than file-by-file. Minimises API round-trips and avoids rate limiting.

---

## 12. Verification Checklist

After running `python main.py update-knowledge-base --dir data`:

- [ ] `chroma_store/` directory exists and has non-zero size
- [ ] `knowledge_base.db` exists
- [ ] `SELECT company, COUNT(*) FROM chunks GROUP BY company;` → non-zero, stable counts for all 3 companies
- [ ] Spot-check chunk: `sqlite_store.get_chunk(1)` — verify `heading_path` is well-formed and `content` has no YAML frontmatter or image markdown
- [ ] Test query: embed `"how to invite a candidate to a test"` → `chroma_store.query("hackerrank", embedding, top_k=5)` → top results include chunks from `screen/invite-candidates/`
- [ ] Idempotency: run command twice → `COUNT(*)` values are identical (upsert not insert)
- [ ] Doc type distribution: `SELECT doc_type, COUNT(*) FROM chunks GROUP BY doc_type;` — verify `how-to`, `integration`, `faq` are all represented
