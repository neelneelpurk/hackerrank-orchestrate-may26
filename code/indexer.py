"""Knowledge base indexer: walks data/, chunks, embeds via Jina, persists to SQLite + ChromaDB.

Resumable by design (per spec — see SqliteStore docstring):
  - chunk text is hashed and stored in SQLite alongside an `embedded_at` timestamp
  - on re-run, any chunk whose (source_path, chunk_index, content_hash) already exists
    AND has embedded_at != NULL is **skipped** (no re-embedding cost)
  - if Jina rate-limits halfway through, the run can resume by simply re-invoking
    `update-knowledge-base` — only the chunks that didn't make it to Chroma will be
    embedded the second time

Dual-text path:
  - SQLite stores the canonical chunk text (`chunks.content`) — the Solver always
    reads from here (via tools.retrieve's _refresh_from_sqlite step).
  - Chroma stores the **embedding text** — for FAQ chunks this is the chunk text
    PLUS heuristic question variants (stop-word-stripped keyword form). The
    extra phrasings widen the embedding's match surface so a query like
    "cancel test" still hits a chunk whose original question is "How Can I
    Cancel a Test Invite?".
  - At retrieve time, evidence text is refreshed from SQLite so the Solver
    never sees the variant noise — only the clean canonical chunk.

Pass `force=True` to do a clean rebuild (clears the company in both stores first).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path

import embedder
from chunker import chunk_document, detect_doc_type
from models import IndexResult
from preprocessor import clean
from storage.chroma_store import ChromaStore
from storage.sqlite_store import SqliteStore


_KEYWORD_STOP = frozenset({
    "how", "what", "where", "when", "why", "who", "which",
    "can", "do", "does", "did", "is", "are", "was", "were", "be",
    "i", "you", "we", "us", "my", "your", "our",
    "the", "a", "an", "to", "of", "in", "on", "at", "for", "with",
    "and", "or", "but", "this", "that", "these", "those", "it",
    "please", "kindly", "if",
})

# Overlap between adjacent chunks within the same source file. Standard RAG
# best-practice — when a topic spans a chunk boundary, the second chunk has
# the previous chunk's tail prepended so the reranker can still pull it on
# queries that hit near the join. 300 chars ≈ 2-3 sentences, ~10% of the soft
# target chunk size.
CHUNK_OVERLAP_CHARS = int(os.environ.get("CHUNK_OVERLAP_CHARS", "300"))


def _normalize_for_dedup(text: str) -> str:
    """Hash key for content-level deduplication. Collapses whitespace + drops
    the bracket-prefix so two copies of the same paragraph hash equal even
    when their heading_paths differ."""
    body = re.sub(r"^\[[^\]]+\]\s*\n+", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", body).strip().lower()


def _add_intra_file_overlap(
    chunks: list[tuple[str, str]],
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[tuple[str, str]]:
    """Prepend the tail of chunk N-1 to chunk N as continuation context.

    Operates on the chunker's `(heading_path, text)` output for one file. The
    `text` shape is `[heading_path]\\n\\n<body>` — we splice the overlap right
    after the bracket header so the heading prefix stays at the top.

    Chunks of < 200 chars (typical for atomic Q&A pairs and short FAQ files)
    are left untouched — the heading already encodes everything they need
    and overlap there is just noise.
    """
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    out = [chunks[0]]
    for i in range(1, len(chunks)):
        path, text = chunks[i]
        if len(text) < 200:
            out.append((path, text))
            continue
        prev_text = chunks[i - 1][1]
        # Skip the prev chunk's bracket header when grabbing the tail.
        prev_body = re.sub(r"^(?:\[[^\]]+\]\s*\n+)+", "", prev_text)
        tail = prev_body[-overlap_chars:] if len(prev_body) > overlap_chars else prev_body
        # Snap to a sentence start so the continuation reads naturally.
        sentence_start = re.search(r"(?<=[.!?])\s+(?=[A-Z\-\d])", tail)
        if sentence_start and sentence_start.start() < overlap_chars * 0.5:
            tail = tail[sentence_start.end():]
        tail = tail.strip()
        if not tail:
            out.append((path, text))
            continue
        # Inject the overlap right after the existing bracket header(s).
        m = re.match(r"^((?:\[[^\]]+\]\s*\n+)+)", text)
        if m:
            header = m.group(1)
            body = text[m.end():]
            new_text = f"{header}[Continued from previous chunk] …{tail}\n\n{body}"
        else:
            new_text = f"[Continued from previous chunk] …{tail}\n\n{text}"
        out.append((path, new_text))
    return out

_HEADING_QUESTION_RE = re.compile(r"^#{1,6}\s+(.+\?)\s*$", re.MULTILINE)
_QA_LINE_RE = re.compile(r"^Q\s*:\s*(.+\??)\s*$", re.MULTILINE | re.IGNORECASE)


def _extract_question(chunk_text: str) -> str:
    """Pull the most likely 'question' string out of a FAQ-style chunk.

    Looks at the leading H?/Q: lines after the [heading_path] prefix.
    Returns "" when nothing question-like is found.
    """
    lines = chunk_text.splitlines()
    # Skip the leading [heading_path] line.
    body = "\n".join(lines[1:] if lines and lines[0].startswith("[") and lines[0].endswith("]") else lines)
    m = _QA_LINE_RE.search(body)
    if m:
        return m.group(1).strip()
    m = _HEADING_QUESTION_RE.search(body)
    if m:
        return m.group(1).strip()
    # Fallback: first non-empty line if it ends in '?'
    for ln in body.splitlines():
        s = ln.strip()
        if s.endswith("?") and len(s) < 200:
            return s
    return ""


def _keyword_form(question: str) -> str:
    """Strip question-stem words + stopwords; return a keyword phrase."""
    tokens = [t for t in re.split(r"\W+", question.lower()) if t and t not in _KEYWORD_STOP]
    return " ".join(tokens)


def _augment_for_faq_embedding(chunk_text: str, doc_type_value: str) -> str:
    """For FAQ-style chunks, append heuristic question variants to the embedding text.

    SQLite still stores `chunk_text` (untouched). Chroma stores the augmented
    string returned here, and that's what gets embedded. The Solver fetches the
    clean version from SQLite at retrieve time so it never sees the variants.
    """
    if doc_type_value != "faq":
        return chunk_text
    question = _extract_question(chunk_text)
    if not question:
        return chunk_text
    keyword = _keyword_form(question)
    variants = [question]
    if keyword and keyword != question.lower():
        variants.append(keyword)
    if len(variants) <= 1:
        return chunk_text
    # Embedding-only block — never written to SQLite.
    return f"{chunk_text}\n\nRelated phrasings: {' | '.join(variants)}"


log = logging.getLogger("indexer")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(h)
log.setLevel(os.environ.get("INDEXER_LOG_LEVEL", "INFO").upper())


PRODUCT_AREA_MAP_HACKERRANK = [
    ("integrations/applicant-tracking-systems/", "integrations_ats"),
    ("integrations/single-sign-on", "integrations_sso"),
    ("integrations/scheduling", "integrations_scheduling"),
    ("integrations/productivity", "integrations_productivity"),
    ("integrations/getting-started-with-integrations", "integrations_overview"),
    ("integrations/", "integrations"),
    ("screen/", "screen"),
    ("interviews/", "interviews"),
    ("hackerrank_community/", "community"),
    ("settings/", "settings"),
    ("library/", "library"),
    ("engage/", "engage"),
    ("chakra/", "chakra"),
    ("skillup/", "skillup"),
    ("general-help/", "general_help"),
    ("uncategorized", "uncategorized"),
]

PRODUCT_AREA_MAP_CLAUDE = [
    ("claude/account-management", "account_management"),
    ("claude/conversation-management", "conversation_management"),
    ("claude/features-and-capabilities", "features"),
    ("claude/troubleshooting", "troubleshooting"),
    ("claude/usage-and-limits", "usage_limits"),
    ("claude/get-started-with-claude", "getting_started"),
    ("claude/personalization-and-settings", "personalization"),
    ("claude-api-and-console/", "api_console"),
    ("claude-code/", "claude_code"),
    ("claude-desktop/", "claude_desktop"),
    ("claude-mobile-apps/", "claude_mobile"),
    ("claude-in-chrome", "claude_in_chrome"),
    ("team-and-enterprise-plans/", "enterprise"),
    ("pro-and-max-plans/", "pro_max_plans"),
    ("identity-management-sso-jit-scim", "identity_sso"),
    ("privacy-and-legal", "privacy"),
    ("amazon-bedrock/", "amazon_bedrock"),
    ("connectors/", "connectors"),
    ("safeguards", "safeguards"),
    ("claude-for-education", "education"),
    ("claude-for-government", "government"),
    ("claude-for-nonprofits", "nonprofits"),
    ("claude/", "general"),
]

PRODUCT_AREA_MAP_VISA = [
    ("support/consumer/travel-support", "travel_support"),
    ("support/consumer", "consumer_support"),
    ("support/small-business", "small_business"),
    ("support", "general_support"),
]


def resolve_product_area(rel_path: str, company: str) -> str:
    p = rel_path.lower().replace("\\", "/")
    if company == "hackerrank":
        for prefix, label in PRODUCT_AREA_MAP_HACKERRANK:
            if prefix in p:
                return label
        return "general_help"
    if company == "claude":
        for prefix, label in PRODUCT_AREA_MAP_CLAUDE:
            if prefix in p:
                return label
        return "general"
    if company == "visa":
        for prefix, label in PRODUCT_AREA_MAP_VISA:
            if prefix in p:
                return label
        return "general_support"
    return "uncategorized"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def index_company(
    company: str,
    data_dir: Path,
    sqlite_store: SqliteStore,
    chroma_store: ChromaStore,
    force: bool = False,
) -> IndexResult:
    t0 = time.time()
    company_dir = data_dir / company
    if not company_dir.exists():
        return IndexResult(company=company, files_processed=0, chunks_created=0, duration_seconds=0.0)

    if force:
        log.info("[%s] --force: clearing SQLite + Chroma rows", company)
        sqlite_store.clear_company(company)
        chroma_store.clear_company(company)

    md_files = sorted(company_dir.rglob("*.md"))
    md_files = [f for f in md_files if f.name != "index.md"]

    files_processed = 0
    n_skipped = 0           # already embedded with same content
    n_to_embed = 0          # need (re)embedding
    n_deduped = 0           # dropped by content-level dedup (boilerplate appearing in many files)
    pending_records: list[dict] = []  # only the ones that need an actual Jina call

    # Content-level dedup: chunks whose normalised body has been seen before
    # are dropped from this run. Saves Jina cost AND prevents the retriever
    # from returning N near-identical hits for boilerplate paragraphs that
    # appear across articles ("contact support at...", license disclaimers).
    seen_body_hashes: set[str] = set()

    for md_file in md_files:
        try:
            raw = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            log.warning("read failed %s: %s", md_file, e)
            continue
        cleaned = clean(raw)
        if not cleaned.content.strip():
            continue
        rel_path = str(md_file.relative_to(data_dir.parent))
        product_area = resolve_product_area(str(md_file.relative_to(data_dir)), company)
        doc_type = detect_doc_type(cleaned.content, str(md_file), cleaned.breadcrumbs)
        try:
            chunks = chunk_document(
                cleaned.content,
                doc_type,
                cleaned.breadcrumbs,
                cleaned.title,
                company,
                product_area,
            )
        except Exception as e:
            log.warning("chunker failed %s: %s", md_file, e)
            chunks = []
        if not chunks:
            continue
        files_processed += 1

        # Add 300-char overlap between adjacent chunks from the same file so
        # the reranker can still surface chunks for queries that hit near a
        # chunk boundary.
        chunks = _add_intra_file_overlap(chunks)

        for idx, (heading_path, text) in enumerate(chunks):
            # Content-level dedup: hash the normalised body (heading_path stripped,
            # whitespace collapsed, lower-cased) so the same paragraph appearing in
            # multiple files is indexed once.
            body_hash = _hash(_normalize_for_dedup(text))
            if body_hash in seen_body_hashes:
                n_deduped += 1
                continue
            seen_body_hashes.add(body_hash)

            chash = _hash(text)
            existing = sqlite_store.lookup_chunk(company, rel_path, idx)

            if (
                not force
                and existing is not None
                and existing.get("content_hash") == chash
                and existing.get("embedded_at")
            ):
                # Already embedded with identical content → skip Jina entirely.
                # Still refresh metadata in SQLite (cheap, in case product_area mapping changed).
                sqlite_store.upsert_chunk(
                    company=company, source_path=rel_path, product_area=product_area,
                    doc_type=doc_type.value, heading_path=heading_path, content=text,
                    chunk_index=idx, content_hash=chash,
                )
                n_skipped += 1
                continue

            # Either new, content-changed, or never finished embedding → queue for Jina.
            sqlite_id = sqlite_store.upsert_chunk(
                company=company, source_path=rel_path, product_area=product_area,
                doc_type=doc_type.value, heading_path=heading_path, content=text,
                chunk_index=idx, content_hash=chash,
            )
            embedding_text = _augment_for_faq_embedding(text, doc_type.value)
            pending_records.append({
                "id": f"{rel_path}:{idx}",
                "sqlite_id": sqlite_id,
                "text": text,                       # clean canonical (for Chroma documents)
                "embedding_text": embedding_text,   # variant-augmented (for embedding only)
                "metadata": {
                    "company": company,
                    "product_area": product_area,
                    "doc_type": doc_type.value,
                    "source_path": rel_path,
                    "heading_path": heading_path,
                    "sqlite_id": sqlite_id,
                },
            })
            n_to_embed += 1

    sqlite_store.commit()

    log.info(
        "[%s] %d files scanned, %d chunks already embedded (skipped), %d pending, %d deduped",
        company, files_processed, n_skipped, n_to_embed, n_deduped,
    )

    if pending_records:
        log.info("[%s] embedding %d new/changed chunks via Jina…", company, n_to_embed)
        try:
            # Embed the variant-augmented `embedding_text` so the vector picks up
            # alternate phrasings, but store the clean `text` as the Chroma
            # document — the Solver only ever sees the SQLite-authoritative copy
            # via the retrieve refresh step in code/tools.py.
            embeddings = embedder.embed_texts([r["embedding_text"] for r in pending_records])
        except Exception as e:
            log.error("[%s] embedding failed mid-run — %s. Re-run to resume; "
                      "%d chunks already in SQLite without embeddings.",
                      company, e, n_to_embed)
            raise
        if len(embeddings) != len(pending_records):
            raise RuntimeError(
                f"jina returned {len(embeddings)} vectors for {len(pending_records)} inputs"
            )
        chroma_store.add_chunks(
            company=company,
            ids=[r["id"] for r in pending_records],
            embeddings=embeddings,
            documents=[r["text"] for r in pending_records],
            metadatas=[r["metadata"] for r in pending_records],
        )
        sqlite_store.mark_embedded([r["sqlite_id"] for r in pending_records])
        log.info("[%s] embedded + persisted %d chunks", company, n_to_embed)
    else:
        log.info("[%s] nothing to embed — fully up to date", company)

    return IndexResult(
        company=company,
        files_processed=files_processed,
        chunks_created=n_skipped + n_to_embed,
        duration_seconds=time.time() - t0,
    )


def run(data_dir: str = "data", force: bool = False) -> list[IndexResult]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    sqlite_store = SqliteStore("knowledge_base.db")
    chroma_store = ChromaStore("chroma_store")
    results = []
    for company in ("hackerrank", "claude", "visa"):
        log.info("[index] %s%s", company, " (force rebuild)" if force else "")
        pending_before = sqlite_store.counts_pending(company)
        if pending_before:
            log.info("[%s] %d chunks pending embedding from prior run", company, pending_before)
        r = index_company(company, data_path, sqlite_store, chroma_store, force=force)
        log.info(
            "[%s] done — files=%d chunks=%d time=%.1fs",
            company, r.files_processed, r.chunks_created, r.duration_seconds,
        )
        results.append(r)
    return results
