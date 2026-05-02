"""Retrieve tool — used by ReWoo workers. Dense-only retrieval (Jina v3 + Chroma + Jina rerank).

Why "the same chunks are coming back regardless of query" was happening:
  - Initial Chroma pool was 8–15 candidates → most queries pull the same H2 cluster.
  - The reranker just re-orders that small cluster, so different queries → very
    similar top-N.
  - With `top_k=3` of duplicates from the same `source_path`, the Solver sees no
    real diversity and the product_area vote is dominated by one document.

Fixes here, in order of impact:

1. **Wider initial pool** (`top_k=50` default). Cheap — vector search is
   logarithmic in collection size. Gives the reranker actual choices.

2. **Source-path diversification before rerank.** No more than `MAX_PER_SOURCE`
   chunks from the same `metadata.source_path` survive into the rerank input.
   This is a coarse but parameter-free MMR proxy; it's what's actually causing
   "same chunks every time" in this corpus where one large how-to article can
   produce 6+ adjacent chunks.

3. **Three-stage fallback** preserved: filtered → unfiltered → paraphrased.

4. **Drop-floor on rerank score.** Anything below `LOW_SIGNAL_DROP` is dropped
   before the Solver sees it (keeps at least `MIN_KEEP`).
"""

from __future__ import annotations

import os
import re
from typing import Optional

import logging

import embedder
from storage.chroma_store import ChromaStore
from storage.sqlite_store import SqliteStore


log = logging.getLogger("retrieve")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(h)
log.setLevel(os.environ.get("RETRIEVE_LOG_LEVEL", "INFO").upper())


_CHROMA: Optional[ChromaStore] = None
_SQLITE: Optional[SqliteStore] = None

# Score-gap heuristic: doubles as the HIGH/MEDIUM confidence boundary.
# Jina v3 reranker scores typically span [0.0, 0.5]; loosened from 0.12 to 0.08
# so weak-but-on-topic retrievals land in MEDIUM rather than LOW.
SCORE_GAP_AMBIGUOUS = float(os.environ.get("RETRIEVE_SCORE_GAP_AMBIGUOUS", "0.08"))

# Bigger pool than the previous 8–15. Reranker is cheap; vector search at
# top_k=50 over ~1000 chunks per company is sub-50ms.
DEFAULT_TOP_K = int(os.environ.get("RETRIEVE_TOP_K", "50"))
DEFAULT_FINAL_N = int(os.environ.get("RETRIEVE_FINAL_N", "5"))

# Cap how many chunks from the same source_path can compete in the rerank
# input. 2 lets a long article contribute its head + a sub-section, without
# letting it monopolise the candidate pool.
MAX_PER_SOURCE = int(os.environ.get("RETRIEVE_MAX_PER_SOURCE", "2"))

# Post-rerank cap: after the cross-encoder ranks, force the final-N to come
# from distinct source_paths so the Solver always sees diverse anchors.
MAX_PER_SOURCE_FINAL = int(os.environ.get("RETRIEVE_MAX_PER_SOURCE_FINAL", "1"))

# Drop-floor: chunks with rerank scores below this are dropped before being
# sent to the Solver. With wider pool + multi-query, weak chunks are cheap
# to drop; 0.05 reflects "rerank gave up" rather than the genuine-noise 0.02.
LOW_SIGNAL_DROP = float(os.environ.get("RETRIEVE_LOW_SIGNAL_DROP", "0.05"))
# Multi-query is the default; legacy paraphrase retry should rarely fire.
LOW_SIGNAL_RETRY = float(os.environ.get("RETRIEVE_LOW_SIGNAL_RETRY", "0.10"))
MIN_KEEP = int(os.environ.get("RETRIEVE_MIN_KEEP", "2"))

# Confidence band thresholds (Jina v3 score range). Loosened from 0.20/0.10 to
# 0.15/0.05 — even small-magnitude on-topic matches now land in MEDIUM, so the
# Solver doesn't bail just because absolute scores are low.
CONFIDENCE_HIGH_TOP = float(os.environ.get("RETRIEVE_CONFIDENCE_HIGH_TOP", "0.15"))
CONFIDENCE_MEDIUM_TOP = float(os.environ.get("RETRIEVE_CONFIDENCE_MEDIUM_TOP", "0.05"))


def _store() -> ChromaStore:
    global _CHROMA
    if _CHROMA is None:
        _CHROMA = ChromaStore("chroma_store")
    return _CHROMA


def _sqlite() -> SqliteStore:
    """SQLite is the authoritative source of truth for chunk text. Chroma stores
    the embedding-time text (which may include question variants for FAQs);
    SQLite stores the canonical document. At retrieve time we refresh the chunk
    text from SQLite by `metadata.sqlite_id` so the Solver always sees the
    clean version, never the variant-augmented embedding text.
    """
    global _SQLITE
    if _SQLITE is None:
        _SQLITE = SqliteStore("knowledge_base.db")
    return _SQLITE


def _refresh_from_sqlite(hits: list[dict]) -> list[dict]:
    """Replace each hit's `text` with the authoritative SQLite content,
    keyed by `metadata.sqlite_id`. Falls back to the original on any miss.
    """
    if not hits:
        return hits
    store = _sqlite()
    for h in hits:
        sid = (h.get("metadata") or {}).get("sqlite_id")
        if sid is None:
            continue
        try:
            row = store.get_chunk(int(sid))
        except Exception:
            row = None
        if row and row.get("content"):
            h["text"] = row["content"]
            # Also refresh metadata fields that might have drifted (rare).
            h["metadata"]["heading_path"] = row.get("heading_path", h["metadata"].get("heading_path", ""))
    return hits


_PRODUCT_WORDS = (
    "hackerrank", "hacker rank", "hr-", "hr ",
    "claude", "anthropic",
    "visa", "visacard",
)
_STOPWORDS = (
    "please", "kindly", "asap", "urgent", "urgently",
    "the", "a", "an",
    "i", "me", "my", "we", "our", "us",
    "is", "are", "was", "were",
)


def _paraphrase_query(query: str) -> Optional[str]:
    """Heuristic paraphrase: strip product names + stopwords, expand acronyms."""
    base = query.lower()
    for word in _PRODUCT_WORDS:
        base = base.replace(word, " ")
    tokens = [t for t in re.split(r"\W+", base) if t and t not in _STOPWORDS]
    expansions = {
        "lti": "learning tools interoperability",
        "sso": "single sign on",
        "ats": "applicant tracking system",
        "scim": "system for cross-domain identity management",
        "jit": "just in time provisioning",
    }
    expanded = [expansions.get(t, t) for t in tokens]
    paraphrased = " ".join(expanded).strip()
    if not paraphrased or paraphrased == query.strip().lower():
        return None
    return paraphrased


def _diversify_by_source(hits: list[dict], max_per_source: int) -> list[dict]:
    """Cap how many chunks from the same source_path appear in the candidate pool.

    Vector search routinely returns 4–8 adjacent chunks from one large article
    because they all embed similarly. Capping per-source forces the reranker to
    consider chunks from *other* docs that might be more relevant.
    """
    if max_per_source <= 0:
        return hits
    seen: dict[str, int] = {}
    out: list[dict] = []
    for h in hits:
        sp = (h.get("metadata") or {}).get("source_path", "")
        seen[sp] = seen.get(sp, 0) + 1
        if seen[sp] <= max_per_source:
            out.append(h)
    return out


def _dense_search(
    company: str,
    query: str,
    doc_type_filter: Optional[list[str]],
    top_k: int,
) -> list[dict]:
    qvec = list(embedder.embed_query(query))
    where = None
    if doc_type_filter:
        if len(doc_type_filter) == 1:
            where = {"doc_type": doc_type_filter[0]}
        else:
            where = {"doc_type": {"$in": list(doc_type_filter)}}
    return _store().query(company, qvec, top_k=top_k, where=where)


def _union_candidates(
    company: str,
    query_variants: list[str],
    doc_type_filter: Optional[list[str]],
    top_k: int,
) -> list[dict]:
    """Run dense search per variant and union by sqlite_id, keeping the
    closest (smallest distance) duplicate. Falls back to chroma_id for hits
    missing sqlite_id metadata.
    """
    candidates: dict = {}
    for q in query_variants:
        for h in _dense_search(company, q, doc_type_filter, top_k):
            md = h.get("metadata") or {}
            key = md.get("sqlite_id")
            if key is None:
                key = ("path", md.get("source_path", ""), md.get("chunk_index", -1))
            existing = candidates.get(key)
            if existing is None or h.get("distance", 1.0) < existing.get("distance", 1.0):
                candidates[key] = h
    return list(candidates.values())


def _confidence_band(top1: float, gap: float) -> str:
    if top1 >= CONFIDENCE_HIGH_TOP and gap >= SCORE_GAP_AMBIGUOUS:
        return "HIGH"
    if top1 >= CONFIDENCE_MEDIUM_TOP:
        return "MEDIUM"
    return "LOW"


def retrieve(
    company: str,
    query_variants,
    doc_type_filter: Optional[list[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    final_n: int = DEFAULT_FINAL_N,
) -> dict:
    """Multi-query dense retrieve → union → diversify → rerank → drop-floor → diversify.

    `query_variants` is a list of semantically-equivalent reformulations; the
    Planner emits 2–3 by default so the dense step pulls a recall-rich union
    pool. The reranker then runs ONCE against the primary (first) variant.

    For backwards compatibility, a bare string is accepted and treated as a
    single-variant list.

    Returns: {status, evidence, fallback, confidence, top_score, score_gap,
              n_variants, n_candidates_after_union, n_after_diversify,
              n_after_rerank, n_after_floor}.
    Each evidence chunk: {text, metadata, rerank_score, distance}.
    """
    # Coerce legacy string input to the variants shape.
    if isinstance(query_variants, str):
        query_variants = [query_variants]
    query_variants = [q for q in (query_variants or []) if isinstance(q, str) and q.strip()]
    if not query_variants:
        return {
            "status": "empty", "evidence": [], "fallback": "filtered",
            "confidence": "LOW", "top_score": 0.0, "score_gap": 0.0,
            "n_variants": 0, "n_candidates_after_union": 0,
            "n_after_diversify": 0, "n_after_rerank": 0, "n_after_floor": 0,
        }
    primary = query_variants[0]

    # Stage 1: filtered union across variants
    raw_hits = _union_candidates(company, query_variants, doc_type_filter, top_k)
    fallback = "filtered"

    # Stage 2: drop the doc_type filter if too aggressive
    if len(raw_hits) < 3 and doc_type_filter:
        raw_hits = _union_candidates(company, query_variants, None, top_k)
        fallback = "unfiltered"

    n_after_union = len(raw_hits)
    if not raw_hits:
        return {
            "status": "empty", "evidence": [], "fallback": fallback,
            "confidence": "LOW", "top_score": 0.0, "score_gap": 0.0,
            "n_variants": len(query_variants), "n_candidates_after_union": 0,
            "n_after_diversify": 0, "n_after_rerank": 0, "n_after_floor": 0,
        }

    # Cap chunks from the same source_path to force diversity into the rerank input.
    diverse = _diversify_by_source(raw_hits, MAX_PER_SOURCE)
    n_after_diversify = len(diverse)

    docs = [h["text"] for h in diverse]
    rerank_top_n = min(max(2 * final_n, 10), len(docs))
    try:
        reranked = embedder.rerank(primary, docs, top_n=rerank_top_n)
    except Exception:
        # Fallback ordering: keep dense order, synthetic descending scores.
        reranked = [
            {"index": i, "relevance_score": 1.0 / (i + 1)}
            for i in range(min(rerank_top_n, len(docs)))
        ]
    n_after_rerank = len(reranked)

    # Stage 3: legacy paraphrase retry — rare with multi-query default, kept as safety net.
    top_score = reranked[0]["relevance_score"] if reranked else 0.0
    if top_score < LOW_SIGNAL_RETRY:
        paraphrased = _paraphrase_query(primary)
        if paraphrased and paraphrased not in query_variants:
            raw_hits2 = _union_candidates(company, [paraphrased], None, top_k)
            if raw_hits2:
                diverse2 = _diversify_by_source(raw_hits2, MAX_PER_SOURCE)
                docs2 = [h["text"] for h in diverse2]
                rerank_top_n2 = min(max(2 * final_n, 10), len(docs2))
                try:
                    reranked2 = embedder.rerank(paraphrased, docs2, top_n=rerank_top_n2)
                except Exception:
                    reranked2 = [
                        {"index": i, "relevance_score": 1.0 / (i + 1)}
                        for i in range(min(rerank_top_n2, len(docs2)))
                    ]
                if reranked2 and reranked2[0]["relevance_score"] > top_score:
                    diverse = diverse2
                    reranked = reranked2
                    fallback = "paraphrased"
                    n_after_rerank = len(reranked)

    # Build evidence; drop very-weak chunks (keep at least MIN_KEEP).
    evidence = []
    for r in reranked:
        h = diverse[r["index"]]
        evidence.append({
            "text": h["text"],
            "metadata": h["metadata"],
            "rerank_score": float(r["relevance_score"]),
            "distance": h.get("distance", 0.0),
        })

    if len(evidence) > MIN_KEEP:
        kept = [e for e in evidence if e["rerank_score"] >= LOW_SIGNAL_DROP]
        if len(kept) < MIN_KEEP:
            kept = evidence[:MIN_KEEP]
        evidence = kept
    n_after_floor = len(evidence)

    # Post-rerank source diversification: cap to MAX_PER_SOURCE_FINAL per source_path
    # so the Solver never sees two adjacent chunks from the same article in the final-N.
    if MAX_PER_SOURCE_FINAL > 0 and len(evidence) > MIN_KEEP:
        seen: dict[str, int] = {}
        diversified: list[dict] = []
        for ev in evidence:
            sp = (ev.get("metadata") or {}).get("source_path", "")
            seen[sp] = seen.get(sp, 0) + 1
            if seen[sp] <= MAX_PER_SOURCE_FINAL:
                diversified.append(ev)
        if len(diversified) >= MIN_KEEP:
            evidence = diversified

    evidence = evidence[:final_n]

    # Refresh chunk text from SQLite (the authoritative source). Important when
    # Chroma was indexed with variant-augmented embedding text — we want the
    # Solver to see the clean canonical document, not the variants.
    evidence = _refresh_from_sqlite(evidence)

    # Confidence band derived from rerank top-1 magnitude + spread. Surfaced to
    # downstream agents (Solver/Reflector); also gates the calibration log.
    top1 = evidence[0]["rerank_score"] if evidence else 0.0
    score_gap = (top1 - evidence[-1]["rerank_score"]) if len(evidence) >= 2 else 0.0
    confidence = _confidence_band(top1, score_gap)

    if len(evidence) >= 2 and score_gap < SCORE_GAP_AMBIGUOUS:
        log.info(
            "retrieve(%s, %r): ambiguous (top1=%.3f, gap=%.3f<%.3f) confidence=%s",
            company, primary[:60], top1, score_gap, SCORE_GAP_AMBIGUOUS, confidence,
        )

    return {
        "status": "success",
        "evidence": evidence,
        "fallback": fallback,
        "confidence": confidence,
        "top_score": top1,
        "score_gap": score_gap,
        "n_variants": len(query_variants),
        "n_candidates_after_union": n_after_union,
        "n_after_diversify": n_after_diversify,
        "n_after_rerank": n_after_rerank,
        "n_after_floor": n_after_floor,
    }
