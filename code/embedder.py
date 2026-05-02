"""Jina API client — embeddings (jina-embeddings-v3) and reranker (jina-reranker-v2-base-multilingual).

Both endpoints share JINA_API_KEY. Two tiers of rate-limit handling:

1. **Token-rate pacer** (rolling 60s window): before every embed POST we estimate
   the batch's tokens (chars/4) and block until the rolling window has room. This
   prevents the bursty "send 40 batches as fast as we can" pattern that trips
   Jina's "tokens per minute" quota even when individual requests are small.

2. **429 with body-aware backoff**: if Jina still returns 429 (e.g. another
   process is sharing the key), we parse "X/Y tokens per minute" from the body
   and sleep ≥ the time it takes the oldest tokens to roll out of the window.

An in-process LRU cache covers query embeddings (handy during calibration sweeps
when the same query string repeats).

Logging philosophy: print every HTTP failure with status code + response body so
you can debug from a single stderr stream — no silent retries.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from collections import deque
from functools import lru_cache
from threading import Lock

import requests


EMBED_MODEL = os.environ.get("JINA_EMBED_MODEL", "jina-embeddings-v3")
# v3 reranker introduces "last but not late" listwise interaction — best in class
# on BEIR (61.85 nDCG@10) and noticeably stronger than v2-base on multi-hop queries.
# Override via env if v3 is unavailable on your account: JINA_RERANK_MODEL=jina-reranker-v2-base-multilingual
RERANK_MODEL = os.environ.get("JINA_RERANK_MODEL", "jina-reranker-v3")
EMBED_URL = "https://api.jina.ai/v1/embeddings"
RERANK_URL = "https://api.jina.ai/v1/rerank"

# Free-tier Jina is ~100k tokens/min. Default to 80k for safety margin; override
# upward if you have a paid key, or downward if multiple processes share the key.
TOKEN_LIMIT_PER_MIN = int(os.environ.get("JINA_TOKENS_PER_MINUTE", "80000"))

# Conservative chars-per-token fallback (used when tiktoken unavailable or
# fails). Real measured ratio on this corpus is ~2.5–2.9 chars/token —
# markdown tables, code blocks and en-dashes tokenise denser than prose. The
# previous 3.5 default under-counted and produced batches that Jina rejected
# as over the 8194-token per-request budget.
CHARS_PER_TOKEN = float(os.environ.get("JINA_CHARS_PER_TOKEN", "2.5"))

# Jina v3 uses a SentencePiece tokenizer (xlm-roberta-base derived) that
# counts ~70-80% MORE tokens than OpenAI's cl100k_base BPE on multilingual
# markdown with code/tables/dashes. Empirically: a 27k-char batch that
# tiktoken counts as ~4775 tokens, Jina rejects as 8194+ — real ratio ≈ 1.72.
# We over-multiply slightly (1.8) for safety so the budget batcher stays
# strictly under Jina's per-request cap.
JINA_TIKTOKEN_SAFETY = float(os.environ.get("JINA_TIKTOKEN_SAFETY", "1.8"))

# Per-text size cap. Jina v3 max is 8194 tokens per text. Worst-case token
# density on this corpus is ~2 chars/token, so 16000 chars caps a single chunk
# at the model limit. **No truncation** — anything exceeding the cap raises a
# RuntimeError pointing back to the chunker. Truncation lossily drops corpus
# content; splitting upstream in `chunker._split_oversized` is the right fix.
PER_TEXT_CHAR_CAP = int(os.environ.get("JINA_PER_TEXT_CHAR_CAP", "16000"))

# Jina v3 enforces a per-REQUEST token budget of 8194 tokens summed across
# the batch's `input` array (NOT per-text). We dynamically pack chunks into
# batches whose Jina-equivalent token total (tiktoken count × safety
# multiplier) stays under BATCH_TOKEN_BUDGET. Setting it to 6500 gives a
# real margin even when our tokenizer estimate is slightly off on a
# particular batch's content mix.
BATCH_TOKEN_BUDGET = int(os.environ.get("JINA_BATCH_TOKEN_BUDGET", "6500"))

# Hard ceiling on texts per batch — even when chunks are tiny, capping the
# array size keeps a single failure from forcing a re-embed of 100s of texts.
BATCH_MAX_TEXTS = int(os.environ.get("JINA_BATCH_MAX_TEXTS", "32"))

# Jina v3 supports Matryoshka truncation (128/256/512/768/1024). 512 is the
# documented sweet spot for support-agent KBs — ~92–95% of full-dim retrieval
# quality at half the storage and faster cosine ops. Bump to 1024 only if you
# are bench-tuning on hard semantic tickets where the last 5% matters.
JINA_DIMENSIONS = int(os.environ.get("JINA_DIMENSIONS", "512"))

# `late_chunking=true` on the PASSAGE side conceptually improves recall by
# encoding the full chunk with surrounding context and pooling. In practice
# Jina's late-chunking tokenizer rejects inputs it considers borderline ("empty,
# whitespace-only, or beyond the truncation window") with HTTP 422, and the
# `truncate=true` parameter does NOT apply when late_chunking is on — so a
# single long markdown chunk in a batch fails the entire batch.
#
# Default OFF. We rely on regular per-chunk embedding (which respects
# `truncate=true`) plus our chunker's MAX_CHUNK_CHARS=12000 cap to keep inputs
# well under the model limit. Turn back on with JINA_LATE_CHUNKING=1 if you
# pre-validated your corpus survives Jina's late-chunking tokenizer.
JINA_LATE_CHUNKING_PASSAGE = os.environ.get("JINA_LATE_CHUNKING", "0") not in ("0", "false", "False", "")

log = logging.getLogger("embedder")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(h)
log.setLevel(os.environ.get("EMBEDDER_LOG_LEVEL", "INFO").upper())


# -------------------- Rolling-window token pacer --------------------

_RATE_LOCK = Lock()
_TOKEN_HISTORY: "deque[tuple[float, int]]" = deque()  # (timestamp, tokens)


def _prune_window(now: float) -> int:
    cutoff = now - 60.0
    while _TOKEN_HISTORY and _TOKEN_HISTORY[0][0] < cutoff:
        _TOKEN_HISTORY.popleft()
    return sum(t for _, t in _TOKEN_HISTORY)


def _reserve_tokens(estimated: int) -> None:
    """Block until estimated tokens fit in the rolling 60s window, then reserve them."""
    while True:
        with _RATE_LOCK:
            now = time.time()
            current = _prune_window(now)
            if current + estimated <= TOKEN_LIMIT_PER_MIN:
                _TOKEN_HISTORY.append((now, estimated))
                return
            oldest_ts = _TOKEN_HISTORY[0][0]
            sleep_for = max(1.0, oldest_ts + 60.0 - now + 0.5)
        log.info(
            "rate pacer: window=%d/%d, +%d would exceed — sleeping %.1fs",
            current, TOKEN_LIMIT_PER_MIN, estimated, sleep_for,
        )
        time.sleep(sleep_for)


def _record_actual_tokens(estimated: int, actual: int) -> None:
    """Replace the most recent estimated reservation with actual usage."""
    if actual <= 0 or actual == estimated:
        return
    with _RATE_LOCK:
        if not _TOKEN_HISTORY:
            return
        ts, _ = _TOKEN_HISTORY[-1]
        _TOKEN_HISTORY[-1] = (ts, actual)


def _release_tokens(estimated: int) -> None:
    """On a definitive failure we'd otherwise double-count; pop the most recent reservation."""
    with _RATE_LOCK:
        if _TOKEN_HISTORY:
            _TOKEN_HISTORY.pop()


_TIKTOKEN_ENC = None
_TIKTOKEN_FAILED = False


def _tiktoken_count(text: str) -> int | None:
    """Accurate token count via tiktoken cl100k_base. Returns None if tiktoken
    is unavailable or errors — caller falls back to chars/CHARS_PER_TOKEN.

    We use cl100k_base (OpenAI's multilingual BPE) as a proxy for Jina v3's
    SentencePiece tokenizer. They agree to within ~10% on English markdown
    and ~20% on mixed scripts; close enough that tiktoken+a 7000-token budget
    keeps us safely under Jina's 8194 hard limit on every batch.
    """
    global _TIKTOKEN_ENC, _TIKTOKEN_FAILED
    if _TIKTOKEN_FAILED:
        return None
    if _TIKTOKEN_ENC is None:
        try:
            import tiktoken
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_FAILED = True
            return None
    try:
        return len(_TIKTOKEN_ENC.encode(text, disallowed_special=()))
    except Exception:
        return None


def _count_tokens(text: str) -> int:
    """Best-effort token count, *inflated to track Jina v3's actual tokenizer*.

    Order of preference:
      1. tiktoken cl100k_base × JINA_TIKTOKEN_SAFETY (the +35% multiplier
         compensates for Jina's SentencePiece tokenizer producing more tokens
         on multilingual/code/markdown content).
      2. chars / CHARS_PER_TOKEN (already conservative at 2.5).
    """
    n = _tiktoken_count(text)
    if n is not None:
        return max(1, int(math.ceil(n * JINA_TIKTOKEN_SAFETY)))
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def _estimate_tokens(payload: dict) -> int:
    inputs = payload.get("input") or payload.get("documents") or []
    if not isinstance(inputs, list):
        inputs = [inputs]
    total = 0
    for x in inputs:
        if isinstance(x, str):
            total += _count_tokens(x)
    q = payload.get("query")
    if isinstance(q, str):
        total += _count_tokens(q)
    return max(1, total)


# -------------------- HTTP --------------------

def _api_key() -> str:
    key = os.environ.get("JINA_API_KEY")
    if not key:
        raise EnvironmentError("JINA_API_KEY is not set")
    return key


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }


def _summarise_payload(payload: dict) -> str:
    inputs = payload.get("input") or payload.get("documents") or []
    if not isinstance(inputs, list):
        inputs = [inputs]
    n = len(inputs)
    if n == 0:
        return "payload: empty input"
    chars = [len(x) if isinstance(x, str) else 0 for x in inputs]
    est = _estimate_tokens(payload)
    return (
        f"payload: model={payload.get('model')} task={payload.get('task')} "
        f"n={n} chars(min/median/max/total)="
        f"{min(chars)}/{sorted(chars)[n // 2]}/{max(chars)}/{sum(chars)} "
        f"est_tokens≈{est}"
    )


def _check_input_sizes(payload: dict) -> None:
    """Strict size pre-flight: raise if any input would otherwise need truncation.

    We do not truncate — silently dropping characters from a corpus chunk loses
    information that the Solver later cites. Instead, we fail the run with the
    list of offending inputs so the operator can lower `CHUNK_MAX_CHARS` in the
    chunker and re-index.
    """
    items: list[str] = []
    if "input" in payload and isinstance(payload["input"], list):
        items = [s for s in payload["input"] if isinstance(s, str)]
    elif "documents" in payload and isinstance(payload["documents"], list):
        items = [s for s in payload["documents"] if isinstance(s, str)]
    over = [(i, len(s)) for i, s in enumerate(items) if len(s) > PER_TEXT_CHAR_CAP]
    if over:
        details = ", ".join(f"idx {i}={chars} chars" for i, chars in over[:5])
        more = "" if len(over) <= 5 else f" (+{len(over) - 5} more)"
        raise RuntimeError(
            f"Refusing to embed: {len(over)} input(s) exceed PER_TEXT_CHAR_CAP="
            f"{PER_TEXT_CHAR_CAP} chars. Lower CHUNK_MAX_CHARS in chunker.py and "
            f"re-run `update-knowledge-base --force`. First offenders: {details}{more}"
        )


_RATE_BODY_RE = re.compile(r"(\d[\d,]*)\s*/\s*(\d[\d,]*)\s*tokens? per minute", re.IGNORECASE)


def _parse_rate_body(body: str) -> tuple[int, int] | None:
    m = _RATE_BODY_RE.search(body or "")
    if not m:
        return None
    used = int(m.group(1).replace(",", ""))
    cap = int(m.group(2).replace(",", ""))
    return used, cap


def _post(url: str, payload: dict, max_retries: int = 6, paced: bool = True) -> dict:
    """POST with retry. Logs the response body on every non-2xx so failures are visible."""
    _check_input_sizes(payload)  # raises on oversized inputs (no silent truncation)

    estimated_tokens = _estimate_tokens(payload)
    if paced:
        _reserve_tokens(estimated_tokens)

    last_err_msg = None
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            r = requests.post(url, json=payload, headers=_headers(), timeout=180)
        except requests.RequestException as e:
            last_err_msg = f"network error: {type(e).__name__}: {e}"
            log.warning(
                "POST %s attempt=%d/%d FAILED (%.1fs) — %s | %s",
                url, attempt + 1, max_retries, time.time() - t0, last_err_msg, _summarise_payload(payload),
            )
            if attempt == max_retries - 1:
                if paced:
                    _release_tokens(estimated_tokens)
                raise RuntimeError(f"jina POST {url} failed: {last_err_msg}")
            time.sleep(2 ** attempt)
            continue

        dt = time.time() - t0
        if r.status_code == 200:
            try:
                data = r.json()
            except Exception as e:
                if paced:
                    _release_tokens(estimated_tokens)
                raise RuntimeError(f"jina POST {url} returned 200 but bad JSON: {e}")
            usage = (data.get("usage") or {}) if isinstance(data, dict) else {}
            actual_tokens = int(usage.get("total_tokens") or 0)
            if paced and actual_tokens:
                _record_actual_tokens(estimated_tokens, actual_tokens)
            log.info(
                "POST %s OK %.2fs (%s, actual_tokens=%s)",
                url, dt, _summarise_payload(payload), actual_tokens or "n/a",
            )
            return data

        body = (r.text or "")[:800]
        last_err_msg = f"HTTP {r.status_code}: {body!r}"

        if r.status_code == 429:
            parsed = _parse_rate_body(body)
            if parsed:
                used, cap = parsed
                # Wait long enough that 'used' could meaningfully decay. Heuristic:
                # need to drop below cap - estimated_tokens; assume roughly linear
                # decay so sleep proportionally, with a 5–60s floor/ceiling.
                overage = max(0, used + estimated_tokens - cap)
                # Conservative: sleep until 60s wraps the worst case.
                sleep_for = min(60.0, max(5.0, overage / max(1, cap) * 60.0 + 5.0))
                log.warning(
                    "POST %s 429 attempt=%d/%d — quota %d/%d, batch≈%d, sleeping %.1fs",
                    url, attempt + 1, max_retries, used, cap, estimated_tokens, sleep_for,
                )
                time.sleep(sleep_for)
                continue
            # Generic 429 backoff
            sleep_for = min(60.0, 2 ** attempt)
            log.warning(
                "POST %s 429 attempt=%d/%d — sleeping %.1fs | body=%s",
                url, attempt + 1, max_retries, sleep_for, body,
            )
            time.sleep(sleep_for)
            continue

        log.warning(
            "POST %s attempt=%d/%d %.2fs — HTTP %d | %s | body=%s",
            url, attempt + 1, max_retries, dt, r.status_code,
            _summarise_payload(payload), body,
        )

        if r.status_code in (500, 502, 503, 504):
            if attempt == max_retries - 1:
                if paced:
                    _release_tokens(estimated_tokens)
                raise RuntimeError(f"jina POST {url} exhausted retries on {r.status_code}: {body!r}")
            time.sleep(2 ** attempt)
            continue

        # Non-retryable 4xx
        if paced:
            _release_tokens(estimated_tokens)
        raise RuntimeError(
            f"jina POST {url} non-retryable {r.status_code}: {body!r} | {_summarise_payload(payload)}"
        )

    if paced:
        _release_tokens(estimated_tokens)
    raise RuntimeError(f"jina POST {url} failed after {max_retries} retries: {last_err_msg}")


# -------------------- Public API --------------------

def _pack_batches(
    texts: list[str],
    token_budget: int = BATCH_TOKEN_BUDGET,
    max_texts: int = BATCH_MAX_TEXTS,
) -> list[tuple[int, list[str]]]:
    """Pack `texts` into batches whose token totals stay under `token_budget`.
    Returns a list of `(start_index, batch_texts)` tuples so the caller can log
    absolute positions.

    Token counting uses tiktoken (cl100k_base) when available — that's much
    more accurate than chars/CHARS_PER_TOKEN on token-dense markdown (tables,
    code, special chars). Falls back to the char ratio if tiktoken is missing.

    Single inputs that already exceed the budget become solo batches — they
    will hit the per-text cap check in `_check_input_sizes` (which raises if
    over PER_TEXT_CHAR_CAP) so they're a guaranteed loud failure rather than
    a silent over-budget batch.
    """
    out: list[tuple[int, list[str]]] = []
    if not texts:
        return out
    cur: list[str] = []
    cur_tokens = 0
    cur_start = 0
    for i, text in enumerate(texts):
        est = _count_tokens(text)
        # If adding this text would tip us over either limit, flush.
        if cur and (cur_tokens + est > token_budget or len(cur) >= max_texts):
            out.append((cur_start, cur))
            cur = []
            cur_tokens = 0
            cur_start = i
        cur.append(text)
        cur_tokens += est
    if cur:
        out.append((cur_start, cur))
    return out


def embed_texts(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    """Batch-embed a list of texts. Returns a list of embedding vectors in input order.

    Tuning notes:
    - `task="retrieval.passage"` for indexing (asymmetric retrieval — passages
      are encoded with a prefix that complements the `retrieval.query` side).
    - `dimensions=512` (Matryoshka sweet spot for support KBs) — explicit so a
      future Jina default change can't silently shrink us further.
    - **No truncation** — neither at our layer (`_check_input_sizes` raises) nor
      at Jina's (`truncate` is not sent). The chunker is the single place where
      chunk size is enforced; truncation would silently drop characters.
    - **Token-budget batching** — Jina v3 has a per-REQUEST token budget
      (~8194). We pack chunks dynamically so each batch's estimated total
      stays under BATCH_TOKEN_BUDGET. Small chunks share large batches; large
      chunks form smaller batches, automatically.
    - `late_chunking` defaults to **OFF** — its tokenizer is stricter than the
      regular path and 422s on borderline inputs.
    - `embedding_type="float"` and `normalized=True` so cosine similarity in
      Chroma is well-defined without further normalisation on our side.

    Rate-paced via the 60s rolling token window so a long indexing run doesn't
    hammer Jina; logs per-batch progress.
    """
    if not texts:
        return []
    n_total = len(texts)
    batches = _pack_batches(texts)
    n_batches = len(batches)
    log.info(
        "embed_texts: %d texts → %d token-budget batches (budget=%d tok/req, "
        "rate_cap=%d tok/min, dim=%d, late_chunking=%s)",
        n_total, n_batches, BATCH_TOKEN_BUDGET, TOKEN_LIMIT_PER_MIN,
        JINA_DIMENSIONS, JINA_LATE_CHUNKING_PASSAGE,
    )

    out: list[list[float]] = []
    t_start = time.time()
    for bi, (start_idx, batch) in enumerate(batches, start=1):
        payload = {
            "model": EMBED_MODEL,
            "task": task,
            "input": batch,
            "dimensions": JINA_DIMENSIONS,
            "embedding_type": "float",
            "normalized": True,
        }
        if JINA_LATE_CHUNKING_PASSAGE:
            payload["late_chunking"] = True
        total_chars = sum(len(t) for t in batch)
        est_tokens = math.ceil(total_chars / CHARS_PER_TOKEN)
        log.info(
            "embed_texts: batch %d/%d (texts %d–%d, n=%d, total chars=%d, est_tokens≈%d)",
            bi, n_batches, start_idx, start_idx + len(batch) - 1,
            len(batch), total_chars, est_tokens,
        )
        try:
            data = _post(EMBED_URL, payload)
        except Exception:
            log.error(
                "embed_texts: batch %d/%d FAILED (texts %d–%d, n=%d, est_tokens≈%d). "
                "Lower JINA_BATCH_TOKEN_BUDGET (current=%d) if Jina is stricter than expected.",
                bi, n_batches, start_idx, start_idx + len(batch) - 1,
                len(batch), est_tokens, BATCH_TOKEN_BUDGET,
            )
            raise
        if "data" not in data:
            log.error("embed_texts: batch %d response missing 'data' — full=%r", bi, data)
            raise RuntimeError(f"unexpected Jina response shape: {data}")
        out.extend([item["embedding"] for item in data["data"]])
    log.info("embed_texts: done — %d vectors in %.1fs", len(out), time.time() - t_start)
    return out


@lru_cache(maxsize=512)
def embed_query(query: str) -> tuple:
    """Embed a search query. `task='retrieval.query'` is the asymmetric pair to the
    `retrieval.passage` task used at index time — the two encoders are trained jointly
    so query↔passage similarity is meaningful even though the surface forms differ.
    Late chunking is irrelevant for short queries; truncation is intentionally absent.
    """
    payload = {
        "model": EMBED_MODEL,
        "task": "retrieval.query",
        "input": [query],
        "dimensions": JINA_DIMENSIONS,
        "embedding_type": "float",
        "normalized": True,
    }
    data = _post(EMBED_URL, payload)
    return tuple(data["data"][0]["embedding"])


def rerank(query: str, documents: list[str], top_n: int = 3) -> list[dict]:
    """Cross-encoder rerank over `documents`. No truncation — caller must ensure
    each document is within the reranker's per-doc token cap (which is what the
    chunker's MAX_CHUNK_CHARS already guarantees).
    """
    if not documents:
        return []
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
    }
    data = _post(RERANK_URL, payload)
    return [
        {"index": r["index"], "relevance_score": r["relevance_score"]}
        for r in data["results"]
    ]
