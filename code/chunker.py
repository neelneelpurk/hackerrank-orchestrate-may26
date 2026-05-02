"""Doc-type detection and per-type chunking."""

from __future__ import annotations

import os
import re
from typing import Iterable

from models import DocType


_STEP_RE = re.compile(r"\*\*Step\s+\d+:", re.IGNORECASE)
_NUMBERED_RE = re.compile(r"^\d+\.", re.MULTILINE)
_LINK_LINE_RE = re.compile(r"^\s*-\s*\[.+?\]\(.+?\)\s*$", re.MULTILINE)

# Q&A detection patterns. A document containing 2+ of any of these is treated
# as a multi-Q&A and split per pair, even if it isn't tagged FAQ. The corpus
# has both `**Q:** / **A:**` style and heading-question style.
_QA_BOLD_Q_RE = re.compile(r"^\s*\*\*\s*(?:Q|Question)\s*[:.]?\s*\*\*", re.MULTILINE | re.IGNORECASE)
_QA_HEADING_Q_RE = re.compile(r"^(#{2,4})\s+(.+\?)\s*$", re.MULTILINE)

# Fenced code block + markdown table detection. We treat these as atomic in
# the paragraph splitter so a fenced ``` block or a table never gets cut
# mid-content (which would produce gibberish in retrieved evidence).
_FENCE_RE = re.compile(r"```")
_TABLE_LINE_RE = re.compile(r"^\s*\|")

# Hard cap so no single chunk approaches Jina v3's 8194-token limit. Worst-case
# token density on this corpus is ~2 chars/token (markdown tables, code, heavy
# punctuation) so 4_000 chars ≈ 2_000 tokens — comfortably under the model
# limit AND aligned with RAG best-practice for cross-encoder rerank (chunks of
# ~512–1024 tokens score most reliably). Smaller chunks also give the
# product_area weighted vote more independent signal per ticket.
MAX_CHUNK_CHARS = int(os.environ.get("CHUNK_MAX_CHARS", "4000"))

# Soft target — the contextual splitter aims for chunks roughly this size
# rather than packing right up to MAX_CHUNK_CHARS. Below the target, no split
# is attempted; above it, the cascade runs even if the chunk is technically
# under MAX_CHUNK_CHARS.
CHUNK_TARGET_CHARS = int(os.environ.get("CHUNK_TARGET_CHARS", "3000"))


def detect_doc_type(content: str, rel_path: str, breadcrumbs: list[str]) -> DocType:
    p = rel_path.lower()
    if "release-notes" in p or "release_notes" in p:
        return DocType.RELEASE_NOTES
    if "frequently-asked-questions" in p or "/faqs/" in p:
        return DocType.FAQ
    if "troubleshooting" in p:
        return DocType.TROUBLESHOOTING
    if "privacy-and-legal" in p or "/policy" in p:
        return DocType.POLICY_LEGAL
    if "getting-started" in p and "integrations" not in p:
        return DocType.CONCEPTUAL
    if "integrations/" in p and any(
        s in p for s in ("applicant-tracking-systems", "single-sign-on", "scheduling", "productivity")
    ):
        return DocType.INTEGRATION

    bc = " > ".join(breadcrumbs).lower()
    if "release notes" in bc:
        return DocType.RELEASE_NOTES
    if "frequently asked" in bc or "faq" in bc:
        return DocType.FAQ
    if "troubleshooting" in bc:
        return DocType.TROUBLESHOOTING
    if "integrations" in bc:
        return DocType.INTEGRATION

    head = "\n".join(content.splitlines()[:15]).lower()
    if re.search(r"-\s*hackerrank\s+integration", head):
        return DocType.INTEGRATION
    if "introduction to" in head or "what is" in head or "overview" in head:
        return DocType.CONCEPTUAL
    if "frequently asked" in head:
        return DocType.FAQ
    if "troubleshoot" in head or "error messages" in head:
        return DocType.TROUBLESHOOTING

    if len(_STEP_RE.findall(content)) >= 3:
        return DocType.HOW_TO
    if len(_NUMBERED_RE.findall(content)) >= 3 and "prerequisite" in content.lower():
        return DocType.HOW_TO

    return DocType.REFERENCE


def _word_count(s: str) -> int:
    return len(s.split())


def _detect_qa_pairs(content: str) -> list[tuple[str, str]]:
    """Detect multi-Q&A patterns inside a single file.

    Returns a list of (question_text, answer_text) tuples. Empty list if no
    multi-Q&A structure is found (caller falls back to normal chunking).

    Two patterns are recognised:
      1. `**Q:** ... **A:** ...` style (or `**Question:** / **Answer:**`).
      2. H2/H3/H4 heading lines whose text ends with `?` (treated as questions);
         the answer is the body until the next question heading.

    A file is treated as multi-Q&A only when it contains 2+ pairs — single-Q&A
    files (most FAQ entries in this corpus) keep their existing single-chunk
    behaviour.
    """
    # --- Pattern 1: bold **Q:** / **A:** pairs ---
    q_starts = [m.start() for m in _QA_BOLD_Q_RE.finditer(content)]
    if len(q_starts) >= 2:
        a_re = re.compile(r"\*\*\s*(?:A|Answer)\s*[:.]?\s*\*\*", re.IGNORECASE)
        pairs: list[tuple[str, str]] = []
        for i, qs in enumerate(q_starts):
            qe = q_starts[i + 1] if i + 1 < len(q_starts) else len(content)
            block = content[qs:qe]
            am = a_re.search(block)
            if not am:
                continue
            question = block[:am.start()].strip().lstrip("*").strip(": *").strip()
            answer = block[am.end():].strip()
            if question and answer:
                pairs.append((question, answer))
        if len(pairs) >= 2:
            return pairs

    # --- Pattern 2: heading lines that are questions ---
    matches = list(_QA_HEADING_Q_RE.finditer(content))
    if len(matches) >= 2:
        pairs = []
        for i, m in enumerate(matches):
            level_marker, q_text = m.group(1), m.group(2).strip()
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            answer = content[body_start:body_end].strip()
            if q_text and answer:
                pairs.append((q_text, answer))
        if len(pairs) >= 2:
            return pairs

    return []


def _split_at_heading(content: str, level: int) -> list[str]:
    pattern = re.compile(rf"^{'#' * level} ", re.MULTILINE)
    parts: list[str] = []
    last = 0
    for m in pattern.finditer(content):
        if m.start() == 0:
            continue
        parts.append(content[last:m.start()].rstrip())
        last = m.start()
    parts.append(content[last:].rstrip())
    return [p for p in parts if p.strip()]


def _extract_h1_title(content: str) -> str:
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _h2_heading(section: str) -> str:
    for line in section.splitlines():
        if line.startswith("## "):
            return line[3:].strip()
    return ""


def _h1_heading(section: str) -> str:
    for line in section.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _build_path(breadcrumbs: list[str], extras: list[str]) -> str:
    parts = [b for b in breadcrumbs if b] + [e for e in extras if e]
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return " > ".join(out)


def _prepend_path(text: str, heading_path: str) -> str:
    return f"[{heading_path}]\n\n{text.strip()}"


def _group_pieces(pieces: list[str], max_chars: int, joiner: str = "") -> list[str]:
    """Greedy-pack consecutive pieces into chunks ≤ max_chars. Single oversized
    pieces are emitted as-is — caller is expected to recurse / hard-split them.
    """
    out: list[str] = []
    cur = ""
    for p in pieces:
        if not p.strip():
            continue
        candidate = cur + (joiner if cur else "") + p
        if not cur or len(candidate) <= max_chars:
            cur = candidate
        else:
            out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out


def _split_on_heading_level(text: str, level: int, max_chars: int) -> list[str]:
    """Split right before each `### `/`#### `/etc. line, then greedy-group."""
    marker = "#" * level + " "
    pattern = re.compile(rf"(?m)(?=^{re.escape(marker)})")
    pieces = [p for p in pattern.split(text) if p.strip()]
    if len(pieces) <= 1:
        return [text]
    return _group_pieces(pieces, max_chars, joiner="")


def _split_on_steps(text: str, max_chars: int) -> list[str]:
    """Split on `**Step N:` bold-step markers, then greedy-group up to max_chars."""
    pieces = [p for p in re.split(r"(?m)(?=^\*\*Step\s+\d+:)", text) if p.strip()]
    if len(pieces) <= 1:
        return [text]
    return _group_pieces(pieces, max_chars, joiner="")


def _segment_with_atomic_blocks(text: str) -> list[str]:
    """Walk lines and emit a sequence of segments where each fenced code block
    or markdown table is one atomic segment, and runs of regular lines are
    grouped between blank-line boundaries.

    Returns a list of paragraph-equivalent strings — used by the paragraph
    splitter so a code fence or table can never get cut mid-content.
    """
    lines = text.splitlines()
    segments: list[str] = []
    buf: list[str] = []

    def flush_buf() -> None:
        if buf:
            joined = "\n".join(buf).strip()
            if joined:
                segments.append(joined)
            buf.clear()

    in_fence = False
    fence_buf: list[str] = []
    in_table = False
    table_buf: list[str] = []

    for line in lines:
        # Fenced code blocks — treat the entire ```...``` region as ONE segment.
        if _FENCE_RE.match(line.lstrip()):
            if not in_fence:
                flush_buf()
                in_fence = True
                fence_buf = [line]
            else:
                fence_buf.append(line)
                segments.append("\n".join(fence_buf))
                in_fence = False
                fence_buf = []
            continue
        if in_fence:
            fence_buf.append(line)
            continue

        # Markdown tables — consecutive `|` lines form one atomic segment.
        if _TABLE_LINE_RE.match(line):
            if not in_table:
                flush_buf()
                in_table = True
                table_buf = [line]
            else:
                table_buf.append(line)
            continue
        if in_table:
            segments.append("\n".join(table_buf))
            in_table = False
            table_buf = []

        # Blank line → paragraph boundary.
        if not line.strip():
            flush_buf()
            continue

        buf.append(line)

    # Tail buffers — handle unclosed fence / open table / trailing paragraph.
    if in_fence and fence_buf:
        segments.append("\n".join(fence_buf))
    if in_table and table_buf:
        segments.append("\n".join(table_buf))
    flush_buf()
    return segments


def _split_on_paragraphs(text: str, max_chars: int) -> list[str]:
    """Blank-line paragraph split — but with fenced code blocks and markdown
    tables preserved as atomic units so they're never cut mid-content."""
    paras = _segment_with_atomic_blocks(text)
    if len(paras) <= 1:
        return [text]
    return _group_pieces(paras, max_chars, joiner="\n\n")


# Sentence boundary: terminator (.!?) + whitespace + capital/dash/digit/quote.
# Conservative — won't split mid-sentence on abbreviations like "Dr. Smith"
# because we require a *new sentence* opener after the terminator.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\-\d\"'])")


def _split_on_sentences(text: str, max_chars: int) -> list[str]:
    """Sentence-level split — final structural granularity before hard char cut.
    Greedy-packs consecutive sentences up to max_chars so a chunk stays a coherent
    short paragraph rather than a single sentence."""
    sents = [s for s in _SENTENCE_BOUNDARY_RE.split(text) if s.strip()]
    if len(sents) <= 1:
        return [text]
    return _group_pieces(sents, max_chars, joiner=" ")


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Last-resort character split, preferring sentence / line / space boundaries."""
    out: list[str] = []
    rest = text
    while len(rest) > max_chars:
        cut = rest.rfind(". ", 0, max_chars)
        if cut < max_chars * 0.5:
            cut = rest.rfind("\n", 0, max_chars)
        if cut < max_chars * 0.5:
            cut = rest.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        else:
            cut += 1  # keep the delimiter on the left
        out.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        out.append(rest)
    return out


def _split_oversized(body: str, max_chars: int) -> list[str]:
    """Contextual split of a too-long block.

    Cascade, applied only to pieces still over `max_chars`:
        H3 → H4 → `**Step N:` → blank-line paragraphs → sentences → hard char cut.

    Each strategy is gated per-piece so a giant H3 inside an H2 splits into
    Steps inside that H3, without flattening smaller siblings. The sentence
    splitter (new) handles the long-prose case where there are no intermediate
    structural markers — far better than hard char cutting through a sentence.

    Every emitted chunk preserves its parent context: the caller re-prepends
    the heading_path on each piece, so even a sentence-level split chunk still
    starts with `[Screen > Invite Candidates > Set an Expiry Date]`.
    """
    if len(body) <= max_chars:
        return [body]

    strategies = (
        lambda t: _split_on_heading_level(t, 3, max_chars),
        lambda t: _split_on_heading_level(t, 4, max_chars),
        lambda t: _split_on_steps(t, max_chars),
        lambda t: _split_on_paragraphs(t, max_chars),
        lambda t: _split_on_sentences(t, max_chars),
    )

    pieces = [body]
    for strategy in strategies:
        if all(len(p) <= max_chars for p in pieces):
            return pieces
        next_pieces: list[str] = []
        for p in pieces:
            if len(p) <= max_chars:
                next_pieces.append(p)
                continue
            split = strategy(p)
            # If the strategy didn't divide this piece, defer to the next strategy.
            if len(split) <= 1:
                next_pieces.append(p)
            else:
                next_pieces.extend(split)
        pieces = next_pieces

    # Anything still oversized → hard split as last resort.
    final: list[str] = []
    for p in pieces:
        if len(p) <= max_chars:
            final.append(p)
        else:
            final.extend(_hard_split(p, max_chars))
    return final


def _enforce_max_chunk(
    chunks: list[tuple[str, str]],
    max_chars: int,
    target_chars: int = CHUNK_TARGET_CHARS,
) -> list[tuple[str, str]]:
    """Post-pass over chunk_document output: split any chunk whose text exceeds
    `target_chars` (soft target) using the contextual cascade in `_split_oversized`.

    Two thresholds:
      - `target_chars` (default ~3000): aspirational chunk size — produces
        focused, semantically self-contained pieces ideal for cross-encoder
        rerank and granular product_area voting.
      - `max_chars` (default ~4000): hard ceiling. Anything above is split
        regardless. The cascade actually targets `target_chars` so chunks
        usually land well under the ceiling.

    The chunks are already in `[heading_path]\\n\\n<body>` form. We strip the
    prefix, split the body, then re-prepend per-piece. When split into N>1
    pieces, each piece gets `(part i/N)` appended to its heading_path so the
    Solver can see partial-section provenance during retrieval.
    """
    if target_chars > max_chars:
        target_chars = max_chars

    out: list[tuple[str, str]] = []
    for path, full_text in chunks:
        if len(full_text) <= target_chars:
            out.append((path, full_text))
            continue
        prefix = f"[{path}]\n\n"
        body = full_text[len(prefix):] if full_text.startswith(prefix) else full_text
        # Budget for the BODY of each split piece. Reserve room for the
        # (longer) part-suffixed prefix on each piece.
        prefix_overhead = len(prefix) + len(" (part 99/99)") + 4
        budget = max(800, target_chars - prefix_overhead)
        # Hard ceiling on the body too — even if the target gives a generous
        # budget, never let a body exceed (max_chars - prefix_overhead).
        body_ceiling = max(800, max_chars - prefix_overhead)
        budget = min(budget, body_ceiling)

        pieces = _split_oversized(body, budget)
        if len(pieces) == 1:
            out.append((path, _prepend_path(pieces[0], path)))
        else:
            n = len(pieces)
            for i, piece in enumerate(pieces, start=1):
                sub_path = f"{path} (part {i}/{n})"
                out.append((sub_path, _prepend_path(piece, sub_path)))
    return out


def _merge_short(chunks: list[tuple[str, str]], min_words: int = 50) -> list[tuple[str, str]]:
    if not chunks:
        return []
    out: list[tuple[str, str]] = []
    buffer_text = ""
    buffer_path = ""
    for path, text in chunks:
        if _word_count(text) < min_words and out:
            prev_path, prev_text = out[-1]
            out[-1] = (prev_path, prev_text + "\n\n" + text)
        else:
            out.append((path, text))
    return out


def chunk_document(
    content: str,
    doc_type: DocType,
    breadcrumbs: list[str],
    title: str,
    company: str,
    product_area: str,
) -> list[tuple[str, str]]:
    """Return list of (heading_path, chunk_text). Universal override + per-type rules.

    A final `_enforce_max_chunk` pass guarantees no chunk exceeds MAX_CHUNK_CHARS,
    splitting via H3 → H4 → steps → paragraph → hard cascade if needed. This keeps
    Jina v3 (8194-token cap) from rejecting embeds on long policy / release-notes
    sections that would otherwise be left intact.
    """
    if not breadcrumbs:
        h1 = _extract_h1_title(content) or title or product_area
        breadcrumbs = [company.title(), product_area.replace("_", " ").title(), h1] if h1 else [company.title(), product_area.replace("_", " ").title()]

    # Multi-Q&A detection runs first — applies regardless of doctype because some
    # how-to / reference docs embed a "Frequently Asked Questions" section that
    # benefits from per-question chunking even when the parent doctype isn't FAQ.
    #
    # Dual storage: emit BOTH the atomic Q&A pair chunks (granular retrieval —
    # one chunk = one question's answer) AND the whole-file chunk (broad-
    # context retrieval — when the query is general or asks about the topic
    # rather than a specific Q). The two indexes complement each other; the
    # reranker picks whichever shape matches the query best.
    qa_pairs = _detect_qa_pairs(content)
    if len(qa_pairs) >= 2:
        chunks = []
        for q, a in qa_pairs:
            sub_path = _build_path(breadcrumbs, [q[:80]])
            qa_text = f"Q: {q}\n\nA: {a}"
            chunks.append((sub_path, _prepend_path(qa_text, sub_path)))
        # Additional whole-file chunk for broader-context retrieval. Will be
        # split by _enforce_max_chunk if it exceeds the cap.
        whole_path = _build_path(breadcrumbs, [])
        chunks.append((whole_path, _prepend_path(content, whole_path)))
        return _enforce_max_chunk(chunks, MAX_CHUNK_CHARS)

    total_words = _word_count(content)
    if total_words < 500:
        path = _build_path(breadcrumbs, [])
        chunks = [(path, _prepend_path(content, path))]
    elif doc_type == DocType.FAQ:
        path = _build_path(breadcrumbs, [])
        chunks = [(path, _prepend_path(content, path))]
    elif doc_type == DocType.HOW_TO:
        chunks = _chunk_how_to(content, breadcrumbs)
    elif doc_type == DocType.INTEGRATION:
        chunks = _chunk_integration(content, breadcrumbs)
    elif doc_type == DocType.RELEASE_NOTES:
        chunks = _chunk_release_notes(content, breadcrumbs)
    elif doc_type == DocType.TROUBLESHOOTING:
        chunks = _chunk_h2_split(content, breadcrumbs, merge_short=True)
    elif doc_type == DocType.POLICY_LEGAL:
        chunks = _chunk_h2_split(content, breadcrumbs, merge_short=False)
    elif doc_type == DocType.REFERENCE:
        chunks = _chunk_reference(content, breadcrumbs)
    elif doc_type == DocType.CONCEPTUAL:
        chunks = _chunk_h2_split(content, breadcrumbs, merge_short=True)
    else:
        chunks = _chunk_h2_split(content, breadcrumbs, merge_short=True)

    return _enforce_max_chunk(chunks, MAX_CHUNK_CHARS)


def _chunk_h2_split(content: str, breadcrumbs: list[str], merge_short: bool) -> list[tuple[str, str]]:
    sections = _split_at_heading(content, 2)
    if len(sections) <= 1:
        path = _build_path(breadcrumbs, [])
        return [(path, _prepend_path(content, path))]

    chunks: list[tuple[str, str]] = []
    for section in sections:
        h2 = _h2_heading(section)
        path = _build_path(breadcrumbs, [h2])
        chunks.append((path, _prepend_path(section, path)))
    if merge_short:
        chunks = _merge_short(chunks)
    return chunks


def _chunk_how_to(content: str, breadcrumbs: list[str]) -> list[tuple[str, str]]:
    sections = _split_at_heading(content, 2)
    chunks: list[tuple[str, str]] = []
    if len(sections) <= 1:
        return [(_build_path(breadcrumbs, []), _prepend_path(content, _build_path(breadcrumbs, [])))]
    for section in sections:
        h2 = _h2_heading(section)
        section_path = _build_path(breadcrumbs, [h2])
        if _word_count(section) <= 600:
            chunks.append((section_path, _prepend_path(section, section_path)))
        else:
            step_chunks = _split_step_groups(section, max_steps=3)
            if len(step_chunks) <= 1:
                chunks.append((section_path, _prepend_path(section, section_path)))
            else:
                for i, sc in enumerate(step_chunks):
                    sub_path = _build_path(breadcrumbs, [h2, f"Steps Group {i+1}"])
                    chunks.append((sub_path, _prepend_path(sc, sub_path)))
    return _merge_short(chunks)


def _chunk_integration(content: str, breadcrumbs: list[str]) -> list[tuple[str, str]]:
    h1_lines = [(m.start(), m.group()) for m in re.finditer(r"^# .+$", content, re.MULTILINE)]
    if len(h1_lines) <= 1:
        return _chunk_h2_split(content, breadcrumbs, merge_short=True)

    sections: list[str] = []
    starts = [pos for pos, _ in h1_lines[1:]]
    starts.append(len(content))
    first_section_start = h1_lines[1][0]
    sections.append(content[:first_section_start])
    prev = first_section_start
    for s in starts[1:]:
        sections.append(content[prev:s])
        prev = s

    chunks: list[tuple[str, str]] = []
    title = _extract_h1_title(content)
    if title and title not in breadcrumbs:
        breadcrumbs = breadcrumbs + [title]
    for section in sections[1:]:
        h1 = _h1_heading(section)
        section_path = _build_path(breadcrumbs, [h1])
        if _word_count(section) <= 600:
            chunks.append((section_path, _prepend_path(section, section_path)))
        else:
            step_chunks = _split_step_groups(section, max_steps=2)
            if len(step_chunks) <= 1:
                chunks.append((section_path, _prepend_path(section, section_path)))
            else:
                for i, sc in enumerate(step_chunks):
                    sub_path = _build_path(breadcrumbs, [h1, f"Steps Group {i+1}"])
                    chunks.append((sub_path, _prepend_path(sc, sub_path)))
    return _merge_short(chunks)


def _chunk_release_notes(content: str, breadcrumbs: list[str]) -> list[tuple[str, str]]:
    sections = _split_at_heading(content, 2)
    if len(sections) <= 1:
        path = _build_path(breadcrumbs, [])
        return [(path, _prepend_path(content, path))]
    chunks: list[tuple[str, str]] = []
    for section in sections:
        h2 = _h2_heading(section).replace("**", "").strip()
        path = _build_path(breadcrumbs, [h2])
        chunks.append((path, _prepend_path(section, path)))
    return chunks


def _chunk_reference(content: str, breadcrumbs: list[str]) -> list[tuple[str, str]]:
    lines = [l for l in content.splitlines() if l.strip()]
    link_lines = sum(1 for l in lines if _LINK_LINE_RE.match(l))
    is_index = lines and (link_lines / len(lines)) > 0.3
    chunks = _chunk_h2_split(content, breadcrumbs, merge_short=not is_index)
    return chunks


def _split_step_groups(section: str, max_steps: int) -> list[str]:
    matches = list(_STEP_RE.finditer(section))
    if len(matches) < 2:
        return [section]
    header = section[: matches[0].start()]
    groups: list[str] = []
    for i in range(0, len(matches), max_steps):
        chunk_start = matches[i].start()
        end_idx = i + max_steps
        chunk_end = matches[end_idx].start() if end_idx < len(matches) else len(section)
        groups.append(header + section[chunk_start:chunk_end])
    return groups
