"""Markdown cleaning + frontmatter extraction."""

from __future__ import annotations

import re
from typing import NamedTuple


class Cleaned(NamedTuple):
    content: str
    breadcrumbs: list[str]
    title: str


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HRULE_RE = re.compile(r"^(?:-{4,}|={4,})\s*$", re.MULTILINE)
_CDN_EMAIL_RE = re.compile(r"\[\[email[^\]]*\]\]\(/cdn-cgi/[^)]*\)")
_LINE_CONT_RE = re.compile(r"\\\n")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    block = m.group(1)
    rest = raw[m.end():]
    fields: dict = {}
    current_key = None
    current_list: list[str] | None = None
    for line in block.splitlines():
        if not line.strip():
            continue
        if current_list is not None and line.startswith("  -"):
            item = line.lstrip("- ").strip().strip('"').strip("'")
            current_list.append(item)
            continue
        if current_list is not None:
            fields[current_key] = current_list
            current_list = None
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if not v:
                current_key = k
                current_list = []
            else:
                fields[k] = v.strip('"').strip("'")
    if current_list is not None:
        fields[current_key] = current_list
    return fields, rest


def clean(raw: str) -> Cleaned:
    fm, body = _parse_frontmatter(raw)
    text = body
    text = _IMAGE_RE.sub("", text)
    text = _HRULE_RE.sub("", text)
    text = _CDN_EMAIL_RE.sub("[contact support]", text)
    text = _LINE_CONT_RE.sub("\n", text)
    text = _HTML_COMMENT_RE.sub("", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    text = text.strip()

    breadcrumbs = fm.get("breadcrumbs") or fm.get("breadcrumb") or []
    if isinstance(breadcrumbs, str):
        breadcrumbs = [breadcrumbs]
    title = fm.get("title", "")

    return Cleaned(content=text, breadcrumbs=breadcrumbs, title=title)
