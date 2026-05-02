"""Pre-LLM callbacks: prompt-injection block, language detect + translate.

The injection guard returns a sentinel `LlmResponse` with a structured PreFlight
that escalates immediately so the rest of PreFlight is skipped.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional


_INJECTION_PATTERNS = [
    r"ignore\s+(?:previous|prior|all)\s+instructions?",
    r"show\s+(?:me\s+)?your\s+(?:system\s+)?prompt",
    r"reveal\s+(?:your\s+)?prompt",
    r"affiche\s+(?:toutes?\s+les\s+)?(?:r[eè]gles|logique|documents?)",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|system\|>",
    r"\bDAN\s+mode\b",
    r"jailbreak",
    r"forget\s+(?:everything|all)\s+(?:above|prior|previous)",
    r"print\s+(?:the\s+)?internal\s+(?:rules|policy)",
    r"output\s+the\s+raw\s+(?:retrieved\s+)?documents?",
    r"give\s+me\s+the\s+code\s+to\s+delete",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def detect_prompt_injection(text: str) -> Optional[str]:
    if not text:
        return None
    m = _INJECTION_RE.search(text)
    return m.group(0) if m else None


def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "en"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        return detect(text)
    except Exception:
        return "en"


def translate_to_english(text: str) -> str:
    """Single Kimi call (via OpenRouter) to translate non-English text. Falls back to original on failure."""
    if not text.strip():
        return text
    try:
        from llm import call_text
        return call_text(
            system="Translate the user's text to English. Preserve technical terms and product names verbatim. Return ONLY the translation — no preface, no quotes.",
            user=text,
            max_tokens=600,
            temperature=0.0,
        ).strip()
    except Exception:
        return text


# ---- ADK callback wrappers ----

def block_prompt_injection_callback(callback_context, llm_request):
    """Before-model callback. If injection patterns found in the user content, short-circuit."""
    try:
        text_parts = []
        for content in llm_request.contents or []:
            for part in (content.parts or []):
                if getattr(part, "text", None):
                    text_parts.append(part.text)
        joined = "\n".join(text_parts)
        match = detect_prompt_injection(joined)
        if match:
            from google.genai import types as genai_types
            from google.adk.models.llm_response import LlmResponse
            sentinel = json.dumps({
                "request_type": "invalid",
                "company_hint": "unknown",
                "intent": "policy",
                "is_multi_request": False,
                "language": "en",
                "escalate_now": True,
                "escalate_reason": f"prompt_injection_detected: {match[:80]}",
            })
            return LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=sentinel)],
                )
            )
    except Exception:
        pass
    return None


def detect_and_translate_callback(callback_context, llm_request):
    """Before-model callback. If non-English, translate and substitute into the request."""
    try:
        contents = llm_request.contents or []
        text_parts = []
        for content in contents:
            for part in (content.parts or []):
                if getattr(part, "text", None):
                    text_parts.append(part.text)
        joined = "\n".join(text_parts)
        lang = detect_language(joined)
        if callback_context and getattr(callback_context, "state", None) is not None:
            callback_context.state["language"] = lang
        if lang != "en" and joined.strip():
            translated = translate_to_english(joined)
            if callback_context and getattr(callback_context, "state", None) is not None:
                callback_context.state["original_text"] = joined
                callback_context.state["translated_text"] = translated
            # Mutate request: replace text parts with translated single block
            from google.genai import types as genai_types
            new_contents = list(contents)
            if new_contents:
                last = new_contents[-1]
                new_contents[-1] = genai_types.Content(
                    role=last.role,
                    parts=[genai_types.Part(text=translated)],
                )
                llm_request.contents = new_contents
    except Exception:
        pass
    return None
