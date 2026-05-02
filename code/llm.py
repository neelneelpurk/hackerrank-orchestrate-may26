"""LiteLLM-based Kimi (Moonshot) client with structured output.

We use LiteLLM directly here for the structured calls (PreFlight / Plan / Solver / Reflection)
because they are deterministic, single-shot synchronous calls — wrapping them in ADK LlmAgents
adds session/runner overhead that buys nothing for non-conversational structured generation.

ADK is used at the orchestration layer (SequentialAgent / LoopAgent / before_model_callback) but
the actual LLM I/O is funnelled through this module so we get one place to handle:
- JSON mode + Pydantic validation
- Retry on rate limit
- Optional `response_format=json_object` when the provider supports it
- json_repair fallback
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional, Type

import litellm
from json_repair import repair_json
from pydantic import BaseModel, ValidationError

litellm.drop_params = True
litellm.suppress_debug_info = True


def _model() -> str:
    """Default model: x-ai/grok-4.3 via OpenRouter.

    Override with `OPENROUTER_MODEL` in .env. Legacy `MOONSHOT_MODEL` is also
    honoured so existing configs from the Kimi era keep working.
    """
    name = (
        os.environ.get("OPENROUTER_MODEL")
        or os.environ.get("MOONSHOT_MODEL")
        or "x-ai/grok-4.3"
    )
    return f"openrouter/{name}"


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
    if not key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set")
    return key


_OR_EXTRA_BODY = {
    "provider": {
        "require_parameters": True,
        "allow_fallbacks": True,
    }
}


def call_text(
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    for attempt in range(4):
        try:
            resp = litellm.completion(
                model=_model(),
                api_key=_api_key(),
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42,
                extra_body=_OR_EXTRA_BODY,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("call_text exhausted retries")


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(raw: str) -> str:
    m = _FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw.strip()


def call_structured(
    system: str,
    user: str,
    schema: Type[BaseModel],
    max_tokens: int = 1500,
    temperature: float = 0.0,
) -> BaseModel:
    sys_prompt = (
        f"{system}\n\n"
        "Respond with ONE JSON object matching this schema. No prose, no markdown fences.\n\n"
        f"Schema:\n{json.dumps(schema.model_json_schema(), indent=2)}"
    )
    last_err: Exception | None = None
    current_user = user
    for attempt in range(2):
        for net_attempt in range(4):
            try:
                resp = litellm.completion(
                    model=_model(),
                    api_key=_api_key(),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=42,
                    response_format={"type": "json_object"},
                    extra_body=_OR_EXTRA_BODY,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": current_user},
                    ],
                )
                raw = resp.choices[0].message.content or ""
                break
            except Exception:
                if net_attempt == 3:
                    raise
                time.sleep(1.5 * (net_attempt + 1))
        else:
            raise RuntimeError("call_structured exhausted network retries")
        try:
            obj = json.loads(_extract_json(raw))
            return schema.model_validate(obj)
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            try:
                repaired = repair_json(_extract_json(raw))
                obj = json.loads(repaired) if isinstance(repaired, str) else repaired
                return schema.model_validate(obj)
            except Exception as e2:
                last_err = e2
            current_user = (
                user
                + "\n\nThe previous output was invalid JSON for the schema. Error: "
                + str(last_err)[:300]
                + "\nReturn corrected JSON only."
            )
    raise ValueError(f"call_structured failed: {last_err}")
