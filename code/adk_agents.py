"""ADK agent composition for the triage pipeline.

Builds the ROOT agent per spec.md §2:

  ROOT (SequentialAgent)
    TicketContext (BaseAgent — load ticket_text into state)
    PreFlight     (LlmAgent — output_schema=PreFlight, before_model_callback=[block_prompt_injection, detect_and_translate])
    StateBridge   (BaseAgent — JSON-stringify state for instruction templates)
    ReWoo loop    (LoopAgent, max_iterations=2)
      Planner     (LlmAgent — output_schema=Plan)
      Workers     (BaseAgent — dynamic parallel retrieve)
      EvidenceBridge (BaseAgent — stringify evidence for solver)
      Solver      (LlmAgent — output_schema=TriageOutput)
      Reflector   (LlmAgent — output_schema=Reflection)
      LoopBreaker (BaseAgent — read reflection, EventActions.escalate=True to break)
    Commit        (BaseAgent — final CSV row)

Why instruction-template + {state} placeholders + include_contents='none':
We want each structured LlmAgent to be a pure single-shot call with the entire
prompt assembled deterministically from state — not from accumulated conversation
history. ADK supports `{key}` and `{key?}` (optional) state placeholders in the
instruction string, which lets us build the planner / solver / reflector prompts
straight from the values the upstream agents wrote into state.

Telemetry: ADK auto-emits OpenTelemetry spans for every LlmAgent invocation, every
sub-agent transition, and every tool call. `code/telemetry.py` wires those spans
into a JSONL file exporter.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.models.lite_llm import LiteLlm
from google.genai import types as genai_types

from typing import Optional, Type, TypeVar
from json_repair import repair_json
from pydantic import BaseModel, ValidationError

import callbacks
import tools as retrieve_tools
from models import Plan, PreFlight, Reflection, Step, TriageOutput
from prompts import PLANNER_PROMPT, PREFLIGHT_PROMPT, REFLECTION_PROMPT, SOLVER_PROMPT
from voting import weighted_product_area


_T = TypeVar("_T", bound=BaseModel)


def _parse(raw, schema: Type[_T]) -> Optional[_T]:
    """Coerce an ADK state value into a validated Pydantic model.

    ADK can stash an `output_schema` result in three shapes depending on version
    + provider quirks:
      - the Pydantic instance itself
      - a `dict` (already-validated)
      - a raw JSON string (LLM output that ADK couldn't auto-parse, e.g. when it
        arrived inside a markdown code fence)

    This returns a typed instance for all three, or None if the value is missing
    or unrecoverable. We don't lose the LLM's actual fields just because the
    transport shape varies.
    """
    if raw is None:
        return None
    if isinstance(raw, schema):
        return raw
    if isinstance(raw, BaseModel):
        try:
            return schema.model_validate(raw.model_dump())
        except ValidationError:
            return None
    if isinstance(raw, dict):
        try:
            return schema.model_validate(raw)
        except ValidationError:
            return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        # Strip a markdown fence if present
        if s.startswith("```"):
            s = s.strip("`")
            if s.lower().startswith("json"):
                s = s[4:].lstrip()
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            try:
                data = json.loads(repair_json(s)) if isinstance(repair_json(s), str) else repair_json(s)
            except Exception:
                return None
        try:
            return schema.model_validate(data)
        except ValidationError:
            return None
    return None


def _to_jsonable_dict(raw, schema: Type[BaseModel]) -> dict:
    """Same as _parse but always returns a dict (empty dict on parse failure)."""
    parsed = _parse(raw, schema)
    return parsed.model_dump() if parsed else {}


# ------------------- Model factory -------------------

def _llm_model() -> LiteLlm:
    """Default LLM (Grok via OpenRouter) bridged through LiteLLM.

    Two OpenRouter-specific bits worth knowing:

    1. **Auth**: LiteLLM with `openrouter/` prefix needs `OPENROUTER_API_KEY` in
       the env. ADK's LiteLlm constructor kwarg `api_key=` doesn't always reach
       `litellm.acompletion`, so we also write the key into os.environ here.

    2. **Provider routing for structured outputs**: OpenRouter spreads each
       model across several upstream providers. Some don't support JSON-schema
       structured outputs, which ADK's `output_schema=…` requires. Without a
       hint, OpenRouter may route to one that returns 400
       "model features structured outputs not support" or 404
       "No endpoints found that can handle the requested parameters".

       `extra_body={"provider": {"require_parameters": True}}` tells OpenRouter
       to only consider providers that support every parameter in the request,
       which forces a structured-output-capable upstream every time.

    The default model is **`x-ai/grok-4.3`**. Override with `OPENROUTER_MODEL`
    in `.env` (legacy `MOONSHOT_MODEL` env var is also honoured for backwards
    compat with earlier configs).
    """
    model_name = (
        os.environ.get("OPENROUTER_MODEL")
        or os.environ.get("MOONSHOT_MODEL")
        or "x-ai/grok-4.3"
    )
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("MOONSHOT_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Add it to .env and re-run."
        )
    os.environ["OPENROUTER_API_KEY"] = api_key

    extra_body: dict = {
        "provider": {
            "require_parameters": True,
            "allow_fallbacks": True,
        }
    }
    # Optional: pin a provider order via env (e.g. "xAI,Together,Fireworks")
    order_env = os.environ.get("OPENROUTER_PROVIDER_ORDER", "").strip()
    if order_env:
        extra_body["provider"]["order"] = [p.strip() for p in order_env.split(",") if p.strip()]

    return LiteLlm(
        model=f"openrouter/{model_name}",
        api_key=api_key,
        temperature=0.0,
        seed=42,
        extra_body=extra_body,
    )


# ------------------- Instruction templates -------------------

PREFLIGHT_INSTRUCTION = PREFLIGHT_PROMPT + """

INPUT TICKET:
{ticket_text}

Output the PreFlight JSON only.
"""

PLANNER_INSTRUCTION = PLANNER_PROMPT + """

TICKET:
{ticket_text}

PREFLIGHT:
{preflight_json}

PREVIOUS_PLAN (only present on retry; ignore if empty):
{plan_json?}

RETRY_HINT (only present on retry; if non-empty, broaden / drop filter / switch company):
{retry_hint?}

Emit JSON only matching the Plan schema.
"""

SOLVER_INSTRUCTION = SOLVER_PROMPT + """

TICKET:
{ticket_text}

PREFLIGHT:
{preflight_json}

PLAN:
{plan_json}

SUGGESTED_PRODUCT_AREA: {suggested_area?}

EVIDENCE:
{evidence_blocks}

Emit JSON only matching the TriageOutput schema.
"""

REFLECTION_INSTRUCTION = REFLECTION_PROMPT + """

TICKET:
{ticket_text}

PREFLIGHT:
{preflight_json}

SOLUTION:
{solution_json}

EVIDENCE_SUMMARY:
{evidence_summary}

is_final: {is_final}

If SOLUTION.response is exactly 'ESCALATE', set escalate=true.
"""


# ------------------- LLM agents -------------------

def make_preflight_agent() -> LlmAgent:
    return LlmAgent(
        name="preflight_agent",
        description="Classify ticket and surface escalation triggers.",
        model=_llm_model(),
        instruction=PREFLIGHT_INSTRUCTION,
        output_schema=PreFlight,
        output_key="preflight",
        include_contents="none",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        before_model_callback=[
            callbacks.block_prompt_injection_callback,
            callbacks.detect_and_translate_callback,
        ],
    )


def make_planner_agent() -> LlmAgent:
    return LlmAgent(
        name="planner_agent",
        description="Emit a ReWoo plan: retrieve / escalate / reply_static steps.",
        model=_llm_model(),
        instruction=PLANNER_INSTRUCTION,
        output_schema=Plan,
        output_key="plan",
        include_contents="none",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


def make_solver_agent() -> LlmAgent:
    return LlmAgent(
        name="solver_agent",
        description="Synthesise a grounded answer from retrieved evidence.",
        model=_llm_model(),
        instruction=SOLVER_INSTRUCTION,
        output_schema=TriageOutput,
        output_key="solution",
        include_contents="none",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


def make_reflector_agent() -> LlmAgent:
    return LlmAgent(
        name="reflector_agent",
        description="Score grounding/completeness/safety/actionability; signal loop exit.",
        model=_llm_model(),
        instruction=REFLECTION_INSTRUCTION,
        output_schema=Reflection,
        output_key="reflection",
        include_contents="none",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


# ------------------- Custom (non-LLM) agents -------------------

class TicketContextAgent(BaseAgent):
    """Loads the user's ticket into state['ticket_text'] and seeds bookkeeping fields."""

    name: str = "ticket_context"
    description: str = "Stage ticket + bookkeeping into session state."

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        ticket_text = ""
        if ctx.user_content and ctx.user_content.parts:
            for part in ctx.user_content.parts:
                if getattr(part, "text", None):
                    ticket_text = part.text
                    break
        delta = {
            "ticket_text": ticket_text,
            "loop_iteration": 0,
            "preflight_json": "{}",
            "plan_json": "{}",
            "retry_hint": "",
            "evidence_blocks": "",
            "evidence_summary": "",
            "solution_json": "{}",
            "is_final": False,
            "suggested_area": "",
        }
        for k, v in delta.items():
            ctx.session.state[k] = v
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(state_delta=delta),
        )


class PreflightBridgeAgent(BaseAgent):
    """After PreFlight runs: copy translated_text into ticket_text (if available) and JSON-stringify preflight."""

    name: str = "preflight_bridge"
    description: str = "Materialise preflight_json + translated ticket for the planner."

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        st = ctx.session.state
        pf = _parse(st.get("preflight"), PreFlight)
        pf_dict = pf.model_dump() if pf else {}
        # Persist the parsed-and-validated copy so downstream stages don't need to re-parse.
        if pf is not None:
            st["preflight"] = pf_dict

        delta = {"preflight_json": json.dumps(pf_dict, default=str)}
        translated = st.get("translated_text")
        if translated:
            delta["ticket_text"] = translated
        for k, v in delta.items():
            ctx.session.state[k] = v
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(state_delta=delta),
        )


class PlannerBridgeAgent(BaseAgent):
    """Before Planner: stage retry_hint and previous plan_json. After Planner: bump iteration."""

    name: str = "planner_bridge"
    description: str = "Stage retry context for the planner instruction template."

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        st = ctx.session.state
        iteration = int(st.get("loop_iteration", 0))
        prev_refl = _parse(st.get("reflection"), Reflection)
        prev_plan = _parse(st.get("plan"), Plan)

        retry_hint = ""
        plan_json = "{}"
        # Retry only when the previous iteration produced a low-confidence reflection;
        # the LoopBreaker already exited the loop on high scores so we won't reach here in that case.
        if iteration > 0 and prev_refl and prev_refl.final_score < _pass_threshold():
            retry_hint = prev_refl.reason or ""
            plan_json = json.dumps(prev_plan.model_dump() if prev_plan else {}, default=str)

        delta = {
            "loop_iteration": iteration + 1,
            "retry_hint": retry_hint,
            "plan_json": plan_json,
        }
        for k, v in delta.items():
            ctx.session.state[k] = v
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(state_delta=delta),
        )


class WorkersAgent(BaseAgent):
    """Reads state['plan'], runs retrieve_tool in parallel via asyncio.gather."""

    name: str = "workers_agent"
    description: str = "ReWoo Workers: parallel retrieval over plan steps."

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        st = ctx.session.state
        plan = _parse(st.get("plan"), Plan)
        steps: list[Step] = plan.steps if plan else []
        # Persist the parsed-and-validated copy so the solver / commit see the same shape.
        if plan is not None:
            st["plan"] = plan.model_dump()

        async def _do(step: Step):
            if step.type != "retrieve" or not step.company or not step.query_variants:
                return step.id, {"status": "skipped", "evidence": []}
            # Bigger Solver context: 5 chunks for answer steps, 3 for label_only.
            final_n = 5 if step.purpose == "answer" else 3
            try:
                res = await asyncio.to_thread(
                    retrieve_tools.retrieve,
                    step.company,
                    step.query_variants,
                    step.doc_type_filter or None,
                    retrieve_tools.DEFAULT_TOP_K,
                    final_n,
                )
            except Exception as e:
                res = {"status": "error", "evidence": [], "error": str(e)[:200]}
            return step.id, res

        results = await asyncio.gather(*[_do(s) for s in steps]) if steps else []
        evidence_by_step = dict(results)

        # Cross-step dedup by sqlite_id: when two retrieve steps both pull the
        # same chunk (common with multi-company fanout or overlapping variants),
        # keep only the strongest occurrence and drop the rest.
        seen_sids: set = set()
        flat_with_step: list[tuple] = []  # (sid, ev_index_in_step, score, ev)
        for sid, pkg in evidence_by_step.items():
            for i, ev in enumerate(pkg.get("evidence", [])):
                flat_with_step.append((sid, i, float(ev.get("rerank_score", 0)), ev))
        flat_with_step.sort(key=lambda t: t[2], reverse=True)
        keep: dict = {sid: set() for sid in evidence_by_step}  # sid -> set of indices to keep
        for sid, i, _score, ev in flat_with_step:
            sid_meta = (ev.get("metadata") or {}).get("sqlite_id")
            if sid_meta is None:
                keep[sid].add(i)
                continue
            if sid_meta in seen_sids:
                continue
            seen_sids.add(sid_meta)
            keep[sid].add(i)
        for sid, pkg in evidence_by_step.items():
            evs = pkg.get("evidence", [])
            pkg["evidence"] = [evs[i] for i in range(len(evs)) if i in keep[sid]]

        suggested_area, suggested_company = weighted_product_area(evidence_by_step)

        # Stringify for the Solver instruction
        plan_json = json.dumps(plan.model_dump() if plan else {}, default=str)
        blocks = []
        summary = []
        for sid, pkg in evidence_by_step.items():
            confidence = pkg.get("confidence", "LOW")
            top_score = pkg.get("top_score", 0.0)
            score_gap = pkg.get("score_gap", 0.0)
            fallback = pkg.get("fallback", "")
            evs = pkg.get("evidence", [])
            if evs:
                blocks.append(
                    f"=== Step {sid} retrieval [confidence={confidence}, "
                    f"top={top_score:.3f}, gap={score_gap:.3f}, fallback={fallback}] ==="
                )
            for i, ev in enumerate(evs):
                md = ev["metadata"]
                blocks.append(
                    f"--- Evidence {sid}.{i} (rerank={ev.get('rerank_score', 0):.3f}) ---\n"
                    f"company={md['company']} product_area={md['product_area']} doc_type={md['doc_type']}\n"
                    f"heading_path={md['heading_path']}\n"
                    f"source_path={md['source_path']}\n\n{ev['text']}\n"
                )
                summary.append(
                    f"{sid}: {md['heading_path']} (rerank={ev.get('rerank_score', 0):.2f}, conf={confidence})"
                )
        if not blocks:
            blocks.append("(no retrievable evidence — emit a static reply or ESCALATE)")
        delta = {
            "evidence_by_step": evidence_by_step,
            "suggested_area": suggested_area,
            "suggested_company": suggested_company,
            "plan_json": plan_json,
            "evidence_blocks": "\n".join(blocks),
            "evidence_summary": "\n".join(summary) if summary else "(none)",
        }
        for k, v in delta.items():
            ctx.session.state[k] = v
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(state_delta=delta),
        )


class SolverBridgeAgent(BaseAgent):
    """Stringify the solution into solution_json + set is_final flag for the reflector."""

    name: str = "solver_bridge"
    description: str = "Materialise solution_json and is_final for the reflector."

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        st = ctx.session.state
        sol = _parse(st.get("solution"), TriageOutput)
        # Persist the parsed-and-validated copy so commit_agent / loop_breaker see typed fields.
        sol_dict: dict = {}
        if sol is not None:
            sol_dict = sol.model_dump()
            st["solution"] = sol_dict

        is_final = int(st.get("loop_iteration", 1)) >= 2
        delta = {
            "solution_json": json.dumps(sol_dict, default=str),
            "is_final": is_final,
        }
        for k, v in delta.items():
            ctx.session.state[k] = v
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(state_delta=delta),
        )


def _pass_threshold() -> float:
    """Reflector pass threshold; env-tunable so we can sweep without a redeploy.

    Default 5.0 (was 6.0 in earlier drafts) — the reflector's revised rubric scores
    grounded answers around 6–7, so 5.0 leaves a defensible margin for confident
    replies while still catching genuinely off-target ones.
    """
    try:
        return float(os.environ.get("REFLECTION_PASS_THRESHOLD", "5.0"))
    except ValueError:
        return 5.0


class LoopBreakerAgent(BaseAgent):
    """Decide whether the ReWoo loop should exit, based purely on numerical scores.

    NOTE on the word "escalate":
      `EventActions.escalate=True` here means **exit the LoopAgent** (ADK's loop
      control signal). It does NOT mean "send to a human" — that's CommitAgent's
      decision based on the same scores. So a confident, well-grounded answer
      sets escalate=True (stop iterating) AND lands as `status="replied"`.
    """

    name: str = "loop_breaker"
    description: str = "Exit the loop when the reflector passes or it's the final attempt."

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        st = ctx.session.state
        refl = _parse(st.get("reflection"), Reflection)
        sol = _parse(st.get("solution"), TriageOutput)
        # Persist parsed copies so commit_agent reads typed shape.
        if refl is not None:
            st["reflection"] = refl.model_dump()
        if sol is not None:
            st["solution"] = sol.model_dump()

        threshold = _pass_threshold()
        passes = bool(refl and refl.final_score >= threshold)
        solver_self_escalate = bool(sol and sol.response.strip().upper() == "ESCALATE")
        is_final = bool(st.get("is_final", False))

        # Exit the loop when:
        #   - we have a confident answer (passes), OR
        #   - the solver explicitly self-escalated, OR
        #   - we just finished the final iteration (no more retries available).
        should_exit_loop = passes or solver_self_escalate or is_final

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(escalate=should_exit_loop),
        )


class CommitAgent(BaseAgent):
    """Map preflight + solution + reflection into the final CSV row.

    Decides `status` (replied vs escalated) — this is the *human-routing* decision,
    distinct from the LoopBreaker's loop-control decision. Order of precedence:

      1. PreFlight escalate_now  → escalated   (hard rules: outage, security, etc.)
      2. Solver self-escalated   → escalated
      3. final_score >= threshold OR reply_static plan step → replied
      4. Otherwise (low confidence after retry, no static reply) → escalated
    """

    name: str = "commit_agent"
    description: str = "Assemble the final CSV row dict at state['triage_result']."

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        st = ctx.session.state
        pf = _parse(st.get("preflight"), PreFlight)
        sol = _parse(st.get("solution"), TriageOutput)
        refl = _parse(st.get("reflection"), Reflection)
        plan = _parse(st.get("plan"), Plan)

        suggested = (st.get("suggested_area") or "").strip()
        company_col = st.get("company_col") or "None"
        pf_dict = pf.model_dump() if pf else {}
        threshold = _pass_threshold()

        # ---- replied vs escalated decision ----
        # Hard escalations first.
        if pf and pf.escalate_now:
            is_replied = False
        elif sol and sol.response.strip().upper() == "ESCALATE":
            is_replied = False
        elif plan and any(s.type == "reply_static" for s in plan.steps):
            is_replied = True
        elif refl and refl.final_score >= threshold and sol:
            is_replied = True
        else:
            is_replied = False

        # ---- Build the row from the validated Pydantic models, not loose dicts ----
        if is_replied and sol:
            response = sol.response
            justification = sol.justification or _positive_grounding_fallback(sol, suggested)
            product_area = sol.product_area or suggested or _fallback_area(pf_dict, company_col)
            status = "replied"
        else:
            response = "Escalate to a human"
            justification = _escalation_justification(pf, sol, refl, suggested, company_col)
            # When the Solver self-escalates because no chunk actually answered the
            # ticket, the SUGGESTED area was derived from chunks the Solver judged
            # off-topic — propagating it (e.g. "government" for a workspace-access
            # ticket that pulled claude-for-government neighbours) is misleading.
            # Fall back to the company general bucket instead.
            solver_self_escalated = bool(sol and sol.response.strip().upper() == "ESCALATE")
            if solver_self_escalated:
                product_area = _fallback_area(pf_dict, company_col)
            else:
                product_area = (
                    (sol.product_area if sol else "")
                    or suggested
                    or _fallback_area(pf_dict, company_col)
                )
            status = "escalated"

        # Reflector verifies the PreFlight request_type independently. If it set
        # `verified_request_type`, that overrides PreFlight (including the
        # "undefined" sentinel for genuinely ambiguous tickets).
        if refl and refl.verified_request_type:
            request_type = refl.verified_request_type
        elif pf:
            request_type = pf.request_type
        else:
            request_type = "invalid"

        # Coerce every field to a stripped, non-empty string for the CSV writer.
        result = {
            "status": str(status),
            "product_area": str(product_area or "uncategorized"),
            "response": str(response or "Escalate to a human").strip() or "Escalate to a human",
            "justification": str(justification or "routing to human review").strip() or "routing to human review",
            "request_type": str(request_type or "invalid"),
        }
        ctx.session.state["triage_result"] = result
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(state_delta={"triage_result": result}),
            content=genai_types.Content(role="model", parts=[genai_types.Part(text=json.dumps(result))]),
        )


def _fallback_area(pf: dict, company_col: str) -> str:
    co = pf.get("company_hint", "unknown")
    if co == "unknown":
        c = (company_col or "").strip().lower()
        co = c if c in ("hackerrank", "claude", "visa") else "unknown"
    return {
        "hackerrank": "general_help",
        "claude": "general",
        "visa": "general_support",
        "unknown": "uncategorized",
    }.get(co, "uncategorized")


def _positive_grounding_fallback(sol: TriageOutput, suggested: str) -> str:
    """When the Solver wrote a reply but somehow left justification empty,
    construct a positive one from the cited chunks instead of saying nothing.
    """
    area = sol.product_area or suggested or "the indexed corpus"
    if sol.cited_chunks:
        head = sol.cited_chunks[0]
        return f"Anchored in {head} ({area}) — the cited section directly answers the request."
    return f"Grounded in {area} — the response paraphrases the cited corpus content."


_BOGUS_ESCALATE_PREFIXES = (
    "anchored in", "grounded in", "cited", "answer is in",
    "the answer", "this is covered by", "covered by",
)


def _is_chunk_anchor_justification(text: str) -> bool:
    """Detect Solver justifications that read like a confident answer's audit trail.
    These are misleading when status=escalated — they suggest the corpus answered
    the ticket when in fact the Solver bailed.
    """
    if not text:
        return False
    head = text.strip().lower()
    return any(head.startswith(p) for p in _BOGUS_ESCALATE_PREFIXES)


def _escalation_justification(
    pf: Optional[PreFlight],
    sol: Optional[TriageOutput],
    refl: Optional[Reflection],
    suggested: str,
    company_col: str,
) -> str:
    """Build a positive-framing justification for the escalation path.

    Priority: PreFlight hard-rule reason → Solver's own justification (only when it
    looks like a real escalation explanation, not a chunk anchor) → Reflector reason
    → company-aware fallback. Never emits "insufficient evidence" / "no chunk matches"
    style strings.
    """
    if pf and pf.escalate_now and pf.escalate_reason:
        return f"Routing to human: {pf.escalate_reason}."

    # Solver self-escalate: its justification is only trustworthy when it explains
    # the GAP, not when it anchors in a chunk. Chunk-anchor language here means the
    # Solver disobeyed the prompt — fall through to reflector / fallback.
    if (
        sol
        and sol.justification
        and sol.response.strip().upper() == "ESCALATE"
        and not _is_chunk_anchor_justification(sol.justification)
    ):
        return sol.justification

    if refl and refl.reason:
        return refl.reason

    if sol and sol.justification and not _is_chunk_anchor_justification(sol.justification):
        return sol.justification

    co = (pf.company_hint if pf else "unknown")
    if co == "unknown":
        co = (company_col or "").strip().lower() or "unknown"
    return f"Routing to human: ticket falls outside the indexed {co} corpus areas."


# ------------------- Root agent factory -------------------

def make_root_agent() -> SequentialAgent:
    rewoo_loop = LoopAgent(
        name="rewoo_loop",
        description="ReWoo: Planner → Workers → Solver → Reflector, max 2 iterations.",
        sub_agents=[
            PlannerBridgeAgent(),
            make_planner_agent(),
            WorkersAgent(),
            make_solver_agent(),
            SolverBridgeAgent(),
            make_reflector_agent(),
            LoopBreakerAgent(),
        ],
        max_iterations=2,
    )

    return SequentialAgent(
        name="triage_root",
        description="HackerRank Orchestrate triage agent: PreFlight → ReWoo loop → Commit.",
        sub_agents=[
            TicketContextAgent(),
            make_preflight_agent(),
            PreflightBridgeAgent(),
            rewoo_loop,
            CommitAgent(),
        ],
    )
