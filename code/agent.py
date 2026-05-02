"""Top-level triage entry point.

Two ways to drive the pipeline:

- `run_triage(issue, subject, company)` — sync, one ticket. Spins up a fresh
  event loop per call. Convenient for interactive single-ticket use.

- `run_triage_batch(rows, on_result)` — sync, many tickets, all under a single
  event loop. The CSV mode in main.py uses this because the cached
  InMemoryRunner is bound to whichever loop first touched it; subsequent
  `asyncio.run()` calls land in *new* loops, the runner's internal coroutines
  reference the dead old loop, and rows fall through to the error path.

Telemetry: every LlmAgent / sub-agent / tool call is captured as an OpenTelemetry
span and exported to LangWatch + `runs/telemetry-<ts>.jsonl` by `code/telemetry.py`.
"""

from __future__ import annotations

import asyncio
import os
import traceback
import uuid
from typing import Callable, Optional

from google.adk.agents import SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from adk_agents import make_root_agent
import telemetry


_RUNNER: Optional[InMemoryRunner] = None
_ROOT: Optional[SequentialAgent] = None
_APP_NAME = "hackerrank_triage"


def _get_runner() -> InMemoryRunner:
    global _RUNNER, _ROOT
    if _RUNNER is None:
        telemetry.init_telemetry()
        _ROOT = make_root_agent()
        _RUNNER = InMemoryRunner(agent=_ROOT, app_name=_APP_NAME)
    return _RUNNER


def _reset_runner() -> None:
    """Drop the cached runner — used by run_triage when entering a new event loop."""
    global _RUNNER, _ROOT
    _RUNNER = None
    _ROOT = None


async def _run_one(issue: str, subject: str, company_col: str) -> dict:
    runner = _get_runner()
    user_id = "triage_user"
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    await runner.session_service.create_session(
        app_name=_APP_NAME, user_id=user_id, session_id=session_id,
        state={"company_col": company_col, "subject": subject},
    )

    ticket_text = ""
    if subject:
        ticket_text += f"Subject: {subject}\n"
    ticket_text += f"Company: {company_col}\nIssue: {issue}".strip()

    user_msg = genai_types.Content(role="user", parts=[genai_types.Part(text=ticket_text)])

    pipeline_error: Optional[Exception] = None
    try:
        async for _ in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_msg):
            # Drain events; the CommitAgent populates state["triage_result"].
            pass
    except Exception as e:
        pipeline_error = e
        print(f"\n[triage] pipeline error: {type(e).__name__}: {str(e)[:400]}", flush=True)
        traceback.print_exc()

    sess = await runner.session_service.get_session(
        app_name=_APP_NAME, user_id=user_id, session_id=session_id
    )
    result = (sess.state or {}).get("triage_result")
    if not result:
        why = (
            f"{type(pipeline_error).__name__}: {str(pipeline_error)[:200]}"
            if pipeline_error else "no triage_result on state — pipeline failed silently"
        )
        result = {
            "status": "escalated",
            "product_area": "uncategorized",
            "response": "Escalate to a human",
            "justification": why,
            "request_type": "invalid",
        }
    # Sanitise: every field must be a non-empty string, no None.
    return {
        "status": str(result.get("status") or "escalated"),
        "product_area": str(result.get("product_area") or "uncategorized"),
        "response": str(result.get("response") or "Escalate to a human"),
        "justification": str(result.get("justification") or "no justification"),
        "request_type": str(result.get("request_type") or "invalid"),
    }


def run_triage(issue: str, subject: str, company_col: str) -> dict:
    """Single-ticket synchronous wrapper. Each call spins up a fresh event loop
    AND a fresh runner so the runner's coroutines never reference a dead loop.
    """
    _reset_runner()
    return asyncio.run(_run_one(issue, subject, company_col))


def run_triage_batch(
    rows: list[dict],
    on_result: Callable[[int, dict, dict], None],
) -> None:
    """Run all rows under a SINGLE event loop with a single shared runner.

    `on_result(i, row, result)` is invoked after each ticket completes — that's
    where the CSV writer flushes its row in main.py. Per-row try/except keeps
    one bad ticket from killing the whole batch.
    """
    _reset_runner()

    async def _drive():
        for i, row in enumerate(rows, 1):
            try:
                result = await _run_one(row["Issue"], row["Subject"], row["Company"])
            except Exception as e:
                print(f"\n[triage] row {i} crashed: {type(e).__name__}: {str(e)[:300]}", flush=True)
                traceback.print_exc()
                result = {
                    "status": "escalated",
                    "product_area": "uncategorized",
                    "response": "Escalate to a human",
                    "justification": f"row_crash: {type(e).__name__}: {str(e)[:140]}",
                    "request_type": "invalid",
                }
            try:
                on_result(i, row, result)
            except Exception as e:
                print(f"[triage] on_result callback failed at row {i}: {e}", flush=True)
                traceback.print_exc()

    asyncio.run(_drive())
