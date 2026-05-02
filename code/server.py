"""FastAPI shim that exposes `agent.run_triage` over HTTP.

This is the target for LangWatch Scenarios — they POST a ticket payload, the
shim runs the full ADK pipeline, and returns the structured result. Telemetry
is initialised at startup so every request lands in LangWatch + the local JSONL
file just like CSV-mode runs do.

Single endpoint, zero conversational state — the agent is invoked as a fresh
session per request via `run_triage`. That keeps the shim trivially horizontally
scalable and matches how the CSV runner uses it.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Make the rest of code/* importable whether the user runs `uvicorn code.server:app`
# or `python code/main.py serve`.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


log = logging.getLogger("server")


class TriageRequest(BaseModel):
    issue: str = Field(..., description="The user's ticket body. May be multi-line.")
    subject: str = Field("", description="Optional subject line.")
    company: str = Field(
        "None",
        description='One of "HackerRank", "Claude", "Visa", or "None" (infer).',
    )


class TriageResponse(BaseModel):
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str


app = FastAPI(
    title="HackerRank Orchestrate triage shim",
    description="HTTP wrapper around agent.run_triage for LangWatch scenarios.",
    version="1.0.0",
)


@app.on_event("startup")
async def _on_startup() -> None:
    """Pre-warm: load .env, init telemetry, build the ADK root agent.

    The first triage call would do this anyway, but doing it at startup means
    LangWatch's first scenario doesn't pay the ~1s warmup latency.
    """
    from dotenv import load_dotenv

    repo_root = HERE.parent
    for candidate in (repo_root / ".env", Path.cwd() / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
            break
    else:
        load_dotenv()

    import telemetry as _telemetry
    _telemetry.init_telemetry()

    # Build the ADK root agent + InMemoryRunner so the first request is hot.
    import agent as _agent
    _agent._get_runner()
    log.info("triage shim ready (model=%s)", os.environ.get("OPENROUTER_MODEL", "x-ai/grok-4.3"))


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResponse)
async def triage(req: TriageRequest) -> TriageResponse:
    # Local import so module load doesn't block on ADK imports.
    from agent import run_triage

    try:
        result = run_triage(req.issue, req.subject, req.company or "None")
    except Exception as e:
        log.exception("triage failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)[:300]}")
    return TriageResponse(**result)
