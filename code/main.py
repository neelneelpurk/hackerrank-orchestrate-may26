"""Entry point: `python code/main.py {update-knowledge-base|triage|calibrate} ...`."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Ensure we can import sibling modules whether the user runs from repo root or code/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Load .env from repo root or current dir
ROOT = HERE.parent
for candidate in (ROOT / ".env", Path.cwd() / ".env"):
    if candidate.exists():
        load_dotenv(candidate)
        break
else:
    load_dotenv()

random.seed(42)
try:
    import numpy
    numpy.random.seed(42)
except Exception:
    pass

# Initialise OpenTelemetry once for the process so every ADK span is captured.
import telemetry as _telemetry
_telemetry.init_telemetry()
import atexit as _atexit
_atexit.register(_telemetry.shutdown_telemetry)


def _check_env() -> None:
    """Print which auth keys are detected (masked) so 401s are obvious before the LLM call."""
    def _mask(v: str) -> str:
        if not v:
            return "(missing)"
        if len(v) <= 10:
            return v[:2] + "…"
        return v[:6] + "…" + v[-4:]
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    jina = os.environ.get("JINA_API_KEY", "")
    lw = os.environ.get("LANGWATCH_API_KEY", "")
    model = (
        os.environ.get("OPENROUTER_MODEL")
        or os.environ.get("MOONSHOT_MODEL")
        or "x-ai/grok-4.3"
    )
    print(
        f"[env] OPENROUTER_API_KEY={_mask(or_key)} | "
        f"JINA_API_KEY={_mask(jina)} | "
        f"LANGWATCH_API_KEY={_mask(lw)} | "
        f"MODEL={model}",
        flush=True,
    )


OUTPUT_HEADER = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]


def cmd_update_kb(args):
    from indexer import run as run_indexer
    results = run_indexer(args.dir, force=args.force)
    print("\nDone. Summary:")
    for r in results:
        print(f"  {r.company:12s} files={r.files_processed:4d} chunks={r.chunks_created:5d} time={r.duration_seconds:.1f}s")


def cmd_triage(args):
    if args.debug:
        try:
            import litellm
            litellm._turn_on_debug()
        except Exception:
            pass
    _check_env()
    from agent import run_triage, run_triage_batch

    if args.csv:
        rows = _read_input_csv(args.csv)
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]
            print(f"[triage] limiting to first {len(rows)} row(s)")
        out_path = args.output or "support_tickets/output.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        f = open(out_path, "w", newline="", encoding="utf-8")
        # QUOTE_MINIMAL: header stays bare (matches sample_support_tickets.csv);
        # only fields containing commas / quotes / newlines get wrapped.
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(OUTPUT_HEADER)
        f.flush()

        n = len(rows)
        start = time.time()
        completed = 0

        def _on_result(i: int, row: dict, result: dict) -> None:
            nonlocal completed
            completed += 1
            try:
                writer.writerow([
                    row["Issue"],
                    row["Subject"],
                    row["Company"],
                    result.get("response", ""),
                    result.get("product_area", ""),
                    result.get("status", ""),
                    result.get("request_type", ""),
                    result.get("justification", ""),
                ])
                f.flush()
                os.fsync(f.fileno())
            except Exception as e:
                print(f"[triage] CSV write failed at row {i}: {e}", flush=True)
                return
            elapsed = time.time() - start
            subj = (row.get("Subject") or row.get("Issue") or "")[:50].replace("\n", " ")
            print(
                f"[{i:2d}/{n}] {elapsed:5.1f}s  {result.get('status', '?'):9s} "
                f"({result.get('product_area', '?'):25s}) — {subj}",
                flush=True,
            )

        try:
            run_triage_batch(rows, _on_result)
        finally:
            f.close()
        print(f"\nWrote {completed}/{n} rows to {out_path}")
        if completed != n:
            print(f"[warn] {n - completed} row(s) did not complete — check stderr for crashes.")
        return

    print("Interactive triage. Empty Issue ends.")
    issue = input("Issue: ").strip()
    if not issue:
        return
    subject = input("Subject: ").strip()
    company = input("Company [HackerRank|Claude|Visa|None]: ").strip() or "None"
    result = run_triage(issue, subject, company)
    print()
    print(f"Status:        {result['status']}")
    print(f"Request type:  {result['request_type']}")
    print(f"Product area:  {result['product_area']}")
    print(f"Justification: {result['justification']}")
    print(f"\nResponse:\n{result['response']}")


def cmd_calibrate(args):
    from calibrator import run as run_cal
    run_cal(args.csv)


def cmd_serve(args):
    """Run the FastAPI triage shim via uvicorn — the target for LangWatch scenarios."""
    _check_env()
    import uvicorn
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        log_level="info",
        # Single worker because the ADK runner caches state on the module level
        # and LangWatch scenarios are sequential anyway.
        workers=1,
        reload=False,
    )


def cmd_diag(args):
    """Bypass ADK and call litellm.completion directly to confirm the auth chain."""
    _check_env()
    if args.debug:
        try:
            import litellm
            litellm._turn_on_debug()
        except Exception:
            pass

    import litellm
    model_name = (
        os.environ.get("OPENROUTER_MODEL")
        or os.environ.get("MOONSHOT_MODEL")
        or "x-ai/grok-4.3"
    )
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("MOONSHOT_API_KEY", "")
    full_model = f"openrouter/{model_name}"
    print(f"[diag] direct litellm.completion(model={full_model!r}) with api_key=…{api_key[-4:] if api_key else 'NONE'}", flush=True)
    try:
        resp = litellm.completion(
            model=full_model,
            api_key=api_key,
            temperature=0,
            max_tokens=20,
            messages=[
                {"role": "system", "content": "Reply with exactly: pong"},
                {"role": "user", "content": "ping"},
            ],
        )
        msg = resp.choices[0].message.content
        print(f"[diag] OK direct call → {msg!r}", flush=True)
    except Exception as e:
        import traceback
        print(f"[diag] FAILED direct call: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return

    # If direct works, also try via ADK's LiteLlm wrapper to isolate the bug.
    print("[diag] now testing via ADK LiteLlm wrapper…", flush=True)
    try:
        import asyncio
        from google.adk.models.lite_llm import LiteLlm
        from google.adk.models.llm_request import LlmRequest
        from google.genai import types as gt

        m = LiteLlm(model=full_model, api_key=api_key, temperature=0)
        req = LlmRequest(
            model=full_model,
            contents=[gt.Content(role="user", parts=[gt.Part(text="ping")])],
            config=gt.GenerateContentConfig(
                system_instruction="Reply with exactly: pong",
                max_output_tokens=20,
            ),
        )

        async def _go():
            async for chunk in m.generate_content_async(req, stream=False):
                return chunk

        resp = asyncio.run(_go())
        if resp and resp.content and resp.content.parts:
            print(f"[diag] OK ADK call → {resp.content.parts[0].text!r}", flush=True)
        else:
            print(f"[diag] ADK returned empty response: {resp}", flush=True)
    except Exception as e:
        import traceback
        print(f"[diag] FAILED ADK call: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


def _read_input_csv(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "Issue": (r.get("Issue") or "").strip(),
                "Subject": (r.get("Subject") or "").strip(),
                "Company": (r.get("Company") or "").strip() or "None",
            })
    return rows


def main():
    ap = argparse.ArgumentParser(prog="triage", description="HackerRank Orchestrate support triage agent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_kb = sub.add_parser("update-knowledge-base", help="Build SQLite + ChromaDB from data/")
    p_kb.add_argument("--dir", default="data")
    p_kb.add_argument(
        "--force",
        action="store_true",
        help="Clear existing SQLite + Chroma rows for each company and re-embed everything (ignores resume state).",
    )
    p_kb.set_defaults(func=cmd_update_kb)

    p_tr = sub.add_parser("triage", help="Triage tickets (CSV mode or interactive)")
    p_tr.add_argument("--csv", help="Path to support_tickets.csv")
    p_tr.add_argument("--output", help="Path to output.csv (CSV mode)")
    p_tr.add_argument("--limit", type=int, default=0, help="Process only the first N rows (0 = all).")
    p_tr.add_argument("--debug", action="store_true", help="Enable LiteLLM debug logging.")
    p_tr.set_defaults(func=cmd_triage)

    p_cal = sub.add_parser("calibrate", help="Sweep thresholds against the labeled sample")
    p_cal.add_argument("--csv", default="support_tickets/sample_support_tickets.csv")
    p_cal.set_defaults(func=cmd_calibrate)

    p_di = sub.add_parser("diag", help="Sanity-check the LLM auth chain (direct LiteLLM + via ADK).")
    p_di.add_argument("--debug", action="store_true", help="Enable LiteLLM debug logging.")
    p_di.set_defaults(func=cmd_diag)

    p_sv = sub.add_parser("serve", help="Run the FastAPI triage shim — target for LangWatch scenarios.")
    p_sv.add_argument("--host", default="0.0.0.0")
    p_sv.add_argument("--port", type=int, default=8000)
    p_sv.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
