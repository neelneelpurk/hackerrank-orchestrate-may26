"""OpenTelemetry telemetry setup for ADK runs.

Two exporters, both attached to the same TracerProvider so every span is fanned out:

1. **LangWatch (cloud)** — primary backend. If `LANGWATCH_API_KEY` is set, we call
   `langwatch.setup(tracer_provider=<our provider>)` which adds LangWatch's OTLP
   span processor. Spans appear in the LangWatch dashboard with the full trace
   tree (root SequentialAgent → preflight → ReWoo loop iterations → solver / etc.).

2. **JsonlFileSpanExporter (local)** — `runs/telemetry-<ts>.jsonl`, one JSON object
   per span. Survives even when LangWatch is unreachable; great for `jq` queries.

Span schema we emit (one line per span in the JSONL file):
{
  "trace_id": str, "span_id": str, "parent_span_id": str|null,
  "name": str, "kind": str,
  "start": iso8601, "end": iso8601, "duration_ms": float,
  "status": "OK"|"ERROR",
  "attributes": {...},
  "events": [{name, timestamp, attributes}]
}

Order matters: we build our own TracerProvider FIRST, attach the JSONL processor,
publish it as the global provider, THEN call `langwatch.setup(tracer_provider=...)`
passing in our provider so LangWatch adds its processor to the same provider rather
than replacing it. If we let langwatch.setup() create its own provider, our local
JSONL exporter would be orphaned.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


_TELEMETRY_INITIALIZED = False
_LOG_PATH: Optional[Path] = None
_LANGWATCH_ATTACHED = False
_LANGWATCH_CLIENT = None


class JsonlFileSpanExporter(SpanExporter):
    """Write OTel spans as JSONL — one line per span, append-only."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            try:
                self._fh.write(self._serialize(span) + "\n")
            except Exception as e:
                self._fh.write(json.dumps({"error": f"serialize_failed: {e}"}) + "\n")
        self._fh.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            self._fh.flush()
        except Exception:
            return False
        return True

    @staticmethod
    def _ns_to_iso(ns: int) -> str:
        return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc).isoformat()

    def _serialize(self, span: ReadableSpan) -> str:
        ctx = span.get_span_context()
        parent = span.parent
        attrs = {}
        if span.attributes:
            for k, v in span.attributes.items():
                # Keep things JSON-safe; trim very long values
                if isinstance(v, (str, int, float, bool)) or v is None:
                    if isinstance(v, str) and len(v) > 4000:
                        attrs[k] = v[:4000] + "…[truncated]"
                    else:
                        attrs[k] = v
                else:
                    try:
                        attrs[k] = json.loads(json.dumps(v, default=str))
                    except Exception:
                        attrs[k] = str(v)[:1000]
        events = []
        for ev in (span.events or []):
            events.append({
                "name": ev.name,
                "timestamp": self._ns_to_iso(ev.timestamp),
                "attributes": dict(ev.attributes) if ev.attributes else {},
            })
        record = {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
            "parent_span_id": format(parent.span_id, "016x") if parent else None,
            "name": span.name,
            "kind": str(span.kind).rsplit(".", 1)[-1],
            "start": self._ns_to_iso(span.start_time),
            "end": self._ns_to_iso(span.end_time),
            "duration_ms": (span.end_time - span.start_time) / 1_000_000,
            "status": str(span.status.status_code).rsplit(".", 1)[-1] if span.status else "UNSET",
            "attributes": attrs,
            "events": events,
        }
        return json.dumps(record, default=str)


def _attach_langwatch(provider: TracerProvider) -> bool:
    """If LANGWATCH_API_KEY is set, attach LangWatch's OTLP processor to our provider."""
    global _LANGWATCH_ATTACHED, _LANGWATCH_CLIENT
    api_key = os.environ.get("LANGWATCH_API_KEY", "").strip()
    if not api_key:
        return False
    try:
        # The langwatch SDK warns when an existing TracerProvider is found — that's
        # exactly our situation by design (we built it to host the JSONL exporter).
        # The behaviour is correct (it adds its processor instead of overriding) so
        # the warning is just noise. Silence it.
        import logging
        logging.getLogger("langwatch.client").setLevel(logging.ERROR)
        import langwatch
        endpoint = os.environ.get("LANGWATCH_ENDPOINT", "").strip() or None
        _LANGWATCH_CLIENT = langwatch.setup(
            api_key=api_key,
            endpoint_url=endpoint,
            tracer_provider=provider,
            base_attributes={
                "service.name": "hackerrank-orchestrate-triage",
                "deployment.environment": os.environ.get("LANGWATCH_ENV", "hackathon"),
            },
        )
        _LANGWATCH_ATTACHED = True
        return True
    except Exception as e:
        # Soft-fail: local JSONL still works even if LangWatch import / network breaks.
        print(f"[telemetry] LangWatch attach failed: {e}", flush=True)
        return False


def init_telemetry(log_path: Optional[str] = None) -> Path:
    """Idempotent: build our TracerProvider, attach JSONL exporter, then attach LangWatch."""
    global _TELEMETRY_INITIALIZED, _LOG_PATH
    if _TELEMETRY_INITIALIZED and _LOG_PATH is not None:
        return _LOG_PATH

    resolved_path = Path(
        log_path
        or os.environ.get("TELEMETRY_LOG")
        or f"runs/telemetry-{int(time.time())}.jsonl"
    )

    resource = Resource.create({"service.name": "hackerrank-orchestrate-triage"})
    provider = TracerProvider(resource=resource)
    exporter = JsonlFileSpanExporter(resolved_path)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    attached = _attach_langwatch(provider)
    if attached:
        print(f"[telemetry] LangWatch cloud + JSONL ({resolved_path})", flush=True)
    else:
        print(f"[telemetry] JSONL only ({resolved_path}) — set LANGWATCH_API_KEY to enable cloud", flush=True)

    _TELEMETRY_INITIALIZED = True
    _LOG_PATH = resolved_path
    return resolved_path


def get_log_path() -> Optional[Path]:
    return _LOG_PATH


def is_langwatch_attached() -> bool:
    return _LANGWATCH_ATTACHED


def shutdown_telemetry() -> None:
    """Force-flush + shutdown so all spans are persisted before exit (both local + cloud)."""
    try:
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=10000)
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception:
        pass
