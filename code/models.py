"""Pydantic models for the triage pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class DocType(str, Enum):
    HOW_TO = "how-to"
    FAQ = "faq"
    REFERENCE = "reference"
    CONCEPTUAL = "conceptual"
    INTEGRATION = "integration"
    TROUBLESHOOTING = "troubleshooting"
    RELEASE_NOTES = "release-notes"
    POLICY_LEGAL = "policy-legal"


VALID_COMPANIES = {"hackerrank", "claude", "visa"}


class Chunk(BaseModel):
    text: str
    company: str
    product_area: str
    doc_type: DocType
    source_path: str
    chunk_index: int
    heading_path: str
    sqlite_id: Optional[int] = None

    @model_validator(mode="after")
    def _validate(self):
        if not self.text or not self.text.strip():
            raise ValueError("Chunk.text cannot be empty")
        if self.company not in VALID_COMPANIES:
            raise ValueError(f"company must be one of {VALID_COMPANIES}")
        return self


class IndexResult(BaseModel):
    company: str
    files_processed: int
    chunks_created: int
    duration_seconds: float


class PreFlight(BaseModel):
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"]
    company_hint: Literal["hackerrank", "claude", "visa", "unknown"]
    intent: Literal["how_to", "factual", "conceptual", "complaint", "policy", "credential"]
    is_multi_request: bool = False
    language: str = "en"
    escalate_now: bool = False
    escalate_reason: Optional[str] = None


class Step(BaseModel):
    id: str
    type: Literal["retrieve", "escalate", "reply_static"]
    company: Optional[Literal["hackerrank", "claude", "visa"]] = None
    doc_type_filter: list[str] = Field(default_factory=list)
    query_variants: list[str] = Field(default_factory=list, max_length=3)
    purpose: Literal["answer", "label_only"] = "answer"
    message: Optional[str] = None
    reason: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_query(cls, values):
        if isinstance(values, dict) and "query_variants" not in values:
            q = values.get("query")
            if isinstance(q, str) and q.strip():
                values = {**values, "query_variants": [q.strip()]}
            elif isinstance(q, list):
                values = {**values, "query_variants": [s for s in q if isinstance(s, str) and s.strip()]}
        return values


class Plan(BaseModel):
    steps: list[Step]
    rationale: str = ""


class TriageOutput(BaseModel):
    response: str
    justification: str
    product_area: str
    cited_chunks: list[str] = Field(default_factory=list)


class Reflection(BaseModel):
    """Pure scoring model — the Reflector has no opinion on loop control or human
    routing. LoopBreakerAgent + CommitAgent consume final_score and decide.

    The legacy `escalate` field was a source of inverted-logic confusion (it meant
    `EventActions.escalate` = "exit the loop", not `status="escalated"` = "send to
    human") and has been removed.
    """

    grounding: float
    completeness: float
    safety: float
    actionability: float
    final_score: Optional[float] = None  # computed in code from the four dims; LLM may omit
    reason: str = ""
    # Reflector's independent verification of the PreFlight request_type. If it
    # agrees with PreFlight, emit the same value; if it disagrees, emit the
    # corrected value; if the ticket is genuinely ambiguous, emit "undefined".
    # CommitAgent uses this to override request_type when set.
    verified_request_type: Optional[
        Literal["product_issue", "feature_request", "bug", "invalid", "undefined"]
    ] = None

    @model_validator(mode="after")
    def _compute_final_score(self):
        # Single source of truth: code-computed weighted score.
        # Weights match the prompt-stated formula: g*0.4 + c*0.3 + s*0.2 + a*0.1.
        self.final_score = (
            self.grounding * 0.4
            + self.completeness * 0.3
            + self.safety * 0.2
            + self.actionability * 0.1
        )
        return self
