"""
src/router.py

Stage 5 of the pipeline: classify an incoming question into a route (see
APPROACH.md §3.5), so agent.py knows whether to call structured_store.py
(exact) or vectorstore.py (semantic), and whether to scope to one form.

Design mirrors extraction.py's detect_form_type: keyword/regex fast-path
first, LLM classification only as a fallback. This keeps routing testable
without Ollama running (test_router.py can assert on the keyword path
alone) and matches APPROACH.md §4's "never trust the small model with
what a heuristic can do reliably" principle.

Four routes (APPROACH.md §3.5):
  - aggregate            -> structured_store.count_forms / list_forms
  - single_form_lookup   -> structured_store.get_form
  - single_form_semantic -> vectorstore.similarity_search(where={"form_id": ...})
  - multi_form_semantic  -> vectorstore.similarity_search(where=None)

retrieval.py consumes RouteDecision directly; it should not need to
re-inspect the question text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import llm_client
from schemas import FORM_SCHEMAS

FORM_ID_RE = re.compile(r"\b([a-zA-Z]+_\d{3,})\b")

_AGGREGATE_KEYWORDS = [
    "how many", "how much", "count", "number of",
    "which forms", "list all", "list the", "all forms",
]

_LOOKUP_KEYWORDS = [
    "what is the", "what's the", "when was", "who is the", "who was the",
]

# Field-value aliases per form_type, used to build an exact filter dict for
# the aggregate route without a text-to-SQL LLM call (APPROACH.md §3.6:
# "map recognized keywords to column filters ... safer and faster since
# the schema is fixed and small"). Extend this as new templates/fields
# are added — it's the one place that needs manual upkeep per schema.
FIELD_ALIASES: dict[str, dict[str, tuple[str, str]]] = {
    "membership": {
        "approved": ("status", "Approved"),
        "pending": ("status", "Pending"),
        "rejected": ("status", "Rejected"),
    },
    "hospital": {
        "cardiology": ("department", "Cardiology"),
        "discharged": ("discharge_status", "Stable, discharged with medication"),
    },
}


class RouteType(str, Enum):
    aggregate = "aggregate"
    single_form_lookup = "single_form_lookup"
    single_form_semantic = "single_form_semantic"
    multi_form_semantic = "multi_form_semantic"


class RoutingError(RuntimeError):
    """Raised when a question can't be routed (keyword miss + LLM fallback unavailable/unparseable)."""


@dataclass
class RouteDecision:
    route: RouteType
    detection_method: str          # "keyword" or "llm"
    form_id: str | None = None     # set for single_form_* routes when known
    form_type: str | None = None   # set for aggregate when a type is implied
    filters: dict = field(default_factory=dict)  # aggregate: {column: value}


def _extract_form_id(question: str) -> str | None:
    match = FORM_ID_RE.search(question)
    return match.group(1) if match else None


def _extract_form_type(question: str) -> str | None:
    lowered = question.lower()
    for form_type in FORM_SCHEMAS:
        if form_type in lowered:
            return form_type
    return None


def _extract_filters(question: str, form_type: str | None) -> dict:
    """Scans FIELD_ALIASES for keyword hits, scoped to form_type if known
    (else checks all registered types and merges — fine since alias
    vocabularies don't currently overlap across form types)."""
    lowered = question.lower()
    types_to_check = [form_type] if form_type else list(FIELD_ALIASES)
    filters: dict = {}
    for ft in types_to_check:
        for keyword, (column, value) in FIELD_ALIASES.get(ft, {}).items():
            if keyword in lowered:
                filters[column] = value
    return filters


def _keyword_route(question: str, form_id: str | None) -> RouteDecision | None:
    """Fast-path classification. Returns None if nothing matches, so the
    caller can fall back to the LLM."""
    lowered = question.lower()

    if any(kw in lowered for kw in _AGGREGATE_KEYWORDS):
        form_type = _extract_form_type(question)
        filters = _extract_filters(question, form_type)
        return RouteDecision(
            route=RouteType.aggregate,
            detection_method="keyword",
            form_type=form_type,
            filters=filters,
        )

    if form_id is not None and any(kw in lowered for kw in _LOOKUP_KEYWORDS):
        return RouteDecision(
            route=RouteType.single_form_lookup,
            detection_method="keyword",
            form_id=form_id,
        )

    if form_id is not None:
        # A form_id is present but the question isn't an obvious exact
        # lookup -> treat as an open-ended question about that one form's
        # free text (safer default than guessing single_form_lookup).
        return RouteDecision(
            route=RouteType.single_form_semantic,
            detection_method="keyword",
            form_id=form_id,
        )

    return None


def _llm_route(question: str) -> RouteDecision:
    """Fallback: one cheap classification call (APPROACH.md §3.5). Only
    reached when no form_id is present in the question and no aggregate
    keyword matched — i.e. genuinely ambiguous between single-form-lookup-
    without-an-id (rare/invalid, since we need a form_id to look up) and
    multi_form_semantic. In practice this mostly resolves to
    multi_form_semantic; the LLM call exists for future question phrasings
    APPROACH.md's keyword lists don't anticipate."""
    categories = [r.value for r in RouteType]
    prompt = (
        "Classify this question into exactly one category:\n"
        "- aggregate: asks to count or list forms matching some criteria\n"
        "- single_form_lookup: asks for one exact field of one specific form\n"
        "- single_form_semantic: an open-ended question about one specific form\n"
        "- multi_form_semantic: an open-ended question across many/all forms\n\n"
        f"Categories: {', '.join(categories)}\n"
        "Respond with only the category name, nothing else.\n\n"
        f"Question: {question}"
    )
    try:
        resp = llm_client.generate(prompt, temperature=0.0)
    except llm_client.LLMError as e:
        raise RoutingError(f"Keyword routing found no match and LLM fallback failed: {e}") from e

    guess = resp.text.strip().lower().strip(".")
    for route in RouteType:
        if route.value in guess:
            return RouteDecision(route=route, detection_method="llm")

    raise RoutingError(f"LLM returned an unrecognized route label: {resp.text.strip()!r}")


def route_question(question: str, form_id: str | None = None) -> RouteDecision:
    """
    Main entry point. `form_id` should be passed by agent.py when the
    caller already has one form open (e.g. a UI showing a single form's
    detail page); otherwise it's inferred from a form_id-shaped token in
    the question text itself.
    """
    if form_id is None:
        form_id = _extract_form_id(question)

    decision = _keyword_route(question, form_id)
    if decision is not None:
        return decision

    return _llm_route(question)


if __name__ == "__main__":
    # Manual smoke test — keyword path only, no Ollama needed.
    samples = [
        ("How many membership applications are approved?", None),
        ("How many hospital forms are in Cardiology?", None),
        ("What is the diagnosis for hospital_003?", None),
        ("Summarize the doctor's notes for hospital_003", None),
        ("What patterns show up across all rejected applicants?", None),
        ("Tell me about this applicant's credit history", "membership_001"),
    ]
    for question, fid in samples:
        try:
            decision = route_question(question, form_id=fid)
            print(f"{question!r}\n  -> {decision}\n")
        except RoutingError as e:
            print(f"{question!r}\n  -> FAILED: {e}\n")