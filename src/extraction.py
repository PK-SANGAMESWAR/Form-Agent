"""
src/extraction.py

Stage 2 of the pipeline: RawForm -> validated structured JSON (see
APPROACH.md §3.2).

Two responsibilities live here:
  1. Form-type detection — which schema in schemas.FORM_SCHEMAS applies.
  2. Schema-guided extraction — call the local LLM (via llm_client) to fill
     that schema from the raw text, validate against Pydantic, and retry
     once with the validation error appended if it fails.

Design choice (form-type detection): keyword heuristic FIRST, LLM
classification only as a fallback. Form headers are reliable literal
signals ("MEMBERSHIP APPLICATION FORM" vs "HOSPITAL ADMISSION FORM"), so a
regex/keyword check is both faster and more deterministic than an LLM call
for the templates we know about. The LLM fallback exists so a form that
doesn't match either keyword set still gets routed instead of hard-failing
— useful once more templates are added later.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

import llm_client
from ingestion import RawForm
from schemas import FORM_SCHEMAS, schema_example

MAX_EXTRACTION_RETRIES = 1

# Keyword fast-path for form-type detection. Order matters only in that
# each entry is checked independently — a form could theoretically match
# both if templates were less distinct, so keep these header phrases
# specific to each template.
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "membership": ["membership application"],
    "hospital": ["hospital admission"],
}


class ExtractionError(RuntimeError):
    """Raised when form-type detection or field extraction can't produce a usable result."""


@dataclass
class ExtractedForm:
    form_id: str
    form_type: str
    data: BaseModel
    detection_method: str  # "keyword" or "llm"
    extraction_attempts: int


def detect_form_type(text: str) -> tuple[str, str]:
    """
    Returns (form_type, detection_method). Tries the keyword fast-path
    first; falls back to a one-word LLM classification call if no keyword
    set matches.
    """
    lowered = text.lower()
    for form_type, keywords in _TYPE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return form_type, "keyword"

    # Fallback: ask the LLM to pick from the known types.
    known_types = list(FORM_SCHEMAS.keys())
    prompt = (
        "You are classifying a scanned/extracted form's raw text into exactly "
        f"one of these categories: {', '.join(known_types)}.\n"
        "Respond with only the single matching category word, nothing else.\n\n"
        f"--- FORM TEXT ---\n{text[:2000]}\n--- END ---"
    )
    try:
        resp = llm_client.generate(prompt, temperature=0.0)
    except llm_client.LLMError as e:
        raise ExtractionError(
            f"Keyword detection failed and LLM fallback is unavailable: {e}"
        ) from e

    guess = resp.text.strip().lower().strip(".")
    for form_type in known_types:
        if form_type in guess:
            return form_type, "llm"

    raise ExtractionError(
        f"Could not determine form type. Keyword match failed and LLM "
        f"returned an unrecognized category: {resp.text.strip()!r}"
    )


def _build_extraction_prompt(form_type: str, raw_text: str, prior_error: str | None = None) -> str:
    example = schema_example(form_type)
    prompt = (
        "Extract structured fields from the form text below into JSON that "
        "matches this exact shape (same keys, same types). Use the example "
        "only to see the expected shape — extract the ACTUAL values from the "
        "form text, do not copy the example's values.\n\n"
        f"Expected JSON shape (example):\n{example}\n\n"
        f"--- FORM TEXT ---\n{raw_text}\n--- END ---\n\n"
        "Respond with only the JSON object, no other text."
    )
    if prior_error:
        prompt += (
            "\n\nYour previous attempt failed validation with this error — "
            f"fix it and try again:\n{prior_error}"
        )
    return prompt


def extract_fields(form_type: str, raw_text: str) -> tuple[BaseModel, int]:
    """
    Runs the schema-guided extraction call, validates against the
    registered Pydantic schema, and retries once (with the validation
    error appended to the prompt) if validation fails.

    Returns (validated_model_instance, attempts_used).
    """
    if form_type not in FORM_SCHEMAS:
        raise ExtractionError(f"No schema registered for form_type={form_type!r}")
    schema_cls = FORM_SCHEMAS[form_type]

    prior_error: str | None = None
    last_exc: Exception | None = None

    for attempt in range(1, MAX_EXTRACTION_RETRIES + 2):  # e.g. 1 try + 1 retry = 2 total
        prompt = _build_extraction_prompt(form_type, raw_text, prior_error)
        try:
            raw_json = llm_client.generate_json(prompt)
        except llm_client.LLMError as e:
            raise ExtractionError(f"LLM call failed during extraction: {e}") from e

        try:
            validated = schema_cls.model_validate(raw_json)
            return validated, attempt
        except ValidationError as e:
            last_exc = e
            prior_error = str(e)

    raise ExtractionError(
        f"Extraction failed validation after {MAX_EXTRACTION_RETRIES + 1} attempts "
        f"for form_type={form_type!r}: {last_exc}"
    )


def extract_form(raw_form: RawForm) -> ExtractedForm:
    """Full stage-2 pipeline for one RawForm: detect type, then extract + validate."""
    form_type, detection_method = detect_form_type(raw_form.text)
    validated, attempts = extract_fields(form_type, raw_form.text)
    return ExtractedForm(
        form_id=raw_form.form_id,
        form_type=form_type,
        data=validated,
        detection_method=detection_method,
        extraction_attempts=attempts,
    )


if __name__ == "__main__":
    # Quick manual smoke test against the sample forms.
    import os

    from ingestion import ingest_directory

    sample_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "sample_forms",
    )
    raw_forms = ingest_directory(sample_dir)
    for rf in raw_forms:
        try:
            result = extract_form(rf)
            print(f"{result.form_id} -> type={result.form_type} "
                  f"(detected via {result.detection_method}, "
                  f"{result.extraction_attempts} attempt(s))")
            print("  ", result.data.model_dump())
        except ExtractionError as e:
            print(f"{rf.form_id} -> FAILED: {e}")
        print()