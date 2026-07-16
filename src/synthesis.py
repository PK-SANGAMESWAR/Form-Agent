"""
src/synthesis.py

Stage 7 of the pipeline: RetrievalResult -> grounded natural-language
answer with citations (see APPROACH.md §3.7).

Three entry points, matching agent.py's public API (APPROACH.md §3.8):
  - answer_question()  -> single_form_lookup / single_form_semantic /
                           multi_form_semantic RetrievalResults
  - summarize_form()    -> one form's full content (structured fields +
                           free text), no retrieval involved
  - format_aggregate()  -> aggregate RetrievalResults never need an LLM
                           call — structured_store's count/rows ARE the
                           answer (APPROACH.md §4 "never ask the LLM to
                           count"); this just wraps them in the same
                           SynthesisResult shape as the other two.

Grounding mechanism: the prompt instructs the LLM to answer only from the
supplied context and to tag every claim with the form_id(s) it came from,
e.g. "[membership_002]". _parse_citations() pulls those tags back out so
callers (agent.py / a UI) can show "answer, backed by these forms"
without re-parsing prose themselves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import llm_client
from retrieval import RetrievalResult
from router import RouteType

CITATION_RE = re.compile(r"\[([a-zA-Z]+_\d{3,})\]")

_NO_ANSWER_MARKER = "NOT_FOUND_IN_CONTEXT"


class SynthesisError(RuntimeError):
    """Raised when the LLM call itself fails (context-building failures are ValueErrors)."""


@dataclass
class SynthesisResult:
    answer: str
    cited_form_ids: list[str] = field(default_factory=list)
    grounded: bool = True   # False if the LLM reported it couldn't answer from context


def _parse_citations(text: str) -> list[str]:
    # dict.fromkeys dedupes while preserving first-seen order — nicer for
    # display than a set, which would scramble the order forms were cited in.
    return list(dict.fromkeys(CITATION_RE.findall(text)))


def _build_context_from_chunks(result: RetrievalResult) -> str:
    if not result.chunks:
        raise ValueError("RetrievalResult has no chunks to build context from.")
    blocks = []
    for chunk in result.chunks:
        blocks.append(f"[{chunk.form_id}] ({chunk.field_name}): {chunk.text}")
    return "\n\n".join(blocks)


def _build_context_from_lookup(result: RetrievalResult) -> str:
    if not result.form:
        raise ValueError("RetrievalResult has no form to build context from.")
    form_id = result.form.get("form_id", "unknown")
    fields = "\n".join(f"{k}: {v}" for k, v in result.form.items() if k != "form_id")
    return f"[{form_id}]\n{fields}"


def _answer_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using ONLY the information in the context below. "
        "Every claim you make must be tagged with the form_id it came from, "
        "using the exact format shown in the context, e.g. [membership_002]. "
        f"If the context does not contain the answer, respond with exactly: {_NO_ANSWER_MARKER}\n\n"
        f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Question: {question}"
    )


def answer_question(result: RetrievalResult, question: str) -> SynthesisResult:
    """
    Builds a grounded prompt from a semantic or lookup RetrievalResult and
    calls the LLM. Callers should route RouteType.aggregate results
    directly to format_aggregate() instead (see module docstring) — this
    function raises if handed one, since there's no free text to ground
    an LLM answer in.
    """
    if result.route == RouteType.aggregate:
        raise ValueError(
            "aggregate RetrievalResults don't need synthesis — call "
            "format_aggregate() instead, the numbers are already exact."
        )
    elif result.route == RouteType.single_form_lookup:
        context = _build_context_from_lookup(result)
    else:  # single_form_semantic or multi_form_semantic
        context = _build_context_from_chunks(result)

    prompt = _answer_prompt(question, context)
    try:
        resp = llm_client.generate(prompt, temperature=0.0)
    except llm_client.LLMError as e:
        raise SynthesisError(f"LLM call failed during answer synthesis: {e}") from e

    text = resp.text.strip()
    if _NO_ANSWER_MARKER in text:
        return SynthesisResult(answer="I couldn't find this in the provided forms.", grounded=False)

    return SynthesisResult(answer=text, cited_form_ids=_parse_citations(text))


def _summary_prompt(form_id: str, fields: dict, free_text_field: str, free_text: str) -> str:
    structured_lines = "\n".join(
        f"{k}: {v}" for k, v in fields.items() if k not in ("form_id", free_text_field)
    )
    return (
        f"Write a concise summary (3-5 sentences) of the following form [{form_id}], "
        "highlighting the most important details. Tag the summary with the form_id "
        f"in the format [{form_id}] at the end.\n\n"
        f"--- STRUCTURED FIELDS ---\n{structured_lines}\n\n"
        f"--- {free_text_field.upper()} ---\n{free_text}\n--- END ---"
    )


def summarize_form(form: dict, free_text_field: str) -> SynthesisResult:
    """
    Summarizes one form's full content (all structured fields + its
    free-text field). `form` is structured_store.get_form()'s return dict;
    `free_text_field` comes from schemas.FREE_TEXT_FIELD[form["form_type"]]
    — agent.py looks that up before calling this, so this module doesn't
    need to import schemas just for one dict lookup.
    """
    form_id = form.get("form_id", "unknown")
    free_text = form.get(free_text_field, "") or ""
    prompt = _summary_prompt(form_id, form, free_text_field, free_text)

    try:
        resp = llm_client.generate(prompt, temperature=0.0)
    except llm_client.LLMError as e:
        raise SynthesisError(f"LLM call failed during summarization: {e}") from e

    text = resp.text.strip()
    return SynthesisResult(answer=text, cited_form_ids=_parse_citations(text) or [form_id])


def format_aggregate(result: RetrievalResult) -> SynthesisResult:
    """
    No LLM call — aggregate counts are already exact (APPROACH.md §4).
    Exists so agent.py has one uniform SynthesisResult-shaped return
    across all four routes, instead of special-casing aggregate.
    """
    form_ids = [row.get("form_id", "") for row in result.rows]
    answer = f"{result.count} matching form(s): {', '.join(form_ids) if form_ids else 'none'}."
    return SynthesisResult(answer=answer, cited_form_ids=form_ids)


if __name__ == "__main__":
    # Manual smoke test — needs Ollama (this stage IS the LLM call, like
    # vectorstore.py's embedding smoke test).
    from vectorstore import RetrievedChunk

    if not llm_client.is_available():
        print(f"Ollama/{llm_client.DEFAULT_MODEL} not reachable — run `ollama serve` first.")
    else:
        fake_result = RetrievalResult(
            route=RouteType.multi_form_semantic,
            chunks=[
                RetrievedChunk(
                    chunk_id="membership_001::remarks::0", form_id="membership_001",
                    form_type="membership", field_name="remarks",
                    text="Applicant has a strong credit history and no prior defaults.",
                    metadata={}, distance=0.1,
                ),
                RetrievedChunk(
                    chunk_id="membership_002::remarks::0", form_id="membership_002",
                    form_type="membership", field_name="remarks",
                    text="Applicant has two prior credit defaults on file.",
                    metadata={}, distance=0.2,
                ),
            ],
        )
        result = answer_question(fake_result, "Which applicants have credit defaults?")
        print("Answer:", result.answer)
        print("Cited:", result.cited_form_ids)
        print("Grounded:", result.grounded)