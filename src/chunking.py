"""
src/chunking.py

Stage 3a of the pipeline: ExtractedForm -> list[TextChunk] (see
APPROACH.md §3.3).

Only the free-text field per form type (schemas.FREE_TEXT_FIELD — e.g.
`remarks` for membership forms, `doctors_notes` for hospital forms) gets
chunked and embedded. Everything else is an exact structured field and
belongs in structured_store.py instead.

Chunking strategy: paragraph/sentence-group chunks of roughly
`target_words`-`max_words` words rather than a fixed character window —
this avoids splitting a single claim/sentence across two chunks, which
would hurt retrieval precision more than it helps recall for text this
short (see APPROACH.md §3.3). Most individual form remarks fields are one
paragraph, so in practice most forms produce exactly one chunk; longer
free-text fields (e.g. a lengthy doctor's note) split cleanly on sentence
boundaries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from extraction import ExtractedForm
from schemas import FREE_TEXT_FIELD

# Word counts, not strict LLM tokens — close enough approximation for
# chunk sizing without pulling in a tokenizer dependency here.
DEFAULT_TARGET_WORDS = 150
DEFAULT_MAX_WORDS = 300

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TextChunk:
    chunk_id: str          # f"{form_id}::{field_name}::{index}"
    form_id: str
    form_type: str
    field_name: str        # which free-text field this came from
    text: str
    metadata: dict = field(default_factory=dict)  # other extracted fields, for filtering


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def chunk_text(
    text: str,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
) -> list[str]:
    """
    Groups sentences into chunks of ~target_words, never exceeding
    max_words unless a single sentence alone is longer (in which case
    that sentence becomes its own chunk rather than being split
    mid-sentence).
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # Flush current chunk if adding this sentence would blow past max_words
        # and we already have at least target_words collected.
        if current and current_words + sentence_words > max_words and current_words >= target_words:
            chunks.append(" ".join(current))
            current, current_words = [], 0

        current.append(sentence)
        current_words += sentence_words

        # Natural break: we've hit target size, flush now.
        if current_words >= target_words:
            chunks.append(" ".join(current))
            current, current_words = [], 0

    if current:
        chunks.append(" ".join(current))

    return chunks


def build_chunks(
    extracted: ExtractedForm,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
) -> list[TextChunk]:
    """
    Builds all TextChunks for one ExtractedForm. Metadata attached to each
    chunk is every OTHER extracted field (i.e. everything except the
    free-text field itself) — this is what lets retrieval.py filter by
    e.g. {"status": "Rejected"} or {"department": "Cardiology"} before or
    after the similarity search.
    """
    form_type = extracted.form_type
    if form_type not in FREE_TEXT_FIELD:
        raise KeyError(f"No free-text field registered for form_type={form_type!r}")
    field_name = FREE_TEXT_FIELD[form_type]

    full_data = extracted.data.model_dump()
    free_text = full_data.get(field_name, "") or ""
    metadata = {k: v for k, v in full_data.items() if k != field_name}
    metadata["form_type"] = form_type

    pieces = chunk_text(free_text, target_words=target_words, max_words=max_words)
    return [
        TextChunk(
            chunk_id=f"{extracted.form_id}::{field_name}::{i}",
            form_id=extracted.form_id,
            form_type=form_type,
            field_name=field_name,
            text=piece,
            metadata=metadata,
        )
        for i, piece in enumerate(pieces)
    ]


if __name__ == "__main__":
    # Manual smoke test using a hand-built ExtractedForm (no LLM/Ollama
    # needed — this stage only depends on already-extracted data).
    from schemas import MembershipForm

    fake = ExtractedForm(
        form_id="membership_001",
        form_type="membership",
        data=MembershipForm(
            name="Ananya Rao",
            date_of_birth="14-03-1994",
            email="ananya.rao@example.com",
            phone="+91-9876543210",
            occupation="Software Engineer",
            application_date="02-01-2026",
            status="Approved",
            remarks=(
                "Applicant has a strong credit history and was approved after "
                "standard verification. No prior defaults on record. Recommended "
                "for the premium membership tier due to consistent income "
                "documentation."
            ),
        ),
        detection_method="keyword",
        extraction_attempts=1,
    )

    for chunk in build_chunks(fake, target_words=20, max_words=40):
        print(chunk.chunk_id)
        print("  text:", chunk.text)
        print("  metadata:", chunk.metadata)
        print()