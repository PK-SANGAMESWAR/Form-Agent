"""
tests/test_chunking.py

Tests for chunking.py — sentence splitting, word-count-based grouping,
TextChunk metadata, and build_chunks() output.
No LLM / Ollama required.
"""
import pytest


# ── chunk_text ────────────────────────────────────────────────────────────────

def test_chunk_text_empty():
    from chunking import chunk_text
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_single_short_sentence():
    from chunking import chunk_text
    result = chunk_text("Hello world.", target_words=50, max_words=100)
    assert result == ["Hello world."]


def test_chunk_text_produces_at_least_one_chunk():
    from chunking import chunk_text
    text = "Patient admitted with chest pain. Angioplasty performed. Discharged after five days."
    chunks = chunk_text(text, target_words=5, max_words=10)
    assert len(chunks) >= 1


def test_chunk_text_respects_max_words():
    from chunking import chunk_text
    # Each sentence is ~6 words; max_words=8 means at most 1 full sentence per chunk
    text = "Patient admitted with chest pain. Angioplasty was performed successfully. Discharged after five days."
    chunks = chunk_text(text, target_words=5, max_words=8)
    for chunk in chunks:
        word_count = len(chunk.split())
        assert word_count <= 20, f"Chunk too long: {word_count} words"


def test_chunk_text_no_splits_below_target():
    from chunking import chunk_text
    # If total text is below target_words, should be one chunk
    text = "Short note here."
    chunks = chunk_text(text, target_words=100, max_words=200)
    assert len(chunks) == 1
    assert chunks[0] == "Short note here."


# ── build_chunks ──────────────────────────────────────────────────────────────

def test_build_chunks_membership(extracted_membership):
    from chunking import build_chunks
    chunks = build_chunks(extracted_membership)
    assert len(chunks) >= 1
    # chunk_id format: {form_id}::{field_name}::{index}
    assert chunks[0].chunk_id.startswith("membership_001::remarks::")
    assert chunks[0].form_id == "membership_001"
    assert chunks[0].form_type == "membership"
    assert chunks[0].field_name == "remarks"


def test_build_chunks_hospital(extracted_hospital):
    from chunking import build_chunks
    chunks = build_chunks(extracted_hospital)
    assert len(chunks) >= 1
    assert chunks[0].chunk_id.startswith("hospital_001::doctors_notes::")
    assert chunks[0].field_name == "doctors_notes"


def test_build_chunks_metadata_excludes_free_text_field(extracted_membership):
    """Metadata on each chunk should NOT include the free-text field itself."""
    from chunking import build_chunks
    chunks = build_chunks(extracted_membership)
    for chunk in chunks:
        assert "remarks" not in chunk.metadata


def test_build_chunks_metadata_includes_form_type(extracted_hospital):
    from chunking import build_chunks
    chunks = build_chunks(extracted_hospital)
    for chunk in chunks:
        assert chunk.metadata.get("form_type") == "hospital"


def test_build_chunks_chunk_ids_are_unique(extracted_membership):
    """Chunk IDs must be unique even if the text is split into multiple pieces."""
    from chunking import build_chunks
    chunks = build_chunks(extracted_membership, target_words=2, max_words=4)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected"


def test_build_chunks_unknown_form_type_raises():
    from chunking import build_chunks
    from extraction import ExtractedForm
    from schemas import MembershipForm
    bad = ExtractedForm(
        form_id="x_001",
        form_type="unknown_type",
        data=MembershipForm(
            name="X", date_of_birth="01-01-2000",
            email="x@example.com", phone="+911234567890",
            occupation="X", application_date="01-01-2026",
            status="Pending", remarks="test",
        ),
        detection_method="keyword",
        extraction_attempts=1,
    )
    with pytest.raises(KeyError):
        build_chunks(bad)


def test_build_chunks_empty_free_text_returns_no_chunks(hospital_form):
    """If the free-text field is empty, no chunks should be produced."""
    from chunking import build_chunks
    from extraction import ExtractedForm
    from schemas import HospitalForm
    empty_notes = HospitalForm(
        **{**hospital_form.model_dump(), "doctors_notes": ""}
    )
    extracted = ExtractedForm(
        form_id="hospital_empty",
        form_type="hospital",
        data=empty_notes,
        detection_method="keyword",
        extraction_attempts=1,
    )
    chunks = build_chunks(extracted)
    assert chunks == []
