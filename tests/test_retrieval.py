"""
tests/test_retrieval.py

Tests for retrieval.py — aggregate, lookup, semantic routes (vector store mocked),
and the SQLite fallback when Chroma has no chunks.
No LLM / Ollama required.
"""
import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_decision(route_name, form_id=None, form_type=None, filters=None):
    from router import RouteDecision, RouteType
    return RouteDecision(
        route=RouteType[route_name],
        detection_method="keyword",
        form_id=form_id,
        form_type=form_type,
        filters=filters or {},
    )


def _fake_chunk(form_id="membership_001"):
    from vectorstore import RetrievedChunk
    return RetrievedChunk(
        chunk_id=f"{form_id}::remarks::0",
        form_id=form_id,
        form_type="membership",
        field_name="remarks",
        text="Applicant has strong credit history.",
        metadata={},
        distance=0.1,
    )


# ── aggregate route ───────────────────────────────────────────────────────────

def test_retrieve_aggregate(mem_conn, extracted_membership):
    import structured_store, retrieval
    structured_store.insert_form(mem_conn, extracted_membership)

    decision = _make_decision("aggregate", form_type="membership", filters={"status": "Approved"})
    result = retrieval.retrieve(decision, "how many approved?", mem_conn, collection=None)

    assert result.count == 1
    assert result.rows[0]["status"] == "Approved"


def test_retrieve_aggregate_zero_results(mem_conn):
    import retrieval
    decision = _make_decision("aggregate", form_type="membership", filters={"status": "Rejected"})
    result = retrieval.retrieve(decision, "how many rejected?", mem_conn, collection=None)
    assert result.count == 0
    assert result.rows == []


def test_retrieve_aggregate_missing_form_type_raises(mem_conn):
    import retrieval
    decision = _make_decision("aggregate", form_type=None)
    with pytest.raises(retrieval.RetrievalError, match="form_type"):
        retrieval.retrieve(decision, "how many?", mem_conn, collection=None)


# ── single_form_lookup route ──────────────────────────────────────────────────

def test_retrieve_single_form_lookup(mem_conn, extracted_membership):
    import structured_store, retrieval
    structured_store.insert_form(mem_conn, extracted_membership)

    decision = _make_decision("single_form_lookup", form_id="membership_001")
    result = retrieval.retrieve(decision, "what is status?", mem_conn, collection=None)

    assert result.form is not None
    assert result.form["name"] == "Ananya Rao"


def test_retrieve_single_form_lookup_missing_raises(mem_conn):
    import retrieval
    decision = _make_decision("single_form_lookup", form_id="does_not_exist")
    with pytest.raises(retrieval.RetrievalError, match="No form found"):
        retrieval.retrieve(decision, "status?", mem_conn, collection=None)


def test_retrieve_single_form_lookup_no_form_id_raises(mem_conn):
    import retrieval
    decision = _make_decision("single_form_lookup", form_id=None)
    with pytest.raises(retrieval.RetrievalError, match="requires a form_id"):
        retrieval.retrieve(decision, "status?", mem_conn, collection=None)


# ── single_form_semantic route ────────────────────────────────────────────────

def test_retrieve_single_form_semantic_with_chunks(mem_conn, extracted_membership):
    import structured_store, retrieval

    structured_store.insert_form(mem_conn, extracted_membership)
    mock_col = MagicMock()

    chunk = _fake_chunk("membership_001")
    with patch("retrieval.vectorstore.similarity_search", return_value=[chunk]):
        decision = _make_decision("single_form_semantic", form_id="membership_001")
        result = retrieval.retrieve(decision, "credit history?", mem_conn, mock_col)

    assert len(result.chunks) == 1
    assert result.chunks[0].form_id == "membership_001"


def test_retrieve_single_form_semantic_fallback_to_lookup(mem_conn, extracted_membership):
    """When Chroma returns no chunks, should fall back to structured store."""
    import structured_store, retrieval
    from router import RouteType

    structured_store.insert_form(mem_conn, extracted_membership)
    mock_col = MagicMock()

    with patch("retrieval.vectorstore.similarity_search", return_value=[]):
        decision = _make_decision("single_form_semantic", form_id="membership_001")
        result = retrieval.retrieve(decision, "anything?", mem_conn, mock_col)

    # Should have fallen back to lookup route
    assert result.route == RouteType.single_form_lookup
    assert result.form is not None
    assert result.form["name"] == "Ananya Rao"


def test_retrieve_single_form_semantic_no_chunks_no_conn_raises():
    """No chunks + no conn → should raise RetrievalError."""
    import retrieval
    mock_col = MagicMock()
    with patch("retrieval.vectorstore.similarity_search", return_value=[]):
        decision = _make_decision("single_form_semantic", form_id="hospital_001")
        with pytest.raises(retrieval.RetrievalError, match="No chunks found"):
            retrieval.retrieve(decision, "anything?", conn=None, collection=mock_col)


# ── multi_form_semantic route ─────────────────────────────────────────────────

def test_retrieve_multi_form_semantic(mem_conn):
    import retrieval
    mock_col = MagicMock()

    chunks = [_fake_chunk("membership_001"), _fake_chunk("membership_002")]
    with patch("retrieval.vectorstore.similarity_search", return_value=chunks), \
         patch("retrieval._bm25_search", return_value=chunks):
        decision = _make_decision("multi_form_semantic")
        result = retrieval.retrieve(decision, "credit defaults?", mem_conn, mock_col)

    assert len(result.chunks) > 0


def test_retrieve_multi_form_semantic_empty_corpus(mem_conn):
    """Empty Chroma + empty BM25 → zero chunks, no crash."""
    import retrieval
    mock_col = MagicMock()

    with patch("retrieval.vectorstore.similarity_search", return_value=[]), \
         patch("retrieval._bm25_search", return_value=[]):
        decision = _make_decision("multi_form_semantic")
        result = retrieval.retrieve(decision, "anything?", mem_conn, mock_col)

    assert result.chunks == []
