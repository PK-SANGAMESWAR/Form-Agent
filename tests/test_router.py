"""
tests/test_router.py

Tests for router.py — keyword routing (no LLM) and LLM fallback (mocked).
The keyword path is fully deterministic so we can assert exact RouteType values.
"""
import pytest
from unittest.mock import patch, MagicMock


# ── Keyword fast-path ─────────────────────────────────────────────────────────

def test_route_aggregate_how_many():
    from router import route_question, RouteType
    d = route_question("How many membership applications are approved?")
    assert d.route == RouteType.aggregate
    assert d.detection_method == "keyword"
    assert d.form_type == "membership"
    assert d.filters == {"status": "Approved"}


def test_route_aggregate_count():
    from router import route_question, RouteType
    d = route_question("Count the hospital forms in Cardiology")
    assert d.route == RouteType.aggregate
    assert d.form_type == "hospital"


def test_route_aggregate_list_all():
    from router import route_question, RouteType
    d = route_question("List all membership forms")
    assert d.route == RouteType.aggregate


def test_route_single_form_lookup_with_form_id():
    from router import route_question, RouteType
    d = route_question("What is the diagnosis?", form_id="hospital_001")
    assert d.route == RouteType.single_form_lookup
    assert d.form_id == "hospital_001"
    assert d.detection_method == "keyword"


def test_route_single_form_lookup_whats_the():
    from router import route_question, RouteType
    d = route_question("What's the status?", form_id="membership_002")
    assert d.route == RouteType.single_form_lookup
    assert d.form_id == "membership_002"


def test_route_single_form_semantic_open_question():
    """Open-ended question with a form_id → single_form_semantic."""
    from router import route_question, RouteType
    d = route_question("Tell me about the credit history", form_id="membership_001")
    assert d.route == RouteType.single_form_semantic
    assert d.form_id == "membership_001"


def test_route_filters_extracted_correctly():
    from router import route_question
    d = route_question("How many membership forms are rejected?")
    assert d.filters.get("status") == "Rejected"


def test_route_no_form_id_in_question_no_keyword():
    """No form_id, no aggregate/lookup keyword → falls through to LLM route."""
    from router import route_question, RouteType
    # Mock the LLM to return multi_form_semantic
    mock_resp = MagicMock()
    mock_resp.text = "multi_form_semantic"
    with patch("router.llm_client.generate", return_value=mock_resp):
        d = route_question("What patterns show up across applicants?")
    assert d.route == RouteType.multi_form_semantic
    assert d.detection_method == "llm"


# ── Form ID extraction from question text ─────────────────────────────────────

def test_form_id_extracted_from_question():
    from router import route_question, RouteType
    # Form ID embedded in question text → should be auto-extracted
    d = route_question("What is the status of membership_002?")
    assert d.form_id == "membership_002"


def test_explicit_form_id_overrides_question_text():
    """Explicitly passed form_id takes precedence over text extraction."""
    from router import route_question
    d = route_question("What is the diagnosis for hospital_001?", form_id="hospital_002")
    # form_id passed explicitly should win
    assert d.form_id == "hospital_002"


# ── LLM fallback (mocked) ─────────────────────────────────────────────────────

def test_llm_fallback_aggregate():
    from router import route_question, RouteType
    mock_resp = MagicMock()
    mock_resp.text = "aggregate"
    with patch("router.llm_client.generate", return_value=mock_resp):
        d = route_question("Some ambiguous aggregate question?")
    assert d.route == RouteType.aggregate


def test_llm_fallback_unrecognized_raises():
    from router import route_question, RoutingError
    mock_resp = MagicMock()
    mock_resp.text = "banana"  # not a valid route
    with patch("router.llm_client.generate", return_value=mock_resp):
        with pytest.raises(RoutingError):
            route_question("Something completely ambiguous?")


def test_llm_fallback_unavailable_raises():
    from router import route_question, RoutingError
    import llm_client
    with patch("router.llm_client.generate", side_effect=llm_client.LLMError("Ollama down")):
        with pytest.raises(RoutingError):
            route_question("Something completely ambiguous?")
