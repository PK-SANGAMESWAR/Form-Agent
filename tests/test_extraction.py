"""
tests/test_extraction.py

Tests for extraction.py — form-type detection (keyword fast-path) and
schema-guided extraction (LLM call mocked so no Ollama needed).
"""
import pytest
import json
from unittest.mock import patch, MagicMock


# ── detect_form_type — keyword path ──────────────────────────────────────────

def test_detect_membership_keyword():
    from extraction import detect_form_type
    text = "MEMBERSHIP APPLICATION FORM\nApplicant Name: Ananya Rao"
    form_type, method = detect_form_type(text)
    assert form_type == "membership"
    assert method == "keyword"


def test_detect_hospital_keyword():
    from extraction import detect_form_type
    text = "HOSPITAL ADMISSION FORM\nPatient: Rahul Menon"
    form_type, method = detect_form_type(text)
    assert form_type == "hospital"
    assert method == "keyword"


def test_detect_case_insensitive():
    from extraction import detect_form_type
    text = "Membership Application form\nname: test"
    form_type, method = detect_form_type(text)
    assert form_type == "membership"


def test_detect_fallback_llm_membership():
    """When keywords miss, the LLM classification fallback is used."""
    from extraction import detect_form_type
    mock_resp = MagicMock()
    mock_resp.text = "membership"
    with patch("extraction.llm_client.generate", return_value=mock_resp):
        form_type, method = detect_form_type("Some form without a clear header")
    assert form_type == "membership"
    assert method == "llm"


def test_detect_fallback_unknown_raises():
    from extraction import detect_form_type, ExtractionError
    mock_resp = MagicMock()
    mock_resp.text = "banana"
    with patch("extraction.llm_client.generate", return_value=mock_resp):
        with pytest.raises(ExtractionError):
            detect_form_type("Totally ambiguous text")


# ── extract_fields — mocked LLM ───────────────────────────────────────────────

def _mock_llm_json(data: dict):
    """Returns a mock llm_client.generate_json that yields data once."""
    return MagicMock(return_value=data)


VALID_MEMBERSHIP_JSON = {
    "name": "Ananya Rao",
    "date_of_birth": "14-03-1994",
    "email": "ananya.rao@example.com",
    "phone": "+91-9876543210",
    "occupation": "Software Engineer",
    "application_date": "02-01-2026",
    "status": "Approved",
    "remarks": "Strong credit history.",
}

VALID_HOSPITAL_JSON = {
    "patient_name": "Rahul Menon",
    "patient_id": "H-2026-0091",
    "date_of_birth": "09-11-1975",
    "doctor": "Dr. Lakshmi Iyer",
    "department": "Cardiology",
    "admission_date": "10-02-2026",
    "discharge_date": "15-02-2026",
    "diagnosis": "Acute myocardial infarction",
    "discharge_status": "Stable, discharged with medication",
    "doctors_notes": "Angioplasty performed. Follow-up in 4 weeks.",
}


def test_extract_fields_membership():
    from extraction import extract_fields
    from schemas import MembershipForm
    with patch("extraction.llm_client.generate_json", _mock_llm_json(VALID_MEMBERSHIP_JSON)):
        model, attempts = extract_fields("membership", "some raw text")
    assert isinstance(model, MembershipForm)
    assert model.name == "Ananya Rao"
    assert model.status == "Approved"
    assert attempts == 1


def test_extract_fields_hospital():
    from extraction import extract_fields
    from schemas import HospitalForm
    with patch("extraction.llm_client.generate_json", _mock_llm_json(VALID_HOSPITAL_JSON)):
        model, attempts = extract_fields("hospital", "some raw text")
    assert isinstance(model, HospitalForm)
    assert model.patient_name == "Rahul Menon"
    assert model.department == "Cardiology"


def test_extract_fields_unknown_type_raises():
    from extraction import extract_fields, ExtractionError
    with pytest.raises(ExtractionError, match="No schema registered"):
        extract_fields("unknown_type", "some text")


def test_extract_fields_retry_on_validation_error():
    """First LLM call returns invalid JSON, second call returns valid — should succeed in 2 attempts."""
    from extraction import extract_fields

    invalid_json = {"name": "X"}  # missing required fields → ValidationError
    call_count = {"n": 0}

    def side_effect(prompt):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return invalid_json
        return VALID_MEMBERSHIP_JSON

    with patch("extraction.llm_client.generate_json", side_effect=side_effect):
        model, attempts = extract_fields("membership", "raw text")
    assert attempts == 2
    assert model.name == "Ananya Rao"


def test_extract_fields_fails_after_all_retries():
    """If all retries return invalid JSON, ExtractionError is raised."""
    from extraction import extract_fields, ExtractionError

    bad_json = {"name": "X"}  # always invalid
    with patch("extraction.llm_client.generate_json", return_value=bad_json):
        with pytest.raises(ExtractionError, match="failed validation"):
            extract_fields("membership", "raw text")


# ── extract_form (full pipeline, mocked) ─────────────────────────────────────

def test_extract_form_membership_end_to_end():
    from extraction import extract_form
    from ingestion import RawForm

    raw = RawForm(
        form_id="membership_test",
        source_path="/fake/path.pdf",
        text="MEMBERSHIP APPLICATION FORM\nName: Ananya Rao",
    )
    with patch("extraction.llm_client.generate_json", _mock_llm_json(VALID_MEMBERSHIP_JSON)):
        result = extract_form(raw)

    assert result.form_id == "membership_test"
    assert result.form_type == "membership"
    assert result.detection_method == "keyword"
    assert result.data.name == "Ananya Rao"
