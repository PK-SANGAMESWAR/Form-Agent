"""
tests/test_schemas.py

Tests for schemas.py — Pydantic validation, FREE_TEXT_FIELD registry,
FORM_SCHEMAS registry, and schema_example().
No LLM / Ollama required.
"""
import pytest
from pydantic import ValidationError


# ── MembershipForm validation ─────────────────────────────────────────────────

def test_membership_form_valid(membership_form):
    assert membership_form.name == "Ananya Rao"
    assert membership_form.status == "Approved"
    assert "@" in membership_form.email


def test_membership_form_invalid_status():
    from schemas import MembershipForm
    with pytest.raises(ValidationError):
        MembershipForm(
            name="Test", date_of_birth="01-01-2000",
            email="t@example.com", phone="+911234567890",
            occupation="X", application_date="01-01-2026",
            status="Unknown",   # not in Literal["Approved","Pending","Rejected"]
            remarks="",
        )


def test_membership_form_invalid_email():
    from schemas import MembershipForm
    with pytest.raises(ValidationError):
        MembershipForm(
            name="Test", date_of_birth="01-01-2000",
            email="not-an-email", phone="+911234567890",
            occupation="X", application_date="01-01-2026",
            status="Pending", remarks="",
        )


def test_membership_form_short_phone():
    from schemas import MembershipForm
    with pytest.raises(ValidationError):
        MembershipForm(
            name="Test", date_of_birth="01-01-2000",
            email="t@example.com", phone="123",   # min_length=10
            occupation="X", application_date="01-01-2026",
            status="Pending", remarks="",
        )


# ── HospitalForm validation ───────────────────────────────────────────────────

def test_hospital_form_valid(hospital_form):
    assert hospital_form.patient_name == "Rahul Menon"
    assert hospital_form.department == "Cardiology"
    assert hospital_form.discharge_date == "15-02-2026"


def test_hospital_form_nullable_discharge_date():
    """discharge_date is optional (ongoing admission)."""
    from schemas import HospitalForm
    form = HospitalForm(
        patient_name="Test", patient_id="H-001",
        date_of_birth="01-01-1990", doctor="Dr. X",
        department="General", admission_date="01-01-2026",
        discharge_date=None,   # still valid
        diagnosis="Fever", discharge_status="Ongoing",
        doctors_notes="Under observation.",
    )
    assert form.discharge_date is None


# ── Registry completeness ─────────────────────────────────────────────────────

def test_form_schemas_keys():
    from schemas import FORM_SCHEMAS
    assert "membership" in FORM_SCHEMAS
    assert "hospital" in FORM_SCHEMAS


def test_free_text_field_keys():
    from schemas import FREE_TEXT_FIELD, FORM_SCHEMAS
    # Every registered form type must have a free-text field mapped
    for ftype in FORM_SCHEMAS:
        assert ftype in FREE_TEXT_FIELD, f"No FREE_TEXT_FIELD entry for {ftype!r}"


def test_free_text_field_is_real_field():
    """The mapped field must actually exist on the schema."""
    from schemas import FREE_TEXT_FIELD, FORM_SCHEMAS
    for ftype, field_name in FREE_TEXT_FIELD.items():
        schema_cls = FORM_SCHEMAS[ftype]
        assert field_name in schema_cls.model_fields, (
            f"FREE_TEXT_FIELD[{ftype!r}]={field_name!r} not found in {schema_cls.__name__}"
        )


def test_schema_example_membership():
    from schemas import schema_example
    ex = schema_example("membership")
    assert "name" in ex and "status" in ex and "remarks" in ex


def test_schema_example_hospital():
    from schemas import schema_example
    ex = schema_example("hospital")
    assert "patient_name" in ex and "doctors_notes" in ex


def test_schema_example_unknown_raises():
    from schemas import schema_example
    with pytest.raises(KeyError):
        schema_example("unknown_type")
