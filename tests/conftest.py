"""
tests/conftest.py

Shared pytest fixtures for the Form-Agent test suite.
Adds src/ to sys.path so all modules are importable without installation.
"""
import sys
import sqlite3
from pathlib import Path

import pytest

# Make src/ importable in every test module
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── Shared in-memory fixtures ────────────────────────────────────────────────

@pytest.fixture
def mem_conn():
    """SQLite connection using :memory: — fast, isolated, auto-cleaned."""
    import structured_store
    conn = structured_store.connect(":memory:")
    structured_store.init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def membership_form():
    """A fully populated MembershipForm for reuse across tests."""
    from schemas import MembershipForm
    return MembershipForm(
        name="Ananya Rao",
        date_of_birth="14-03-1994",
        email="ananya.rao@example.com",
        phone="+91-9876543210",
        occupation="Software Engineer",
        application_date="02-01-2026",
        status="Approved",
        remarks="Applicant has a strong credit history and no prior defaults on record.",
    )


@pytest.fixture
def hospital_form():
    """A fully populated HospitalForm for reuse across tests."""
    from schemas import HospitalForm
    return HospitalForm(
        patient_name="Rahul Menon",
        patient_id="H-2026-0091",
        date_of_birth="09-11-1975",
        doctor="Dr. Lakshmi Iyer",
        department="Cardiology",
        admission_date="10-02-2026",
        discharge_date="15-02-2026",
        diagnosis="Acute myocardial infarction",
        discharge_status="Stable, discharged with medication",
        doctors_notes="Patient underwent angioplasty on the second day. Follow-up in 4 weeks.",
    )


@pytest.fixture
def extracted_membership(membership_form):
    from extraction import ExtractedForm
    return ExtractedForm(
        form_id="membership_001",
        form_type="membership",
        data=membership_form,
        detection_method="keyword",
        extraction_attempts=1,
    )


@pytest.fixture
def extracted_hospital(hospital_form):
    from extraction import ExtractedForm
    return ExtractedForm(
        form_id="hospital_001",
        form_type="hospital",
        data=hospital_form,
        detection_method="keyword",
        extraction_attempts=1,
    )
