"""
tests/test_structured_store.py

Tests for structured_store.py — DB init, insert, get, list, count, all_form_ids.
Uses the in-memory :memory: fixture from conftest.py.
No LLM / Ollama required.
"""
import pytest


# ── init_db ───────────────────────────────────────────────────────────────────

def test_init_db_creates_tables(mem_conn):
    import structured_store
    # forms_index must exist
    tables = mem_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {r["name"] for r in tables}
    assert "forms_index" in names
    assert "form_membership" in names
    assert "form_hospital" in names


def test_init_db_idempotent(mem_conn):
    """Calling init_db twice must not raise."""
    import structured_store
    structured_store.init_db(mem_conn)  # second call
    # If no exception, we're good


# ── insert_form / get_form ────────────────────────────────────────────────────

def test_insert_and_get_membership(mem_conn, extracted_membership):
    import structured_store
    structured_store.insert_form(mem_conn, extracted_membership)
    result = structured_store.get_form(mem_conn, "membership_001")
    assert result is not None
    assert result["name"] == "Ananya Rao"
    assert result["status"] == "Approved"
    assert result["form_type"] == "membership"


def test_insert_and_get_hospital(mem_conn, extracted_hospital):
    import structured_store
    structured_store.insert_form(mem_conn, extracted_hospital)
    result = structured_store.get_form(mem_conn, "hospital_001")
    assert result is not None
    assert result["patient_name"] == "Rahul Menon"
    assert result["department"] == "Cardiology"
    assert result["form_type"] == "hospital"


def test_get_form_nonexistent_returns_none(mem_conn):
    import structured_store
    assert structured_store.get_form(mem_conn, "does_not_exist") is None


def test_insert_overwrite(mem_conn, extracted_membership):
    """Re-inserting the same form_id should replace the record."""
    import structured_store
    from extraction import ExtractedForm
    from schemas import MembershipForm

    structured_store.insert_form(mem_conn, extracted_membership)

    updated = ExtractedForm(
        form_id="membership_001",
        form_type="membership",
        data=MembershipForm(
            name="Ananya Rao Updated",
            date_of_birth="14-03-1994",
            email="ananya.rao@example.com",
            phone="+91-9876543210",
            occupation="Manager",
            application_date="02-01-2026",
            status="Rejected",
            remarks="Updated remarks.",
        ),
        detection_method="keyword",
        extraction_attempts=1,
    )
    structured_store.insert_form(mem_conn, updated)

    result = structured_store.get_form(mem_conn, "membership_001")
    assert result["name"] == "Ananya Rao Updated"
    assert result["status"] == "Rejected"


# ── count_forms / list_forms ──────────────────────────────────────────────────

def test_count_forms_total(mem_conn, extracted_membership, extracted_hospital):
    import structured_store
    structured_store.insert_form(mem_conn, extracted_membership)
    # Insert a second membership
    from extraction import ExtractedForm
    from schemas import MembershipForm
    second = ExtractedForm(
        form_id="membership_002",
        form_type="membership",
        data=MembershipForm(
            name="Vikram Nair", date_of_birth="22-07-1988",
            email="v@example.com", phone="+91-9000000002",
            occupation="Designer", application_date="15-01-2026",
            status="Rejected", remarks="Two prior credit defaults.",
        ),
        detection_method="keyword", extraction_attempts=1,
    )
    structured_store.insert_form(mem_conn, second)
    assert structured_store.count_forms(mem_conn, "membership") == 2


def test_count_forms_with_filter(mem_conn, extracted_membership):
    import structured_store
    structured_store.insert_form(mem_conn, extracted_membership)
    assert structured_store.count_forms(mem_conn, "membership", {"status": "Approved"}) == 1
    assert structured_store.count_forms(mem_conn, "membership", {"status": "Rejected"}) == 0


def test_list_forms_returns_all(mem_conn, extracted_membership, extracted_hospital):
    import structured_store
    structured_store.insert_form(mem_conn, extracted_membership)
    rows = structured_store.list_forms(mem_conn, "membership")
    assert len(rows) == 1
    assert rows[0]["form_id"] == "membership_001"


def test_list_forms_with_filter(mem_conn, extracted_membership):
    import structured_store
    structured_store.insert_form(mem_conn, extracted_membership)
    approved = structured_store.list_forms(mem_conn, "membership", {"status": "Approved"})
    rejected = structured_store.list_forms(mem_conn, "membership", {"status": "Rejected"})
    assert len(approved) == 1
    assert len(rejected) == 0


# ── all_form_ids ──────────────────────────────────────────────────────────────

def test_all_form_ids_empty(mem_conn):
    import structured_store
    assert structured_store.all_form_ids(mem_conn) == []


def test_all_form_ids_all_types(mem_conn, extracted_membership, extracted_hospital):
    import structured_store
    structured_store.insert_form(mem_conn, extracted_membership)
    structured_store.insert_form(mem_conn, extracted_hospital)
    ids = structured_store.all_form_ids(mem_conn)
    assert "membership_001" in ids
    assert "hospital_001" in ids


def test_all_form_ids_filtered_by_type(mem_conn, extracted_membership, extracted_hospital):
    import structured_store
    structured_store.insert_form(mem_conn, extracted_membership)
    structured_store.insert_form(mem_conn, extracted_hospital)
    mem_ids = structured_store.all_form_ids(mem_conn, "membership")
    hosp_ids = structured_store.all_form_ids(mem_conn, "hospital")
    assert mem_ids == ["membership_001"]
    assert hosp_ids == ["hospital_001"]


# ── get_form_type ─────────────────────────────────────────────────────────────

def test_get_form_type(mem_conn, extracted_membership):
    import structured_store
    structured_store.insert_form(mem_conn, extracted_membership)
    assert structured_store.get_form_type(mem_conn, "membership_001") == "membership"
    assert structured_store.get_form_type(mem_conn, "nonexistent") is None
