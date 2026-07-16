"""
src/structured_store.py

Stage 3b of the pipeline: SQLite storage for the exact structured fields
of each ExtractedForm (see APPROACH.md §3.4).

This is what makes "how many applicants are approved?" and similar
aggregate/filter questions EXACT rather than an LLM guess — router.py
routes those questions here instead of to the vector store (§3.5), and
retrieval.py builds a SQL WHERE clause from recognized fields (§3.6).

Design: one table PER form type (wide format), since the set of form
types is small and known in advance from schemas.FORM_SCHEMAS — this is
simpler to query than a long-format single table and needs no JOINs for
the common case. A separate `forms_index` table maps form_id -> form_type
so callers can look up a form without knowing its type ahead of time.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from extraction import ExtractedForm
from schemas import FORM_SCHEMAS, FREE_TEXT_FIELD

DEFAULT_DB_PATH = "forms.sqlite"


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    # check_same_thread=False allows the cached connection to be used across
    # Streamlit's worker threads (safe because Streamlit serialises UI reruns).
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _table_name(form_type: str) -> str:
    return f"form_{form_type}"


def init_db(conn: sqlite3.Connection) -> None:
    """
    Creates forms_index plus one table per registered form type, with a
    column per Pydantic field. All columns are TEXT — every field in the
    current schemas (schemas.py) is string-based (dates are stored as the
    DD-MM-YYYY strings as written on the form; convert at query time if
    date arithmetic is ever needed). form_id is the primary key in both
    forms_index and each per-type table.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS forms_index (
            form_id TEXT PRIMARY KEY,
            form_type TEXT NOT NULL
        )
        """
    )

    for form_type, schema_cls in FORM_SCHEMAS.items():
        table = _table_name(form_type)
        columns = ["form_id TEXT PRIMARY KEY"]
        for name in schema_cls.model_fields:
            if name == "form_id":
                continue
            columns.append(f"{name} TEXT")
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(columns)})")

    conn.commit()


def insert_form(conn: sqlite3.Connection, extracted: ExtractedForm, overwrite: bool = True) -> None:
    """Inserts (or replaces, if overwrite=True) one ExtractedForm's fields."""
    form_type = extracted.form_type
    if form_type not in FORM_SCHEMAS:
        raise KeyError(f"No table registered for form_type={form_type!r}. Call init_db() first.")

    table = _table_name(form_type)
    data = extracted.data.model_dump(mode="json")
    data["form_id"] = extracted.form_id

    columns = list(data.keys())
    placeholders = ", ".join("?" for _ in columns)
    verb = "INSERT OR REPLACE" if overwrite else "INSERT"

    conn.execute(
        f"{verb} INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        [data[c] for c in columns],
    )
    conn.execute(
        "INSERT OR REPLACE INTO forms_index (form_id, form_type) VALUES (?, ?)",
        (extracted.form_id, form_type),
    )
    conn.commit()


def get_form_type(conn: sqlite3.Connection, form_id: str) -> str | None:
    row = conn.execute(
        "SELECT form_type FROM forms_index WHERE form_id = ?", (form_id,)
    ).fetchone()
    return row["form_type"] if row else None


def get_form(conn: sqlite3.Connection, form_id: str) -> dict | None:
    """Fetches all structured fields for one form_id, looking up its type first."""
    form_type = get_form_type(conn, form_id)
    if form_type is None:
        return None
    table = _table_name(form_type)
    row = conn.execute(f"SELECT * FROM {table} WHERE form_id = ?", (form_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["form_type"] = form_type
    return result


def _build_where(filters: dict | None) -> tuple[str, list]:
    if not filters:
        return "", []
    clauses = [f"{col} = ?" for col in filters]
    return " WHERE " + " AND ".join(clauses), list(filters.values())


def list_forms(
    conn: sqlite3.Connection,
    form_type: str,
    filters: dict | None = None,
) -> list[dict]:
    """
    Exact filtered listing, e.g. list_forms(conn, "membership", {"status": "Rejected"}).
    Excludes the free-text field from `filters` keys silently isn't enforced
    here — filtering ON the free-text field by exact match is legal SQL,
    just usually not what you want (use retrieval.py's semantic search
    instead for that).
    """
    if form_type not in FORM_SCHEMAS:
        raise KeyError(f"Unknown form_type={form_type!r}")
    table = _table_name(form_type)
    where_sql, params = _build_where(filters)
    rows = conn.execute(f"SELECT * FROM {table}{where_sql}", params).fetchall()
    return [dict(r) for r in rows]


def count_forms(
    conn: sqlite3.Connection,
    form_type: str,
    filters: dict | None = None,
) -> int:
    """Exact count — this is the function router.py's `aggregate` path calls
    for "how many X are Y?"-style questions, instead of asking the LLM to count."""
    if form_type not in FORM_SCHEMAS:
        raise KeyError(f"Unknown form_type={form_type!r}")
    table = _table_name(form_type)
    where_sql, params = _build_where(filters)
    row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}{where_sql}", params).fetchone()
    return row["n"]


def all_form_ids(conn: sqlite3.Connection, form_type: str | None = None) -> list[str]:
    if form_type is None:
        rows = conn.execute("SELECT form_id FROM forms_index").fetchall()
    else:
        rows = conn.execute(
            "SELECT form_id FROM forms_index WHERE form_type = ?", (form_type,)
        ).fetchall()
    return [r["form_id"] for r in rows]


if __name__ == "__main__":
    # Manual smoke test: build the DB in-memory, insert two fake
    # ExtractedForms, and run an exact count — no LLM/Ollama needed.
    from schemas import MembershipForm, HospitalForm

    conn = connect(":memory:")
    init_db(conn)

    forms = [
        ExtractedForm(
            form_id="membership_001",
            form_type="membership",
            data=MembershipForm(
                name="Ananya Rao", date_of_birth="14-03-1994",
                email="a@example.com", phone="+91-9000000001", occupation="Engineer",
                application_date="02-01-2026", status="Approved",
                remarks="Strong credit history.",
            ),
            detection_method="keyword", extraction_attempts=1,
        ),
        ExtractedForm(
            form_id="membership_002",
            form_type="membership",
            data=MembershipForm(
                name="Vikram Nair", date_of_birth="22-07-1988",
                email="v@example.com", phone="+91-9000000002", occupation="Designer",
                application_date="15-01-2026", status="Rejected",
                remarks="Two prior credit defaults.",
            ),
            detection_method="keyword", extraction_attempts=1,
        ),
    ]
    for f in forms:
        insert_form(conn, f)

    print("Total membership forms:", count_forms(conn, "membership"))
    print("Approved:", count_forms(conn, "membership", {"status": "Approved"}))
    print("Rejected:", count_forms(conn, "membership", {"status": "Rejected"}))
    print("get_form('membership_002'):", get_form(conn, "membership_002"))
    print("all_form_ids():", all_form_ids(conn))