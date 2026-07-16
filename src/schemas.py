"""
src/schemas.py

Pydantic schemas for each known form template (see APPROACH.md §3.2).

extraction.py uses these to:
  1. show the LLM the target JSON shape (via `.model_json_schema()` or a
     hand-written example — see extraction.py),
  2. validate the LLM's JSON output, raising a ValidationError that gets
     fed back into a retry prompt on failure.

Adding a new form template later = add a new BaseModel here + register it
in FORM_SCHEMAS. Nothing else in the pipeline needs to change (this is the
"generalizes to new templates" creativity point in APPROACH.md §8).
"""
from __future__ import annotations
from typing import Literal
from pydantic import EmailStr, BaseModel, Field


class MembershipForm(BaseModel):
    name: str
    date_of_birth: str = Field(description="DD-MM-YYYY as written on the form")
    email: EmailStr
    phone: str = Field(min_length=10)
    occupation: str
    application_date: str = Field(description="DD-MM-YYYY as written on the form")
    status: Literal["Approved", "Pending", "Rejected"]
    remarks: str = Field(description="Free-text remarks/notes field, verbatim")


class HospitalForm(BaseModel):
    patient_name: str
    patient_id: str
    date_of_birth: str = Field(description="DD-MM-YYYY as written on the form")
    doctor: str
    department: str
    admission_date: str = Field(description="DD-MM-YYYY as written on the form")
    discharge_date: str | None = None
    diagnosis: str
    discharge_status: str
    doctors_notes: str = Field(description="Free-text clinical notes field, verbatim")


# Registry: form_type label -> schema class.
# Keys here are the exact strings the form-type-detection step in
# extraction.py must produce (keyword heuristic or LLM classification).
FORM_SCHEMAS: dict[str, type[BaseModel]] = {
    "membership": MembershipForm,
    "hospital": HospitalForm,
}

# The free-text field per form type that gets chunked + embedded downstream
# (chunking.py / vectorstore.py) rather than treated as an exact-match field.
FREE_TEXT_FIELD: dict[str, str] = {
    "membership": "remarks",
    "hospital": "doctors_notes",
}


def schema_example(form_type: str) -> dict:
    """
    A hand-written example instance per schema, used in extraction.py's
    prompt to show the LLM the exact expected JSON shape (more reliable
    for small local models than a raw JSON-schema dump).
    """
    examples = {
        "membership": {
            "name": "Ananya Rao",
            "date_of_birth": "14-03-1994",
            "email": "ananya.rao@example.com",
            "phone": "+91-9876543210",
            "occupation": "Software Engineer",
            "application_date": "02-01-2026",
            "status": "Approved",
            "remarks": "Applicant has a strong credit history...",
        },
        "hospital": {
            "patient_name": "Rahul Menon",
            "patient_id": "H-2026-0091",
            "date_of_birth": "09-11-1975",
            "doctor": "Dr. Lakshmi Iyer",
            "department": "Cardiology",
            "admission_date": "10-02-2026",
            "discharge_date": "15-02-2026",
            "diagnosis": "Acute myocardial infarction",
            "discharge_status": "Stable, discharged with medication",
            "doctors_notes": "Patient underwent angioplasty...",
        },
    }
    if form_type not in examples:
        raise KeyError(f"No example for form_type={form_type!r}. Known: {list(examples)}")
    return examples[form_type]