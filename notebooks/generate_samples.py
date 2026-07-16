"""
One-off generator for mock test forms.
Not part of the shipped pipeline — just creates realistic PDFs so
ingestion.py (and later stages) have something real to chew on.
Run once: python data/sample_forms/_generate_samples.py
"""
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
# pyrefly: ignore [missing-import]
from PIL import Image, ImageDraw
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def make_text_pdf(path, lines):
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    y = height - 72
    for line in lines:
        c.drawString(72, y, line)
        y -= 18
    c.save()


# --- Membership form 1 (text-based) ---
make_text_pdf(
    os.path.join(HERE, "membership_001.pdf"),
    [
        "MEMBERSHIP APPLICATION FORM",
        "",
        "Name: Ananya Rao",
        "Date of Birth: 14-03-1994",
        "Email: ananya.rao@example.com",
        "Phone: +91-9876543210",
        "Occupation: Software Engineer",
        "Application Date: 02-01-2026",
        "Status: Approved",
        "Remarks: Applicant has a strong credit history and was approved after",
        "standard verification. No prior defaults on record. Recommended for",
        "the premium membership tier due to consistent income documentation.",
    ],
)

# --- Membership form 2 (text-based, different status) ---
make_text_pdf(
    os.path.join(HERE, "membership_002.pdf"),
    [
        "MEMBERSHIP APPLICATION FORM",
        "",
        "Name: Vikram Nair",
        "Date of Birth: 22-07-1988",
        "Email: vikram.nair@example.com",
        "Phone: +91-9812345678",
        "Occupation: Freelance Designer",
        "Application Date: 15-01-2026",
        "Status: Rejected",
        "Remarks: Application rejected due to two prior credit defaults flagged",
        "during verification. Applicant may reapply after 12 months with updated",
        "financial documentation.",
    ],
)

# --- Hospital form (text-based) ---
make_text_pdf(
    os.path.join(HERE, "hospital_001.pdf"),
    [
        "HOSPITAL ADMISSION FORM",
        "",
        "Patient Name: Rahul Menon",
        "Patient ID: H-2026-0091",
        "Date of Birth: 09-11-1975",
        "Doctor: Dr. Lakshmi Iyer",
        "Department: Cardiology",
        "Admission Date: 10-02-2026",
        "Discharge Date: 15-02-2026",
        "Diagnosis: Acute myocardial infarction",
        "Discharge Status: Stable, discharged with medication",
        "Doctor's Notes: Patient underwent angioplasty on the second day of",
        "admission. Recovery was uneventful. Advised to follow a low-sodium diet",
        "and attend cardiac rehab sessions twice weekly for the next two months.",
    ],
)

# --- Hospital form 2, rendered as an IMAGE-ONLY pdf to simulate a scan ---
img = Image.new("RGB", (850, 1100), "white")
d = ImageDraw.Draw(img)
lines = [
    "HOSPITAL ADMISSION FORM",
    "",
    "Patient Name: Fathima Sheikh",
    "Patient ID: H-2026-0114",
    "Date of Birth: 03-05-1990",
    "Doctor: Dr. Arjun Kapoor",
    "Department: Orthopedics",
    "Admission Date: 20-02-2026",
    "Discharge Date: 24-02-2026",
    "Diagnosis: Fractured tibia (left leg)",
    "Discharge Status: Stable, discharged with cast",
    "Doctor's Notes: Surgical fixation performed. No signs of infection.",
    "Follow-up X-ray scheduled in 4 weeks. Patient advised to avoid weight",
    "bearing on the left leg until cleared by physiotherapy.",
]
y = 60
for line in lines:
    d.text((60, y), line, fill="black")
    y += 30
img.save(os.path.join(HERE, "_scan_page.png"))

img_path = os.path.join(HERE, "_scan_page.png")
scan_pdf = os.path.join(HERE, "hospital_002_scanned.pdf")
im = Image.open(img_path).convert("RGB")
im.save(scan_pdf)
os.remove(img_path)

print("Generated sample forms in", HERE)
for f in sorted(os.listdir(HERE)):
    print(" -", f)