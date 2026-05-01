from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Catalog: all extractable fields per document type
# ---------------------------------------------------------------------------

FIELD_CATALOG: dict[str, list[dict[str, Any]]] = {
    "invoice": [
        {"key": "vendor_name",        "label": "Vendor Name",              "db_type": "TEXT", "required": True,  "default": True},
        {"key": "invoice_number",     "label": "Invoice Number",           "db_type": "TEXT", "required": True,  "default": True},
        {"key": "invoice_date",       "label": "Invoice Date",             "db_type": "TEXT", "required": True,  "default": True},
        {"key": "due_date",           "label": "Due Date",                 "db_type": "TEXT", "required": False, "default": True},
        {"key": "bill_to",            "label": "Bill To",                  "db_type": "TEXT", "required": False, "default": True},
        {"key": "po_number",          "label": "PO Number",                "db_type": "TEXT", "required": False, "default": True},
        {"key": "payment_terms",      "label": "Payment Terms",            "db_type": "TEXT", "required": False, "default": True},
        {"key": "subtotal",           "label": "Subtotal",                 "db_type": "REAL", "required": False, "default": True},
        {"key": "tax",                "label": "Tax",                      "db_type": "REAL", "required": False, "default": True},
        {"key": "shipping_handling",  "label": "Shipping & Handling",      "db_type": "REAL", "required": False, "default": True},
        {"key": "total_amount",       "label": "Total Amount",             "db_type": "REAL", "required": True,  "default": True},
        {"key": "currency",           "label": "Currency",                 "db_type": "TEXT", "required": False, "default": True},
        {"key": "line_items",         "label": "Line Items (JSON)",        "db_type": "TEXT", "required": False, "default": True},
    ],
    "medical_discharge": [
        {"key": "patient_name",             "label": "Patient Name",                    "db_type": "TEXT",    "required": True,  "default": True},
        {"key": "date_of_birth",            "label": "Date of Birth",                   "db_type": "TEXT",    "required": False, "default": True},
        {"key": "mrn",                      "label": "Medical Record Number (MRN)",     "db_type": "TEXT",    "required": False, "default": True},
        {"key": "gender",                   "label": "Gender",                          "db_type": "TEXT",    "required": False, "default": True},
        {"key": "facility_name",            "label": "Facility Name",                   "db_type": "TEXT",    "required": False, "default": True},
        {"key": "admission_date",           "label": "Admission Date",                  "db_type": "TEXT",    "required": True,  "default": True},
        {"key": "discharge_date",           "label": "Discharge Date",                  "db_type": "TEXT",    "required": True,  "default": True},
        {"key": "length_of_stay_days",      "label": "Length of Stay (days)",           "db_type": "INTEGER", "required": False, "default": True},
        {"key": "attending_physician",      "label": "Attending Physician",             "db_type": "TEXT",    "required": False, "default": True},
        {"key": "specialty",                "label": "Physician Specialty",             "db_type": "TEXT",    "required": False, "default": False},
        {"key": "insurance",                "label": "Insurance",                       "db_type": "TEXT",    "required": False, "default": False},
        {"key": "primary_diagnosis",        "label": "Primary Diagnosis",               "db_type": "TEXT",    "required": True,  "default": True},
        {"key": "primary_diagnosis_icd10",  "label": "Primary Diagnosis ICD-10 Code",   "db_type": "TEXT",    "required": False, "default": True},
        {"key": "secondary_diagnoses",      "label": "Secondary Diagnoses (JSON)",      "db_type": "TEXT",    "required": False, "default": True},
        {"key": "hospital_course",          "label": "Hospital Course (narrative)",     "db_type": "TEXT",    "required": False, "default": False},
        {"key": "discharge_condition",      "label": "Discharge Condition",             "db_type": "TEXT",    "required": False, "default": True},
        {"key": "discharge_instructions",   "label": "Discharge Instructions",          "db_type": "TEXT",    "required": False, "default": False},
        {"key": "follow_up_date",           "label": "Follow-up Date",                  "db_type": "TEXT",    "required": False, "default": True},
        {"key": "follow_up_provider",       "label": "Follow-up Provider",              "db_type": "TEXT",    "required": False, "default": False},
        {"key": "medications",              "label": "Medications (JSON)",              "db_type": "TEXT",    "required": False, "default": True},
        {"key": "vitals_at_discharge",      "label": "Vitals at Discharge (JSON)",      "db_type": "TEXT",    "required": False, "default": False},
    ],
    "nda": [
        {"key": "disclosing_party",       "label": "Disclosing Party",                   "db_type": "TEXT", "required": True,  "default": True},
        {"key": "receiving_party",        "label": "Receiving Party",                    "db_type": "TEXT", "required": True,  "default": True},
        {"key": "agreement_date",         "label": "Agreement Date",                     "db_type": "TEXT", "required": True,  "default": True},
        {"key": "effective_date",         "label": "Effective Date",                     "db_type": "TEXT", "required": False, "default": True},
        {"key": "expiration_date",        "label": "Expiration Date",                    "db_type": "TEXT", "required": False, "default": True},
        {"key": "agreement_type",         "label": "Agreement Type (mutual / one-way)",  "db_type": "TEXT", "required": False, "default": True},
        {"key": "confidentiality_period", "label": "Confidentiality Period",             "db_type": "TEXT", "required": False, "default": True},
        {"key": "governing_law",          "label": "Governing Law",                      "db_type": "TEXT", "required": False, "default": True},
        {"key": "permitted_use",          "label": "Permitted Use / Purpose",            "db_type": "TEXT", "required": False, "default": True},
        {"key": "signatory_disclosing",   "label": "Signatory — Disclosing Party",       "db_type": "TEXT", "required": False, "default": False},
        {"key": "signatory_receiving",    "label": "Signatory — Receiving Party",        "db_type": "TEXT", "required": False, "default": False},
    ],
    "lab_report": [
        {"key": "patient_name",           "label": "Patient Name",                   "db_type": "TEXT", "required": True,  "default": True},
        {"key": "date_of_birth",          "label": "Date of Birth",                  "db_type": "TEXT", "required": False, "default": True},
        {"key": "mrn",                    "label": "Medical Record Number (MRN)",    "db_type": "TEXT", "required": False, "default": True},
        {"key": "gender",                 "label": "Gender",                         "db_type": "TEXT", "required": False, "default": True},
        {"key": "lab_name",               "label": "Lab / Facility Name",            "db_type": "TEXT", "required": False, "default": True},
        {"key": "clia_number",            "label": "CLIA Number",                    "db_type": "TEXT", "required": False, "default": False},
        {"key": "ordering_physician",     "label": "Ordering Physician",             "db_type": "TEXT", "required": False, "default": True},
        {"key": "ordering_specialty",     "label": "Ordering Specialty",             "db_type": "TEXT", "required": False, "default": False},
        {"key": "accession_number",       "label": "Accession Number",               "db_type": "TEXT", "required": False, "default": True},
        {"key": "specimen_type",          "label": "Specimen Type",                  "db_type": "TEXT", "required": False, "default": True},
        {"key": "collected_date",         "label": "Specimen Collected Date",        "db_type": "TEXT", "required": False, "default": True},
        {"key": "reported_date",          "label": "Report Date",                    "db_type": "TEXT", "required": False, "default": True},
        {"key": "report_id",              "label": "Report ID",                      "db_type": "TEXT", "required": False, "default": True},
        {"key": "reviewing_pathologist",  "label": "Reviewing Pathologist",          "db_type": "TEXT", "required": False, "default": True},
        {"key": "clinical_interpretation","label": "Clinical Interpretation",        "db_type": "TEXT", "required": False, "default": True},
        {"key": "lab_panels",             "label": "All Lab Results (JSON)",         "db_type": "TEXT", "required": False, "default": True},
        {"key": "abnormal_results",       "label": "Abnormal / Flagged Results (JSON)", "db_type": "TEXT", "required": False, "default": True},
    ],
    "business_doc": [
        {"key": "company_name",       "label": "Company Name",               "db_type": "TEXT", "required": True,  "default": True},
        {"key": "document_subtype",   "label": "Document Subtype",           "db_type": "TEXT", "required": False, "default": True},
        {"key": "report_period",      "label": "Reporting Period",           "db_type": "TEXT", "required": False, "default": True},
        {"key": "report_date",        "label": "Report Date",                "db_type": "TEXT", "required": False, "default": True},
        {"key": "report_id",          "label": "Report ID",                  "db_type": "TEXT", "required": False, "default": True},
        {"key": "prepared_by",        "label": "Prepared By",                "db_type": "TEXT", "required": False, "default": True},
        {"key": "approved_by",        "label": "Approved By",                "db_type": "TEXT", "required": False, "default": True},
        {"key": "classification",     "label": "Document Classification",    "db_type": "TEXT", "required": False, "default": True},
        {"key": "executive_summary",  "label": "Executive Summary",          "db_type": "TEXT", "required": False, "default": True},
        {"key": "kpis",               "label": "KPIs (JSON)",                "db_type": "TEXT", "required": False, "default": True},
        {"key": "recommendations",    "label": "Recommendations (JSON)",     "db_type": "TEXT", "required": False, "default": False},
    ],
}

# Maps document_type value → SQLite table name
TABLE_NAMES: dict[str, str] = {
    "invoice":           "invoices",
    "medical_discharge": "discharge_summaries",
    "nda":               "ndas",
    "lab_report":        "lab_reports",
    "business_doc":      "business_docs",
}

# Columns added to every per-type table automatically (not user-selectable)
_SYSTEM_COLUMNS: list[tuple[str, str]] = [
    ("id",                "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("source_file",       "TEXT NOT NULL"),
    ("original_filename", "TEXT"),
    ("content_hash",      "TEXT"),
    ("processed_at",      "TEXT DEFAULT (datetime('now'))"),
]


class SchemaConfig:
    """Loads and persists per-type field selections; generates DDL."""

    def __init__(self, settings_path: Path) -> None:
        self._path = settings_path
        self._selections: dict[str, list[str]] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, list[str]]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {
            doc_type: [f["key"] for f in fields if f["default"]]
            for doc_type, fields in FIELD_CATALOG.items()
        }

    def save(self) -> None:
        self._path.write_text(json.dumps(self._selections, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Field selection
    # ------------------------------------------------------------------

    def get_selected_fields(self, doc_type: str) -> list[str]:
        default = [f["key"] for f in FIELD_CATALOG.get(doc_type, []) if f["default"]]
        return self._selections.get(doc_type, default)

    def set_selected_fields(self, doc_type: str, field_keys: list[str]) -> None:
        # Required fields are always included regardless of user selection.
        required = {f["key"] for f in FIELD_CATALOG.get(doc_type, []) if f["required"]}
        self._selections[doc_type] = sorted(set(field_keys) | required, key=lambda k: _catalog_order(doc_type, k))

    # ------------------------------------------------------------------
    # DDL generation
    # ------------------------------------------------------------------

    def get_ddl(self, doc_type: str) -> str:
        table = TABLE_NAMES.get(doc_type, doc_type)
        selected = set(self.get_selected_fields(doc_type))
        col_defs: list[str] = [f"    {name} {defn}" for name, defn in _SYSTEM_COLUMNS]
        for f in FIELD_CATALOG.get(doc_type, []):
            if f["key"] in selected:
                null_clause = " NOT NULL" if f["required"] else ""
                col_defs.append(f"    {f['key']} {f['db_type']}{null_clause}")
        return f"CREATE TABLE IF NOT EXISTS {table} (\n" + ",\n".join(col_defs) + "\n);"


def get_required_fields(doc_type: str) -> frozenset[str]:
    """Return the set of required field keys for a document type per FIELD_CATALOG."""
    return frozenset(f["key"] for f in FIELD_CATALOG.get(doc_type, []) if f["required"])


def _catalog_order(doc_type: str, key: str) -> int:
    for i, f in enumerate(FIELD_CATALOG.get(doc_type, [])):
        if f["key"] == key:
            return i
    return 9999
