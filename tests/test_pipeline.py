"""
Pipeline integration tests.

Drop real invoice PDFs into tests/fixtures/ and they will be picked up
automatically by the parametrized tests below.  The suite also covers
core behaviour with synthetic text so it can run without any fixtures.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(tmp_path):
    """Return a DocumentPipeline wired to a throwaway data directory."""
    from src.doc_ai.config import get_settings
    from src.doc_ai.pipeline import DocumentPipeline

    get_settings.cache_clear()
    os.environ["APP_ENV"] = "test"
    os.environ["APP_DATA_ROOT"] = str(tmp_path)
    get_settings.cache_clear()
    settings = get_settings()
    return DocumentPipeline(settings)


def _txt_bytes(text: str) -> bytes:
    return text.encode()


SAMPLE_INVOICE_TEXT = """\
Orion Data Systems
Invoice Number: INV-56528
Invoice Date: 2026-03-13
Due Date: 2026-04-12
Bill To: BrightPath Marketing
Item Qty Unit Price Total
Managed Network Switch 4 $1,250.00 $5,000.00
Data Backup Appliance 4 $1,800.00 $7,200.00
Office Chair Ergonomic 2 $295.00 $590.00
Data Backup Appliance 4 $1,800.00 $7,200.00
Subtotal $19,990.00
Tax $1,649.18
Total $21,639.18
Payment Terms: Net 30.
"""

SAMPLE_WITH_SHIPPING = """\
Acme Supplies
Invoice Number: INV-99001
Invoice Date: 2026-01-15
Due Date: 2026-02-14
Subtotal $500.00
Tax $45.00
Shipping & Handling $25.00
Total $570.00
"""


# ---------------------------------------------------------------------------
# Extraction — rule-based
# ---------------------------------------------------------------------------

class TestRuleBasedExtraction:
    def test_standard_fields(self):
        from src.doc_ai.extractors import RuleBasedInvoiceExtractor
        from src.doc_ai.schemas import ParsedDocument

        doc = ParsedDocument(
            file_name="test.txt",
            file_path=Path("test.txt"),
            raw_text=SAMPLE_INVOICE_TEXT,
            sections=SAMPLE_INVOICE_TEXT.splitlines(),
        )
        result = RuleBasedInvoiceExtractor().extract(doc)

        assert result["invoice_number"] == "INV-56528"
        assert result["invoice_date"] == "2026-03-13"
        assert result["due_date"] == "2026-04-12"
        assert result["total_amount"] == pytest.approx(21639.18)
        assert result["tax"] == pytest.approx(1649.18)
        assert result["subtotal"] == pytest.approx(19990.0)

    def test_line_items_extracted(self):
        from src.doc_ai.extractors import RuleBasedInvoiceExtractor
        from src.doc_ai.schemas import ParsedDocument

        doc = ParsedDocument(
            file_name="test.txt",
            file_path=Path("test.txt"),
            raw_text=SAMPLE_INVOICE_TEXT,
            sections=SAMPLE_INVOICE_TEXT.splitlines(),
        )
        result = RuleBasedInvoiceExtractor().extract(doc)
        items = result.get("line_items", [])

        assert len(items) >= 3
        descriptions = [i["description"] for i in items]
        assert any("Network Switch" in d for d in descriptions)
        assert any("Office Chair" in d for d in descriptions)
        for item in items:
            assert item["quantity"] > 0
            assert item["unit_price"] > 0
            assert item["total"] == pytest.approx(item["quantity"] * item["unit_price"])

    def test_shipping_handling_extracted(self):
        from src.doc_ai.extractors import RuleBasedInvoiceExtractor
        from src.doc_ai.schemas import ParsedDocument

        doc = ParsedDocument(
            file_name="shipping.txt",
            file_path=Path("shipping.txt"),
            raw_text=SAMPLE_WITH_SHIPPING,
            sections=SAMPLE_WITH_SHIPPING.splitlines(),
        )
        result = RuleBasedInvoiceExtractor().extract(doc)

        assert result["shipping_handling"] == pytest.approx(25.0)

    def test_missing_fields_return_none(self):
        from src.doc_ai.extractors import RuleBasedInvoiceExtractor
        from src.doc_ai.schemas import ParsedDocument

        doc = ParsedDocument(
            file_name="empty.txt",
            file_path=Path("empty.txt"),
            raw_text="Nothing useful here.",
            sections=["Nothing useful here."],
        )
        result = RuleBasedInvoiceExtractor().extract(doc)

        assert result["vendor_name"] is None
        assert result["invoice_number"] is None
        assert result["line_items"] == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_pass_when_totals_match(self):
        from src.doc_ai.validators import InvoiceValidator

        data = {
            "vendor_name": "Acme",
            "invoice_number": "INV-001",
            "invoice_date": "2026-01-01",
            "due_date": "2026-02-01",
            "subtotal": 100.0,
            "tax": 10.0,
            "shipping_handling": None,
            "total_amount": 110.0,
            "currency": "USD",
        }
        checks = InvoiceValidator().validate(data)
        statuses = {c.field: c.status for c in checks}
        assert statuses["total_consistency"] == "pass"

    def test_fail_when_totals_mismatch(self):
        from src.doc_ai.validators import InvoiceValidator

        data = {
            "vendor_name": "Acme",
            "invoice_number": "INV-001",
            "invoice_date": "2026-01-01",
            "total_amount": 999.0,
            "subtotal": 100.0,
            "tax": 10.0,
            "shipping_handling": None,
        }
        checks = InvoiceValidator().validate(data)
        statuses = {c.field: c.status for c in checks}
        assert statuses["total_consistency"] == "fail"

    def test_shipping_included_in_total_check(self):
        from src.doc_ai.validators import InvoiceValidator

        data = {
            "vendor_name": "Acme",
            "invoice_number": "INV-001",
            "invoice_date": "2026-01-01",
            "subtotal": 500.0,
            "tax": 45.0,
            "shipping_handling": 25.0,
            "total_amount": 570.0,
        }
        checks = InvoiceValidator().validate(data)
        statuses = {c.field: c.status for c in checks}
        assert statuses["total_consistency"] == "pass"

    def test_missing_required_fields_fail(self):
        from src.doc_ai.validators import InvoiceValidator

        checks = InvoiceValidator().validate({})
        fail_fields = {c.field for c in checks if c.status == "fail"}
        assert {"vendor_name", "invoice_number", "invoice_date", "total_amount"}.issubset(fail_fields)

    def test_all_invoice_fields_have_validation_row(self):
        from src.doc_ai.validators import InvoiceValidator

        data = {
            "vendor_name": "Acme",
            "invoice_number": "INV-001",
            "invoice_date": "2026-01-01",
            "due_date": "2026-02-01",
            "subtotal": 100.0,
            "tax": 10.0,
            "shipping_handling": 5.0,
            "total_amount": 115.0,
            "currency": "USD",
            "line_items": [{"description": "Widget", "quantity": 1, "unit_price": 100.0, "total": 100.0}],
        }
        checks = InvoiceValidator().validate(data)
        covered = {c.field for c in checks}
        assert covered >= {
            "vendor_name", "invoice_number", "invoice_date", "due_date",
            "total_amount", "subtotal", "tax", "shipping_handling",
            "currency", "line_items",
        }

    def test_missing_optional_fields_produce_warn_not_fail(self):
        from src.doc_ai.validators import InvoiceValidator

        data = {
            "vendor_name": "Acme",
            "invoice_number": "INV-001",
            "invoice_date": "2026-01-01",
            "total_amount": 100.0,
            "subtotal": None,
            "tax": None,
            "shipping_handling": None,
            "currency": None,
            "line_items": [],
        }
        checks = InvoiceValidator().validate(data)
        statuses = {c.field: c.status for c in checks}
        assert statuses["subtotal"] == "warn"
        assert statuses["tax"] == "warn"
        assert statuses["currency"] == "warn"
        assert statuses["line_items"] == "warn"
        assert "shipping_handling" not in statuses  # optional, no check when absent


# ---------------------------------------------------------------------------
# Content hash / deduplication
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_same_text_same_hash(self):
        from src.doc_ai.pipeline import DocumentPipeline
        h1 = DocumentPipeline._compute_text_hash("Hello World")
        h2 = DocumentPipeline._compute_text_hash("Hello World")
        assert h1 == h2

    def test_whitespace_normalised(self):
        from src.doc_ai.pipeline import DocumentPipeline
        h1 = DocumentPipeline._compute_text_hash("Hello  World")
        h2 = DocumentPipeline._compute_text_hash("Hello World")
        assert h1 == h2

    def test_case_normalised(self):
        from src.doc_ai.pipeline import DocumentPipeline
        h1 = DocumentPipeline._compute_text_hash("HELLO WORLD")
        h2 = DocumentPipeline._compute_text_hash("hello world")
        assert h1 == h2

    def test_different_text_different_hash(self):
        from src.doc_ai.pipeline import DocumentPipeline
        h1 = DocumentPipeline._compute_text_hash("Invoice A")
        h2 = DocumentPipeline._compute_text_hash("Invoice B")
        assert h1 != h2


# ---------------------------------------------------------------------------
# Concurrent writes — regression for "table already exists" race condition
# ---------------------------------------------------------------------------

class TestConcurrentWrites:
    def test_concurrent_bulk_processing_no_crash(self, tmp_path):
        """Multiple files processed concurrently must not crash with a SQLite race condition."""
        pipeline = _make_pipeline(tmp_path)
        invoices = [
            ("a.txt", _txt_bytes(SAMPLE_INVOICE_TEXT)),
            ("b.txt", _txt_bytes(SAMPLE_WITH_SHIPPING)),
            ("c.txt", _txt_bytes(SAMPLE_INVOICE_TEXT.replace("INV-56528", "INV-99999"))),
            ("d.txt", _txt_bytes(SAMPLE_WITH_SHIPPING.replace("INV-99001", "INV-11111"))),
        ]

        errors = []
        def process(item):
            name, data = item
            try:
                pipeline.process_bytes(name, data)
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(process, invoices))

        assert errors == [], f"Concurrent writes produced errors: {errors}"


# ---------------------------------------------------------------------------
# Rate limit retry — unit tests for parsing and exception behaviour
# ---------------------------------------------------------------------------

class TestRateLimitRetry:
    def test_parse_retry_after_from_message(self):
        from src.doc_ai.extractors import _parse_retry_after

        class FakeExc(Exception):
            pass

        exc = FakeExc("Rate limit reached. Please try again in 15s.")
        assert _parse_retry_after(exc) == 16  # rounded up + 1

    def test_parse_retry_after_from_429_keyword(self):
        from src.doc_ai.extractors import _parse_retry_after

        exc = Exception("Error code: 429 - rate limit exceeded")
        assert _parse_retry_after(exc) == 15

    def test_parse_retry_after_returns_none_for_non_rate_limit(self):
        from src.doc_ai.extractors import _parse_retry_after

        exc = Exception("Invalid API key")
        assert _parse_retry_after(exc) is None

    def test_rate_limit_retry_carries_retry_after(self):
        from src.doc_ai.extractors import RateLimitRetry

        rl = RateLimitRetry("too many requests", retry_after=45)
        assert rl.retry_after == 45
        assert "too many requests" in str(rl)

    def test_parse_retry_after_ignores_reset_window_headers(self):
        """x-ratelimit-reset-requests reports window reset time, not retry delay — must be ignored."""
        from src.doc_ai.extractors import _parse_retry_after

        class FakeResponse:
            status_code = 429
            headers = {"x-ratelimit-reset-requests": "552"}

        class FakeExc(Exception):
            response = FakeResponse()

        # Falls through to message parsing and hits the generic 429 fallback (15s)
        result = _parse_retry_after(FakeExc("rate limit exceeded"))
        assert result == 15

    def test_parse_retry_after_uses_retry_after_header(self):
        from src.doc_ai.extractors import _parse_retry_after

        class FakeResponse:
            status_code = 429
            headers = {"retry-after": "20"}

        class FakeExc(Exception):
            response = FakeResponse()

        assert _parse_retry_after(FakeExc("429")) == 20

    def test_parse_retry_after_caps_large_message_value(self):
        from src.doc_ai.extractors import _MAX_RETRY_AFTER_SECONDS, _parse_retry_after

        exc = Exception("Please try again in 600s.")
        assert _parse_retry_after(exc) == _MAX_RETRY_AFTER_SECONDS

    def test_parse_retry_after_handles_minute_format(self):
        """Groq reports 'try again in 9m44.879s' for longer-period limits."""
        from src.doc_ai.extractors import _MAX_RETRY_AFTER_SECONDS, _parse_retry_after

        exc = Exception("Rate limit reached. Please try again in 9m44.879s.")
        # 9*60 + 44 + 1 = 585s → capped
        assert _parse_retry_after(exc) == _MAX_RETRY_AFTER_SECONDS

    def test_parse_retry_after_handles_short_minute_format(self):
        from src.doc_ai.extractors import _parse_retry_after

        exc = Exception("Rate limit reached. Please try again in 0m15.5s.")
        assert _parse_retry_after(exc) == 16  # 0*60 + 15 + 1


# ---------------------------------------------------------------------------
# Document type detection
# ---------------------------------------------------------------------------

SAMPLE_MEDICAL_DISCHARGE = """\
City General Hospital
Patient Name: Jane Doe
Date of Birth: 1985-06-15
Admission Date: 2026-04-01
Discharge Date: 2026-04-05
Primary Diagnosis: Community-acquired pneumonia
Treating Physician: Dr. Smith
Discharge Condition: Stable
Medications:
- Amoxicillin 500mg twice daily for 7 days
- Ibuprofen 400mg as needed
Follow-up Date: 2026-04-19
Discharge Instructions: Rest, drink fluids, return if fever exceeds 38.5C.
"""

SAMPLE_NDA = """\
NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("Agreement") is entered into as of April 1, 2026
between Acme Corp ("Disclosing Party") and Beta LLC ("Receiving Party").

The Receiving Party agrees to keep all Confidential Information secret for a period
of 3 years from the Effective Date.

This Agreement shall be governed by the laws of the State of Delaware.

Agreement Type: Mutual
"""


class TestDocumentTypeDetection:
    def test_detects_invoice(self):
        from src.doc_ai.extractors import detect_document_type
        assert detect_document_type(SAMPLE_INVOICE_TEXT) == "invoice"

    def test_detects_medical_discharge(self):
        from src.doc_ai.extractors import detect_document_type
        assert detect_document_type(SAMPLE_MEDICAL_DISCHARGE) == "medical_discharge"

    def test_detects_nda(self):
        from src.doc_ai.extractors import detect_document_type
        assert detect_document_type(SAMPLE_NDA) == "nda"

    def test_defaults_to_invoice_for_unknown(self):
        from src.doc_ai.extractors import detect_document_type
        assert detect_document_type("some random text with no keywords") == "invoice"


class TestMedicalDischargeExtraction:
    def _doc(self, text):
        from src.doc_ai.schemas import ParsedDocument
        return ParsedDocument(
            file_name="discharge.txt",
            file_path=Path("discharge.txt"),
            raw_text=text,
            sections=text.splitlines(),
        )

    def test_extracts_required_fields(self):
        from src.doc_ai.extractors import RuleBasedMedicalDischargeExtractor
        result = RuleBasedMedicalDischargeExtractor().extract(self._doc(SAMPLE_MEDICAL_DISCHARGE))
        assert result["document_type"] == "medical_discharge"
        assert result["patient_name"] == "Jane Doe"
        assert result["admission_date"] == "2026-04-01"
        assert result["discharge_date"] == "2026-04-05"
        assert result["primary_diagnosis"] is not None

    def test_extracts_medications_as_list(self):
        from src.doc_ai.extractors import RuleBasedMedicalDischargeExtractor
        result = RuleBasedMedicalDischargeExtractor().extract(self._doc(SAMPLE_MEDICAL_DISCHARGE))
        assert isinstance(result["medications"], list)
        assert len(result["medications"]) >= 1

    def test_rule_based_extractor_routes_to_medical(self):
        from src.doc_ai.extractors import RuleBasedInvoiceExtractor
        result = RuleBasedInvoiceExtractor().extract(self._doc(SAMPLE_MEDICAL_DISCHARGE))
        assert result["document_type"] == "medical_discharge"


class TestNDAExtraction:
    def _doc(self, text):
        from src.doc_ai.schemas import ParsedDocument
        return ParsedDocument(
            file_name="nda.txt",
            file_path=Path("nda.txt"),
            raw_text=text,
            sections=text.splitlines(),
        )

    def test_extracts_parties(self):
        from src.doc_ai.extractors import RuleBasedNDAExtractor
        result = RuleBasedNDAExtractor().extract(self._doc(SAMPLE_NDA))
        assert result["document_type"] == "nda"
        assert result["disclosing_party"] is not None
        assert result["receiving_party"] is not None

    def test_detects_mutual_agreement_type(self):
        from src.doc_ai.extractors import RuleBasedNDAExtractor
        result = RuleBasedNDAExtractor().extract(self._doc(SAMPLE_NDA))
        assert result["agreement_type"] == "mutual"

    def test_extracts_governing_law(self):
        from src.doc_ai.extractors import RuleBasedNDAExtractor
        result = RuleBasedNDAExtractor().extract(self._doc(SAMPLE_NDA))
        assert result["governing_law"] is not None


class TestMedicalDischargeValidation:
    def test_pass_when_required_fields_present(self):
        from src.doc_ai.validators import MedicalDischargeValidator
        data = {
            "patient_name": "Jane Doe",
            "admission_date": "2026-04-01",
            "discharge_date": "2026-04-05",
            "primary_diagnosis": "Pneumonia",
            "medications": ["Amoxicillin"],
        }
        checks = MedicalDischargeValidator().validate(data)
        statuses = {c.field: c.status for c in checks}
        assert statuses["patient_name"] == "pass"
        assert statuses["primary_diagnosis"] == "pass"

    def test_fail_when_discharge_before_admission(self):
        from src.doc_ai.validators import MedicalDischargeValidator
        data = {
            "patient_name": "Jane Doe",
            "admission_date": "2026-04-05",
            "discharge_date": "2026-04-01",
            "primary_diagnosis": "Pneumonia",
        }
        checks = MedicalDischargeValidator().validate(data)
        statuses = {c.field: c.status for c in checks}
        assert statuses["date_order"] == "fail"

    def test_get_validator_routes_correctly(self):
        from src.doc_ai.validators import get_validator, MedicalDischargeValidator, NDAValidator, InvoiceValidator
        assert isinstance(get_validator("medical_discharge"), MedicalDischargeValidator)
        assert isinstance(get_validator("nda"), NDAValidator)
        assert isinstance(get_validator("invoice"), InvoiceValidator)
        assert isinstance(get_validator("unknown"), InvoiceValidator)


# ---------------------------------------------------------------------------
# Lab report extraction
# ---------------------------------------------------------------------------

SAMPLE_LAB_REPORT = """\
NEXUS CLINICAL LABORATORIES
LABORATORY REPORT
Patient: Jane Smith
DOB: March 15, 1980
MRN: MRN-1234567
Gender: Female
Ordering Physician: Dr. Alan Grant, MD — Oncology
Accession #: NX-99887766
Specimen Type: Venous Blood
Collected: April 10, 2025 at 09:00 AM
Reported: April 11, 2025 at 08:00 AM
COMPLETE BLOOD COUNT (CBC)
Test
Result
Units
Hemoglobin
9.5
g/dL
Platelets
450
x10^3/uL
Glucose (Fasting)
115
mg/dL
CLINICAL INTERPRETATION / COMMENTS
Low hemoglobin consistent with anemia. Elevated platelets noted.
Reviewed & Verified by: Dr. Patricia Lee, MD | Pathology | April 11, 2025
"""

SAMPLE_BUSINESS_DOC = """\
Acme Corporation
123 Main Street, Springfield, IL 62701 | Confidential Internal Document
Period Ending: March 31, 2026 | Report ID: RPT-1122334
PROJECT STATUS UPDATE
EXECUTIVE SUMMARY
Strong quarter with revenue up 15% year-over-year. Costs reduced by 8%.
KEY PERFORMANCE INDICATORS
Metric
Current Period
Prior Period
Variance
Status
Revenue ($M)
52.3
45.5
+15%
s On Track
EBITDA Margin (%)
28.5
24.1
+4.4 pts
s Strong
STRATEGIC RECOMMENDATIONS
1. Continue investment in cloud infrastructure. 2. Expand the partner network in APAC.
Prepared by: Sarah Connor, CFO
Approved by: John Smith, CEO | Date: April 5, 2026
Document Classification: Confidential
"""


class TestLabReportExtraction:
    def _doc(self, text):
        from src.doc_ai.schemas import ParsedDocument
        return ParsedDocument(
            file_name="lab.txt",
            file_path=Path("lab.txt"),
            raw_text=text,
            sections=text.splitlines(),
        )

    def test_detects_lab_report(self):
        from src.doc_ai.extractors import detect_document_type
        assert detect_document_type(SAMPLE_LAB_REPORT) == "lab_report"

    def test_extracts_patient_fields(self):
        from src.doc_ai.extractors import RuleBasedLabReportExtractor
        result = RuleBasedLabReportExtractor().extract(self._doc(SAMPLE_LAB_REPORT))
        assert result["document_type"] == "lab_report"
        assert result["patient_name"] == "Jane Smith"
        assert result["mrn"] == "MRN-1234567"
        assert result["gender"] == "Female"

    def test_extracts_lab_panels_as_list(self):
        from src.doc_ai.extractors import RuleBasedLabReportExtractor
        result = RuleBasedLabReportExtractor().extract(self._doc(SAMPLE_LAB_REPORT))
        assert isinstance(result["lab_panels"], list)
        assert len(result["lab_panels"]) >= 3
        assert all("test" in p and "value" in p and "units" in p for p in result["lab_panels"])

    def test_extracts_clinical_interpretation(self):
        from src.doc_ai.extractors import RuleBasedLabReportExtractor
        result = RuleBasedLabReportExtractor().extract(self._doc(SAMPLE_LAB_REPORT))
        assert result["clinical_interpretation"] is not None
        assert "anemia" in result["clinical_interpretation"].lower()

    def test_routes_from_rule_based_invoice_extractor(self):
        from src.doc_ai.extractors import RuleBasedInvoiceExtractor
        result = RuleBasedInvoiceExtractor().extract(self._doc(SAMPLE_LAB_REPORT))
        assert result["document_type"] == "lab_report"


class TestLabReportValidation:
    def test_pass_when_patient_and_panels_present(self):
        from src.doc_ai.validators import LabReportValidator
        data = {
            "patient_name": "Jane Smith",
            "collected_date": "2025-04-10",
            "reported_date": "2025-04-11",
            "lab_panels": [{"test": "Hemoglobin", "value": "9.5", "units": "g/dL", "reference_range": "", "flag": "L"}],
            "abnormal_results": [{"test": "Hemoglobin", "value": "9.5", "units": "g/dL", "reference_range": "", "flag": "L"}],
        }
        checks = LabReportValidator().validate(data)
        statuses = {c.field: c.status for c in checks}
        assert statuses["patient_name"] == "pass"
        assert statuses["lab_panels"] == "pass"
        assert statuses["abnormal_results"] == "warn"  # abnormal flags trigger a review warning

    def test_fail_when_patient_missing(self):
        from src.doc_ai.validators import LabReportValidator
        checks = LabReportValidator().validate({"patient_name": None, "lab_panels": []})
        statuses = {c.field: c.status for c in checks}
        assert statuses["patient_name"] == "fail"

    def test_get_validator_routes_to_lab_report(self):
        from src.doc_ai.validators import get_validator, LabReportValidator
        assert isinstance(get_validator("lab_report"), LabReportValidator)


class TestBusinessDocExtraction:
    def _doc(self, text):
        from src.doc_ai.schemas import ParsedDocument
        return ParsedDocument(
            file_name="bizreport.txt",
            file_path=Path("bizreport.txt"),
            raw_text=text,
            sections=text.splitlines(),
        )

    def test_detects_business_doc(self):
        from src.doc_ai.extractors import detect_document_type
        assert detect_document_type(SAMPLE_BUSINESS_DOC) == "business_doc"

    def test_extracts_company_and_subtype(self):
        from src.doc_ai.extractors import RuleBasedBusinessDocExtractor
        result = RuleBasedBusinessDocExtractor().extract(self._doc(SAMPLE_BUSINESS_DOC))
        assert result["document_type"] == "business_doc"
        assert result["company_name"] == "Acme Corporation"
        assert result["document_subtype"] == "Project Status Update"

    def test_document_subtype_normalized_to_title_case(self):
        # Regression: ALL-CAPS PDF header must be title-cased for consistency with reviewed outputs
        from src.doc_ai.extractors import RuleBasedBusinessDocExtractor
        result = RuleBasedBusinessDocExtractor().extract(self._doc(SAMPLE_BUSINESS_DOC))
        assert result["document_subtype"] == result["document_subtype"].title()

    def test_executive_summary_has_no_embedded_newlines(self):
        # Regression: pdfplumber word-wraps long lines with \n; joining must produce a single space-separated string
        from src.doc_ai.extractors import RuleBasedBusinessDocExtractor
        wrapped = SAMPLE_BUSINESS_DOC.replace(
            "Strong quarter with revenue up 15% year-over-year. Costs reduced by 8%.",
            "Strong quarter with revenue\nup 15% year-over-year.\nCosts reduced by 8%.",
        )
        result = RuleBasedBusinessDocExtractor().extract(self._doc(wrapped))
        assert "\n" not in (result["executive_summary"] or "")
        assert "Strong quarter" in result["executive_summary"]

    def test_unstructured_merged_lines_company_name(self):
        # Regression: unstructured merges company name + address onto one line without a newline.
        # The extractor must strip the trailing address portion.
        from src.doc_ai.extractors import RuleBasedBusinessDocExtractor
        merged = SAMPLE_BUSINESS_DOC.replace(
            "Acme Corporation\n123 Main Street, Springfield, IL 62701 | Confidential Internal Document",
            "Acme Corporation 123 Main Street, Springfield, IL 62701 | Confidential Internal Document",
        )
        result = RuleBasedBusinessDocExtractor().extract(self._doc(merged))
        assert result["company_name"] == "Acme Corporation"

    def test_unstructured_merged_lines_document_subtype(self):
        # Regression: unstructured merges the ALL-CAPS subtype with metadata on the same line.
        # The extractor must extract only the uppercase leading words.
        from src.doc_ai.extractors import RuleBasedBusinessDocExtractor
        merged = SAMPLE_BUSINESS_DOC.replace(
            "PROJECT STATUS UPDATE\n",
            "PROJECT STATUS UPDATE Period Ending: March 31, 2026 | Report ID: RPT-1122334\n",
        )
        result = RuleBasedBusinessDocExtractor().extract(self._doc(merged))
        assert result["document_subtype"] == "Project Status Update"

    def test_unstructured_merged_lines_executive_summary(self):
        # Regression: unstructured omits the newline after the EXECUTIVE SUMMARY header,
        # joining it directly to the paragraph text with a space.
        from src.doc_ai.extractors import RuleBasedBusinessDocExtractor
        merged = SAMPLE_BUSINESS_DOC.replace(
            "EXECUTIVE SUMMARY\nStrong quarter",
            "EXECUTIVE SUMMARY Strong quarter",
        )
        result = RuleBasedBusinessDocExtractor().extract(self._doc(merged))
        assert result["executive_summary"] is not None
        assert "Strong quarter" in result["executive_summary"]

    def test_extracts_report_metadata(self):
        from src.doc_ai.extractors import RuleBasedBusinessDocExtractor
        result = RuleBasedBusinessDocExtractor().extract(self._doc(SAMPLE_BUSINESS_DOC))
        assert result["report_id"] == "RPT-1122334"
        assert result["prepared_by"] is not None
        assert "Sarah Connor" in result["prepared_by"]

    def test_extracts_kpis_as_list(self):
        from src.doc_ai.extractors import RuleBasedBusinessDocExtractor
        result = RuleBasedBusinessDocExtractor().extract(self._doc(SAMPLE_BUSINESS_DOC))
        assert isinstance(result["kpis"], list)
        assert len(result["kpis"]) >= 2
        assert all("metric" in k for k in result["kpis"])

    def test_extracts_recommendations_as_list(self):
        from src.doc_ai.extractors import RuleBasedBusinessDocExtractor
        result = RuleBasedBusinessDocExtractor().extract(self._doc(SAMPLE_BUSINESS_DOC))
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) >= 2

    def test_routes_from_rule_based_invoice_extractor(self):
        from src.doc_ai.extractors import RuleBasedInvoiceExtractor
        result = RuleBasedInvoiceExtractor().extract(self._doc(SAMPLE_BUSINESS_DOC))
        assert result["document_type"] == "business_doc"


class TestBusinessDocValidation:
    def test_pass_when_required_fields_present(self):
        from src.doc_ai.validators import BusinessDocValidator
        data = {
            "company_name": "Acme Corp",
            "document_subtype": "Quarterly Report",
            "report_period": "Q1 2026",
            "prepared_by": "CFO",
            "kpis": [{"metric": "Revenue", "current_period": "52M", "prior_period": "45M", "variance": "+15%"}],
        }
        checks = BusinessDocValidator().validate(data)
        statuses = {c.field: c.status for c in checks}
        assert statuses["company_name"] == "pass"
        assert statuses["kpis"] == "pass"

    def test_fail_when_company_missing(self):
        from src.doc_ai.validators import BusinessDocValidator
        checks = BusinessDocValidator().validate({"company_name": None})
        statuses = {c.field: c.status for c in checks}
        assert statuses["company_name"] == "fail"

    def test_get_validator_routes_to_business_doc(self):
        from src.doc_ai.validators import get_validator, BusinessDocValidator
        assert isinstance(get_validator("business_doc"), BusinessDocValidator)


class TestSchemaConfig:
    def test_default_selections_include_defaults(self):
        from src.doc_ai.schema_config import SchemaConfig, FIELD_CATALOG
        import tempfile
        cfg = SchemaConfig(Path(tempfile.mktemp(suffix=".json")))
        for doc_type, fields in FIELD_CATALOG.items():
            selected = set(cfg.get_selected_fields(doc_type))
            for f in fields:
                if f["default"]:
                    assert f["key"] in selected, f"{f['key']} should be selected by default for {doc_type}"

    def test_required_fields_always_included(self):
        from src.doc_ai.schema_config import SchemaConfig, FIELD_CATALOG
        import tempfile
        cfg = SchemaConfig(Path(tempfile.mktemp(suffix=".json")))
        cfg.set_selected_fields("invoice", [])  # attempt to clear all
        selected = set(cfg.get_selected_fields("invoice"))
        for f in FIELD_CATALOG["invoice"]:
            if f["required"]:
                assert f["key"] in selected, f"Required field {f['key']} must always be included"

    def test_ddl_contains_table_name(self):
        from src.doc_ai.schema_config import SchemaConfig
        import tempfile
        cfg = SchemaConfig(Path(tempfile.mktemp(suffix=".json")))
        ddl = cfg.get_ddl("invoice")
        assert "CREATE TABLE IF NOT EXISTS invoices" in ddl
        assert "vendor_name" in ddl

    def test_save_and_reload(self):
        from src.doc_ai.schema_config import SchemaConfig
        import tempfile
        path = Path(tempfile.mktemp(suffix=".json"))
        cfg = SchemaConfig(path)
        cfg.set_selected_fields("nda", ["disclosing_party", "receiving_party"])
        cfg.save()
        cfg2 = SchemaConfig(path)
        selected = cfg2.get_selected_fields("nda")
        assert "disclosing_party" in selected
        assert "receiving_party" in selected


# ---------------------------------------------------------------------------
# Pipeline — end-to-end with .txt fixtures
# ---------------------------------------------------------------------------

class TestPipelineEndToEnd:
    def test_process_txt_invoice(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.process_bytes("invoice.txt", _txt_bytes(SAMPLE_INVOICE_TEXT))

        assert result.extracted_data["invoice_number"] == "INV-56528"
        assert result.extracted_data["total_amount"] == pytest.approx(21639.18)
        assert isinstance(result.extracted_data.get("line_items"), list)
        assert result.content_hash != ""

    def test_duplicate_detection(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        r1 = pipeline.process_bytes("invoice.txt", _txt_bytes(SAMPLE_INVOICE_TEXT))
        assert not r1.summary.get("duplicate")

        # Same content, different filename
        r2 = pipeline.process_bytes("invoice_renamed.txt", _txt_bytes(SAMPLE_INVOICE_TEXT))
        assert r2.summary.get("duplicate") is True

    def test_different_content_not_duplicate(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        pipeline.process_bytes("a.txt", _txt_bytes(SAMPLE_INVOICE_TEXT))

        other = SAMPLE_INVOICE_TEXT.replace("INV-56528", "INV-99999")
        r2 = pipeline.process_bytes("b.txt", _txt_bytes(other))
        assert not r2.summary.get("duplicate")

    def test_content_hash_persisted(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.process_bytes("invoice.txt", _txt_bytes(SAMPLE_INVOICE_TEXT))
        assert pipeline.is_already_processed(result.content_hash)

    def test_shipping_in_result(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.process_bytes("shipping.txt", _txt_bytes(SAMPLE_WITH_SHIPPING))
        assert result.extracted_data.get("shipping_handling") == pytest.approx(25.0)

    def test_validation_runs(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.process_bytes("invoice.txt", _txt_bytes(SAMPLE_INVOICE_TEXT))
        assert len(result.validation_results) > 0


# ---------------------------------------------------------------------------
# Parametrized tests against real fixture files
# ---------------------------------------------------------------------------

_PDF_LIBS_AVAILABLE = any(
    __import__("importlib").util.find_spec(lib) is not None
    for lib in ("pypdf", "pdfplumber", "unstructured")
)


def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

_TESSERACT_AVAILABLE = _tesseract_available()


def _fixture_pdfs():
    if not FIXTURES.exists():
        return []
    return list(FIXTURES.glob("*.pdf")) + list(FIXTURES.glob("*.txt"))


# ---------------------------------------------------------------------------
# Truth-data loader — used by TestFixtureGroundTruth and TestFixtureMissingData
# ---------------------------------------------------------------------------

_TRUTH_DATA: dict[str, dict] = {}
_TRUTH_DIR = FIXTURES / "truth_data"
if _TRUTH_DIR.exists():
    for _f in sorted(_TRUTH_DIR.glob("*_truth.json")):
        try:
            _TRUTH_DATA.update(json.loads(_f.read_text()))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed truth-data JSON in {_f.name}: {exc}") from exc


def _truth_params():
    """Return pytest.param entries for every fixture that has truth data."""
    return [
        pytest.param(FIXTURES / name, expected, id=name)
        for name, expected in _TRUTH_DATA.items()
        if (FIXTURES / name).exists()
    ]


@pytest.mark.parametrize("fixture_path", _fixture_pdfs(), ids=lambda p: p.name)
def test_fixture_file_extracts_without_error(fixture_path, tmp_path):
    """Every file in tests/fixtures/ must process without a pipeline crash."""
    if fixture_path.suffix == ".pdf" and not _PDF_LIBS_AVAILABLE:
        pytest.skip("PDF parsing libraries not installed in this environment")
    pipeline = _make_pipeline(tmp_path)
    file_bytes = fixture_path.read_bytes()
    result = pipeline.process_bytes(fixture_path.name, file_bytes)

    assert result.content_hash != "" or result.summary.get("duplicate")
    assert isinstance(result.extracted_data, dict)
    assert "document_type" in result.extracted_data or result.errors


@pytest.mark.parametrize("fixture_path", _fixture_pdfs(), ids=lambda p: p.name)
def test_fixture_file_no_crash_on_duplicate(fixture_path, tmp_path):
    """Processing the same fixture twice must return duplicate=True, not crash."""
    if fixture_path.suffix == ".pdf" and not _PDF_LIBS_AVAILABLE:
        pytest.skip("PDF parsing libraries not installed in this environment")
    pipeline = _make_pipeline(tmp_path)
    file_bytes = fixture_path.read_bytes()
    pipeline.process_bytes(fixture_path.name, file_bytes)
    copy_name = f"copy_of_{fixture_path.name}"
    r2 = pipeline.process_bytes(copy_name, file_bytes)
    assert r2.summary.get("duplicate") is True


# ---------------------------------------------------------------------------
# Ground-truth assertions — document type and field presence
# ---------------------------------------------------------------------------

class TestFixtureGroundTruth:
    """Each 'full' and 'similar' fixture must extract the correct document_type
    and produce non-None values for every field that truth data marks non-null."""

    @pytest.mark.parametrize("fixture_path,expected", _truth_params())
    def test_document_type_identified_correctly(self, fixture_path, expected, tmp_path):
        if not _PDF_LIBS_AVAILABLE:
            pytest.skip("PDF libraries not installed")
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.process_bytes(fixture_path.name, fixture_path.read_bytes())
        assert not result.summary.get("duplicate"), "Fixture should not be a duplicate on first run"
        assert result.extracted_data.get("document_type") == expected["document_type"], (
            f"{fixture_path.name}: expected document_type={expected['document_type']!r}, "
            f"got {result.extracted_data.get('document_type')!r}"
        )

    @pytest.mark.parametrize("fixture_path,expected", _truth_params())
    def test_non_null_truth_fields_are_extracted(self, fixture_path, expected, tmp_path):
        if not _PDF_LIBS_AVAILABLE:
            pytest.skip("PDF libraries not installed")
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.process_bytes(fixture_path.name, fixture_path.read_bytes())
        extracted = result.extracted_data
        missing = [
            field for field, val in expected.items()
            if field != "document_type"
            and val is not None
            and extracted.get(field) in (None, "", [], {})
        ]
        assert missing == [], (
            f"{fixture_path.name}: fields in truth data but not extracted: {missing}\n"
            f"Trace: {result.extraction_trace}"
        )


# ---------------------------------------------------------------------------
# Missing-data fixture assertions
# ---------------------------------------------------------------------------

class TestFixtureMissingData:
    """Missing-data fixtures must have their null truth fields absent in extraction
    and must be flagged needs_review=True."""

    _MISSING_FIXTURES = [
        pytest.param(FIXTURES / name, expected, id=name)
        for name, expected in _TRUTH_DATA.items()
        if "_missing" in name and (FIXTURES / name).exists()
    ]

    @pytest.mark.parametrize("fixture_path,expected", _MISSING_FIXTURES)
    def test_missing_fixture_flagged_needs_review(self, fixture_path, expected, tmp_path):
        if not _PDF_LIBS_AVAILABLE:
            pytest.skip("PDF libraries not installed")
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.process_bytes(fixture_path.name, fixture_path.read_bytes())
        assert result.needs_review is True, (
            f"{fixture_path.name}: expected needs_review=True but got False. "
            f"Validation: {result.validation_results}"
        )

    @pytest.mark.parametrize("fixture_path,expected", _MISSING_FIXTURES)
    def test_null_truth_fields_are_none_in_extraction(self, fixture_path, expected, tmp_path):
        if not _PDF_LIBS_AVAILABLE:
            pytest.skip("PDF libraries not installed")
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.process_bytes(fixture_path.name, fixture_path.read_bytes())
        extracted = result.extracted_data
        wrongly_extracted = [
            field for field, val in expected.items()
            if val is None
            and field not in ("document_type",)
            and extracted.get(field) not in (None, "", [], {})
        ]
        assert wrongly_extracted == [], (
            f"{fixture_path.name}: fields truth marks null but extractor found values for: "
            f"{wrongly_extracted} — check if the PDF truly has those fields blank"
        )


# ---------------------------------------------------------------------------
# Duplicate detection assertions
# ---------------------------------------------------------------------------

class TestFixtureDuplicateDetection:
    """_dup fixtures are exact copies of their originals.
    Processing original then dup must flag dup as duplicate.
    Processing dup alone must NOT flag it as duplicate."""

    _DUP_PAIRS = [
        pytest.param(
            FIXTURES / f"{doc_type}_format_a_full.pdf",
            FIXTURES / f"{doc_type}_format_a_full_dup.pdf",
            id=f"{doc_type}_searchable_dup",
        )
        for doc_type in ("invoice", "medical_discharge", "nda", "lab_report", "business_doc")
        if (FIXTURES / f"{doc_type}_format_a_full.pdf").exists()
        and (FIXTURES / f"{doc_type}_format_a_full_dup.pdf").exists()
    ] + [
        pytest.param(
            FIXTURES / f"{doc_type}_no_text_full.pdf",
            FIXTURES / f"{doc_type}_no_text_dup.pdf",
            id=f"{doc_type}_no_text_dup",
        )
        for doc_type in ("invoice", "medical_discharge", "nda", "lab_report", "business_doc")
        if (FIXTURES / f"{doc_type}_no_text_full.pdf").exists()
        and (FIXTURES / f"{doc_type}_no_text_dup.pdf").exists()
    ]

    @pytest.mark.parametrize("original,dup", _DUP_PAIRS)
    def test_dup_detected_after_original_processed(self, original, dup, tmp_path):
        if not _PDF_LIBS_AVAILABLE:
            pytest.skip("PDF libraries not installed")
        pipeline = _make_pipeline(tmp_path)
        r1 = pipeline.process_bytes(original.name, original.read_bytes())
        assert not r1.summary.get("duplicate"), "Original should not be flagged as duplicate"
        r2 = pipeline.process_bytes(dup.name, dup.read_bytes())
        assert r2.summary.get("duplicate") is True, (
            f"{dup.name}: expected duplicate=True after processing original {original.name}. "
            f"content_hash original={r1.content_hash!r}"
        )

    @pytest.mark.parametrize("original,dup", _DUP_PAIRS)
    def test_dup_alone_is_not_flagged_as_duplicate(self, original, dup, tmp_path):
        if not _PDF_LIBS_AVAILABLE:
            pytest.skip("PDF libraries not installed")
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.process_bytes(dup.name, dup.read_bytes())
        assert not result.summary.get("duplicate"), (
            f"{dup.name}: should not be flagged as duplicate when processed fresh"
        )


# ---------------------------------------------------------------------------
# Template matching assertions
# ---------------------------------------------------------------------------

class TestFixtureTemplateMatching:
    """Process format_a_full then format_a_similar for the same doc type.
    The second document must show Template sources (template memory fired)."""

    _TEMPLATE_PAIRS = [
        pytest.param(
            FIXTURES / f"{doc_type}_format_a_full.pdf",
            FIXTURES / f"{doc_type}_format_a_similar.pdf",
            id=doc_type,
        )
        for doc_type in ("invoice", "medical_discharge", "nda", "lab_report", "business_doc")
        if (FIXTURES / f"{doc_type}_format_a_full.pdf").exists()
        and (FIXTURES / f"{doc_type}_format_a_similar.pdf").exists()
    ]

    @pytest.mark.parametrize("first,second", _TEMPLATE_PAIRS)
    def test_similar_fixture_uses_template_source(self, first, second, tmp_path):
        if not _PDF_LIBS_AVAILABLE:
            pytest.skip("PDF libraries not installed")
        pipeline = _make_pipeline(tmp_path)

        r1 = pipeline.process_bytes(first.name, first.read_bytes())
        assert not r1.summary.get("duplicate")
        assert r1.summary.get("learned_template") is not None, (
            f"{first.name}: expected a template to be learned after first pass. "
            f"Trace: {r1.extraction_trace}"
        )

        r2 = pipeline.process_bytes(second.name, second.read_bytes())
        assert not r2.summary.get("duplicate")
        template_sources = [
            field for field, src in r2.field_sources.items()
            if src == "Template"
        ]
        assert len(template_sources) > 0, (
            f"{second.name}: expected at least one Template-sourced field after processing "
            f"{first.name} first. field_sources={r2.field_sources}. "
            f"Trace: {r2.extraction_trace}"
        )

    @pytest.mark.parametrize("first,second", _TEMPLATE_PAIRS)
    def test_template_hit_boosts_confidence(self, first, second, tmp_path):
        if not _PDF_LIBS_AVAILABLE:
            pytest.skip("PDF libraries not installed")
        pipeline = _make_pipeline(tmp_path)
        pipeline.process_bytes(first.name, first.read_bytes())
        r2 = pipeline.process_bytes(second.name, second.read_bytes())

        template_fields = [f for f, s in r2.field_sources.items() if s == "Template"]
        if not template_fields:
            pytest.skip("No template fields found — template matching did not fire")

        for field in template_fields:
            conf = r2.field_confidence.get(field, 0.0)
            assert conf >= 0.82, (
                f"{field}: Template-sourced field has confidence {conf:.3f} < 0.82 baseline"
            )


# ---------------------------------------------------------------------------
# OCR path assertions
# ---------------------------------------------------------------------------

class TestFixtureOCR:
    """No-text fixtures must be processed via the OCR fallback path.
    Parsed text must be non-empty and a document type must be assigned."""

    _NO_TEXT_FIXTURES = [
        pytest.param(FIXTURES / f"{doc_type}_no_text_full.pdf", doc_type, id=f"{doc_type}_no_text")
        for doc_type in ("invoice", "medical_discharge", "nda", "lab_report", "business_doc")
        if (FIXTURES / f"{doc_type}_no_text_full.pdf").exists()
    ]

    @pytest.mark.parametrize("fixture_path,expected_type", _NO_TEXT_FIXTURES)
    def test_no_text_pdf_produces_nonempty_parsed_text(self, fixture_path, expected_type, tmp_path):
        if not _PDF_LIBS_AVAILABLE:
            pytest.skip("PDF libraries not installed")
        if not _TESSERACT_AVAILABLE:
            pytest.skip("Tesseract binary not installed — OCR unavailable")
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.process_bytes(fixture_path.name, fixture_path.read_bytes())
        assert result.parsed_text.strip() != "", (
            f"{fixture_path.name}: OCR produced no text. "
            f"Errors: {result.errors}"
        )

    @pytest.mark.parametrize("fixture_path,expected_type", _NO_TEXT_FIXTURES)
    def test_no_text_pdf_document_type_identified(self, fixture_path, expected_type, tmp_path):
        if not _PDF_LIBS_AVAILABLE:
            pytest.skip("PDF libraries not installed")
        if not _TESSERACT_AVAILABLE:
            pytest.skip("Tesseract binary not installed — OCR unavailable")
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.process_bytes(fixture_path.name, fixture_path.read_bytes())
        doc_type = result.extracted_data.get("document_type", "unknown")
        assert doc_type != "unknown", (
            f"{fixture_path.name}: document type not identified after OCR. "
            f"parsed_text preview: {result.parsed_text[:200]!r}"
        )

    @pytest.mark.parametrize("fixture_path,expected_type", _NO_TEXT_FIXTURES)
    def test_no_text_pdf_is_truly_image_only(self, fixture_path, expected_type, tmp_path):
        """Verify the fixture has no extractable text (confirming it will exercise OCR)."""
        try:
            import pypdf
        except ImportError:
            pytest.skip("pypdf not available")
        reader = pypdf.PdfReader(str(fixture_path))
        direct_text = "".join(p.extract_text() or "" for p in reader.pages).strip()
        assert len(direct_text) < 100, (
            f"{fixture_path.name}: pypdf extracted {len(direct_text)} chars — "
            "this PDF is NOT image-only. Regenerate it with no searchable text."
        )


# ---------------------------------------------------------------------------
# Unexpected extraction error handling (e.g. Groq "Execution failed")
# ---------------------------------------------------------------------------

class TestUnexpectedExtractionError:
    def test_unexpected_api_error_returns_result_flagged_for_review(self, tmp_path):
        """Non-ExtractionError exceptions (e.g. Groq 'Execution failed') must not
        crash the pipeline — the document should come back needs_review=True."""
        from unittest.mock import MagicMock, patch

        pipeline = _make_pipeline(tmp_path)
        mock_extractor = MagicMock()
        mock_extractor.extract_with_trace.side_effect = RuntimeError("Execution failed")

        with patch("src.doc_ai.pipeline.build_extractor", return_value=mock_extractor):
            result = pipeline.process_bytes("doc.txt", _txt_bytes(SAMPLE_INVOICE_TEXT))

        assert result.needs_review is True
        assert any("Extraction error" in e or "Execution failed" in e for e in result.errors)

    def test_rate_limit_retry_still_propagates(self, tmp_path):
        """RateLimitRetry must bubble up through process_bytes so app.py can handle it."""
        from unittest.mock import MagicMock, patch
        from src.doc_ai.extractors import RateLimitRetry

        pipeline = _make_pipeline(tmp_path)
        mock_extractor = MagicMock()
        mock_extractor.extract_with_trace.side_effect = RateLimitRetry("too many requests", retry_after=5)

        with patch("src.doc_ai.pipeline.build_extractor", return_value=mock_extractor):
            with pytest.raises(RateLimitRetry):
                pipeline.process_bytes("doc.txt", _txt_bytes(SAMPLE_INVOICE_TEXT))


# ---------------------------------------------------------------------------
# Field confidence scoring
# ---------------------------------------------------------------------------

class TestFieldConfidence:
    def test_confidence_dict_present_on_result(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.process_bytes("invoice.txt", _txt_bytes(SAMPLE_INVOICE_TEXT))
        assert isinstance(result.field_confidence, dict)
        assert len(result.field_confidence) > 0

    def test_missing_field_has_zero_confidence(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        # SAMPLE_INVOICE_TEXT has no vendor_name extractable via rule-based
        result = pipeline.process_bytes("invoice.txt", _txt_bytes(SAMPLE_INVOICE_TEXT))
        conf = result.field_confidence
        # Any field that is None in extracted_data must have 0.0 confidence
        for field, value in result.extracted_data.items():
            if value in (None, "", [], {}):
                assert conf.get(field, 0.0) == 0.0, (
                    f"Field '{field}' is empty but confidence={conf.get(field)}"
                )

    def test_present_field_has_nonzero_confidence(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.process_bytes("invoice.txt", _txt_bytes(SAMPLE_INVOICE_TEXT))
        conf = result.field_confidence
        # total_amount is present in sample — must have > 0 confidence
        assert result.extracted_data.get("total_amount") is not None
        assert conf.get("total_amount", 0.0) > 0.0

    def test_confidence_values_are_in_valid_range(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        result = pipeline.process_bytes("invoice.txt", _txt_bytes(SAMPLE_INVOICE_TEXT))
        for field, score in result.field_confidence.items():
            assert 0.0 <= score <= 1.0, f"'{field}' confidence {score} out of [0, 1]"

    def test_validation_fail_lowers_confidence(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        # Use the medical discharge sample — discharge_date before admission triggers fail
        text = SAMPLE_MEDICAL_DISCHARGE.replace("Admission Date: 2026-04-01", "Admission Date: 2026-04-10")
        result = pipeline.process_bytes("discharge.txt", _txt_bytes(text))
        conf = result.field_confidence
        # date_order check fails → the involved date fields should have lower confidence
        # At minimum: no date field involved in a fail should be at >= 0.85
        fails = {c["field"] for c in result.validation_results if c["status"] == "fail"}
        for f in fails:
            if f in conf:
                assert conf[f] < 0.85, (
                    f"Field '{f}' has a fail validation but confidence={conf[f]:.2f}"
                )

    def test_finalize_review_returns_confidence(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        r = pipeline.process_bytes("invoice.txt", _txt_bytes(SAMPLE_INVOICE_TEXT))
        reviewed = pipeline.finalize_review(
            source_file=r.source_file,
            upload_path=r.upload_path,
            parsed_text=r.parsed_text,
            corrected_data=r.extracted_data,
            content_hash=r.content_hash,
        )
        assert isinstance(reviewed.field_confidence, dict)
        assert len(reviewed.field_confidence) > 0


# ---------------------------------------------------------------------------
# _compute_field_confidence — unit tests
# ---------------------------------------------------------------------------

class TestComputeFieldConfidence:
    """Tests for DocumentPipeline._compute_field_confidence directly."""

    @staticmethod
    def _run(extracted_data, validation_checks, trace):
        from src.doc_ai.pipeline import DocumentPipeline
        return DocumentPipeline._compute_field_confidence(
            extracted_data, validation_checks, trace
        )

    # --- baseline selection from trace ---

    def test_llm_trace_gives_highest_baseline(self):
        trace = ["Used the LLM reasoning layer for an unseen format."]
        conf = self._run({"vendor_name": "Acme"}, [], trace)
        assert conf["vendor_name"] == pytest.approx(0.88)

    def test_openai_keyword_in_trace_triggers_llm_baseline(self):
        trace = ["openai returned the extraction result"]
        conf = self._run({"invoice_number": "INV-1"}, [], trace)
        assert conf["invoice_number"] == pytest.approx(0.88)

    def test_groq_keyword_in_trace_triggers_llm_baseline(self):
        trace = ["groq provider returned data"]
        conf = self._run({"total_amount": 100.0}, [], trace)
        assert conf["total_amount"] == pytest.approx(0.88)

    def test_learned_template_trace_gives_template_baseline(self):
        trace = ["Applied learned template anchors to extract fields."]
        conf = self._run({"invoice_date": "2026-01-01"}, [], trace)
        assert conf["invoice_date"] == pytest.approx(0.82)

    def test_rule_based_trace_gives_rule_baseline(self):
        trace = ["Fell back to rule-based label and regex extraction."]
        conf = self._run({"due_date": "2026-02-01"}, [], trace)
        assert conf["due_date"] == pytest.approx(0.72)

    def test_unknown_trace_gives_default_baseline(self):
        trace = ["Some other extraction step."]
        conf = self._run({"currency": "USD"}, [], trace)
        assert conf["currency"] == pytest.approx(0.75)

    def test_empty_trace_gives_default_baseline(self):
        conf = self._run({"currency": "USD"}, [], [])
        assert conf["currency"] == pytest.approx(0.75)

    # --- validation multipliers ---

    def test_pass_multiplier_is_1x(self):
        from src.doc_ai.schemas import ValidationCheck
        trace = ["Some extraction step."]
        checks = [ValidationCheck(field="vendor_name", status="pass", message="ok")]
        conf = self._run({"vendor_name": "Acme"}, checks, trace)
        assert conf["vendor_name"] == pytest.approx(0.75 * 1.0)

    def test_warn_multiplier_is_0_75x(self):
        from src.doc_ai.schemas import ValidationCheck
        trace = []
        checks = [ValidationCheck(field="due_date", status="warn", message="missing")]
        conf = self._run({"due_date": "2026-01-01"}, checks, trace)
        assert conf["due_date"] == pytest.approx(0.75 * 0.75, abs=0.001)

    def test_fail_multiplier_is_0_35x(self):
        from src.doc_ai.schemas import ValidationCheck
        trace = []
        checks = [ValidationCheck(field="total_amount", status="fail", message="bad")]
        conf = self._run({"total_amount": -5.0}, checks, trace)
        assert conf["total_amount"] == pytest.approx(0.75 * 0.35, abs=0.001)

    def test_worst_validation_wins_when_multiple_checks_for_same_field(self):
        from src.doc_ai.schemas import ValidationCheck
        trace = []
        checks = [
            ValidationCheck(field="invoice_date", status="pass", message="required ok"),
            ValidationCheck(field="invoice_date", status="fail", message="bad format"),
        ]
        conf = self._run({"invoice_date": "not-a-date"}, checks, trace)
        # fail wins over pass
        assert conf["invoice_date"] == pytest.approx(0.75 * 0.35, abs=0.001)

    def test_warn_beats_pass_but_loses_to_fail(self):
        from src.doc_ai.schemas import ValidationCheck
        trace = []
        checks = [
            ValidationCheck(field="invoice_date", status="pass", message="required ok"),
            ValidationCheck(field="invoice_date", status="warn", message="no date"),
        ]
        conf_warn = self._run({"invoice_date": "2026-01-01"}, checks, trace)
        assert conf_warn["invoice_date"] == pytest.approx(0.75 * 0.75, abs=0.001)

    # --- missing / empty fields always produce 0 ---

    def test_none_value_gives_zero(self):
        conf = self._run({"vendor_name": None}, [], [])
        assert conf["vendor_name"] == 0.0

    def test_empty_string_gives_zero(self):
        conf = self._run({"vendor_name": ""}, [], [])
        assert conf["vendor_name"] == 0.0

    def test_empty_list_gives_zero(self):
        conf = self._run({"line_items": []}, [], [])
        assert conf["line_items"] == 0.0

    def test_empty_dict_gives_zero(self):
        conf = self._run({"metadata": {}}, [], [])
        assert conf["metadata"] == 0.0

    # --- system fields excluded ---

    def test_document_type_is_excluded(self):
        conf = self._run({"document_type": "invoice", "vendor_name": "X"}, [], [])
        assert "document_type" not in conf

    def test_source_file_is_excluded(self):
        conf = self._run({"source_file": "file.pdf", "vendor_name": "X"}, [], [])
        assert "source_file" not in conf

    # --- dict-style validation checks (from validation_results list) ---

    def test_accepts_dict_checks_from_validation_results(self):
        trace = []
        checks = [{"field": "vendor_name", "status": "pass", "message": "ok"}]
        conf = self._run({"vendor_name": "Acme"}, checks, trace)
        assert conf["vendor_name"] == pytest.approx(0.75 * 1.0)

    # --- combined baseline + multiplier sanity ---

    def test_llm_pass_is_highest_possible_score(self):
        from src.doc_ai.schemas import ValidationCheck
        trace = ["Used the LLM reasoning layer."]
        checks = [ValidationCheck(field="vendor_name", status="pass", message="ok")]
        conf = self._run({"vendor_name": "Acme"}, checks, trace)
        assert conf["vendor_name"] == pytest.approx(0.88)

    def test_rule_based_fail_is_near_floor(self):
        from src.doc_ai.schemas import ValidationCheck
        trace = ["rule-based extraction"]
        checks = [ValidationCheck(field="total_amount", status="fail", message="bad")]
        conf = self._run({"total_amount": -1.0}, checks, trace)
        assert conf["total_amount"] == pytest.approx(0.72 * 0.35, abs=0.001)

    def test_all_values_remain_in_unit_interval(self):
        from src.doc_ai.schemas import ValidationCheck
        trace = ["LLM groq extraction"]
        data = {"f1": "val", "f2": None, "f3": [], "f4": 100.0, "f5": ""}
        checks = [
            ValidationCheck(field="f1", status="pass", message="ok"),
            ValidationCheck(field="f4", status="fail", message="bad"),
        ]
        conf = self._run(data, checks, trace)
        for field, score in conf.items():
            assert 0.0 <= score <= 1.0, f"{field}: {score}"


# ---------------------------------------------------------------------------
# Spatial extractor unit tests
# ---------------------------------------------------------------------------

class TestSpatialExtractor:
    """Unit tests for spatial_extractor helpers — no pdfplumber required."""

    def _make_word(self, text, x0, top, x1=None, bottom=None):
        from src.doc_ai.spatial_extractor import SpatialWord
        return SpatialWord(
            text=text,
            x0=float(x0),
            top=float(top),
            x1=float(x1 if x1 is not None else x0 + len(text) * 6),
            bottom=float(bottom if bottom is not None else top + 12),
        )

    def _make_row(self, words):
        from src.doc_ai.spatial_extractor import SpatialRow
        return SpatialRow(words=words)

    def test_group_into_rows_single_row(self):
        from src.doc_ai.spatial_extractor import _group_into_rows
        words = [
            self._make_word("Invoice", 10, 50),
            self._make_word("Number:", 60, 51),
            self._make_word("INV-001", 120, 50),
        ]
        rows = _group_into_rows(words)
        assert len(rows) == 1
        assert rows[0].text == "Invoice Number: INV-001"

    def test_group_into_rows_two_rows(self):
        from src.doc_ai.spatial_extractor import _group_into_rows
        words = [
            self._make_word("Invoice:", 10, 50),
            self._make_word("Total:", 10, 100),
        ]
        rows = _group_into_rows(words)
        assert len(rows) == 2

    def test_row_to_pair_inline_colon(self):
        from src.doc_ai.spatial_extractor import SpatialWord, SpatialRow, _row_to_pair
        words = [
            self._make_word("Invoice", 10, 50),
            self._make_word("Number:", 60, 50),
            self._make_word("INV-001", 120, 50),
        ]
        row = SpatialRow(words=words)
        pair = _row_to_pair(row)
        assert pair is not None
        label, value = pair
        assert "invoice" in label
        assert value == "INV-001"

    def test_row_to_pair_no_colon_returns_none(self):
        from src.doc_ai.spatial_extractor import SpatialRow, _row_to_pair
        words = [self._make_word("just", 10, 50), self._make_word("text", 50, 50)]
        row = SpatialRow(words=words)
        assert _row_to_pair(row) is None

    def test_extract_fields_from_layout_label_value(self):
        from src.doc_ai.spatial_extractor import (
            SpatialWord, SpatialRow, SpatialLayout, extract_fields_from_layout
        )
        words = [
            self._make_word("Invoice", 10, 50, x1=60),
            self._make_word("Number:", 62, 50, x1=110),
            self._make_word("INV-999", 112, 50, x1=160),
        ]
        row = SpatialRow(words=words)
        layout = SpatialLayout(page_width=600, page_height=800, rows=[row], page_number=0)
        fields = extract_fields_from_layout([layout])
        assert "invoice_number" in fields
        assert fields["invoice_number"] == "INV-999"

    def test_build_spatial_anchors_records_position(self):
        from src.doc_ai.spatial_extractor import (
            SpatialWord, SpatialRow, SpatialLayout, build_spatial_anchors
        )
        words = [
            self._make_word("Invoice", 10, 50, x1=60),
            self._make_word("Number:", 62, 50, x1=110),
            self._make_word("INV-001", 112, 50, x1=162),
        ]
        row = SpatialRow(words=words)
        layout = SpatialLayout(page_width=600, page_height=800, rows=[row], page_number=0)
        extracted = {"invoice_number": "INV-001"}
        anchors = build_spatial_anchors([layout], extracted)
        assert len(anchors) == 1
        assert anchors[0]["field"] == "invoice_number"
        assert "norm_x" in anchors[0]
        assert "norm_y" in anchors[0]
        assert 0.0 <= anchors[0]["norm_x"] <= 1.0
        assert 0.0 <= anchors[0]["norm_y"] <= 1.0

    def test_build_spatial_anchors_skips_label_less_value_scan(self):
        """Fields that appear as standalone text (no label) are NOT anchored — position-only anchors are unreliable."""
        from src.doc_ai.spatial_extractor import (
            SpatialWord, SpatialRow, SpatialLayout, build_spatial_anchors
        )
        row1 = SpatialRow(words=[
            self._make_word("Company", 10, 50, x1=80),
            self._make_word("Name:", 82, 50, x1=120),
            self._make_word("Acme", 122, 50, x1=160),
            self._make_word("Corp", 162, 50, x1=200),
        ])
        # Standalone row with no label — value-scan pass must NOT store this anchor
        row2 = SpatialRow(words=[
            self._make_word("Q1-2026", 10, 100, x1=80),
        ])
        layout = SpatialLayout(page_width=600, page_height=800, rows=[row1, row2], page_number=0)
        extracted = {"company_name": "Acme Corp", "report_period": "Q1-2026"}
        anchors = build_spatial_anchors([layout], extracted)
        fields = {a["field"] for a in anchors}
        # company_name is anchored via Pass 1 (label "company name" is in _LABEL_TO_FIELD)
        assert "company_name" in fields
        assert anchors[0]["label_text"] == "company name"
        # report_period appears as bare text with no label — must not be stored
        assert "report_period" not in fields

    def test_extract_by_spatial_anchors_finds_value(self):
        from src.doc_ai.spatial_extractor import (
            SpatialWord, SpatialRow, SpatialLayout, extract_by_spatial_anchors
        )
        words = [
            self._make_word("Invoice", 10, 50, x1=60),
            self._make_word("Number:", 62, 50, x1=110),
            self._make_word("INV-001", 112, 50, x1=162),
        ]
        row = SpatialRow(words=words)
        layout = SpatialLayout(page_width=600, page_height=800, rows=[row], page_number=0)
        anchors = [
            {"field": "invoice_number", "label_text": "invoice number", "norm_x": 0.016, "norm_y": 0.0625},
        ]
        result = extract_by_spatial_anchors([layout], anchors, x_tol=0.15, y_tol=0.10)
        assert "invoice_number" in result
        assert result["invoice_number"] == "INV-001"

    def test_empty_layouts_returns_empty(self):
        from src.doc_ai.spatial_extractor import (
            extract_fields_from_layout, build_spatial_anchors, extract_by_spatial_anchors
        )
        assert extract_fields_from_layout([]) == {}
        assert build_spatial_anchors([], {"invoice_number": "X"}) == []
        assert extract_by_spatial_anchors([], [{"field": "invoice_number"}]) == {}

    def test_column_gap_pairs_detects_two_column_layout(self):
        from src.doc_ai.spatial_extractor import SpatialWord, SpatialRow, _column_gap_pairs
        # Label on left, value on right with >20pt gap
        words = [
            self._make_word("Total", 10, 50, x1=50),
            self._make_word("1,234.56", 120, 50, x1=200),
        ]
        row = SpatialRow(words=words)
        pairs = _column_gap_pairs([row])
        assert "total" in pairs
        assert pairs["total"] == "1,234.56"

    def test_stacked_pairs_detects_label_then_value(self):
        from src.doc_ai.spatial_extractor import SpatialWord, SpatialRow, _stacked_pairs
        label_words = [self._make_word("Invoice Number:", 10, 50, x1=120)]
        value_words = [self._make_word("INV-777", 10, 65, x1=80)]
        label_row = SpatialRow(words=label_words)
        value_row = SpatialRow(words=value_words)
        pairs = _stacked_pairs([label_row, value_row])
        assert "invoice number" in pairs
        assert pairs["invoice number"] == "INV-777"


class TestSpatialExtractionFromFixturePDF:
    """Integration tests using a real fixture PDF — skipped if pdfplumber absent."""

    FIXTURE = FIXTURES / "invoice_format_a_full.pdf"

    def test_extract_spatial_layout_returns_data_for_pdf(self):
        pytest.importorskip("pdfplumber")
        if not self.FIXTURE.exists():
            pytest.skip("invoice_format_a_full.pdf fixture not present")
        from src.doc_ai.spatial_extractor import extract_spatial_layout
        layouts = extract_spatial_layout(self.FIXTURE)
        assert isinstance(layouts, list)
        assert len(layouts) >= 1
        assert layouts[0].page_width > 0
        assert layouts[0].page_height > 0

    def test_extract_fields_from_pdf_fixture_gets_some_fields(self):
        pytest.importorskip("pdfplumber")
        if not self.FIXTURE.exists():
            pytest.skip("invoice_format_a_full.pdf fixture not present")
        from src.doc_ai.spatial_extractor import extract_spatial_layout, extract_fields_from_layout
        layouts = extract_spatial_layout(self.FIXTURE)
        fields = extract_fields_from_layout(layouts)
        # Searchable PDF should yield at least one recognised field
        assert isinstance(fields, dict)
        assert len(fields) >= 1


class TestTemplateSpatialAnchors:
    """Verify TemplateMemory stores and merges spatial anchors correctly."""

    def test_learn_template_stores_spatial_anchors(self, tmp_path):
        from src.doc_ai.template_memory import TemplateMemory
        store = tmp_path / "templates.json"
        mem = TemplateMemory(store)
        anchors = [{"field": "invoice_number", "label_text": "invoice number", "norm_x": 0.05, "norm_y": 0.1}]
        lines = ["Invoice Number: INV-001", "Total: 100.00"]
        data = {"document_type": "invoice", "invoice_number": "INV-001", "total_amount": "100.00"}
        sig = TemplateMemory.build_signature(lines)
        template = mem.learn_template("test_inv.pdf", sig, data, lines, spatial_anchors=anchors)
        assert "spatial_anchors" in template
        assert template["spatial_anchors"][0]["field"] == "invoice_number"
        stored = mem.load_templates()
        assert stored[0].get("spatial_anchors") is not None

    def test_learn_template_without_spatial_anchors_still_works(self, tmp_path):
        from src.doc_ai.template_memory import TemplateMemory
        store = tmp_path / "templates.json"
        mem = TemplateMemory(store)
        lines = ["Invoice Number: INV-002", "Total: 200.00"]
        data = {"document_type": "invoice", "invoice_number": "INV-002"}
        sig = TemplateMemory.build_signature(lines)
        template = mem.learn_template("test_inv2.pdf", sig, data, lines, spatial_anchors=None)
        assert template.get("template_name") == "test_inv2"
        stored = mem.load_templates()
        assert len(stored) == 1

    def test_update_template_merges_spatial_anchors(self, tmp_path):
        from src.doc_ai.template_memory import TemplateMemory
        store = tmp_path / "templates.json"
        mem = TemplateMemory(store)
        lines = ["Invoice Number: INV-003", "Total: 300.00", "vendor acme corp"]
        data = {"document_type": "invoice", "invoice_number": "INV-003", "total_amount": "300.00"}
        # Identical non-zero zone_density vectors so the merge threshold (0.85) is met.
        zone = [0.5] * 80
        sig = {"zone_density": zone}
        anchors_first = [{"field": "invoice_number", "label_text": "invoice number", "norm_x": 0.05, "norm_y": 0.1}]
        mem.learn_template("t.pdf", sig, data, lines, spatial_anchors=anchors_first)

        # Second learn with same signature and a new anchor field
        anchors_second = [{"field": "total_amount", "label_text": "total", "norm_x": 0.05, "norm_y": 0.8}]
        mem.learn_template("t.pdf", sig, data, lines, spatial_anchors=anchors_second)

        stored = mem.load_templates()
        assert len(stored) == 1  # merged, not duplicated
        fields_stored = {a["field"] for a in stored[0].get("spatial_anchors", [])}
        assert "invoice_number" in fields_stored
        assert "total_amount" in fields_stored


# ---------------------------------------------------------------------------
# Type-scoped template matching
# ---------------------------------------------------------------------------

class TestTypeScopedTemplateMatching:
    """Templates should only match documents of the same type."""

    def _make_mem(self, tmp_path):
        from src.doc_ai.template_memory import TemplateMemory
        mem = TemplateMemory(tmp_path / "templates.json")
        inv_lines = ["Invoice Number: INV-001", "Total: 100.00", "vendor acme corp", "invoice"]
        biz_lines = ["Company Report Q1", "Prepared by: Finance", "Period: 2026-Q1", "business"]
        inv_sig = TemplateMemory.build_signature(inv_lines)
        biz_sig = TemplateMemory.build_signature(biz_lines)
        mem.learn_template("inv.pdf", inv_sig, {"document_type": "invoice", "invoice_number": "INV-001"}, inv_lines)
        mem.learn_template("biz.pdf", biz_sig, {"document_type": "business_doc", "company_name": "Acme"}, biz_lines)
        return mem, inv_sig

    def test_find_best_match_filters_by_type(self, tmp_path):
        mem, sig = self._make_mem(tmp_path)
        match = mem.find_best_match(sig, document_type="invoice")
        assert match is not None
        assert match.template["document_type"] == "invoice"

    def test_find_best_match_no_type_returns_any(self, tmp_path):
        mem, sig = self._make_mem(tmp_path)
        match = mem.find_best_match(sig)
        assert match is not None  # still works without filter

    def test_find_best_match_wrong_type_returns_none(self, tmp_path):
        mem, sig = self._make_mem(tmp_path)
        # Only invoice and business_doc templates exist; nda should find nothing
        match = mem.find_best_match(sig, document_type="nda")
        assert match is None


class TestGetRequiredFields:
    """get_required_fields must derive required keys from FIELD_CATALOG."""

    def test_invoice_required_keys(self):
        from src.doc_ai.schema_config import get_required_fields
        assert get_required_fields("invoice") == {"vendor_name", "invoice_number", "invoice_date", "total_amount"}

    def test_nda_required_keys_match_actual_field_names(self):
        from src.doc_ai.schema_config import get_required_fields
        assert get_required_fields("nda") == {"disclosing_party", "receiving_party", "agreement_date"}

    def test_business_doc_required_keys(self):
        from src.doc_ai.schema_config import get_required_fields
        assert get_required_fields("business_doc") == {"company_name"}

    def test_unknown_type_returns_empty(self):
        from src.doc_ai.schema_config import get_required_fields
        assert get_required_fields("future_type") == frozenset()


class TestRequiredFieldPassesByDocType:
    """_has_required_field_passes should use per-type required fields."""

    def _make_checks(self, fields: list[str]) -> list:
        from src.doc_ai.schemas import ValidationCheck
        return [ValidationCheck(field=f, status="pass", message="ok") for f in fields]

    def test_invoice_requires_four_fields(self):
        from src.doc_ai.pipeline import DocumentPipeline
        checks = self._make_checks(["vendor_name", "invoice_number", "invoice_date", "total_amount"])
        assert DocumentPipeline._has_required_field_passes(checks, "invoice") is True

    def test_invoice_fails_without_invoice_number(self):
        from src.doc_ai.pipeline import DocumentPipeline
        checks = self._make_checks(["vendor_name", "invoice_date", "total_amount"])
        assert DocumentPipeline._has_required_field_passes(checks, "invoice") is False

    def test_business_doc_only_requires_company_name(self):
        from src.doc_ai.pipeline import DocumentPipeline
        # Only company_name needed — no invoice fields required
        checks = self._make_checks(["company_name"])
        assert DocumentPipeline._has_required_field_passes(checks, "business_doc") is True

    def test_business_doc_fails_without_company_name(self):
        from src.doc_ai.pipeline import DocumentPipeline
        checks = self._make_checks(["document_subtype", "report_period"])
        assert DocumentPipeline._has_required_field_passes(checks, "business_doc") is False

    def test_nda_requires_both_parties(self):
        from src.doc_ai.pipeline import DocumentPipeline
        checks = self._make_checks(["disclosing_party", "receiving_party", "agreement_date"])
        assert DocumentPipeline._has_required_field_passes(checks, "nda") is True

    def test_nda_fails_with_only_one_party(self):
        from src.doc_ai.pipeline import DocumentPipeline
        checks = self._make_checks(["disclosing_party"])
        assert DocumentPipeline._has_required_field_passes(checks, "nda") is False


class TestInferMissingFieldsDocTypeGating:
    """_infer_missing_fields must not stamp invoice fields onto non-invoice documents."""

    def test_invoice_infers_currency_and_date(self):
        from src.doc_ai.extractors import _infer_missing_fields
        extracted = {"document_type": "invoice", "currency": None, "invoice_date": None}
        result = _infer_missing_fields("Total: $1,200.00  Date: 2026-01-15", extracted)
        assert result.get("currency") == "USD"
        assert result.get("invoice_date") == "2026-01-15"

    def test_business_doc_gets_no_invoice_fields(self):
        from src.doc_ai.extractors import _infer_missing_fields
        # Raw text contains dates and dollar signs — should not infer invoice fields
        extracted = {"document_type": "business_doc", "company_name": "Acme Corp"}
        result = _infer_missing_fields("Revenue: $500,000  Period ending 2026-03-31", extracted)
        assert result == {}

    def test_medical_discharge_gets_no_invoice_fields(self):
        from src.doc_ai.extractors import _infer_missing_fields
        extracted = {"document_type": "medical_discharge", "patient_name": "Jane Doe"}
        result = _infer_missing_fields("Discharged: 2026-02-10  Bill: $250.00", extracted)
        assert result == {}

    def test_unknown_type_defaults_to_no_infer(self):
        from src.doc_ai.extractors import _infer_missing_fields
        extracted = {"document_type": "future_type"}
        result = _infer_missing_fields("$999.00  INV-123  2026-01-01", extracted)
        assert result == {}


class TestZoneDensityFingerprint:
    """Zone-density fingerprint and cosine similarity for layout matching."""

    def _make_layout(self, word_positions: list[tuple[str, float, float]]) -> "SpatialLayout":
        from src.doc_ai.spatial_extractor import SpatialWord, SpatialRow, SpatialLayout
        rows = []
        for text, x, y in word_positions:
            w = SpatialWord(text=text, x0=x, top=y, x1=x + len(text) * 6, bottom=y + 10)
            rows.append(SpatialRow(words=[w]))
        return SpatialLayout(page_width=600.0, page_height=800.0, rows=rows, page_number=0)

    def test_build_zone_density_returns_correct_length(self):
        from src.doc_ai.spatial_extractor import build_zone_density, _ZONE_COLS, _ZONE_ROWS
        layout = self._make_layout([("Hello", 100, 50)])
        density = build_zone_density([layout])
        assert len(density) == _ZONE_COLS * _ZONE_ROWS

    def test_build_zone_density_empty_layouts(self):
        from src.doc_ai.spatial_extractor import build_zone_density, _ZONE_COLS, _ZONE_ROWS
        density = build_zone_density([])
        assert density == [0.0] * (_ZONE_COLS * _ZONE_ROWS)
        assert all(v == 0.0 for v in density)

    def test_build_zone_density_values_in_range(self):
        from src.doc_ai.spatial_extractor import build_zone_density
        layout = self._make_layout([
            ("Company Name", 50, 60),
            ("Total Amount", 300, 400),
            ("Invoice Number", 50, 200),
        ])
        density = build_zone_density([layout])
        assert all(0.0 <= v <= 1.0 for v in density)
        assert max(density) == 1.0  # peak zone is always 1.0

    def test_cosine_similarity_identical_vectors(self):
        from src.doc_ai.spatial_extractor import cosine_similarity
        v = [0.5, 0.0, 1.0, 0.3, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=0.001)

    def test_cosine_similarity_orthogonal_vectors(self):
        from src.doc_ai.spatial_extractor import cosine_similarity
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=0.001)

    def test_cosine_similarity_mismatched_lengths(self):
        from src.doc_ai.spatial_extractor import cosine_similarity
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_same_layout_different_content_scores_high(self):
        """Two docs with identical layout but different company names should match well."""
        from src.doc_ai.template_memory import TemplateMemory
        # Simulate two business reports: same word positions, different text
        layout_a = self._make_layout([
            ("Phoenix Energy Solutions", 50, 60),
            ("Project Status Update", 200, 110),
            ("Prepared by: Alice Smith", 50, 515),
        ])
        layout_b = self._make_layout([
            ("Meridian Cloud Inc", 50, 60),
            ("Strategic Initiative Briefing", 200, 110),
            ("Prepared by: Bob Jones", 50, 515),
        ])
        lines_a = ["Phoenix Energy Solutions", "Project Status Update", "Prepared by: Alice Smith"]
        lines_b = ["Meridian Cloud Inc", "Strategic Initiative Briefing", "Prepared by: Bob Jones"]
        sig_a = TemplateMemory.build_signature(lines_a, layouts=[layout_a])
        sig_b = TemplateMemory.build_signature(lines_b, layouts=[layout_b])
        score = TemplateMemory._score_signature(sig_a, sig_b)
        # Zone density (spatial structure) is identical so score should be high
        assert score >= 0.80, f"Expected >= 0.80, got {score}"

    def test_different_layouts_score_low(self):
        """An invoice layout vs a sparse one-page letter should score low."""
        from src.doc_ai.template_memory import TemplateMemory
        # Dense invoice: words spread across many zones
        invoice_positions = [(f"word{i}", (i % 8) * 70 + 10, (i // 8) * 80 + 20) for i in range(40)]
        layout_invoice = self._make_layout(invoice_positions)
        # Sparse letter: all words clustered in the top-left zone
        layout_letter = self._make_layout([("Dear", 10, 10), ("Sir", 10, 25), ("Regards", 10, 40)])
        sig_invoice = TemplateMemory.build_signature(["invoice total due"], layouts=[layout_invoice])
        sig_letter = TemplateMemory.build_signature(["Dear Sir Regards"], layouts=[layout_letter])
        score = TemplateMemory._score_signature(sig_invoice, sig_letter)
        assert score < 0.70, f"Expected < 0.70, got {score}"

    def test_signature_without_layouts_scores_zero(self):
        """Signatures with no zone_density (text-only upload) score 0.0 — no layout to compare."""
        from src.doc_ai.template_memory import TemplateMemory
        sig_a = TemplateMemory.build_signature(["invoice number total amount due"])
        sig_b = TemplateMemory.build_signature(["invoice number total amount due"])
        assert sig_a == {"zone_density": []}
        score = TemplateMemory._score_signature(sig_a, sig_b)
        assert score == 0.0


class TestNeedsLlmFallback:
    """_needs_llm_fallback must use per-type required fields, not hardcoded invoice fields."""

    def test_invoice_fallback_when_missing_required(self):
        from src.doc_ai.extractors import _needs_llm_fallback
        extracted = {"document_type": "invoice", "vendor_name": "Acme", "invoice_number": None,
                     "invoice_date": "2026-01-01", "total_amount": 100.0}
        assert _needs_llm_fallback(extracted) is True

    def test_invoice_no_fallback_when_complete(self):
        from src.doc_ai.extractors import _needs_llm_fallback
        extracted = {"document_type": "invoice", "vendor_name": "Acme", "invoice_number": "INV-1",
                     "invoice_date": "2026-01-01", "total_amount": 100.0}
        assert _needs_llm_fallback(extracted) is False

    def test_business_doc_no_fallback_with_company_name(self):
        from src.doc_ai.extractors import _needs_llm_fallback
        # business_doc only needs company_name — invoice fields being None must not trigger fallback
        extracted = {"document_type": "business_doc", "company_name": "Acme Corp",
                     "vendor_name": None, "invoice_number": None, "total_amount": None}
        assert _needs_llm_fallback(extracted) is False

    def test_business_doc_fallback_without_company_name(self):
        from src.doc_ai.extractors import _needs_llm_fallback
        extracted = {"document_type": "business_doc", "company_name": None}
        assert _needs_llm_fallback(extracted) is True

    def test_medical_discharge_no_fallback_when_complete(self):
        from src.doc_ai.extractors import _needs_llm_fallback
        extracted = {
            "document_type": "medical_discharge",
            "patient_name": "Jane Doe",
            "admission_date": "2026-01-10",
            "discharge_date": "2026-01-15",
            "primary_diagnosis": "Pneumonia",
        }
        assert _needs_llm_fallback(extracted) is False


class TestBuildAnchorsQuality:
    """_build_anchors must not create anchors that would extract wrong data."""

    def _make_template_memory(self, tmp_path):
        from src.doc_ai.template_memory import TemplateMemory
        return TemplateMemory(tmp_path / "templates.json")

    def test_anchor_rejected_when_captured_value_differs(self, tmp_path):
        """Value buried in a compound line must not become an anchor for that field."""
        tm = self._make_template_memory(tmp_path)
        # "2025-04-24" appears after the "Prepared by" value — pattern would capture the wrong text
        lines = ["Prepared by: Amanda Thomas | Date: 2025-04-24"]
        extracted = {"document_type": "business_doc", "report_date": "2025-04-24"}
        anchors = tm._build_anchors(extracted, lines)
        assert "report_date" not in anchors

    def test_anchor_accepted_when_value_is_sole_content(self, tmp_path):
        """A clean label: value line creates a valid anchor."""
        tm = self._make_template_memory(tmp_path)
        lines = ["Report Date: 2025-04-24"]
        extracted = {"document_type": "business_doc", "report_date": "2025-04-24"}
        anchors = tm._build_anchors(extracted, lines)
        assert "report_date" in anchors
        assert "(?P<value>" in anchors["report_date"]["pattern"]

    def test_duplicate_pattern_only_used_once(self, tmp_path):
        """When two fields would share the same anchor pattern, only the first gets it."""
        tm = self._make_template_memory(tmp_path)
        # Both fields have values that produce the same label:pattern
        lines = ["Period Ending: Q1 2025"]
        extracted = {
            "document_type": "business_doc",
            "report_period": "Q1 2025",
            "report_date": "Q1 2025",  # same value on same line
        }
        anchors = tm._build_anchors(extracted, lines)
        # Exactly one field should own the pattern
        assert sum(1 for f in ("report_period", "report_date") if f in anchors) == 1

    def test_no_anchor_when_value_not_in_any_line(self, tmp_path):
        """Fields whose values don't appear in any line produce no anchor."""
        tm = self._make_template_memory(tmp_path)
        lines = ["Company Name: Acme Corp"]
        extracted = {"document_type": "business_doc", "report_date": "2025-04-24"}
        anchors = tm._build_anchors(extracted, lines)
        assert "report_date" not in anchors


class TestExtractFromTemplateDocTypeGating:
    """_extract_from_template must not write invoice-only fields to non-invoice results."""

    def _make_invoice_template(self) -> dict:
        return {
            "document_type": "invoice",
            "anchors": {"vendor_name": {"pattern": r"Vendor\s*:\s*(?P<value>.+)"}},
        }

    def _make_business_template(self) -> dict:
        return {
            "document_type": "business_doc",
            "anchors": {"company_name": {"pattern": r"Company\s*:\s*(?P<value>.+)"}},
        }

    def test_invoice_template_writes_line_items(self):
        from src.doc_ai.extractors import _extract_from_template
        text = "Vendor: Acme Corp\nItem  Description  Qty  Price\nWidget A  10  $5.00"
        result = _extract_from_template(self._make_invoice_template(), text)
        assert "line_items" in result

    def test_non_invoice_template_does_not_write_line_items(self):
        from src.doc_ai.extractors import _extract_from_template
        text = "Company: Acme Corp\nItem  Description  Qty  Price\nWidget A  10  $5.00"
        result = _extract_from_template(self._make_business_template(), text)
        assert "line_items" not in result


class TestSpatialAnchorLabelClearing:
    """Pass 2 spatial anchors must not store a label belonging to a different field."""

    def _make_layout(self, rows_of_words: list[list[tuple[str, float, float]]]) -> "SpatialLayout":
        """Each inner list becomes one SpatialRow (words grouped by caller)."""
        from src.doc_ai.spatial_extractor import SpatialWord, SpatialRow, SpatialLayout
        rows = []
        for word_list in rows_of_words:
            words = [
                SpatialWord(text=t, x0=x, top=y, x1=x + len(t) * 6, bottom=y + 10)
                for t, x, y in word_list
            ]
            rows.append(SpatialRow(words=words))
        return SpatialLayout(page_width=600.0, page_height=800.0, rows=rows, page_number=0)

    def test_foreign_label_not_stored(self):
        """If the row containing the value has 'prepared by' label, report_date anchor is skipped."""
        from src.doc_ai.spatial_extractor import build_spatial_anchors
        # Single row: "Prepared by: 2025-04-24" — label "prepared by" maps to prepared_by, not report_date
        layout = self._make_layout([[
            ("Prepared", 10, 100), ("by:", 75, 100), ("2025-04-24", 130, 100),
        ]])
        extracted = {"document_type": "business_doc", "report_date": "2025-04-24"}
        anchors = build_spatial_anchors([layout], extracted)
        # The anchor must NOT be stored — position-only anchors without a label are unreliable
        date_anchor = next((a for a in anchors if a["field"] == "report_date"), None)
        assert date_anchor is None

    def test_matching_label_kept(self):
        """If the row label maps to the correct field, label_text is preserved."""
        from src.doc_ai.spatial_extractor import build_spatial_anchors
        # Single row: "Prepared by: Amanda Thomas" — label "prepared by" maps to prepared_by ✓
        layout = self._make_layout([[
            ("Prepared", 10, 200), ("by:", 75, 200), ("Amanda", 130, 200), ("Thomas", 195, 200),
        ]])
        extracted = {"document_type": "business_doc", "prepared_by": "Amanda Thomas"}
        anchors = build_spatial_anchors([layout], extracted)
        prep_anchor = next((a for a in anchors if a["field"] == "prepared_by"), None)
        assert prep_anchor is not None
        assert prep_anchor["label_text"] == "prepared by"


class TestDeriveBadPattern:
    """_derive_bad_pattern must generalise repeated-char values and exact-match others."""

    def test_all_underscores_generalises(self):
        from src.doc_ai.template_memory import _derive_bad_pattern
        import re
        pat = _derive_bad_pattern("___________________________")
        assert re.search(pat, "___", re.IGNORECASE)
        assert re.search(pat, "_____________", re.IGNORECASE)
        assert not re.search(pat, "John Smith", re.IGNORECASE)

    def test_all_dashes_generalises(self):
        from src.doc_ai.template_memory import _derive_bad_pattern
        import re
        pat = _derive_bad_pattern("-----")
        assert re.search(pat, "---", re.IGNORECASE)
        assert not re.search(pat, "Approved", re.IGNORECASE)

    def test_mixed_value_exact_match(self):
        from src.doc_ai.template_memory import _derive_bad_pattern
        import re
        pat = _derive_bad_pattern("QUARTERLY PERFORMANCE REVIEW")
        assert re.search(pat, "QUARTERLY PERFORMANCE REVIEW", re.IGNORECASE)
        assert re.search(pat, "quarterly performance review", re.IGNORECASE)
        assert not re.search(pat, "ANNUAL PERFORMANCE REVIEW", re.IGNORECASE)


class TestBadPatternStore:
    """BadPatternStore learns from corrections and cleans future extractions."""

    def test_add_pattern_persisted(self, tmp_path):
        from src.doc_ai.template_memory import BadPatternStore
        store = BadPatternStore(tmp_path / "bad_patterns.json")
        pattern = store.add_pattern("business_doc", "approved_by", "___________________________")
        assert pattern is not None
        loaded = store.load()
        assert "business_doc.approved_by" in loaded
        assert pattern in loaded["business_doc.approved_by"]

    def test_duplicate_pattern_not_added_twice(self, tmp_path):
        from src.doc_ai.template_memory import BadPatternStore
        store = BadPatternStore(tmp_path / "bad_patterns.json")
        store.add_pattern("business_doc", "approved_by", "___")
        store.add_pattern("business_doc", "approved_by", "_______")  # same generalised pattern
        loaded = store.load()
        assert len(loaded["business_doc.approved_by"]) == 1

    def test_apply_clears_matching_field(self, tmp_path):
        from src.doc_ai.template_memory import BadPatternStore
        store = BadPatternStore(tmp_path / "bad_patterns.json")
        store.add_pattern("business_doc", "approved_by", "___")
        extracted = {"document_type": "business_doc", "approved_by": "__________", "company_name": "Acme"}
        cleared = store.apply(extracted)
        assert "approved_by" in cleared
        assert extracted["approved_by"] is None
        assert extracted["company_name"] == "Acme"

    def test_apply_skips_non_matching_value(self, tmp_path):
        from src.doc_ai.template_memory import BadPatternStore
        store = BadPatternStore(tmp_path / "bad_patterns.json")
        store.add_pattern("business_doc", "approved_by", "___")
        extracted = {"document_type": "business_doc", "approved_by": "Jane Smith"}
        cleared = store.apply(extracted)
        assert "approved_by" not in cleared
        assert extracted["approved_by"] == "Jane Smith"

    def test_apply_skips_none_values(self, tmp_path):
        from src.doc_ai.template_memory import BadPatternStore
        store = BadPatternStore(tmp_path / "bad_patterns.json")
        store.add_pattern("business_doc", "approved_by", "___")
        extracted = {"document_type": "business_doc", "approved_by": None}
        cleared = store.apply(extracted)
        assert cleared == []

    def test_pattern_scoped_to_doc_type(self, tmp_path):
        from src.doc_ai.template_memory import BadPatternStore
        store = BadPatternStore(tmp_path / "bad_patterns.json")
        store.add_pattern("business_doc", "approved_by", "___")
        # Same field value in a different doc type must NOT be cleared
        extracted = {"document_type": "nda", "approved_by": "___"}
        cleared = store.apply(extracted)
        assert cleared == []

    def test_empty_rejected_value_returns_none(self, tmp_path):
        from src.doc_ai.template_memory import BadPatternStore
        store = BadPatternStore(tmp_path / "bad_patterns.json")
        result = store.add_pattern("business_doc", "approved_by", "")
        assert result is None


class TestFinalizeReviewLearnsPatterns:
    """finalize_review must learn bad patterns when original fields are cleared."""

    def test_cleared_field_adds_bad_pattern(self, tmp_path):
        from src.doc_ai.pipeline import DocumentPipeline
        from src.doc_ai.config import Settings
        from src.doc_ai.template_memory import BadPatternStore
        settings = Settings(
            app_env="test",
            data_dir=tmp_path,
            upload_dir=tmp_path / "uploads",
            output_dir=tmp_path / "outputs",
            database_path=tmp_path / "db.db",
            template_store_path=tmp_path / "templates.json",
            promoted_template_store_path=tmp_path / "promoted.json",
            bad_patterns_path=tmp_path / "bad_patterns.json",
            review_export_dir=tmp_path / "exports",
            enable_template_learning=False,
            min_learning_pass_ratio=0.6,
            llm_provider="openai",
            llm_base_url=None,
            openai_api_key=None,
            openai_model="gpt-4.1-mini",
        )
        (tmp_path / "uploads").mkdir()
        (tmp_path / "outputs").mkdir()
        (tmp_path / "exports").mkdir()
        upload_path = tmp_path / "uploads" / "doc.txt"
        upload_path.write_text("Company: Acme")

        pipeline = DocumentPipeline(settings)
        original = {"document_type": "business_doc", "company_name": "Acme Corp", "approved_by": "___________"}
        corrected = {"document_type": "business_doc", "company_name": "Acme Corp", "approved_by": None}

        pipeline.finalize_review(
            source_file="doc.txt",
            upload_path=str(upload_path),
            parsed_text="Company: Acme",
            corrected_data=corrected,
            original_extracted=original,
        )

        store = BadPatternStore(settings.bad_patterns_path)
        loaded = store.load()
        assert "business_doc.approved_by" in loaded

    def test_unchanged_field_not_stored(self, tmp_path):
        from src.doc_ai.pipeline import DocumentPipeline
        from src.doc_ai.config import Settings
        from src.doc_ai.template_memory import BadPatternStore
        settings = Settings(
            app_env="test",
            data_dir=tmp_path,
            upload_dir=tmp_path / "uploads",
            output_dir=tmp_path / "outputs",
            database_path=tmp_path / "db.db",
            template_store_path=tmp_path / "templates.json",
            promoted_template_store_path=tmp_path / "promoted.json",
            bad_patterns_path=tmp_path / "bad_patterns.json",
            review_export_dir=tmp_path / "exports",
            enable_template_learning=False,
            min_learning_pass_ratio=0.6,
            llm_provider="openai",
            llm_base_url=None,
            openai_api_key=None,
            openai_model="gpt-4.1-mini",
        )
        (tmp_path / "uploads").mkdir()
        (tmp_path / "outputs").mkdir()
        (tmp_path / "exports").mkdir()
        upload_path = tmp_path / "uploads" / "doc.txt"
        upload_path.write_text("Company: Acme")

        pipeline = DocumentPipeline(settings)
        original = {"document_type": "business_doc", "company_name": "Acme Corp"}
        corrected = {"document_type": "business_doc", "company_name": "Acme Corp"}

        pipeline.finalize_review(
            source_file="doc.txt",
            upload_path=str(upload_path),
            parsed_text="Company: Acme",
            corrected_data=corrected,
            original_extracted=original,
        )

        store = BadPatternStore(settings.bad_patterns_path)
        assert store.load() == {}


class TestFinalizeReviewPreservesProcessingTrace:
    """finalize_review must not wipe the original processing trace from the DB."""

    def _settings(self, tmp_path):
        from src.doc_ai.config import Settings
        (tmp_path / "uploads").mkdir()
        (tmp_path / "outputs").mkdir()
        (tmp_path / "exports").mkdir()
        return Settings(
            app_env="test",
            data_dir=tmp_path,
            upload_dir=tmp_path / "uploads",
            output_dir=tmp_path / "outputs",
            database_path=tmp_path / "db.db",
            template_store_path=tmp_path / "templates.json",
            promoted_template_store_path=tmp_path / "promoted.json",
            bad_patterns_path=tmp_path / "bad_patterns.json",
            review_export_dir=tmp_path / "exports",
            enable_template_learning=False,
            min_learning_pass_ratio=0.6,
            llm_provider="openai",
            llm_base_url=None,
            openai_api_key=None,
            openai_model="gpt-4.1-mini",
        )

    def test_processing_trace_survives_finalize_review(self, tmp_path):
        # Regression: finalize_review used to overwrite extraction_traces with only review
        # steps, discarding "Used the LLM reasoning layer" and similar processing messages.
        from src.doc_ai.pipeline import DocumentPipeline
        from src.doc_ai.storage import ResultStore
        from src.doc_ai.schemas import ValidationCheck
        settings = self._settings(tmp_path)
        upload_path = tmp_path / "uploads" / "doc.txt"
        upload_path.write_text("Company: Acme")

        store = ResultStore(settings)
        # Simulate a processing run that used the LLM.
        store.persist(
            "doc.txt",
            {"document_type": "business_doc", "company_name": "Acme Corp"},
            [ValidationCheck(field="company_name", status="pass", message="ok")],
            ["Detected document type: business_doc.", "Used the LLM reasoning layer as baseline (no matching template found)."],
            content_hash="abc123",
        )

        pipeline = DocumentPipeline(settings)
        pipeline.finalize_review(
            source_file="doc.txt",
            upload_path=str(upload_path),
            parsed_text="Company: Acme",
            corrected_data={"document_type": "business_doc", "company_name": "Acme Corp"},
        )

        final_trace = store.get_processing_trace("doc.txt")
        llm_steps = [s for s in final_trace if "llm" in s.lower()]
        assert llm_steps, "LLM processing step must be preserved after finalize_review"
        assert any("human-reviewed" in s.lower() for s in final_trace), "Review step must also be present"
