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
        assert _parse_retry_after(exc) == 30

    def test_parse_retry_after_returns_none_for_non_rate_limit(self):
        from src.doc_ai.extractors import _parse_retry_after

        exc = Exception("Invalid API key")
        assert _parse_retry_after(exc) is None

    def test_rate_limit_retry_carries_retry_after(self):
        from src.doc_ai.extractors import RateLimitRetry

        rl = RateLimitRetry("too many requests", retry_after=45)
        assert rl.retry_after == 45
        assert "too many requests" in str(rl)

    def test_parse_retry_after_caps_large_header_value(self):
        """Provider headers like x-ratelimit-reset-requests can return 500+s; cap at 90."""
        from src.doc_ai.extractors import _MAX_RETRY_AFTER_SECONDS, _parse_retry_after

        class FakeResponse:
            status_code = 429
            headers = {"x-ratelimit-reset-requests": "552"}

        class FakeExc(Exception):
            response = FakeResponse()

        result = _parse_retry_after(FakeExc("rate limit"))
        assert result == _MAX_RETRY_AFTER_SECONDS

    def test_parse_retry_after_caps_large_message_value(self):
        from src.doc_ai.extractors import _MAX_RETRY_AFTER_SECONDS, _parse_retry_after

        exc = Exception("Please try again in 600s.")
        result = _parse_retry_after(exc)
        assert result == _MAX_RETRY_AFTER_SECONDS


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
        assert result["document_subtype"] == "PROJECT STATUS UPDATE"

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


def _fixture_pdfs():
    if not FIXTURES.exists():
        return []
    return list(FIXTURES.glob("*.pdf")) + list(FIXTURES.glob("*.txt"))


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

    FIXTURE = FIXTURES / "invoice_001.pdf"

    def test_extract_spatial_layout_returns_data_for_pdf(self):
        pytest.importorskip("pdfplumber")
        if not self.FIXTURE.exists():
            pytest.skip("invoice_001.pdf fixture not present")
        from src.doc_ai.spatial_extractor import extract_spatial_layout
        layouts = extract_spatial_layout(self.FIXTURE)
        assert isinstance(layouts, list)
        assert len(layouts) >= 1
        assert layouts[0].page_width > 0
        assert layouts[0].page_height > 0

    def test_extract_fields_from_pdf_fixture_gets_some_fields(self):
        pytest.importorskip("pdfplumber")
        if not self.FIXTURE.exists():
            pytest.skip("invoice_001.pdf fixture not present")
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
        sig = TemplateMemory.build_signature(lines)
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
