from __future__ import annotations

import json
import re
from typing import Any

from .config import Settings
from .schemas import ParsedDocument
from .template_memory import TemplateMemory


class ExtractionError(RuntimeError):
    pass


class RateLimitRetry(Exception):
    """Raised when the LLM provider returns a 429 rate-limit response."""
    def __init__(self, message: str, retry_after: int) -> None:
        super().__init__(message)
        self.retry_after = retry_after


_MAX_RETRY_AFTER_SECONDS = 90


def _parse_retry_after(exc: Exception) -> int | None:
    """Extract the suggested wait time (seconds) from a rate-limit error, or None if not a 429.

    Caps the returned value at _MAX_RETRY_AFTER_SECONDS: provider headers like
    x-ratelimit-reset-requests can report the full minute-window reset time
    (e.g. 552s), which is impractical for interactive use.  If the provider
    still rejects after the cap we retry again — that is preferable to blocking
    the UI for 9+ minutes.
    """
    # openai-sdk attaches the raw httpx response
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {})
        for header in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            val = headers.get(header)
            if val:
                try:
                    return min(_MAX_RETRY_AFTER_SECONDS, max(1, int(float(val))))
                except (ValueError, TypeError):
                    pass
        status = getattr(response, "status_code", None)
        if status == 429:
            pass  # fall through to message parsing

    msg = str(exc)
    # "Please try again in 1.5s" or "try again in 30s"
    m = re.search(r"try again in (\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
    if m:
        return min(_MAX_RETRY_AFTER_SECONDS, max(1, int(float(m.group(1))) + 1))
    # "retry after 30 seconds"
    m = re.search(r"retry.{0,15}?(\d+)\s*second", msg, re.IGNORECASE)
    if m:
        return min(_MAX_RETRY_AFTER_SECONDS, int(m.group(1)))
    # Any mention of 429
    if "429" in msg or "rate limit" in msg.lower() or "resource_exhausted" in msg.lower():
        return 30
    return None


class BaseExtractor:
    def extract(self, parsed_document: ParsedDocument) -> dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Document type detection
# ---------------------------------------------------------------------------

_DOC_TYPE_SIGNALS: dict[str, list[str]] = {
    "invoice": [
        "invoice number", "invoice no", "invoice #", "bill to", "vendor",
        "subtotal", "total amount", "payment terms", "amount due", "purchase order",
    ],
    "medical_discharge": [
        "discharge summary", "discharge date", "admission date", "primary diagnosis",
        "diagnosis", "patient name", "date of birth", "treating physician",
        "attending physician", "medications", "follow-up", "hospital", "clinical",
        "discharge condition", "discharge instructions",
    ],
    "nda": [
        "non-disclosure", "nondisclosure", "confidentiality agreement", "nda",
        "disclosing party", "receiving party", "proprietary information",
        "trade secret", "confidential information", "governing law",
        "mutual non-disclosure",
    ],
    "lab_report": [
        "laboratory report", "lab report", "ordering physician", "accession",
        "specimen type", "reference range", "clinical interpretation",
        "complete blood count", "cbc", "comprehensive metabolic", "cmp",
        "pathology", "clia", "mrn", "reported", "collected",
    ],
    "business_doc": [
        "project status", "executive summary", "key performance indicators",
        "kpi", "strategic recommendations", "ebitda", "compliance audit",
        "board report", "quarterly report", "annual report", "period ending",
        "prepared by", "approved by", "document classification",
    ],
}


def detect_document_type(text: str) -> str:
    lower = text.lower()
    scores = {
        doc_type: sum(1 for kw in keywords if kw in lower)
        for doc_type, keywords in _DOC_TYPE_SIGNALS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "invoice"


# ---------------------------------------------------------------------------
# Empty schemas per document type
# ---------------------------------------------------------------------------

_LINE_ITEM_RE = re.compile(
    r"^(?P<description>[A-Za-z][^\n]{2,60}?)\s+"
    r"(?P<qty>\d+)\s+"
    r"\$?(?P<unit_price>[\d,]+\.\d{2})\s+"
    r"\$?(?P<total>[\d,]+\.\d{2})\s*$",
    re.MULTILINE,
)
_LINE_ITEM_SKIP = {"item", "description", "product", "service", "qty", "quantity"}


def _extract_line_items(raw_text: str) -> list[dict[str, Any]]:
    items = []
    for m in _LINE_ITEM_RE.finditer(raw_text):
        description = m.group("description").strip()
        if description.lower() in _LINE_ITEM_SKIP:
            continue
        try:
            items.append({
                "description": description,
                "quantity": int(m.group("qty")),
                "unit_price": float(m.group("unit_price").replace(",", "")),
                "total": float(m.group("total").replace(",", "")),
            })
        except ValueError:
            continue
    return items


class RuleBasedMedicalDischargeExtractor(BaseExtractor):
    FIELD_PATTERNS = {
        "facility_name": [r"(?:Hospital|Clinic|Medical Center|Health System)\s*:\s*(?P<value>.+)"],
        "patient_name": [
            r"Patient\s*(?:Name)?\s*:\s*(?P<value>[A-Za-z ,]+)",
            r"Name\s*:\s*(?P<value>[A-Za-z ,]+)",
        ],
        "date_of_birth": [
            r"(?:Date\s*of\s*Birth|DOB)\s*:\s*(?P<value>[\d/\-]+)",
        ],
        "admission_date": [
            r"(?:Admission|Admitted|Admit)\s*Date\s*:\s*(?P<value>[\d/\-]+)",
            r"Date\s*(?:of\s*)?Admission\s*:\s*(?P<value>[\d/\-]+)",
        ],
        "discharge_date": [
            r"Discharge\s*Date\s*:\s*(?P<value>[\d/\-]+)",
            r"Date\s*(?:of\s*)?Discharge\s*:\s*(?P<value>[\d/\-]+)",
        ],
        "primary_diagnosis": [
            r"Primary\s*Diagnosis\s*:\s*(?P<value>.+)",
            r"Diagnosis\s*:\s*(?P<value>.+)",
            r"Principal\s*Diagnosis\s*:\s*(?P<value>.+)",
        ],
        "treating_physician": [
            r"(?:Treating|Attending|Discharge)\s*Physician\s*:\s*(?P<value>.+)",
            r"Physician\s*:\s*(?P<value>.+)",
            r"Dr\.?\s*(?P<value>[A-Za-z ,]+)",
        ],
        "discharge_condition": [
            r"(?:Discharge\s*)?Condition\s*:\s*(?P<value>.+)",
            r"Condition\s*(?:at\s*Discharge)?\s*:\s*(?P<value>.+)",
        ],
        "follow_up_date": [
            r"Follow[\s\-]*[Uu]p\s*(?:Date|Appointment)?\s*:\s*(?P<value>[\d/\-]+)",
        ],
    }

    def extract(self, parsed_document: ParsedDocument) -> dict[str, Any]:
        text = parsed_document.raw_text
        extracted = _empty_medical_discharge(parsed_document.file_name)

        for field, patterns in self.FIELD_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted[field] = match.group("value").strip()
                    break

        # Extract medications as a list (lines following "Medications:" header)
        med_match = re.search(r"Medications?\s*:(.*?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
        if med_match:
            lines = [l.strip(" -•\t") for l in med_match.group(1).splitlines() if l.strip(" -•\t")]
            extracted["medications"] = [l for l in lines if l]

        # Extract discharge instructions block
        instr_match = re.search(r"Discharge\s*Instructions?\s*:(.*?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
        if instr_match:
            extracted["discharge_instructions"] = instr_match.group(1).strip()

        return extracted


class RuleBasedNDAExtractor(BaseExtractor):
    FIELD_PATTERNS = {
        "agreement_date": [
            r"(?:Agreement|Effective)\s*Date\s*:\s*(?P<value>[\w ,]+\d{4})",
            r"dated\s+(?:as\s+of\s+)?(?P<value>[\w ,]+\d{4})",
        ],
        "effective_date": [
            r"Effective\s*Date\s*:\s*(?P<value>[\w ,]+\d{4})",
        ],
        "expiration_date": [
            r"(?:Expir|Terminat|End)\s*(?:ation)?\s*Date\s*:\s*(?P<value>[\w ,]+\d{4})",
            r"(?:shall\s+expire|expires)\s+on\s+(?P<value>[\w ,]+\d{4})",
        ],
        "disclosing_party": [
            r"Disclosing\s*Party\s*:\s*(?P<value>[^\n\)]+)",
            r'"Discloser"\s*(?:means|shall mean)\s+(?P<value>[^\n,]+)',
            r'(?P<value>[A-Za-z][\w\s,\.]+?)\s*\(?["\']?Disclosing\s*Party["\']?\)?',
        ],
        "receiving_party": [
            r"Receiving\s*Party\s*:\s*(?P<value>[^\n\)]+)",
            r'"Recipient"\s*(?:means|shall mean)\s+(?P<value>[^\n,]+)',
            r'(?P<value>[A-Za-z][\w\s,\.]+?)\s*\(?["\']?Receiving\s*Party["\']?\)?',
        ],
        "governing_law": [
            r"governed\s+by\s+(?:the\s+laws?\s+of\s+)?(?P<value>[^\n,.]+)",
            r"Governing\s*Law\s*:\s*(?P<value>[^\n]+)",
        ],
        "confidentiality_period": [
            r"(?:period|term)\s+of\s+(?P<value>\d+\s*(?:year|month|day)s?)",
            r"(?:for\s+a\s+period\s+of\s+)(?P<value>\d+\s*(?:year|month|day)s?)",
        ],
    }

    def extract(self, parsed_document: ParsedDocument) -> dict[str, Any]:
        text = parsed_document.raw_text
        extracted = _empty_nda(parsed_document.file_name)

        for field, patterns in self.FIELD_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted[field] = match.group("value").strip()
                    break

        lower = text.lower()
        if "mutual" in lower:
            extracted["agreement_type"] = "mutual"
        elif "one-way" in lower or "one way" in lower or "unilateral" in lower:
            extracted["agreement_type"] = "one-way"

        return extracted


class RuleBasedLabReportExtractor(BaseExtractor):
    FIELD_PATTERNS = {
        "patient_name": [
            r"Patient\s*:\s*(?P<value>[A-Za-z][A-Za-z ,\.]+?)(?:\s+DOB|\s+MRN|\s*$)",
        ],
        "date_of_birth": [
            r"DOB\s*:\s*(?P<value>[A-Za-z0-9 ,]+?)(?:\s+\(Age|\s*$)",
            r"Date\s+of\s+Birth\s*:\s*(?P<value>[A-Za-z0-9 ,/\-]+)",
        ],
        "mrn": [
            r"MRN\s*:\s*(?P<value>[\w\-]+)",
        ],
        "gender": [
            r"Gender\s*:\s*(?P<value>\w+)",
        ],
        "ordering_physician": [
            r"Ordering\s+Physician\s*:\s*(?P<value>[^\n\u2014]+?)(?:\s*[\u2014\|]|\s*$)",
        ],
        "ordering_specialty": [
            r"Ordering\s+Physician\s*:[^\u2014\|]+[\u2014\|]\s*(?P<value>[^\n]+)",
        ],
        "accession_number": [
            r"Accession\s*#\s*:\s*(?P<value>[\w\-]+)",
        ],
        "specimen_type": [
            r"Specimen\s+Type\s*:\s*(?P<value>[^\n]+)",
        ],
        "collected_date": [
            r"Collected\s*:\s*(?P<value>[A-Za-z0-9 ,]+?)(?:\s+at\s+[\d:]+\s*[AP]M|\s*$)",
        ],
        "reported_date": [
            r"Reported\s*:\s*(?P<value>[A-Za-z0-9 ,]+?)(?:\s+at\s+[\d:]+\s*[AP]M|\s*$)",
        ],
        "clia_number": [
            r"CLIA\s*#?\s*:\s*(?P<value>[\w]+)",
        ],
        "report_id": [
            r"Report\s+ID\s*:\s*(?P<value>[\w\-]+)",
            r"Accession\s*#\s*:\s*(?P<value>[\w\-]+)",
        ],
        "reviewing_pathologist": [
            r"Reviewed\s*(?:&|and)\s*Verified\s+by\s*:\s*(?P<value>[^|\n]+?)(?:\s*\||\s*$)",
        ],
    }

    def extract(self, parsed_document: "ParsedDocument") -> dict:
        text = parsed_document.raw_text
        extracted = _empty_lab_report(parsed_document.file_name)

        # Lab name from first non-empty line
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("CONFIDENTIAL"):
                extracted["lab_name"] = line
                break

        for field, patterns in self.FIELD_PATTERNS.items():
            for pattern in patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    extracted[field] = m.group("value").strip()
                    break

        # Clinical interpretation block
        interp_m = re.search(
            r"CLINICAL\s+INTERPRETATION[^\n]*\n(.*?)(?:\nReviewed|\Z)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if interp_m:
            extracted["clinical_interpretation"] = interp_m.group(1).strip()

        # Parse lab result rows from multi-line format: test_name / numeric_value / units
        _SKIP_LINES = {"test", "result", "units", "reference range", "flag", "reference", ""}
        panels: list[dict] = []
        abnormal: list[dict] = []
        raw_lines = [l.strip() for l in text.splitlines()]
        i = 0
        while i < len(raw_lines):
            line = raw_lines[i]
            if not line or line.lower() in _SKIP_LINES or (line == line.upper() and len(line) > 4):
                i += 1
                continue
            # Check if next line is numeric (the result value)
            if i + 1 < len(raw_lines):
                candidate_val = raw_lines[i + 1].strip()
                try:
                    float(candidate_val.replace(",", ""))
                    units = raw_lines[i + 2].strip() if i + 2 < len(raw_lines) else ""
                    entry = {
                        "test": line,
                        "value": candidate_val,
                        "units": units,
                        "reference_range": "",
                        "flag": "",
                    }
                    panels.append(entry)
                    i += 3
                    continue
                except ValueError:
                    pass
            i += 1

        extracted["lab_panels"] = panels
        extracted["abnormal_results"] = abnormal
        return extracted


class RuleBasedBusinessDocExtractor(BaseExtractor):
    FIELD_PATTERNS = {
        "report_period": [
            r"Period\s+Ending\s*:\s*(?P<value>[^\|\n]+)",
        ],
        "report_id": [
            r"Report\s+ID\s*:\s*(?P<value>[\w\-]+)",
        ],
        "report_date": [
            r"(?:Date|Dated)\s*:\s*(?P<value>[A-Za-z0-9 ,]+\d{4})",
            r"Period\s+Ending\s*:\s*(?P<value>[A-Za-z0-9 ,]+\d{4})",
        ],
        "prepared_by": [
            r"Prepared\s+by\s*:\s*(?P<value>[^\n,]+)",
        ],
        "approved_by": [
            r"Approved\s+by\s*:\s*(?P<value>[^\|\n]+?)(?:\s*\||\s*$)",
        ],
        "classification": [
            r"Document\s+Classification\s*:\s*(?P<value>[^\|\n]+)",
        ],
    }

    def extract(self, parsed_document: "ParsedDocument") -> dict:
        text = parsed_document.raw_text
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        extracted = _empty_business_doc(parsed_document.file_name)

        # Company name: first substantive non-confidential line
        content_lines = [l for l in lines if not l.startswith("CONFIDENTIAL")]
        if content_lines:
            extracted["company_name"] = content_lines[0]

        # Document subtype: first ALL-CAPS line that isn't a section header or address
        _SKIP_SUBTYPES = {"EXECUTIVE SUMMARY", "KEY PERFORMANCE INDICATORS", "STRATEGIC RECOMMENDATIONS"}
        for cl in content_lines[1:]:
            if cl == cl.upper() and len(cl) > 6 and cl not in _SKIP_SUBTYPES and "|" not in cl and not cl[0].isdigit():
                extracted["document_subtype"] = cl
                break

        for field, patterns in self.FIELD_PATTERNS.items():
            for pattern in patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    extracted[field] = m.group("value").strip()
                    break

        # Executive summary block
        exec_m = re.search(
            r"EXECUTIVE\s+SUMMARY\s*\n(.*?)(?:\n[A-Z][A-Z ]{4,}\n|\Z)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if exec_m:
            extracted["executive_summary"] = exec_m.group(1).strip()

        # KPI table: multi-line format — metric / current / prior / variance (/ status)
        kpis: list[dict] = []
        _KPI_SKIP = {"metric", "current period", "prior period", "variance", "status", ""}
        in_kpi = False
        kpi_lines: list[str] = []
        for line in lines:
            if line.upper() in ("KEY PERFORMANCE INDICATORS",):
                in_kpi = True
                continue
            if in_kpi:
                if line.upper() in ("STRATEGIC RECOMMENDATIONS", "EXECUTIVE SUMMARY") or line.startswith("CONFIDENTIAL"):
                    break
                if line.lower() not in _KPI_SKIP:
                    kpi_lines.append(line)

        # Group every 4-5 lines: metric, current, prior, variance, [status]
        j = 0
        while j < len(kpi_lines):
            metric = kpi_lines[j]
            if j + 3 < len(kpi_lines):
                try:
                    float(kpi_lines[j + 1].replace(",", "").replace("%", "").replace("/", ""))
                    kpis.append({
                        "metric": metric,
                        "current_period": kpi_lines[j + 1],
                        "prior_period": kpi_lines[j + 2],
                        "variance": kpi_lines[j + 3],
                    })
                    j += 5 if (j + 4 < len(kpi_lines) and not kpi_lines[j + 4][0].isdigit()) else 4
                    continue
                except (ValueError, IndexError):
                    pass
            j += 1
        extracted["kpis"] = kpis

        # Numbered recommendations — may all appear on a single long line
        rec_section_m = re.search(
            r"STRATEGIC\s+RECOMMENDATIONS\s*\n(.*?)(?:CONFIDENTIAL|\Z)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if rec_section_m:
            block = rec_section_m.group(1)
            # Split on "N. " to separate inline-concatenated items
            parts = re.split(r"\d+\.\s+", block)
            recs = [p.strip().rstrip(". ") for p in parts if p.strip() and len(p.strip()) > 10]
        else:
            recs = re.findall(r"\d+\.\s+(.{20,}?)(?=\s+\d+\.|\s*$)", text)
        extracted["recommendations"] = recs

        return extracted


class RuleBasedInvoiceExtractor(BaseExtractor):
    FIELD_PATTERNS = {
        "vendor_name": [
            r"Vendor\s*:\s*(?P<value>.+)",
            r"Supplier\s*:\s*(?P<value>.+)",
            r"From\s*:\s*(?P<value>.+)",
        ],
        "invoice_number": [
            r"Invoice\s*(?:Number|No\.?)\s*:\s*(?P<value>[\w\-]+)",
            r"Invoice\s*#\s*(?P<value>[\w\-]+)",
        ],
        "invoice_date": [
            r"Invoice\s*Date\s*:\s*(?P<value>[\d/\-]+)",
            r"Date\s*:\s*(?P<value>[\d/\-]+)",
        ],
        "due_date": [r"Due\s*Date\s*:\s*(?P<value>[\d/\-]+)"],
        "subtotal": [r"Subtotal\s*:?\s*\$?(?P<value>[\d,]+(?:\.\d{1,2})?)"],
        "tax": [r"Tax\s*(?:Rate)?\s*:?\s*\$?(?P<value>[\d,]+(?:\.\d{1,2})?)"],
        "shipping_handling": [
            r"Shipping\s*(?:&|and)?\s*Handling\s*:?\s*\$?(?P<value>[\d,]+(?:\.\d{1,2})?)",
            r"Shipping\s*:?\s*\$?(?P<value>[\d,]+(?:\.\d{1,2})?)",
            r"Freight\s*:?\s*\$?(?P<value>[\d,]+(?:\.\d{1,2})?)",
            r"S\s*&\s*H\s*:?\s*\$?(?P<value>[\d,]+(?:\.\d{1,2})?)",
        ],
        "total_amount": [
            r"(?<![A-Za-z])Total\s*:\s*\$?(?P<value>[\d,]+(?:\.\d{1,2})?)",
            r"(?<![A-Za-z])Total\s+\$(?P<value>[\d,]+(?:\.\d{1,2})?)",
            r"Amount\s*Due\s*:?\s*\$?(?P<value>[\d,]+(?:\.\d{1,2})?)",
        ],
        "currency": [r"Currency\s*:\s*(?P<value>[A-Z]{3})"],
    }

    def extract(self, parsed_document: ParsedDocument) -> dict[str, Any]:
        doc_type = detect_document_type(parsed_document.raw_text)
        if doc_type == "medical_discharge":
            return RuleBasedMedicalDischargeExtractor().extract(parsed_document)
        if doc_type == "nda":
            return RuleBasedNDAExtractor().extract(parsed_document)
        if doc_type == "lab_report":
            return RuleBasedLabReportExtractor().extract(parsed_document)
        if doc_type == "business_doc":
            return RuleBasedBusinessDocExtractor().extract(parsed_document)

        text = parsed_document.raw_text
        extracted: dict[str, Any] = _empty_invoice(parsed_document.file_name)

        for field, patterns in self.FIELD_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted[field] = match.group("value").strip()
                    break

        _coerce_money_fields(extracted)
        extracted["line_items"] = _extract_line_items(text)
        return extracted


class TemplateOnlyExtractor(BaseExtractor):
    def __init__(self, template_memory: TemplateMemory) -> None:
        self._template_memory = template_memory

    def extract(self, parsed_document: ParsedDocument) -> dict[str, Any]:
        result, _ = self.extract_with_trace(parsed_document)
        return result

    def extract_with_trace(self, parsed_document: ParsedDocument) -> tuple[dict[str, Any], list[str]]:
        trace = ["Started template-only extraction."]
        lines = parsed_document.sections
        signature = TemplateMemory.build_signature(lines)
        match = self._template_memory.find_best_match(signature)
        if not match or match.score < 0.55:
            raise ExtractionError("No close learned template was found for this document.")

        trace.append(f"Matched learned template `{match.template.get('template_name', 'unknown')}` with score {match.score}.")
        extracted = _empty_invoice(parsed_document.file_name)
        extracted.update(_extract_from_template(match.template, parsed_document.raw_text))
        return extracted, trace


class AdaptiveInvoiceAgent(BaseExtractor):
    def __init__(self, settings: Settings) -> None:
        self._template_memory = TemplateMemory(settings.template_store_path)
        self._rule_based = RuleBasedInvoiceExtractor()

    def extract(self, parsed_document: ParsedDocument) -> dict[str, Any]:
        result, _ = self.extract_with_trace(parsed_document)
        return result

    def extract_with_trace(self, parsed_document: ParsedDocument) -> tuple[dict[str, Any], list[str]]:
        trace: list[str] = []
        lines = parsed_document.sections
        signature = TemplateMemory.build_signature(lines)
        trace.append("Generated document signature from top lines and keywords.")

        # Spatial extraction from PDF (lazy — only if pdfplumber available)
        spatial_fields: dict[str, Any] = {}
        spatial_layouts = []
        if parsed_document.file_path and parsed_document.file_path.suffix.lower() == ".pdf":
            try:
                from .spatial_extractor import extract_spatial_layout, extract_fields_from_layout
                spatial_layouts = extract_spatial_layout(parsed_document.file_path)
                if spatial_layouts:
                    spatial_fields = extract_fields_from_layout(spatial_layouts)
                    if spatial_fields:
                        trace.append(f"Spatial PDF extraction found {len(spatial_fields)} field(s) by page position.")
            except Exception:
                pass

        template_match = self._template_memory.find_best_match(signature)
        if template_match:
            trace.append(
                f"Best learned template candidate was `{template_match.template.get('template_name', 'unknown')}` "
                f"with similarity {template_match.score}."
            )
        else:
            trace.append("No learned template candidates were available yet.")

        doc_type = detect_document_type(parsed_document.raw_text)
        if template_match and template_match.score >= 0.55:
            extracted = _empty_schema(parsed_document.file_name, doc_type)
            extracted.update(_extract_from_template(template_match.template, parsed_document.raw_text))
            # If the template has spatial anchors, use them to extract / confirm fields
            if spatial_layouts and template_match.template.get("spatial_anchors"):
                try:
                    from .spatial_extractor import extract_by_spatial_anchors
                    anchor_fields = extract_by_spatial_anchors(spatial_layouts, template_match.template["spatial_anchors"])
                    if anchor_fields:
                        extracted.update(anchor_fields)
                        trace.append(f"Spatial template anchors matched {len(anchor_fields)} field(s) at expected page positions.")
                except Exception:
                    pass
            trace.append("Applied learned template anchors to extract fields.")
        else:
            extracted = self._rule_based.extract(parsed_document)
            trace.append("Fell back to rule-based label and regex extraction.")
        trace.append(f"Detected document type: {doc_type}.")

        # Fill any remaining empty fields from spatial extraction
        spatial_filled = [f for f, v in spatial_fields.items() if extracted.get(f) in (None, "", []) and v]
        for f in spatial_filled:
            extracted[f] = spatial_fields[f]
        if spatial_filled:
            trace.append(f"Filled {len(spatial_filled)} missing field(s) from spatial extraction: {', '.join(sorted(spatial_filled))}.")

        inferred = _infer_missing_fields(parsed_document.raw_text, extracted)
        if inferred:
            extracted.update(inferred)
            trace.append(f"Inferred additional fields from semantic heuristics: {', '.join(sorted(inferred))}.")
        else:
            trace.append("No additional missing fields could be inferred.")

        return extracted, trace

    @property
    def template_memory(self) -> TemplateMemory:
        return self._template_memory


class LLMAssistedInvoiceAgent(BaseExtractor):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._template_memory = TemplateMemory(settings.template_store_path)
        self._adaptive_local = AdaptiveInvoiceAgent(settings)

    def extract(self, parsed_document: ParsedDocument) -> dict[str, Any]:
        result, _ = self.extract_with_trace(parsed_document)
        return result

    def extract_with_trace(self, parsed_document: ParsedDocument) -> tuple[dict[str, Any], list[str]]:
        trace: list[str] = []
        lines = parsed_document.sections
        signature = TemplateMemory.build_signature(lines)
        trace.append("Generated document signature from top lines and keywords.")

        # Spatial extraction from PDF (lazy — only if pdfplumber available)
        spatial_fields: dict[str, Any] = {}
        spatial_layouts = []
        if parsed_document.file_path and parsed_document.file_path.suffix.lower() == ".pdf":
            try:
                from .spatial_extractor import extract_spatial_layout, extract_fields_from_layout
                spatial_layouts = extract_spatial_layout(parsed_document.file_path)
                if spatial_layouts:
                    spatial_fields = extract_fields_from_layout(spatial_layouts)
                    if spatial_fields:
                        trace.append(f"Spatial PDF extraction found {len(spatial_fields)} field(s) by page position.")
            except Exception:
                pass

        template_match = self._template_memory.find_best_match(signature)
        if template_match:
            trace.append(
                f"Best learned template candidate was `{template_match.template.get('template_name', 'unknown')}` "
                f"with similarity {template_match.score}."
            )
        else:
            trace.append("No learned template candidates were available yet.")

        doc_type = detect_document_type(parsed_document.raw_text)
        trace.append(f"Detected document type: {doc_type}.")

        # For familiar layouts, keep the faster local path.
        if template_match and template_match.score >= 0.68:
            extracted = _empty_schema(parsed_document.file_name, doc_type)
            extracted.update(_extract_from_template(template_match.template, parsed_document.raw_text))
            # Spatial anchor extraction for matched templates
            if spatial_layouts and template_match.template.get("spatial_anchors"):
                try:
                    from .spatial_extractor import extract_by_spatial_anchors
                    anchor_fields = extract_by_spatial_anchors(spatial_layouts, template_match.template["spatial_anchors"])
                    if anchor_fields:
                        extracted.update(anchor_fields)
                        trace.append(f"Spatial template anchors matched {len(anchor_fields)} field(s) at expected page positions.")
                except Exception:
                    pass
            trace.append("Used learned template anchors because the document format looked familiar enough.")
            # Fill gaps from general spatial extraction before deciding on LLM
            for f, v in spatial_fields.items():
                if extracted.get(f) in (None, "", []) and v:
                    extracted[f] = v
            if _needs_llm_fallback(extracted):
                trace.append("Template extraction was too incomplete, so the pipeline fell back to the LLM.")
                if not self._settings.openai_api_key:
                    trace.append("No API key was available for LLM fallback, so incomplete template output was kept.")
                else:
                    extracted = self._extract_with_llm(parsed_document, doc_type)
                    trace.append("Used the LLM reasoning layer after incomplete template extraction.")
        else:
            if not self._settings.openai_api_key:
                trace.append("OPENAI_API_KEY not set, so the pipeline fell back to adaptive local extraction.")
                return self._adaptive_local.extract_with_trace(parsed_document)

            extracted = self._extract_with_llm(parsed_document, doc_type)
            trace.append("Used the LLM reasoning layer for an unseen or weakly matched document format.")
            # Fill any gaps the LLM missed with spatial fields
            for f, v in spatial_fields.items():
                if extracted.get(f) in (None, "", []) and v:
                    extracted[f] = v

        inferred = _infer_missing_fields(parsed_document.raw_text, extracted)
        if inferred:
            extracted.update(inferred)
            trace.append(f"Inferred additional fields from semantic heuristics: {', '.join(sorted(inferred))}.")
        else:
            trace.append("No additional missing fields could be inferred.")

        return extracted, trace

    def _extract_with_llm(self, parsed_document: ParsedDocument, doc_type: str = "invoice") -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ExtractionError("The `openai` package is not installed.") from exc

        client = self._build_client(OpenAI)

        if doc_type == "medical_discharge":
            prompt = (
                "Extract fields from this medical discharge document and return one JSON object only. "
                "Use exactly these keys: document_type, source_file, facility_name, patient_name, "
                "date_of_birth, admission_date, discharge_date, primary_diagnosis, secondary_diagnoses, "
                "treating_physician, discharge_condition, discharge_instructions, follow_up_date, medications. "
                "secondary_diagnoses and medications must be JSON arrays of strings (use [] if empty). "
                "Set document_type to 'medical_discharge'. "
                "Dates should be YYYY-MM-DD. Use null for missing scalar values. "
                "Do not wrap the JSON in markdown."
            )
            system_msg = "You extract structured data from medical discharge summaries. Return valid JSON only."
        elif doc_type == "nda":
            prompt = (
                "Extract fields from this non-disclosure agreement and return one JSON object only. "
                "Use exactly these keys: document_type, source_file, agreement_date, effective_date, "
                "expiration_date, disclosing_party, receiving_party, agreement_type, "
                "confidentiality_period, governing_law, permitted_use. "
                "Set document_type to 'nda'. "
                "agreement_type should be 'mutual' or 'one-way'. "
                "Dates should be YYYY-MM-DD. Use null for missing values. "
                "Do not wrap the JSON in markdown."
            )
            system_msg = "You extract structured data from non-disclosure agreements. Return valid JSON only."
        elif doc_type == "lab_report":
            prompt = (
                "Extract fields from this laboratory report and return one JSON object only. "
                "Use exactly these keys: document_type, source_file, patient_name, date_of_birth, mrn, gender, "
                "lab_name, clia_number, ordering_physician, ordering_specialty, accession_number, "
                "specimen_type, collected_date, reported_date, report_id, reviewing_pathologist, "
                "clinical_interpretation, lab_panels, abnormal_results. "
                "Set document_type to 'lab_report'. "
                "lab_panels must be a JSON array of objects each with keys: test, value, units, "
                "reference_range, flag (H/L/empty string). "
                "abnormal_results must be the subset of lab_panels entries where flag is H or L. "
                "Dates should be YYYY-MM-DD when possible. Use null for missing scalars, [] for empty arrays. "
                "Do not wrap the JSON in markdown."
            )
            system_msg = "You extract structured data from medical laboratory reports. Return valid JSON only."
        elif doc_type == "business_doc":
            prompt = (
                "Extract fields from this business document and return one JSON object only. "
                "Use exactly these keys: document_type, source_file, company_name, document_subtype, "
                "report_period, report_date, report_id, prepared_by, approved_by, classification, "
                "executive_summary, kpis, recommendations. "
                "Set document_type to 'business_doc'. "
                "kpis must be a JSON array of objects each with keys: metric, current_period, prior_period, variance. "
                "recommendations must be a JSON array of strings (the numbered recommendations). "
                "Dates should be YYYY-MM-DD when possible. Use null for missing scalars, [] for empty arrays. "
                "Do not wrap the JSON in markdown."
            )
            system_msg = "You extract structured data from business reports and documents. Return valid JSON only."
        else:
            prompt = (
                "Extract invoice fields from the document text and return one JSON object only. "
                "Use exactly these keys and no others: "
                "document_type, source_file, vendor_name, invoice_number, invoice_date, due_date, "
                "subtotal, tax, shipping_handling, total_amount, currency, line_items. "
                "Use null for missing scalar values and [] for missing line_items. "
                "Set document_type to invoice unless the text clearly shows otherwise. "
                "Dates should be normalized to YYYY-MM-DD when possible. "
                "Monetary values should be numbers, not strings with currency symbols. "
                "shipping_handling is the shipping and/or handling charge if present, otherwise null. "
                "line_items must be a JSON array of objects, each with keys: "
                "description (string), quantity (number), unit_price (number), total (number). "
                "Do not wrap the JSON in markdown."
            )
            system_msg = "You convert unstructured business documents into structured invoice JSON. Return valid JSON only."

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"{prompt}\n\nDocument text:\n{parsed_document.raw_text[:18000]}"},
        ]
        content = self._request_llm_json(client, messages)
        if not content:
            raise ExtractionError("The LLM returned an empty response.")

        content = _extract_json_object(_strip_json_fence(content))
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"LLM response was not valid JSON: {content}") from exc

        extracted = _empty_schema(parsed_document.file_name, doc_type)
        extracted.update(data)
        extracted["document_type"] = extracted.get("document_type") or doc_type
        extracted["source_file"] = parsed_document.file_name

        if doc_type == "invoice":
            _coerce_money_fields(extracted)
            if not isinstance(extracted.get("line_items"), list):
                extracted["line_items"] = []
        elif doc_type == "medical_discharge":
            for list_field in ("secondary_diagnoses", "medications"):
                if not isinstance(extracted.get(list_field), list):
                    extracted[list_field] = []
        elif doc_type == "lab_report":
            for list_field in ("lab_panels", "abnormal_results"):
                if not isinstance(extracted.get(list_field), list):
                    extracted[list_field] = []
        elif doc_type == "business_doc":
            for list_field in ("kpis", "recommendations"):
                if not isinstance(extracted.get(list_field), list):
                    extracted[list_field] = []

        return extracted

    def _request_llm_json(self, client, messages: list[dict[str, str]]) -> str:
        request = {
            "model": self._settings.openai_model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        try:
            response = client.chat.completions.create(**request)
        except TypeError:
            request.pop("response_format", None)
            try:
                response = client.chat.completions.create(**request)
            except TypeError:
                request.pop("temperature", None)
                response = client.chat.completions.create(**request)
        except Exception as exc:
            retry_after = _parse_retry_after(exc)
            if retry_after is not None:
                raise RateLimitRetry(str(exc), retry_after) from exc
            raise

        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "")
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
        return str(content).strip()

    def _build_client(self, openai_cls):
        provider = (self._settings.llm_provider or "openai").strip().lower()
        base_url = self._settings.llm_base_url
        api_key = self._settings.openai_api_key

        if provider == "groq":
            return openai_cls(
                api_key=api_key,
                base_url=base_url or "https://api.groq.com/openai/v1",
            )
        if provider == "openrouter":
            return openai_cls(
                api_key=api_key,
                base_url=base_url or "https://openrouter.ai/api/v1",
            )
        if provider == "ollama":
            return openai_cls(
                api_key=api_key or "ollama",
                base_url=base_url or "http://localhost:11434/v1/",
            )
        if provider == "gemini":
            return openai_cls(
                api_key=api_key,
                base_url=base_url or "https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        return openai_cls(
            api_key=api_key,
            base_url=base_url,
        )


def build_extractor(mode: str, settings: Settings) -> BaseExtractor:
    template_memory = TemplateMemory(settings.template_store_path)
    if mode == "template-only":
        return TemplateOnlyExtractor(template_memory)
    if mode == "rule-based":
        return RuleBasedInvoiceExtractor()
    if mode == "llm-assisted":
        return LLMAssistedInvoiceAgent(settings)
    return AdaptiveInvoiceAgent(settings)


def _empty_invoice(source_file: str) -> dict[str, Any]:
    return {
        "document_type": "invoice",
        "source_file": source_file,
        "vendor_name": None,
        "invoice_number": None,
        "invoice_date": None,
        "due_date": None,
        "subtotal": None,
        "tax": None,
        "shipping_handling": None,
        "total_amount": None,
        "currency": "USD",
        "line_items": [],
    }


def _empty_medical_discharge(source_file: str) -> dict[str, Any]:
    return {
        "document_type": "medical_discharge",
        "source_file": source_file,
        "facility_name": None,
        "patient_name": None,
        "date_of_birth": None,
        "admission_date": None,
        "discharge_date": None,
        "primary_diagnosis": None,
        "secondary_diagnoses": [],
        "treating_physician": None,
        "discharge_condition": None,
        "discharge_instructions": None,
        "follow_up_date": None,
        "medications": [],
    }


def _empty_nda(source_file: str) -> dict[str, Any]:
    return {
        "document_type": "nda",
        "source_file": source_file,
        "agreement_date": None,
        "effective_date": None,
        "expiration_date": None,
        "disclosing_party": None,
        "receiving_party": None,
        "agreement_type": None,
        "confidentiality_period": None,
        "governing_law": None,
        "permitted_use": None,
    }


def _empty_lab_report(source_file: str) -> dict[str, Any]:
    return {
        "document_type": "lab_report",
        "source_file": source_file,
        "patient_name": None,
        "date_of_birth": None,
        "mrn": None,
        "gender": None,
        "lab_name": None,
        "clia_number": None,
        "ordering_physician": None,
        "ordering_specialty": None,
        "accession_number": None,
        "specimen_type": None,
        "collected_date": None,
        "reported_date": None,
        "report_id": None,
        "reviewing_pathologist": None,
        "clinical_interpretation": None,
        "lab_panels": [],
        "abnormal_results": [],
    }


def _empty_business_doc(source_file: str) -> dict[str, Any]:
    return {
        "document_type": "business_doc",
        "source_file": source_file,
        "company_name": None,
        "document_subtype": None,
        "report_period": None,
        "report_date": None,
        "report_id": None,
        "prepared_by": None,
        "approved_by": None,
        "classification": None,
        "executive_summary": None,
        "kpis": [],
        "recommendations": [],
    }


def _empty_schema(source_file: str, doc_type: str) -> dict[str, Any]:
    if doc_type == "medical_discharge":
        return _empty_medical_discharge(source_file)
    if doc_type == "nda":
        return _empty_nda(source_file)
    if doc_type == "lab_report":
        return _empty_lab_report(source_file)
    if doc_type == "business_doc":
        return _empty_business_doc(source_file)
    return _empty_invoice(source_file)


def _extract_from_template(template: dict[str, Any], raw_text: str) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    for field, anchor in template.get("anchors", {}).items():
        pattern = anchor.get("pattern")
        if not pattern:
            continue
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if not match:
            continue
        value = match.groupdict().get("value") or match.group(0)
        extracted[field] = value.strip()

    _coerce_money_fields(extracted)
    if not extracted.get("line_items"):
        extracted["line_items"] = _extract_line_items(raw_text)

    return extracted


def _infer_missing_fields(raw_text: str, extracted: dict[str, Any]) -> dict[str, Any]:
    inferred: dict[str, Any] = {}

    if extracted.get("currency") in (None, ""):
        if "$" in raw_text:
            inferred["currency"] = "USD"

    if extracted.get("total_amount") is None:
        amounts = [
            float(m.replace(",", ""))
            for m in re.findall(r"\$\s*([\d,]+\.\d{2})", raw_text)
            if m
        ]
        if amounts:
            inferred["total_amount"] = max(amounts)

    if extracted.get("invoice_number") in (None, ""):
        match = re.search(r"\b(?:INV|Invoice)[-\s#:]*(\w+)\b", raw_text, re.IGNORECASE)
        if match:
            inferred["invoice_number"] = match.group(1)

    if extracted.get("invoice_date") in (None, ""):
        match = re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b", raw_text)
        if match:
            inferred["invoice_date"] = match.group(1)

    return inferred


def _needs_llm_fallback(extracted: dict[str, Any]) -> bool:
    required_fields = ("vendor_name", "invoice_number", "invoice_date", "total_amount")
    missing_required = sum(1 for field in required_fields if extracted.get(field) in (None, "", []))
    return missing_required >= 1


_MONEY_FIELDS = ("subtotal", "tax", "shipping_handling", "total_amount")


def _coerce_money_fields(extracted: dict[str, Any]) -> None:
    for field in _MONEY_FIELDS:
        value = extracted.get(field)
        if value in (None, ""):
            extracted[field] = None
            continue
        try:
            extracted[field] = float(str(value).replace(",", "").replace("$", ""))
        except ValueError:
            pass


def _strip_json_fence(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json_object(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        return content.strip()
    return content[start : end + 1].strip()
