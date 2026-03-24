from __future__ import annotations

import re
from typing import Any

from .config import Settings
from .schemas import ParsedDocument
from .template_memory import TemplateMemory


class ExtractionError(RuntimeError):
    pass


class BaseExtractor:
    def extract(self, parsed_document: ParsedDocument) -> dict[str, Any]:
        raise NotImplementedError


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
        "subtotal": [r"Subtotal\s*:\s*\$?(?P<value>[\d,]+(?:\.\d{1,2})?)"],
        "tax": [r"Tax\s*:\s*\$?(?P<value>[\d,]+(?:\.\d{1,2})?)"],
        "total_amount": [
            r"Total\s*:\s*\$?(?P<value>[\d,]+(?:\.\d{1,2})?)",
            r"Amount\s*Due\s*:\s*\$?(?P<value>[\d,]+(?:\.\d{1,2})?)",
        ],
        "currency": [r"Currency\s*:\s*(?P<value>[A-Z]{3})"],
    }

    def extract(self, parsed_document: ParsedDocument) -> dict[str, Any]:
        text = parsed_document.raw_text
        extracted: dict[str, Any] = {
            "document_type": "invoice",
            "source_file": parsed_document.file_name,
            "vendor_name": None,
            "invoice_number": None,
            "invoice_date": None,
            "due_date": None,
            "subtotal": None,
            "tax": None,
            "total_amount": None,
            "currency": "USD",
        }

        for field, patterns in self.FIELD_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted[field] = match.group("value").strip()
                    break

        for money_field in ("subtotal", "tax", "total_amount"):
            value = extracted.get(money_field)
            if value is not None:
                extracted[money_field] = float(str(value).replace(",", ""))

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

        template_match = self._template_memory.find_best_match(signature)
        if template_match:
            trace.append(
                f"Best learned template candidate was `{template_match.template.get('template_name', 'unknown')}` "
                f"with similarity {template_match.score}."
            )
        else:
            trace.append("No learned template candidates were available yet.")

        if template_match and template_match.score >= 0.55:
            extracted = _empty_invoice(parsed_document.file_name)
            extracted.update(_extract_from_template(template_match.template, parsed_document.raw_text))
            trace.append("Applied learned template anchors to extract fields.")
        else:
            extracted = self._rule_based.extract(parsed_document)
            trace.append("Fell back to rule-based label and regex extraction.")

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


def build_extractor(mode: str, settings: Settings) -> BaseExtractor:
    template_memory = TemplateMemory(settings.template_store_path)
    if mode == "template-only":
        return TemplateOnlyExtractor(template_memory)
    if mode == "rule-based":
        return RuleBasedInvoiceExtractor()
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
        "total_amount": None,
        "currency": "USD",
    }


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

    for money_field in ("subtotal", "tax", "total_amount"):
        value = extracted.get(money_field)
        if value not in (None, ""):
            try:
                extracted[money_field] = float(str(value).replace(",", "").replace("$", ""))
            except ValueError:
                continue

    return extracted


def _infer_missing_fields(raw_text: str, extracted: dict[str, Any]) -> dict[str, Any]:
    inferred: dict[str, Any] = {}

    if extracted.get("currency") in (None, ""):
        if "$" in raw_text:
            inferred["currency"] = "USD"

    if extracted.get("total_amount") is None:
        match = re.search(r"\$?\s*(\d[\d,]*\.\d{2})", raw_text)
        if match:
            try:
                inferred["total_amount"] = float(match.group(1).replace(",", ""))
            except ValueError:
                pass

    if extracted.get("invoice_number") in (None, ""):
        match = re.search(r"\b(?:INV|Invoice)[-\s#:]*(\w+)\b", raw_text, re.IGNORECASE)
        if match:
            inferred["invoice_number"] = match.group(1)

    if extracted.get("invoice_date") in (None, ""):
        match = re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b", raw_text)
        if match:
            inferred["invoice_date"] = match.group(1)

    return inferred
