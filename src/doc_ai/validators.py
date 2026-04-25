from __future__ import annotations

from datetime import datetime
from typing import Any

from .schemas import ValidationCheck


def get_validator(doc_type: str):
    if doc_type == "medical_discharge":
        return MedicalDischargeValidator()
    if doc_type == "nda":
        return NDAValidator()
    if doc_type == "lab_report":
        return LabReportValidator()
    if doc_type == "business_doc":
        return BusinessDocValidator()
    return InvoiceValidator()


def _is_valid_date(value: str) -> bool:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue
    return False


class BaseValidator:
    REQUIRED_FIELDS: tuple[str, ...] = ()

    def _check_required_fields(self, extracted_data: dict[str, Any]) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        for field in self.REQUIRED_FIELDS:
            value = extracted_data.get(field)
            if value in (None, "", []):
                checks.append(ValidationCheck(field=field, status="fail", message="Required field is missing."))
            else:
                checks.append(ValidationCheck(field=field, status="pass", message="Required field is present."))
        return checks


class InvoiceValidator(BaseValidator):
    REQUIRED_FIELDS = ("vendor_name", "invoice_number", "invoice_date", "total_amount")

    def validate(self, extracted_data: dict[str, Any]) -> list[ValidationCheck]:
        checks = self._check_required_fields(extracted_data)

        for field in ("invoice_date", "due_date"):
            value = extracted_data.get(field)
            if not value:
                checks.append(ValidationCheck(field=field, status="warn", message="No date provided."))
                continue
            if _is_valid_date(str(value)):
                checks.append(ValidationCheck(field=field, status="pass", message="Date format is valid."))
            else:
                checks.append(
                    ValidationCheck(field=field, status="fail", message="Date should be in YYYY-MM-DD or MM/DD/YYYY.")
                )

        total = extracted_data.get("total_amount")
        if isinstance(total, (int, float)) and total > 0:
            checks.append(ValidationCheck(field="total_amount", status="pass", message="Total amount is positive."))
        else:
            checks.append(ValidationCheck(field="total_amount", status="fail", message="Total amount must be > 0."))

        subtotal = extracted_data.get("subtotal")
        tax = extracted_data.get("tax")
        shipping = extracted_data.get("shipping_handling") or 0
        if isinstance(total, (int, float)) and isinstance(subtotal, (int, float)) and isinstance(tax, (int, float)):
            expected_total = round(subtotal + tax + float(shipping), 2)
            actual_total = round(total, 2)
            label = "Subtotal + tax + shipping" if shipping else "Subtotal + tax"
            if expected_total == actual_total:
                checks.append(
                    ValidationCheck(field="total_consistency", status="pass", message=f"{label} matches total.")
                )
            else:
                checks.append(
                    ValidationCheck(
                        field="total_consistency",
                        status="fail",
                        message=f"Expected total {expected_total:.2f}, but found {actual_total:.2f}.",
                    )
                )

        return checks


class MedicalDischargeValidator(BaseValidator):
    REQUIRED_FIELDS = ("patient_name", "admission_date", "discharge_date", "primary_diagnosis")

    def validate(self, extracted_data: dict[str, Any]) -> list[ValidationCheck]:
        checks = self._check_required_fields(extracted_data)

        for date_field in ("admission_date", "discharge_date", "follow_up_date", "date_of_birth"):
            value = extracted_data.get(date_field)
            if not value:
                if date_field in ("admission_date", "discharge_date"):
                    checks.append(ValidationCheck(field=date_field, status="warn", message="No date provided."))
                continue
            if _is_valid_date(str(value)):
                checks.append(ValidationCheck(field=date_field, status="pass", message="Date format is valid."))
            else:
                checks.append(ValidationCheck(field=date_field, status="fail",
                                               message="Date should be YYYY-MM-DD or MM/DD/YYYY."))

        admit = extracted_data.get("admission_date")
        discharge = extracted_data.get("discharge_date")
        if admit and discharge:
            try:
                fmt = "%Y-%m-%d"
                a = datetime.strptime(str(admit), fmt)
                d = datetime.strptime(str(discharge), fmt)
                if d >= a:
                    checks.append(ValidationCheck(field="date_order", status="pass",
                                                   message="Discharge date is on or after admission date."))
                else:
                    checks.append(ValidationCheck(field="date_order", status="fail",
                                                   message="Discharge date is before admission date."))
            except ValueError:
                pass

        medications = extracted_data.get("medications", [])
        if isinstance(medications, list) and medications:
            checks.append(ValidationCheck(field="medications", status="pass",
                                           message=f"{len(medications)} medication(s) recorded."))
        else:
            checks.append(ValidationCheck(field="medications", status="warn",
                                           message="No medications recorded."))

        return checks


class NDAValidator(BaseValidator):
    REQUIRED_FIELDS = ("disclosing_party", "receiving_party", "agreement_date")

    def validate(self, extracted_data: dict[str, Any]) -> list[ValidationCheck]:
        checks = self._check_required_fields(extracted_data)

        for date_field in ("agreement_date", "effective_date", "expiration_date"):
            value = extracted_data.get(date_field)
            if not value:
                checks.append(ValidationCheck(field=date_field, status="warn", message="No date provided."))
                continue
            if _is_valid_date(str(value)):
                checks.append(ValidationCheck(field=date_field, status="pass", message="Date format is valid."))
            else:
                checks.append(ValidationCheck(field=date_field, status="fail",
                                               message="Date should be YYYY-MM-DD or MM/DD/YYYY."))

        agreement_type = extracted_data.get("agreement_type")
        if agreement_type in ("mutual", "one-way"):
            checks.append(ValidationCheck(field="agreement_type", status="pass",
                                           message=f"Agreement type is {agreement_type}."))
        else:
            checks.append(ValidationCheck(field="agreement_type", status="warn",
                                           message="Agreement type (mutual/one-way) not detected."))

        if extracted_data.get("governing_law"):
            checks.append(ValidationCheck(field="governing_law", status="pass",
                                           message="Governing law is present."))
        else:
            checks.append(ValidationCheck(field="governing_law", status="warn",
                                           message="Governing law not found."))

        return checks


class LabReportValidator(BaseValidator):
    REQUIRED_FIELDS = ("patient_name",)

    def validate(self, extracted_data: dict[str, Any]) -> list[ValidationCheck]:
        checks = self._check_required_fields(extracted_data)

        for date_field in ("collected_date", "reported_date"):
            value = extracted_data.get(date_field)
            if not value:
                checks.append(ValidationCheck(field=date_field, status="warn", message="No date provided."))
                continue
            if _is_valid_date(str(value)):
                checks.append(ValidationCheck(field=date_field, status="pass", message="Date format is valid."))
            else:
                checks.append(ValidationCheck(field=date_field, status="warn",
                                               message="Date could not be parsed as YYYY-MM-DD or MM/DD/YYYY."))

        panels = extracted_data.get("lab_panels", [])
        if isinstance(panels, list) and panels:
            checks.append(ValidationCheck(field="lab_panels", status="pass",
                                           message=f"{len(panels)} lab result(s) recorded."))
        else:
            checks.append(ValidationCheck(field="lab_panels", status="warn",
                                           message="No lab panel results found."))

        abnormal = extracted_data.get("abnormal_results", [])
        if isinstance(abnormal, list) and abnormal:
            checks.append(ValidationCheck(field="abnormal_results", status="warn",
                                           message=f"{len(abnormal)} abnormal result(s) flagged — review recommended."))
        else:
            checks.append(ValidationCheck(field="abnormal_results", status="pass",
                                           message="No abnormal flags detected."))

        return checks


class BusinessDocValidator(BaseValidator):
    REQUIRED_FIELDS = ("company_name",)

    def validate(self, extracted_data: dict[str, Any]) -> list[ValidationCheck]:
        checks = self._check_required_fields(extracted_data)

        for optional in ("document_subtype", "report_period", "prepared_by"):
            value = extracted_data.get(optional)
            if value:
                checks.append(ValidationCheck(field=optional, status="pass", message="Field is present."))
            else:
                checks.append(ValidationCheck(field=optional, status="warn", message="Field not found."))

        kpis = extracted_data.get("kpis", [])
        if isinstance(kpis, list) and kpis:
            checks.append(ValidationCheck(field="kpis", status="pass",
                                           message=f"{len(kpis)} KPI(s) recorded."))
        else:
            checks.append(ValidationCheck(field="kpis", status="warn", message="No KPIs found."))

        return checks
