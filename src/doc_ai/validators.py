from __future__ import annotations

from datetime import datetime
from typing import Any

from .schemas import ValidationCheck


class InvoiceValidator:
    REQUIRED_FIELDS = ("vendor_name", "invoice_number", "invoice_date", "total_amount")

    def validate(self, extracted_data: dict[str, Any]) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []

        for field in self.REQUIRED_FIELDS:
            value = extracted_data.get(field)
            if value in (None, "", []):
                checks.append(ValidationCheck(field=field, status="fail", message="Required field is missing."))
            else:
                checks.append(ValidationCheck(field=field, status="pass", message="Required field is present."))

        for field in ("invoice_date", "due_date"):
            value = extracted_data.get(field)
            if not value:
                checks.append(ValidationCheck(field=field, status="warn", message="No date provided."))
                continue
            if self._is_valid_date(str(value)):
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
        if isinstance(total, (int, float)) and isinstance(subtotal, (int, float)) and isinstance(tax, (int, float)):
            expected_total = round(subtotal + tax, 2)
            actual_total = round(total, 2)
            if expected_total == actual_total:
                checks.append(
                    ValidationCheck(field="total_consistency", status="pass", message="Subtotal + tax matches total.")
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

    @staticmethod
    def _is_valid_date(value: str) -> bool:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue
        return False
