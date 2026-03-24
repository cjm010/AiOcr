from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings
from .extractors import ExtractionError, build_extractor
from .parsers import DocumentParser
from .schemas import PipelineResult
from .storage import ResultStore
from .template_memory import TemplateMemory
from .validators import InvoiceValidator


class DocumentPipeline:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._parser = DocumentParser()
        self._validator = InvoiceValidator()
        self._store = ResultStore(settings)

    def process_upload(
        self,
        uploaded_file: Any,
        extraction_mode: str = "adaptive-local",
        learn_from_upload: bool = True,
    ) -> PipelineResult:
        saved_path = self._save_upload(uploaded_file.name, uploaded_file.getbuffer())
        errors: list[str] = []
        extraction_trace: list[str] = []

        try:
            parsed_document = self._parser.parse(saved_path)
        except Exception as exc:
            errors.append(f"Parsing failed: {exc}")
            return PipelineResult(
                source_file=saved_path.name,
                upload_path=str(saved_path),
                parsed_text="",
                extracted_data={
                    "document_type": "unknown",
                    "source_file": saved_path.name,
                },
                validation_results=[],
                output_files={},
                summary={
                    "source_file": saved_path.name,
                    "extraction_mode": extraction_mode,
                    "validation_passes": 0,
                    "validation_fails": 0,
                    "validation_warnings": 0,
                    "outputs_written": [],
                },
                errors=errors,
                extraction_trace=extraction_trace,
            )

        extractor = build_extractor(extraction_mode, self._settings)
        try:
            if hasattr(extractor, "extract_with_trace"):
                extracted_data, extraction_trace = extractor.extract_with_trace(parsed_document)
            else:
                extracted_data = extractor.extract(parsed_document)
                extraction_trace = [f"Used `{extractor.__class__.__name__}` extraction."]
        except ExtractionError as exc:
            errors.append(str(exc))
            extracted_data = {
                "document_type": "invoice",
                "source_file": saved_path.name,
                "vendor_name": None,
                "invoice_number": None,
                "invoice_date": None,
                "due_date": None,
                "subtotal": None,
                "tax": None,
                "total_amount": None,
                "currency": "USD",
            }
            extraction_trace.append(f"Extraction failed: {exc}")

        validation_checks = self._validator.validate(extracted_data)
        learned_template_name = self._learn_from_result(
            saved_path=saved_path,
            extraction_mode=extraction_mode,
            learn_from_upload=learn_from_upload,
            parsed_lines=parsed_document.sections,
            extracted_data=extracted_data,
            validation_checks=validation_checks,
            extraction_trace=extraction_trace,
        )

        if learned_template_name:
            extraction_trace.append(f"Learned or updated template `{learned_template_name}` from this upload.")

        output_files = self._store.persist(saved_path.name, extracted_data, validation_checks, extraction_trace)

        summary = {
            "source_file": saved_path.name,
            "extraction_mode": extraction_mode,
            "validation_passes": sum(check.status == "pass" for check in validation_checks),
            "validation_fails": sum(check.status == "fail" for check in validation_checks),
            "validation_warnings": sum(check.status == "warn" for check in validation_checks),
            "template_learning_enabled": self._settings.enable_template_learning and learn_from_upload,
            "learned_template": learned_template_name,
            "outputs_written": list(output_files.keys()),
        }

        return PipelineResult(
            source_file=saved_path.name,
            upload_path=str(saved_path),
            parsed_text=parsed_document.raw_text,
            extracted_data=extracted_data,
            validation_results=[check.to_dict() for check in validation_checks],
            output_files=output_files,
            summary=summary,
            errors=errors,
            extraction_trace=extraction_trace,
        )

    def finalize_review(
        self,
        source_file: str,
        upload_path: str,
        parsed_text: str,
        corrected_data: dict[str, Any],
        extraction_mode: str = "adaptive-local",
        learn_from_upload: bool = True,
    ) -> PipelineResult:
        saved_path = Path(upload_path)
        extraction_trace = ["Used human-reviewed corrections from the UI."]
        validation_checks = self._validator.validate(corrected_data)
        learned_template_name = self._learn_from_result(
            saved_path=saved_path,
            extraction_mode=extraction_mode,
            learn_from_upload=learn_from_upload,
            parsed_lines=[line.strip() for line in parsed_text.splitlines() if line.strip()],
            extracted_data=corrected_data,
            validation_checks=validation_checks,
            extraction_trace=extraction_trace,
        )
        if learned_template_name:
            extraction_trace.append(f"Learned or updated template `{learned_template_name}` from reviewed data.")

        output_files = self._store.persist(source_file, corrected_data, validation_checks, extraction_trace)
        summary = {
            "source_file": source_file,
            "extraction_mode": extraction_mode,
            "validation_passes": sum(check.status == "pass" for check in validation_checks),
            "validation_fails": sum(check.status == "fail" for check in validation_checks),
            "validation_warnings": sum(check.status == "warn" for check in validation_checks),
            "template_learning_enabled": self._settings.enable_template_learning and learn_from_upload,
            "learned_template": learned_template_name,
            "outputs_written": list(output_files.keys()),
            "reviewed_by_user": True,
        }

        return PipelineResult(
            source_file=source_file,
            upload_path=upload_path,
            parsed_text=parsed_text,
            extracted_data=corrected_data,
            validation_results=[check.to_dict() for check in validation_checks],
            output_files=output_files,
            summary=summary,
            errors=[],
            extraction_trace=extraction_trace,
        )

    def _save_upload(self, file_name: str, file_bytes: bytes) -> Path:
        destination = self._settings.upload_dir / Path(file_name).name
        destination.write_bytes(file_bytes)
        return destination

    def _learn_from_result(
        self,
        saved_path: Path,
        extraction_mode: str,
        learn_from_upload: bool,
        parsed_lines: list[str],
        extracted_data: dict[str, Any],
        validation_checks: list,
        extraction_trace: list[str],
    ) -> str | None:
        if extraction_mode == "template-only":
            extraction_trace.append("Skipped learning because template-only mode is read-only.")
            return None
        if not self._settings.enable_template_learning or not learn_from_upload:
            extraction_trace.append("Template learning is disabled for this run.")
            return None

        total_checks = len(validation_checks)
        pass_checks = sum(getattr(check, "status", "") == "pass" for check in validation_checks)
        pass_ratio = (pass_checks / total_checks) if total_checks else 0.0
        if pass_ratio < self._settings.min_learning_pass_ratio:
            extraction_trace.append(
                f"Skipped template learning because pass ratio {pass_ratio:.2f} is below "
                f"{self._settings.min_learning_pass_ratio:.2f}."
            )
            return None

        memory = TemplateMemory(self._settings.template_store_path)
        signature = TemplateMemory.build_signature(parsed_lines)
        template = memory.learn_template(saved_path.name, signature, extracted_data, parsed_lines)
        return template.get("template_name")
