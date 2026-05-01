from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .config import Settings
from .extractors import ExtractionError, RateLimitRetry, build_extractor
from .parsers import DocumentParser
from .schemas import PipelineResult
from .storage import ResultStore
from .template_memory import BadPatternStore, TemplateMemory
from .validators import get_validator


def _actual_extraction_mode(requested: str, trace: list[str]) -> str:
    """Infer the mode that was actually used from the extraction trace."""
    combined = " ".join(trace).lower()
    if "used the llm reasoning layer" in combined:
        return "llm-assisted"
    if (
        "applied learned template anchors" in combined
        or "used learned template anchors" in combined
        or "matched learned template" in combined
    ):
        return "template"
    if "fell back to rule-based" in combined or "rule-based label and regex" in combined:
        return "rule-based"
    return requested


# Compiled once — used by DocumentPipeline._build_field_sources on every document.
_GAP_RE = re.compile(r"^(.+?)\s+filled\s+\d+\s+\S+.*?:\s*(.+)\.$", re.IGNORECASE)
_INFER_RE = re.compile(
    r"inferred additional fields from semantic heuristics:\s*(.+)\.$", re.IGNORECASE
)
_BAD_RE = re.compile(
    r"learned bad-value patterns for \d+ field\S*\s+from corrections:\s*(.+)\.$",
    re.IGNORECASE,
)


class DocumentPipeline:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._parser = DocumentParser()
        self._store = ResultStore(settings)

    def process_upload(
        self,
        uploaded_file: Any,
        extraction_mode: str = "adaptive-local",
        learn_from_upload: bool = True,
    ) -> PipelineResult:
        return self.process_bytes(
            uploaded_file.name,
            bytes(uploaded_file.getbuffer()),
            extraction_mode=extraction_mode,
            learn_from_upload=learn_from_upload,
        )

    @staticmethod
    def _compute_text_hash(text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def process_bytes(
        self,
        file_name: str,
        file_bytes: bytes,
        extraction_mode: str = "adaptive-local",
        learn_from_upload: bool = True,
    ) -> PipelineResult:
        saved_path, _file_hash = self._save_upload(file_name, file_bytes)
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
                content_hash="",
                needs_review=True,
            )

        content_hash = self._compute_text_hash(parsed_document.raw_text)

        if self._store.has_been_processed(content_hash):
            return PipelineResult(
                source_file=saved_path.name,
                upload_path=str(saved_path),
                parsed_text=parsed_document.raw_text,
                extracted_data={"document_type": "unknown", "source_file": saved_path.name},
                validation_results=[],
                output_files={},
                summary={
                    "source_file": saved_path.name,
                    "extraction_mode": extraction_mode,
                    "duplicate": True,
                },
                errors=["Duplicate: identical document content was already processed."],
                extraction_trace=["Content hash matched an existing record — skipping re-processing."],
                content_hash=content_hash,
                needs_review=False,
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
        except RateLimitRetry:
            raise  # propagate to app.py rate-limit handler
        except Exception as exc:
            # Unexpected API/runtime errors (e.g. Groq "Execution failed") — degrade
            # gracefully so the document gets flagged for review rather than crashing.
            errors.append(f"Extraction error: {exc}")
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
            extraction_trace.append(f"Extraction failed unexpectedly: {exc}")

        bad_store = BadPatternStore(self._settings.bad_patterns_path)
        cleared_fields = bad_store.apply(extracted_data)
        if cleared_fields:
            extraction_trace.append(
                f"Cleared {len(cleared_fields)} field(s) matching known bad patterns: "
                f"{', '.join(sorted(cleared_fields))}."
            )

        validation_checks = get_validator(extracted_data.get("document_type", "invoice")).validate(extracted_data)
        learned_template_name = self._learn_from_result(
            saved_path=saved_path,
            extraction_mode=extraction_mode,
            learn_from_upload=learn_from_upload,
            parsed_lines=parsed_document.sections,
            extracted_data=extracted_data,
            validation_checks=validation_checks,
            extraction_trace=extraction_trace,
            allow_automatic_learning=extraction_mode != "llm-assisted",
        )

        if learned_template_name:
            extraction_trace.append(f"Learned or updated template `{learned_template_name}` from this upload.")

        rule_based_fallback = (
            extraction_mode in ("adaptive-local", "template-only")
            and not any(
                "learned template" in step.lower()
                and any(kw in step.lower() for kw in ("matched", "applied", "used"))
                for step in extraction_trace
            )
        )
        needs_review = bool(
            errors
            or any(check.status == "fail" for check in validation_checks)
            or rule_based_fallback
        )

        output_files = self._store.persist(
            saved_path.name,
            extracted_data,
            validation_checks,
            extraction_trace,
            content_hash=content_hash,
            original_filename=file_name,
        )

        summary = {
            "source_file": saved_path.name,
            "original_filename": file_name,
            "content_hash": content_hash,
            "extraction_mode": _actual_extraction_mode(extraction_mode, extraction_trace),
            "validation_passes": sum(check.status == "pass" for check in validation_checks),
            "validation_fails": sum(check.status == "fail" for check in validation_checks),
            "validation_warnings": sum(check.status == "warn" for check in validation_checks),
            "template_learning_enabled": self._settings.enable_template_learning and learn_from_upload,
            "learned_template": learned_template_name,
            "outputs_written": list(output_files.keys()),
            "needs_review": needs_review,
        }

        field_sources = self._build_field_sources(extracted_data, extraction_trace)
        field_confidence = self._compute_field_confidence(
            extracted_data, validation_checks, extraction_trace, field_sources
        )

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
            content_hash=content_hash,
            needs_review=needs_review,
            field_confidence=field_confidence,
            field_sources=field_sources,
        )

    def is_already_processed(self, content_hash: str) -> bool:
        return self._store.has_been_processed(content_hash)

    def finalize_review(
        self,
        source_file: str,
        upload_path: str,
        parsed_text: str,
        corrected_data: dict[str, Any],
        extraction_mode: str = "adaptive-local",
        learn_from_upload: bool = True,
        approve_for_future_matching: bool = False,
        content_hash: str = "",
        original_extracted: dict[str, Any] | None = None,
    ) -> PipelineResult:
        saved_path = Path(upload_path)
        # Preserve the original processing trace so metrics (LLM usage, template hits, etc.)
        # remain accurate after the user submits a review.
        prior_trace = self._store.get_processing_trace(source_file)
        extraction_trace = prior_trace + ["Used human-reviewed corrections from the UI."]
        if approve_for_future_matching:
            extraction_trace.append("User explicitly approved this result for future matching.")

        if original_extracted:
            bad_store = BadPatternStore(self._settings.bad_patterns_path)
            doc_type = corrected_data.get("document_type", "invoice")
            learned_patterns: list[str] = []
            for field, orig_val in original_extracted.items():
                if field in {"document_type", "source_file"} or orig_val in (None, "", []):
                    continue
                corrected_val = corrected_data.get(field)
                if corrected_val in (None, "", []):
                    pattern = bad_store.add_pattern(doc_type, field, str(orig_val))
                    if pattern:
                        learned_patterns.append(field)
            if learned_patterns:
                extraction_trace.append(
                    f"Learned bad-value patterns for {len(learned_patterns)} field(s) from corrections: "
                    f"{', '.join(sorted(learned_patterns))}."
                )
        validation_checks = get_validator(corrected_data.get("document_type", "invoice")).validate(corrected_data)
        learned_template_name = self._learn_from_result(
            saved_path=saved_path,
            extraction_mode=extraction_mode,
            learn_from_upload=learn_from_upload,
            parsed_lines=[line.strip() for line in parsed_text.splitlines() if line.strip()],
            extracted_data=corrected_data,
            validation_checks=validation_checks,
            extraction_trace=extraction_trace,
            force_learning=approve_for_future_matching,
            allow_automatic_learning=True,
        )
        if learned_template_name:
            extraction_trace.append(f"Learned or updated template `{learned_template_name}` from reviewed data.")

        output_files = self._store.persist(source_file, corrected_data, validation_checks, extraction_trace, content_hash=content_hash)
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
            "approved_for_future_matching": approve_for_future_matching,
        }

        field_sources = {
            k: "Manual"
            for k, v in corrected_data.items()
            if k not in ("document_type", "source_file") and v not in (None, "", [], {})
        }
        field_confidence = self._compute_field_confidence(
            corrected_data, validation_checks, extraction_trace, field_sources
        )

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
            field_confidence=field_confidence,
            field_sources=field_sources,
        )

    def log_error(self, error_type: str, source: str, message: str, severity: str = "error") -> None:
        self._store.log_error(error_type, source, message, severity)

    def log_upload(self, original_filename: str, upload_path: str, file_size_bytes: int) -> None:
        self._store.log_upload(original_filename, upload_path, file_size_bytes)

    def _save_upload(self, file_name: str, file_bytes: bytes) -> tuple[Path, str]:
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        p = Path(file_name)
        saved_name = f"{p.stem}_{content_hash[:8]}{p.suffix}"
        destination = self._settings.upload_dir / saved_name
        destination.write_bytes(file_bytes)
        return destination, content_hash

    def _learn_from_result(
        self,
        saved_path: Path,
        extraction_mode: str,
        learn_from_upload: bool,
        parsed_lines: list[str],
        extracted_data: dict[str, Any],
        validation_checks: list,
        extraction_trace: list[str],
        force_learning: bool = False,
        allow_automatic_learning: bool = True,
    ) -> str | None:
        if extraction_mode == "template-only":
            extraction_trace.append("Skipped learning because template-only mode is read-only.")
            return None
        if not self._settings.enable_template_learning or not learn_from_upload:
            extraction_trace.append("Template learning is disabled for this run.")
            return None
        if not force_learning and not allow_automatic_learning:
            extraction_trace.append("Skipped automatic template learning for llm-assisted extraction until a user approves the result.")
            return None

        total_checks = len(validation_checks)
        pass_checks = sum(getattr(check, "status", "") == "pass" for check in validation_checks)
        pass_ratio = (pass_checks / total_checks) if total_checks else 0.0
        if force_learning:
            doc_type = extracted_data.get("document_type", "invoice")
            if not self._has_required_field_passes(validation_checks, doc_type):
                extraction_trace.append("Skipped template learning because approved data is still missing required fields.")
                return None
            extraction_trace.append("Bypassed normal learning threshold because the user explicitly approved the result.")
        elif pass_ratio < self._settings.min_learning_pass_ratio:
            extraction_trace.append(
                f"Skipped template learning because pass ratio {pass_ratio:.2f} is below "
                f"{self._settings.min_learning_pass_ratio:.2f}."
            )
            return None

        memory = TemplateMemory(self._settings.template_store_path)

        spatial_anchors: list[dict] = []
        spatial_layouts: list = []
        if saved_path.suffix.lower() == ".pdf":
            try:
                from .spatial_extractor import build_spatial_anchors, extract_spatial_layout
                spatial_layouts = extract_spatial_layout(saved_path)
                if spatial_layouts:
                    spatial_anchors = build_spatial_anchors(spatial_layouts, extracted_data)
            except Exception:
                pass

        signature = TemplateMemory.build_signature(parsed_lines, layouts=spatial_layouts)

        template = memory.learn_template(
            saved_path.name, signature, extracted_data, parsed_lines,
            spatial_anchors=spatial_anchors or None,
        )
        return template.get("template_name")

    # Per-source confidence baselines — Cross-validated is highest because multiple
    # independent methods independently produced the same value.
    _SOURCE_BASELINE: dict[str, float] = {
        "Cross-validated": 0.97,
        "Manual": 0.95,
        "LLM": 0.88,
        "Template": 0.82,
        "Spatial": 0.78,
        "Rule-based": 0.72,
        "Inferred": 0.65,
    }

    @staticmethod
    def _compute_field_confidence(
        extracted_data: dict,
        validation_checks: list,
        extraction_trace: list[str],
        field_sources: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """Return a 0–1 confidence score for every extracted field.

        When *field_sources* is provided, each field's baseline is drawn from
        its specific extraction method.  Cross-validated fields (multiple methods
        agreed) receive the highest baseline.  Falls back to a document-level
        baseline derived from the trace when *field_sources* is absent.
        """
        # Document-level fallback baseline (used when a field has no source entry).
        trace_lower = " ".join(extraction_trace).lower()
        if "llm" in trace_lower or "openai" in trace_lower or "groq" in trace_lower:
            doc_baseline = 0.88
        elif "learned template" in trace_lower or "applied" in trace_lower:
            doc_baseline = 0.82
        elif "spatial" in trace_lower:
            doc_baseline = 0.78
        elif "rule-based" in trace_lower or "regex" in trace_lower:
            doc_baseline = 0.72
        else:
            doc_baseline = 0.75

        # Map field → worst validation status across all checks for that field.
        field_worst: dict[str, str] = {}
        _precedence = {"pass": 2, "warn": 1, "fail": 0}
        for check in validation_checks:
            fname = getattr(check, "field", None) or check.get("field", "")
            status = getattr(check, "status", None) or check.get("status", "warn")
            if fname not in field_worst or _precedence.get(status, 1) < _precedence.get(field_worst[fname], 1):
                field_worst[fname] = status

        _status_multiplier = {"pass": 1.0, "warn": 0.75, "fail": 0.35}
        _source_baseline = DocumentPipeline._SOURCE_BASELINE

        confidence: dict[str, float] = {}
        for key, value in extracted_data.items():
            if key in ("document_type", "source_file"):
                continue
            if value in (None, "", [], {}):
                confidence[key] = 0.0
            else:
                source = (field_sources or {}).get(key)
                base = _source_baseline.get(source, doc_baseline) if source else doc_baseline
                if key in field_worst:
                    confidence[key] = round(base * _status_multiplier[field_worst[key]], 3)
                else:
                    confidence[key] = base

        return confidence

    @staticmethod
    def _build_field_sources(
        extracted_data: dict,
        extraction_trace: list[str],
    ) -> dict[str, str]:
        """Return a {field: method} dict describing how each field was obtained.

        Parses the structured trace messages produced by the extraction functions
        to assign per-field attribution.  Fields whose trace message isn't found
        fall back to the primary extraction method inferred from the full trace.
        """
        sources: dict[str, str] = {}
        llm_replaced = False   # LLM replaced an incomplete template result
        llm_primary = False    # LLM was the very first extraction method
        template_primary = False

        for step in extraction_trace:
            lower = step.lower()

            # LLM replaced the template/rule-based output entirely — reset.
            if "llm reasoning layer after incomplete template" in lower:
                sources = {}
                llm_replaced = True
                continue

            # LLM was the primary method (no prior template pass).
            if "llm reasoning layer for an unseen" in lower:
                llm_primary = True
                continue

            # Template anchors were the primary method (no LLM).
            if ("applied learned template" in lower or
                    "used learned template anchors" in lower):
                template_primary = True
                continue

            # Manual review step — handled at the end.
            if "human-reviewed corrections" in lower:
                continue

            # Gap-fill messages: attribute listed fields to the named source.
            m = _GAP_RE.match(step)
            if m:
                source_label = m.group(1).lower()
                fields = [f.strip() for f in m.group(2).split(",")]
                if "spatial" in source_label:
                    method = "Spatial"
                elif "template" in source_label:
                    method = "Template"
                elif "rule" in source_label or "rule-based" in source_label:
                    method = "Rule-based"
                else:
                    method = m.group(1).strip()
                for f in fields:
                    sources[f] = method
                continue

            # Inferred fields.
            m = _INFER_RE.match(step)
            if m:
                for f in [x.strip() for x in m.group(1).split(",")]:
                    sources[f] = "Inferred"
                continue

            # Cross-validated fields — multiple methods agreed on the value.
            if lower.startswith("cross-validated"):
                fields_str = step.split(":", 1)[-1].strip().rstrip(".")
                for f in [x.strip() for x in fields_str.split(",")]:
                    sources[f] = "Cross-validated"
                continue

            # Restored prior values where LLM returned null — keep prior source.
            # (Source attribution was already set before the LLM call; leave it.)
            if "restored prior extraction values" in lower:
                continue

            # LLM override of a prior value — mark as LLM.
            if lower.startswith("llm value differs"):
                fields_str = step.split("(llm used):", 1)[-1].strip().rstrip(".")
                for f in [x.strip() for x in fields_str.split(",")]:
                    sources[f] = "LLM"
                continue

            # Bad-pattern fields are not a source of value — skip.
            if _BAD_RE.match(step):
                continue

        # Assign default method to every non-null field not yet attributed.
        if llm_replaced or llm_primary:
            default = "LLM"
        elif template_primary:
            default = "Template"
        else:
            default = "Rule-based"

        for key, value in extracted_data.items():
            if key in ("document_type", "source_file"):
                continue
            if key not in sources and value not in (None, "", [], {}):
                sources[key] = default

        return sources

    @staticmethod
    def _has_required_field_passes(validation_checks: list, doc_type: str = "invoice") -> bool:
        from .schema_config import get_required_fields
        required_fields = get_required_fields(doc_type) or {"vendor_name", "total_amount"}
        passing_fields = {
            getattr(check, "field", "")
            for check in validation_checks
            if getattr(check, "status", "") == "pass" and getattr(check, "field", "") in required_fields
        }
        return required_fields.issubset(passing_fields)
