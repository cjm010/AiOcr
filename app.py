from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.doc_ai.config import get_settings
from src.doc_ai.extractors import RateLimitRetry
from src.doc_ai.logging_config import setup_logging
from src.doc_ai.pipeline import DocumentPipeline
from src.doc_ai.schema_config import FIELD_CATALOG, SchemaConfig


st.set_page_config(
    page_title="AI-Powered Data Quality Platform",
    page_icon=":page_facing_up:",
    layout="wide",
)

FIELD_ORDER = [
    "document_type",
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "subtotal",
    "tax",
    "shipping_handling",
    "total_amount",
    "currency",
]

# Fields that are not always present on every invoice — they boost completeness
# when filled but do not count against the score when null.
OPTIONAL_FIELDS = {"shipping_handling"}

# Per-document-type field configs: (ordered fields, optional fields set)
FIELDS_BY_TYPE: dict[str, tuple[list[str], set[str]]] = {
    "invoice": (FIELD_ORDER, OPTIONAL_FIELDS),
    "medical_discharge": (
        [
            "document_type",
            "facility_name",
            "patient_name",
            "date_of_birth",
            "admission_date",
            "discharge_date",
            "primary_diagnosis",
            "treating_physician",
            "discharge_condition",
            "discharge_instructions",
            "follow_up_date",
        ],
        {"facility_name", "discharge_instructions", "follow_up_date", "date_of_birth"},
    ),
    "nda": (
        [
            "document_type",
            "disclosing_party",
            "receiving_party",
            "agreement_date",
            "effective_date",
            "expiration_date",
            "agreement_type",
            "confidentiality_period",
            "governing_law",
            "permitted_use",
        ],
        {"effective_date", "expiration_date", "confidentiality_period", "permitted_use"},
    ),
    "lab_report": (
        [
            "document_type",
            "patient_name",
            "date_of_birth",
            "mrn",
            "gender",
            "lab_name",
            "ordering_physician",
            "accession_number",
            "specimen_type",
            "collected_date",
            "reported_date",
            "report_id",
            "reviewing_pathologist",
            "clinical_interpretation",
        ],
        {"date_of_birth", "mrn", "gender", "clia_number", "ordering_specialty"},
    ),
    "business_doc": (
        [
            "document_type",
            "company_name",
            "document_subtype",
            "report_period",
            "report_date",
            "report_id",
            "prepared_by",
            "approved_by",
            "classification",
            "executive_summary",
        ],
        {"report_date", "report_id", "approved_by", "classification"},
    ),
}

LIST_FIELDS_BY_TYPE: dict[str, list[str]] = {
    "invoice": ["line_items"],
    "medical_discharge": ["secondary_diagnoses", "medications"],
    "nda": [],
    "lab_report": ["lab_panels", "abnormal_results"],
    "business_doc": ["kpis", "recommendations"],
}


def _fields_for(doc_type: str) -> tuple[list[str], set[str]]:
    return FIELDS_BY_TYPE.get(doc_type, FIELDS_BY_TYPE["invoice"])

LLM_PROVIDER_OPTIONS = ["openai", "groq", "openrouter", "ollama", "gemini"]

MODEL_OPTIONS_BY_PROVIDER = {
    "openai": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o", "custom"],
    "groq": ["openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "custom"],
    "openrouter": [
        "openai/gpt-oss-20b:free",
        "meta-llama/llama-3.3-8b-instruct:free",
        "google/gemma-3-27b-it:free",
        "custom",
    ],
    "ollama": ["llama3.2", "mistral", "qwen2.5", "custom"],
    "gemini": ["gemini-2.0-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro", "custom"],
}


@st.cache_data(show_spinner=False)
def render_pdf_pages(upload_path: str) -> list[bytes]:
    path = Path(upload_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        return []

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    pages: list[bytes] = []
    for page_index in range(len(document)):
        page = document[page_index]
        bitmap = page.render(scale=1.5)
        image = bitmap.to_pil()
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            pages.append(buffer.getvalue())
        page.close()
    document.close()
    return pages


def render_pdf_preview(upload_path: str) -> None:
    path = Path(upload_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        st.info("PDF preview is only available for uploaded PDF files.")
        return

    try:
        page_images = render_pdf_pages(upload_path)
    except Exception as exc:
        st.warning(f"PDF preview could not be rendered in-app: {exc}")
        st.download_button(
            "Download PDF",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/pdf",
            use_container_width=True,
        )
        return

    if not page_images:
        st.info("This PDF could not be rendered for preview.")
        return

    st.caption("Preview is rendered as images so it works reliably in the browser.")
    for index, image_bytes in enumerate(page_images, start=1):
        st.image(image_bytes, caption=f"Page {index}", use_container_width=True)

    st.download_button(
        "Download original PDF",
        data=path.read_bytes(),
        file_name=path.name,
        mime="application/pdf",
        use_container_width=True,
    )


def coerce_form_data(source_file: str, values: dict[str, str]) -> dict[str, object]:
    corrected = {"source_file": source_file}
    for field, value in values.items():
        text = value.strip()
        if field in {"subtotal", "tax", "shipping_handling", "total_amount"}:
            if not text:
                corrected[field] = None
            else:
                try:
                    corrected[field] = float(text.replace(",", "").replace("$", ""))
                except ValueError:
                    corrected[field] = text
        else:
            corrected[field] = text or None
    corrected["document_type"] = corrected.get("document_type") or "invoice"
    corrected["currency"] = corrected.get("currency") or "USD"
    return corrected


def _parse_line_items_json(text: str) -> list:
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def _line_items_to_json(items) -> str:
    if not items:
        return "[]"
    if isinstance(items, str):
        return items
    return json.dumps(items, indent=2)


def render_completeness_bar(extracted_data: dict) -> None:
    def _is_filled(v):
        return v not in (None, "", "null")

    doc_type = extracted_data.get("document_type", "invoice")
    field_order, optional_fields = _fields_for(doc_type)
    list_fields = LIST_FIELDS_BY_TYPE.get(doc_type, [])

    filled = sum(1 for f in field_order if _is_filled(extracted_data.get(f)))
    filled += sum(1 for lf in list_fields if extracted_data.get(lf))
    required_total = sum(1 for f in field_order if f not in optional_fields) + len(list_fields)
    optional_present = sum(1 for f in optional_fields if _is_filled(extracted_data.get(f)))
    total = required_total + optional_present
    pct = filled / total if total else 0

    left, right = st.columns([1, 2])
    with left:
        st.metric("Extraction completeness", f"{pct:.0%}", f"{filled}/{total} fields")
    with right:
        st.progress(pct)
        if pct >= 0.8:
            st.caption("Good — most fields extracted successfully.")
        elif pct >= 0.5:
            st.caption("Partial — review and fill in the missing fields below.")
        else:
            st.caption("Low — many fields are missing, manual review needed.")


def _confidence_badge(score: float) -> str:
    """Return a Streamlit-colored inline badge string for a 0–1 confidence score."""
    pct = round(score * 100)
    if score >= 0.85:
        return f":green[{pct}% confidence]"
    if score >= 0.60:
        return f":orange[{pct}% confidence]"
    if score > 0.0:
        return f":red[{pct}% confidence]"
    return ":red[not extracted]"


def render_review_form(
    pipeline: DocumentPipeline,
    result,
    extraction_mode: str,
    learn_from_upload: bool,
) -> None:
    st.subheader("Review and correct fields")
    st.caption("If extraction missed anything, update the values below and save the reviewed result.")

    doc_type = result.extracted_data.get("document_type", "invoice")
    active_fields, _ = _fields_for(doc_type)
    list_fields = LIST_FIELDS_BY_TYPE.get(doc_type, [])

    field_confidence = getattr(result, "field_confidence", {})
    defaults = {field: result.extracted_data.get(field) for field in active_fields}
    with st.form("review_form", clear_on_submit=False):
        form_values: dict[str, str] = {}
        col_left, col_right = st.columns(2)
        for index, field in enumerate(active_fields):
            target_col = col_left if index % 2 == 0 else col_right
            current = defaults.get(field)
            score = field_confidence.get(field)
            label = field.replace("_", " ").title()
            if score is not None:
                label = f"{label}  —  {_confidence_badge(score)}"
            with target_col:
                form_values[field] = st.text_input(
                    label,
                    value="" if current is None else str(current),
                )

        list_field_values: dict[str, str] = {}
        for lf in list_fields:
            st.markdown(f"**{lf.replace('_', ' ').title()}**")
            list_field_values[lf] = st.text_area(
                f"{lf} (JSON array)",
                value=_line_items_to_json(result.extracted_data.get(lf)),
                height=150,
                label_visibility="collapsed",
            )

        approve_for_future_matching = st.checkbox(
            "Approve this reviewed result for future matching",
            value=True,
            help="If checked, the reviewed values will be saved as a stronger template for similar future documents.",
        )

        submitted = st.form_submit_button("Save reviewed result", type="primary")

    if submitted:
        corrected_data = coerce_form_data(result.source_file, form_values)
        for lf, raw in list_field_values.items():
            corrected_data[lf] = _parse_line_items_json(raw)
        reviewed_result = pipeline.finalize_review(
            source_file=result.source_file,
            upload_path=result.upload_path,
            parsed_text=result.parsed_text,
            corrected_data=corrected_data,
            extraction_mode=extraction_mode,
            learn_from_upload=learn_from_upload,
            approve_for_future_matching=approve_for_future_matching,
            content_hash=result.content_hash,
        )
        st.session_state["last_result"] = reviewed_result
        st.session_state.manual_corrections_total += 1
        st.success("Reviewed values saved. The outputs and validation report have been updated.")
        st.rerun()


def render_approval_actions(
    pipeline: DocumentPipeline,
    result,
    extraction_mode: str,
    learn_from_upload: bool,
) -> None:
    st.subheader("Approve current result")
    st.caption("If these extracted values already look right, approve them so the system reuses this format next time.")

    if st.button("Approve current result for future matching", use_container_width=True):
        reviewed_result = pipeline.finalize_review(
            source_file=result.source_file,
            upload_path=result.upload_path,
            parsed_text=result.parsed_text,
            corrected_data=result.extracted_data,
            extraction_mode=extraction_mode,
            learn_from_upload=learn_from_upload,
            approve_for_future_matching=True,
            content_hash=result.content_hash,
        )
        st.session_state["last_result"] = reviewed_result
        st.session_state.approvals_total += 1
        st.success("This result was approved and saved for stronger future matching.")
        st.rerun()


def resolve_runtime_settings(base_settings):
    ui_api_key = st.session_state.get("ui_openai_api_key", "").strip()
    custom_model = st.session_state.get("ui_openai_custom_model", "").strip()
    selected_model = st.session_state.get("ui_openai_model", base_settings.openai_model)
    provider = st.session_state.get("ui_llm_provider", base_settings.llm_provider)
    custom_base_url = st.session_state.get("ui_llm_base_url", "").strip()

    if selected_model == "custom":
        runtime_model = custom_model or base_settings.openai_model
    else:
        runtime_model = selected_model
    runtime_api_key = ui_api_key or base_settings.openai_api_key

    return replace(
        base_settings,
        llm_provider=provider,
        llm_base_url=custom_base_url or base_settings.llm_base_url,
        openai_api_key=runtime_api_key,
        openai_model=runtime_model or base_settings.openai_model,
    )


def compute_upload_signature(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()



def _parse_template_match(trace: list[str]) -> tuple[str, str]:
    for step in trace:
        m = re.search(r"Matched learned template `([^`]+)` with score ([0-9.]+)", step)
        if m:
            return m.group(1), f"{float(m.group(2)):.2f}"
    return "—", "—"


def _llm_used(trace: list[str]) -> bool:
    keywords = ("llm", "language model", "openai", "gemini", "anthropic", "claude")
    return any(any(kw in step.lower() for kw in keywords) for step in trace)


_MAX_RATE_LIMIT_RETRIES = 3


def _rate_limit_countdown(placeholder, retry_after: int, n_files: int = 1) -> None:
    """Block the main thread showing a live countdown, then clear the placeholder."""
    label = f"{n_files} file(s)" if n_files > 1 else "file"
    for remaining in range(retry_after, 0, -1):
        placeholder.warning(
            f"⏳ Rate limit reached for {label} — retrying in **{remaining}s**..."
        )
        time.sleep(1)
    placeholder.empty()


def _run_bulk_processing(
    pipeline: DocumentPipeline,
    file_items: list[dict],
    extraction_mode: str,
    learn_from_upload: bool,
    max_workers: int,
    logger: logging.Logger,
) -> list[dict]:
    total = len(file_items)
    progress_bar = st.progress(0.0, text=f"Starting — 0 / {total} processed")
    status_placeholder = st.empty()
    rate_limit_placeholder = st.empty()
    completed = 0
    status_lines: list[str] = []
    bulk_results: list[dict] = []
    all_status_lines: list[str] = []

    def process_one(item: dict):
        t0 = time.monotonic()
        result = pipeline.process_bytes(
            item["name"],
            item["bytes"],
            extraction_mode=extraction_mode,
            learn_from_upload=learn_from_upload,
        )
        result.summary["processing_time_s"] = round(time.monotonic() - t0, 2)
        return result

    def _run_pass(items: list[dict]) -> list[dict]:
        """Process a list of items concurrently; return any that were rate-limited."""
        nonlocal completed
        rate_limited: list[dict] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(process_one, item): item for item in items}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                completed += 1
                try:
                    result = future.result()
                    is_duplicate = result.summary.get("duplicate", False)
                    if is_duplicate:
                        status = "duplicate"
                    elif result.needs_review:
                        status = "needs_review"
                    else:
                        status = "success"
                    bulk_results.append({
                        "filename": item["name"],
                        "content_hash": result.content_hash,
                        "status": status,
                        "extraction_mode": result.summary.get("extraction_mode", extraction_mode),
                        "validation_fails": result.summary.get("validation_fails", 0),
                        "needs_review": result.needs_review,
                        "errors": "; ".join(result.errors) if result.errors else "",
                        "_result_obj": None if is_duplicate else result,
                    })
                    icon = "↩" if is_duplicate else ("✓" if status == "success" else "⚠")
                    line = (
                        f"{icon} {item['name']}"
                        + (" — duplicate, skipped" if is_duplicate else
                           (" — needs review" if result.needs_review else ""))
                    )
                    status_lines.append(line)
                    all_status_lines.append(line)
                    logger.info("SUCCESS file=%s hash=%s needs_review=%s duplicate=%s",
                                item["name"], result.content_hash, result.needs_review, is_duplicate)
                except RateLimitRetry as rl:
                    rate_limited.append({"item": item, "retry_after": rl.retry_after})
                    line = f"⏳ {item['name']}: rate limited — will retry"
                    status_lines.append(line)
                    all_status_lines.append(line)
                    logger.warning("RATE_LIMITED file=%s retry_after=%s", item["name"], rl.retry_after)
                except Exception as exc:
                    bulk_results.append({
                        "filename": item["name"],
                        "content_hash": item["hash"],
                        "status": "failed",
                        "extraction_mode": extraction_mode,
                        "validation_fails": 0,
                        "needs_review": True,
                        "errors": str(exc),
                        "_result_obj": None,
                    })
                    line = f"✗ {item['name']}: {exc}"
                    status_lines.append(line)
                    all_status_lines.append(line)
                    logger.error("FAILED file=%s error=%s", item["name"], exc)

                progress_bar.progress(
                    min(completed / total, 1.0),
                    text=f"{completed} / {total} processed — {item['name']}",
                )
                status_placeholder.text("\n".join(status_lines[-8:]))
        return rate_limited

    pending = _run_pass(file_items)

    for _attempt in range(_MAX_RATE_LIMIT_RETRIES):
        if not pending:
            break
        max_wait = max(r["retry_after"] for r in pending)
        _rate_limit_countdown(rate_limit_placeholder, max_wait, n_files=len(pending))
        # Remove "will retry" lines and replace with fresh status after retry
        pending = _run_pass([r["item"] for r in pending])

    # Any still rate-limited after all retries → mark as failed
    for r in pending:
        item = r["item"]
        bulk_results.append({
            "filename": item["name"],
            "content_hash": item["hash"],
            "status": "failed",
            "extraction_mode": extraction_mode,
            "validation_fails": 0,
            "needs_review": True,
            "errors": f"Rate limit: exhausted {_MAX_RATE_LIMIT_RETRIES} retries",
            "_result_obj": None,
        })
        line = f"✗ {item['name']}: rate limit — gave up after {_MAX_RATE_LIMIT_RETRIES} retries"
        all_status_lines.append(line)
        logger.error("RATE_LIMIT_EXHAUSTED file=%s", item["name"])

    progress_bar.progress(1.0, text=f"Complete — {total} file(s) processed")
    status_placeholder.empty()

    # Deduplicate within batch by content hash — catches same-content different-filename files
    # that slipped through concurrent processing before either committed to the DB.
    seen_content_hashes: dict[str, str] = {}
    for r in bulk_results:
        ch = r.get("content_hash", "")
        if not ch or r["status"] in ("failed", "duplicate"):
            continue
        if ch in seen_content_hashes:
            r["status"] = "duplicate"
            r["_result_obj"] = None
            r["errors"] = f"Duplicate: same content as '{seen_content_hashes[ch]}' in this batch"
            logger.info("DUPLICATE file=%s matches=%s", r["filename"], seen_content_hashes[ch])
        else:
            seen_content_hashes[ch] = r["filename"]

    return bulk_results, all_status_lines



def _is_bulk_auto_approvable(result, confidence_threshold: float = 0.80, apply_confidence_gate: bool = False) -> bool:
    if result.errors or any(r.get("status") == "fail" for r in result.validation_results):
        return False
    if apply_confidence_gate:
        present = [c for c in result.field_confidence.values() if c > 0]
        if present and min(present) < confidence_threshold:
            return False
    return True


def render_single_tab(
    pipeline: DocumentPipeline,
    extraction_mode: str,
    learn_from_upload: bool,
    logger: logging.Logger,
    *,
    confidence_threshold: float = 0.80,
    auto_approve: bool = False,
    output_format: str = "Both",
) -> None:
    uploaded_file = st.file_uploader(
        "Upload an invoice or similar unstructured business document",
        type=["pdf", "txt", "md", "json"],
        accept_multiple_files=False,
    )

    if not uploaded_file:
        st.info("Upload a file to run the pipeline.")
        with st.expander("Example unstructured invoice text for testing"):
            st.code(
                "\n".join(
                    [
                        "Vendor: Greenleaf Supplies",
                        "Invoice Number: 4587",
                        "Invoice Date: 2024-02-12",
                        "Due Date: 2024-03-12",
                        "Subtotal: 1100.00",
                        "Tax: 100.00",
                        "Total: 1200.00",
                        "Currency: USD",
                    ]
                ),
                language="text",
            )
        return

    uploaded_bytes = uploaded_file.getvalue()
    current_upload_signature = compute_upload_signature(uploaded_bytes)
    previous_upload_signature = st.session_state.get("current_upload_signature")
    if previous_upload_signature != current_upload_signature:
        st.session_state["current_upload_signature"] = current_upload_signature
        if st.session_state.get("last_processed_signature") != current_upload_signature:
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_uploaded_name", None)

    if st.button("Process document", type="primary"):
        _rl_placeholder = st.empty()
        result = None
        for _attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            try:
                with st.spinner("Running parsing, extraction, validation, and storage..."):
                    result = pipeline.process_upload(
                        uploaded_file,
                        extraction_mode=extraction_mode,
                        learn_from_upload=learn_from_upload,
                    )
                logger.info("SUCCESS file=%s hash=%s needs_review=%s", uploaded_file.name, result.content_hash, result.needs_review)
                break
            except RateLimitRetry as rl:
                if _attempt < _MAX_RATE_LIMIT_RETRIES:
                    logger.warning("RATE_LIMITED file=%s retry_after=%s attempt=%s", uploaded_file.name, rl.retry_after, _attempt + 1)
                    _rate_limit_countdown(_rl_placeholder, rl.retry_after, n_files=1)
                else:
                    logger.error("RATE_LIMIT_EXHAUSTED file=%s", uploaded_file.name)
                    st.error(f"Rate limit: exhausted {_MAX_RATE_LIMIT_RETRIES} retries. Try again later.")
                    return
            except Exception as exc:
                logger.error("FAILED file=%s error=%s", uploaded_file.name, exc)
                st.error(f"Pipeline failed unexpectedly: {exc}")
                return
        if result is None:
            return
        st.session_state["last_result"] = result
        st.session_state["last_uploaded_name"] = uploaded_file.name
        st.session_state["last_processed_signature"] = current_upload_signature
        st.session_state.docs_processed_total += 1

        if auto_approve and not result.errors:
            present = [c for c in result.field_confidence.values() if c > 0]
            if present and min(present) >= confidence_threshold:
                reviewed = pipeline.finalize_review(
                    source_file=result.source_file,
                    upload_path=result.upload_path,
                    parsed_text=result.parsed_text,
                    corrected_data=result.extracted_data,
                    extraction_mode=extraction_mode,
                    learn_from_upload=learn_from_upload,
                    approve_for_future_matching=True,
                    content_hash=result.content_hash,
                )
                st.session_state["last_result"] = reviewed
                st.session_state.approvals_total += 1
                st.rerun()


    result = st.session_state.get("last_result")
    if not result:
        return

    if st.session_state.get("last_processed_signature") != current_upload_signature:
        st.info("Press `Process document` to run the pipeline for the currently selected file.")
        return

    st.subheader("Processing summary")
    st.write(result.summary)

    if result.errors:
        st.error("The pipeline completed with errors or warnings.")
        for error in result.errors:
            st.write(f"- {error}")

    doc_col, review_col = st.columns([1.1, 1])

    with doc_col:
        st.subheader("Source document")
        render_pdf_preview(result.upload_path)
        st.subheader("Copyable parsed text")
        st.caption("If the embedded PDF viewer does not let you copy text easily, use this parsed text instead.")
        st.text_area("Parsed document text", result.parsed_text[:20000], height=320)

    with review_col:
        st.subheader("Current extracted fields")
        render_completeness_bar(result.extracted_data)
        st.json(result.extracted_data)
        render_approval_actions(pipeline, result, extraction_mode=extraction_mode, learn_from_upload=learn_from_upload)
        st.subheader("Validation report")
        validation_df = pd.DataFrame(result.validation_results)
        if validation_df.empty:
            st.info("No validation checks were produced.")
        else:
            st.dataframe(validation_df, use_container_width=True)
        render_review_form(pipeline, result, extraction_mode=extraction_mode, learn_from_upload=learn_from_upload)

    st.subheader("Agent trace")
    if result.extraction_trace:
        for step in result.extraction_trace:
            st.write(f"- {step}")
    else:
        st.info("No extraction trace was recorded.")

    st.subheader("Output files")
    for label, path_str in result.output_files.items():
        path = Path(path_str)
        if output_format == "JSON" and path.suffix != ".json":
            continue
        if output_format == "CSV" and path.suffix != ".csv":
            continue
        st.write(f"{label}: `{path}`")
        if path.suffix == ".json" and path.exists():
            st.download_button(
                label=f"Download {path.name}",
                data=json.dumps(result.extracted_data, indent=2),
                file_name=path.name,
                mime="application/json",
            )


def _render_bulk_results(
    approved: list,
    flagged: list,
    pipeline: DocumentPipeline,
    extraction_mode: str,
    learn_from_upload: bool,
    raw_results: list | None = None,
) -> None:
    raw_results = raw_results or []
    duplicates = [r for r in raw_results if r.get("status") == "duplicate"]
    failures = [r for r in raw_results if r.get("status") == "failed"]
    total_processed = len(approved) + len(flagged) + len(duplicates) + len(failures)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", total_processed)
    m2.metric("Auto-approved", len(approved))
    m3.metric("Needs review", len(flagged))
    m4.metric("Skipped / failed", len(duplicates) + len(failures))

    def _summary_row(r, status_label: str) -> dict:
        def _is_filled(v):
            return v not in (None, "", "null")
        _dt = r.extracted_data.get("document_type", "invoice")
        _fo, _opt = _fields_for(_dt)
        filled = sum(1 for f in _fo if _is_filled(r.extracted_data.get(f)))
        req_total = sum(1 for f in _fo if f not in _opt)
        opt_present = sum(1 for f in _opt if _is_filled(r.extracted_data.get(f)))
        denom = req_total + opt_present
        passes = sum(1 for c in r.validation_results if c.get("status") == "pass")
        warns = sum(1 for c in r.validation_results if c.get("status") == "warn")
        fails = sum(1 for c in r.validation_results if c.get("status") == "fail")
        tmpl_name, tmpl_score = _parse_template_match(r.extraction_trace)
        proc_time = r.summary.get("processing_time_s")
        return {
            "File": r.source_file,
            "Status": status_label,
            "Extraction Mode": r.summary.get("extraction_mode", "—"),
            "Template Matched": tmpl_name,
            "Match Score": tmpl_score,
            "LLM Used": "Yes" if _llm_used(r.extraction_trace) else "No",
            "Time (s)": proc_time if proc_time is not None else "—",
            "Completeness": f"{filled}/{denom} ({filled * 100 // denom if denom else 0}%)",
            "✓ Pass": passes,
            "⚠ Warn": warns,
            "✗ Fail": fails,
            "Errors": "; ".join(r.errors) if r.errors else "",
            "Content Hash": r.content_hash,
        }

    rows = (
        [_summary_row(r, "auto-approved") for r in approved]
        + [_summary_row(r, "needs review") for r in flagged]
        + [{"File": r["filename"], "Status": "duplicate", "Errors": r.get("errors", ""), "Content Hash": r.get("content_hash", "")} for r in duplicates]
        + [{"File": r["filename"], "Status": "failed", "Errors": r.get("errors", ""), "Content Hash": r.get("content_hash", "")} for r in failures]
    )

    summary_df = pd.DataFrame(rows)
    with st.expander("Batch summary table", expanded=True):
        st.dataframe(summary_df, use_container_width=True)
        st.download_button(
            "Export summary as CSV",
            data=summary_df.to_csv(index=False),
            file_name="bulk_upload_summary.csv",
            mime="text/csv",
        )

    if not flagged:
        st.success("All documents passed validation and have been auto-approved.")
        return

    review_index = st.session_state.get("bulk_review_index", 0)
    if "bulk_reviewed" not in st.session_state:
        st.session_state["bulk_reviewed"] = []

    if review_index >= len(flagged):
        st.success(f"All {len(flagged)} flagged document(s) have been reviewed.")
        if st.button("← Go back to first item", key="bulk_restart"):
            st.session_state["bulk_review_index"] = 0
            st.rerun()
        return

    st.divider()
    current = flagged[review_index]

    prev_col, info_col, next_col = st.columns([1, 3, 1])
    with prev_col:
        if st.button("← Previous", disabled=(review_index == 0), use_container_width=True, key="bulk_prev"):
            st.session_state["bulk_review_index"] = review_index - 1
            st.rerun()
    with info_col:
        st.markdown(
            f"<p style='text-align:center;padding-top:6px'>"
            f"Item <b>{review_index + 1}</b> of <b>{len(flagged)}</b> flagged</p>",
            unsafe_allow_html=True,
        )
    with next_col:
        if st.button("Next →", disabled=(review_index >= len(flagged) - 1), use_container_width=True, key="bulk_next"):
            st.session_state["bulk_review_index"] = review_index + 1
            st.rerun()

    st.subheader(f"Reviewing: {current.source_file}")
    render_completeness_bar(current.extracted_data)

    fails = [c for c in current.validation_results if c.get("status") == "fail"]
    if fails:
        st.warning(f"{len(fails)} validation check(s) failed — correct or confirm the values below.")
        for c in fails:
            st.write(f"- **{c['field']}**: {c['message']}")

    with st.expander("Source document", expanded=True):
        preview_col, text_col = st.columns([1, 1])
        with preview_col:
            render_pdf_preview(current.upload_path)
        with text_col:
            st.caption("Parsed text — use this if the PDF viewer doesn't let you copy easily.")
            st.text_area("Parsed document text", current.parsed_text[:20000], height=400, key=f"bulk_parsed_{review_index}", label_visibility="collapsed")

    _bulk_doc_type = current.extracted_data.get("document_type", "invoice")
    _bulk_fields, _ = _fields_for(_bulk_doc_type)
    _bulk_list_fields = LIST_FIELDS_BY_TYPE.get(_bulk_doc_type, [])

    _bulk_confidence = getattr(current, "field_confidence", {})
    defaults = {field: current.extracted_data.get(field) for field in _bulk_fields}
    with st.form(f"bulk_review_{review_index}"):
        form_values: dict[str, str] = {}
        left_col, right_col = st.columns(2)
        for idx, field in enumerate(_bulk_fields):
            target_col = left_col if idx % 2 == 0 else right_col
            current_val = defaults.get(field)
            score = _bulk_confidence.get(field)
            label = field.replace("_", " ").title()
            if score is not None:
                label = f"{label}  —  {_confidence_badge(score)}"
            with target_col:
                form_values[field] = st.text_input(
                    label,
                    value="" if current_val is None else str(current_val),
                )
        bulk_list_field_values: dict[str, str] = {}
        for lf in _bulk_list_fields:
            st.markdown(f"**{lf.replace('_', ' ').title()}**")
            bulk_list_field_values[lf] = st.text_area(
                f"{lf} (JSON array)",
                value=_line_items_to_json(current.extracted_data.get(lf)),
                height=150,
                label_visibility="collapsed",
            )
        approve_future = st.checkbox("Approve for future matching", value=True)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            submitted = st.form_submit_button("Save and continue", type="primary", use_container_width=True)
        with btn_col2:
            skipped = st.form_submit_button("Skip", use_container_width=True)

    if submitted:
        corrected = coerce_form_data(current.source_file, form_values)
        for lf, raw in bulk_list_field_values.items():
            corrected[lf] = _parse_line_items_json(raw)
        reviewed = pipeline.finalize_review(
            source_file=current.source_file,
            upload_path=current.upload_path,
            parsed_text=current.parsed_text,
            corrected_data=corrected,
            extraction_mode=extraction_mode,
            learn_from_upload=learn_from_upload,
            approve_for_future_matching=approve_future,
            content_hash=current.content_hash,
        )
        st.session_state["bulk_reviewed"].append(reviewed)
        st.session_state["bulk_review_index"] = review_index + 1
        st.rerun()

    if skipped:
        st.session_state["bulk_review_index"] = review_index + 1
        st.rerun()


def render_bulk_tab(
    pipeline: DocumentPipeline,
    extraction_mode: str,
    learn_from_upload: bool,
    logger: logging.Logger,
    *,
    confidence_threshold: float = 0.80,
    auto_approve: bool = False,
) -> None:
    st.caption(
        "Upload multiple documents at once. Files that pass all validation checks are auto-approved and stored. "
        "Files with failures are queued for manual review."
    )

    uploaded_files = st.file_uploader(
        "Upload documents for bulk processing",
        type=["pdf", "txt", "md", "json"],
        accept_multiple_files=True,
        key=f"bulk_uploader_{st.session_state.get('bulk_batch_id', 0)}",
    )

    if uploaded_files:
        if st.button("Clear files", key="bulk_clear_files"):
            st.session_state["bulk_batch_id"] = st.session_state.get("bulk_batch_id", 0) + 1
            st.session_state.pop("bulk_approved", None)
            st.session_state.pop("bulk_flagged", None)
            st.session_state.pop("bulk_summary_rows", None)
            st.rerun()

    if not uploaded_files:
        st.info("Upload one or more files to begin bulk processing.")
        return

    # If results from a previous run exist, show them before doing the dedup check.
    # Without this, st.rerun() after processing would re-run the dedup check, find
    # all files already in the DB, and return early — hiding the review queue.
    approved = st.session_state.get("bulk_approved")
    flagged = st.session_state.get("bulk_flagged")
    if approved is not None:
        _render_bulk_results(approved, flagged, pipeline, extraction_mode, learn_from_upload,
                             raw_results=st.session_state.get("bulk_summary_rows"))
        if st.button("Process another batch", use_container_width=True):
            st.session_state["bulk_batch_id"] = st.session_state.get("bulk_batch_id", 0) + 1
            st.session_state.pop("bulk_approved", None)
            st.session_state.pop("bulk_flagged", None)
            st.session_state.pop("bulk_summary_rows", None)
            st.rerun()
        return

    # Pre-read bytes and deduplicate within this batch by raw bytes.
    # DB-level dedup (by content hash) is handled inside the pipeline after text parsing,
    # so two files with identical text but different binary bytes are also caught.
    file_items: list[dict] = []
    seen_hashes: dict[str, str] = {}
    for f in uploaded_files:
        fb = f.getvalue()
        h = compute_upload_signature(fb)
        if h in seen_hashes:
            file_items.append({"name": f.name, "bytes": fb, "hash": h, "skip_reason": f"duplicate of '{seen_hashes[h]}' in this batch"})
        else:
            seen_hashes[h] = f.name
            file_items.append({"name": f.name, "bytes": fb, "hash": h, "skip_reason": None})

    to_process = [item for item in file_items if item["skip_reason"] is None]
    to_skip = [item for item in file_items if item["skip_reason"] is not None]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total uploaded", len(uploaded_files))
    c2.metric("To process", len(to_process))
    c3.metric("Skipped (duplicates)", len(to_skip))

    if to_skip:
        with st.expander(f"{len(to_skip)} file(s) skipped"):
            for item in to_skip:
                st.write(f"- **{item['name']}** — {item['skip_reason']}")

    if not to_process:
        st.info("All uploaded files have already been processed or are duplicates.")
        return

    max_workers = st.slider("Parallel workers", min_value=1, max_value=8, value=min(4, os.cpu_count() or 2))

    if st.button("Process all documents", type="primary", key="bulk_process_btn", use_container_width=True):
        raw_results, status_lines = _run_bulk_processing(
            pipeline, to_process, extraction_mode, learn_from_upload, max_workers, logger
        )
        approved: list = []
        flagged: list = []
        for item_result in raw_results:
            if item_result.get("status") in ("duplicate", "failed"):
                pass  # duplicates and hard failures are excluded from review queue
            elif _is_bulk_auto_approvable(item_result["_result_obj"], confidence_threshold, apply_confidence_gate=auto_approve):
                approved.append(item_result["_result_obj"])
            else:
                flagged.append(item_result["_result_obj"])
        processed_count = sum(1 for r in raw_results if r.get("status") not in ("failed", "duplicate"))
        st.session_state.docs_processed_total += processed_count
        st.session_state["bulk_approved"] = approved
        st.session_state["bulk_flagged"] = [r for r in flagged if r is not None]
        st.session_state["bulk_summary_rows"] = raw_results
        st.session_state["bulk_status_lines"] = status_lines
        st.session_state["bulk_review_index"] = 0
        st.session_state.pop("bulk_reviewed", None)
        st.rerun()

    approved_done = st.session_state.get("bulk_approved")
    flagged_done = st.session_state.get("bulk_flagged")
    if approved_done is not None:
        saved_status = st.session_state.get("bulk_status_lines")
        if saved_status:
            with st.expander("Processing log", expanded=False):
                st.text("\n".join(saved_status))
        _render_bulk_results(approved_done, flagged_done, pipeline, extraction_mode, learn_from_upload,
                             raw_results=st.session_state.get("bulk_summary_rows"))
        if st.button("Process another batch", use_container_width=True):
            st.session_state["bulk_batch_id"] = st.session_state.get("bulk_batch_id", 0) + 1
            st.session_state.pop("bulk_approved", None)
            st.session_state.pop("bulk_flagged", None)
            st.session_state.pop("bulk_summary_rows", None)
            st.session_state.pop("bulk_status_lines", None)
            st.rerun()


def main() -> None:
    settings = get_settings()
    logger = setup_logging(settings.data_dir / "logs")

    if "docs_processed_total" not in st.session_state:
        st.session_state.docs_processed_total = 0
    if "manual_corrections_total" not in st.session_state:
        st.session_state.manual_corrections_total = 0
    if "approvals_total" not in st.session_state:
        st.session_state.approvals_total = 0

    st.title("AI-Powered Data Quality Platform for Unstructured Data")
    st.caption(
        "Upload an unstructured business document, convert it into structured data, validate the output, and review any corrections."
    )

    with st.sidebar:
        st.subheader("Pipeline settings")
        extraction_mode = st.selectbox(
            "Extraction mode",
            options=["llm-assisted", "adaptive-local", "template-only", "rule-based"],
            index=0,
            help="Use an LLM for unfamiliar formats, adaptive local extraction, only learned templates, or a fixed rules baseline.",
        )
        learn_from_upload = st.checkbox(
            "Learn from successful uploads",
            value=settings.enable_template_learning,
            help="Stores reusable anchors from high-quality runs so similar future documents are easier to parse.",
        )
        st.write(f"Data directory: `{settings.data_dir}`")
        if extraction_mode == "llm-assisted":
            st.markdown("**LLM settings**")
            provider_index = (
                LLM_PROVIDER_OPTIONS.index(settings.llm_provider)
                if settings.llm_provider in LLM_PROVIDER_OPTIONS
                else 0
            )
            provider = st.selectbox(
                "LLM provider",
                options=LLM_PROVIDER_OPTIONS,
                key="ui_llm_provider",
                index=provider_index,
                help="Choose the provider used for llm-assisted extraction.",
            )
            model_options = MODEL_OPTIONS_BY_PROVIDER.get(provider, ["custom"])
            current_model = st.session_state.get("ui_openai_model")
            if current_model not in model_options:
                st.session_state["ui_openai_model"] = model_options[0]
            st.text_input(
                "API key",
                key="ui_openai_api_key",
                type="password",
                placeholder="sk-... or provider key",
                help="Stored only in this browser session and never written to project files or outputs.",
            )
            st.selectbox(
                "Model",
                options=model_options,
                key="ui_openai_model",
                index=0,
                help="Choose a recommended model for the selected provider or select custom to enter another model id.",
            )
            if st.session_state.get("ui_openai_model") == "custom":
                st.text_input(
                    "Custom model id",
                    key="ui_openai_custom_model",
                    placeholder="gpt-4.1-mini",
                )
            st.text_input(
                "Custom base URL (optional)",
                key="ui_llm_base_url",
                placeholder="Leave blank to use the provider default",
                help="Useful for self-hosted or compatible gateways. Ollama defaults to http://localhost:11434/v1/.",
            )
            if provider == "ollama":
                _ollama_url = (st.session_state.get("ui_llm_base_url") or "http://localhost:11434").rstrip("/")
                try:
                    import urllib.request as _urlreq
                    _urlreq.urlopen(f"{_ollama_url}/api/tags", timeout=2)
                    st.success(f"Ollama reachable at `{_ollama_url}`", icon="✓")
                except Exception:
                    st.warning(
                        f"Ollama not reachable at `{_ollama_url}`. "
                        "Start Ollama before processing documents.",
                        icon="⚠",
                    )
            if st.button("Clear API key", use_container_width=True):
                st.session_state["ui_openai_api_key"] = ""
                st.rerun()
            st.caption("Your API key is used only for the current Streamlit session.")
        st.caption(
            "Approved corrections improve template memory and extraction behavior. They do not fine-tune the foundation model."
        )

        st.divider()
        st.subheader("Reset")
        if st.button("Reset all data", use_container_width=True, type="secondary"):
            st.session_state["confirm_reset"] = True

        if st.session_state.get("confirm_reset"):
            st.warning("This will permanently delete the database and all learned templates.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, reset", type="primary", use_container_width=True):
                    db = settings.database_path
                    tmpl = settings.template_store_path
                    if Path(db).exists():
                        Path(db).unlink()
                    tmpl.write_text("[]", encoding="utf-8")
                    st.session_state.clear()
                    st.success("Database and template memory cleared.")
                    st.rerun()
            with col_no:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.pop("confirm_reset", None)
                    st.rerun()

    runtime_settings = resolve_runtime_settings(settings)
    _match_threshold = st.session_state.get("admin_match_threshold", 0.55)
    runtime_settings = replace(runtime_settings, min_learning_pass_ratio=_match_threshold)
    pipeline = DocumentPipeline(runtime_settings)

    if extraction_mode == "llm-assisted" and runtime_settings.llm_provider not in ("ollama",) and not runtime_settings.openai_api_key:
        st.warning("No API key is active for the selected provider. `llm-assisted` mode will fall back to adaptive local extraction.")

    # Read admin control values from session state so they're available to all tabs.
    # Widgets that write these keys live inside tab_admin below.
    confidence_threshold = st.session_state.get("admin_confidence_threshold", 0.80)
    output_format = st.session_state.get("admin_output_format", "Both")
    auto_approve = st.session_state.get("admin_auto_approve", False)

    tab_single, tab_bulk, tab_schema, tab_admin = st.tabs(
        ["Single Document", "Bulk Upload", "Schema Settings", "Admin Dashboard"]
    )

    with tab_single:
        render_single_tab(pipeline, extraction_mode, learn_from_upload, logger,
                          confidence_threshold=confidence_threshold,
                          auto_approve=auto_approve,
                          output_format=output_format)

    with tab_bulk:
        render_bulk_tab(pipeline, extraction_mode, learn_from_upload, logger,
                        confidence_threshold=confidence_threshold,
                        auto_approve=auto_approve)

    with tab_schema:
        render_schema_settings_tab(runtime_settings)

    with tab_admin:
        render_admin_dashboard_tab(extraction_mode)


def render_admin_dashboard_tab(extraction_mode: str) -> None:
    st.header("Admin Dashboard")

    # ---------------- DERIVED METRICS ---------------- #
    total_docs = st.session_state.docs_processed_total
    manual_reviews = st.session_state.manual_corrections_total
    approvals = st.session_state.approvals_total
    review_rate = round(manual_reviews / max(total_docs, 1) * 100, 1)

    # ---------------- KPI ROW ---------------- #
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Docs Processed", total_docs)
    c2.metric("Manual Corrections", manual_reviews)
    c3.metric("Approvals", approvals)
    c4.metric("Review Rate", f"{review_rate}%")
    c5.metric("System Mode", extraction_mode)

    st.divider()

    # ---------------- CONFIG PANEL ---------------- #
    st.subheader("Configuration")
    st.slider("Template Match Threshold", 0.0, 1.0, 0.55, 0.05, key="admin_match_threshold")
    st.slider("Extraction Confidence Threshold", 0.5, 1.0, 0.80, 0.01, key="admin_confidence_threshold")
    st.selectbox("Output Format", ["JSON", "CSV", "Both"], key="admin_output_format")
    st.checkbox("Auto-approve high confidence results", key="admin_auto_approve")

    st.divider()

    # ---------------- SYSTEM HEALTH ---------------- #
    st.subheader("System Health")
    result = st.session_state.get("last_result")
    if result:
        passed = sum(1 for v in result.validation_results if v.get("status") == "pass")
        failed = sum(1 for v in result.validation_results if v.get("status") == "fail")
        warnings = sum(1 for v in result.validation_results if v.get("status") == "warn")
        st.success(f"Passed: {passed}")
        st.error(f"Failed: {failed}")
        st.warning(f"Warnings: {warnings}")
        st.bar_chart(
            pd.DataFrame({"Status": ["Pass", "Fail", "Warn"], "Count": [passed, failed, warnings]}).set_index("Status")
        )
    else:
        st.info("Process a document to see validation health metrics here.")

    st.divider()

    # ---------------- PROCESSING TREND ---------------- #
    st.subheader("Processing Trend")
    if total_docs > 0:
        trend_df = pd.DataFrame({
            "Run": list(range(1, total_docs + 1)),
            "Processed": [1] * total_docs,
        })
        st.line_chart(trend_df.set_index("Run"))
    else:
        st.info("No documents processed yet this session.")


def render_schema_settings_tab(settings) -> None:
    st.header("Database Schema Settings")
    st.markdown(
        "Select which extracted fields are saved to the per-type database tables. "
        "Required fields are always included and cannot be deselected. "
        "Changes take effect for new documents processed after saving."
    )

    schema_cfg = SchemaConfig(settings.data_dir / "schema_settings.json")

    _DOC_TYPE_LABELS = {
        "invoice":           "Invoices",
        "medical_discharge": "Medical Discharge Summaries",
        "nda":               "Non-Disclosure Agreements (NDA)",
        "lab_report":        "Laboratory Reports",
        "business_doc":      "Business Documents",
    }

    pending_selections: dict[str, list[str]] = {}

    for doc_type, label in _DOC_TYPE_LABELS.items():
        fields = FIELD_CATALOG.get(doc_type, [])
        current_selected = set(schema_cfg.get_selected_fields(doc_type))

        with st.expander(f"{label}  —  {len(current_selected)} field(s) selected", expanded=False):
            st.markdown(f"**Table:** `{_table_name_for(doc_type)}`")
            st.caption("System columns `id`, `source_file`, `original_filename`, `content_hash`, `processed_at` are always present.")
            st.divider()

            new_selected: list[str] = []
            cols = st.columns(2)
            for idx, field in enumerate(fields):
                is_required = field["required"]
                is_checked = field["key"] in current_selected
                col = cols[idx % 2]
                with col:
                    checked = st.checkbox(
                        f"**{field['label']}**  `{field['db_type']}`{'  *(required)*' if is_required else ''}",
                        value=is_checked,
                        disabled=is_required,
                        key=f"schema_{doc_type}_{field['key']}",
                    )
                if checked or is_required:
                    new_selected.append(field["key"])

            pending_selections[doc_type] = new_selected

            st.divider()
            st.markdown("**Resulting DDL preview**")
            # Build a temporary SchemaConfig to generate DDL from current checkbox state
            _tmp = SchemaConfig(settings.data_dir / "schema_settings.json")
            _tmp._selections[doc_type] = new_selected  # noqa: SLF001 — preview only
            st.code(_tmp.get_ddl(doc_type), language="sql")

    st.divider()
    if st.button("Save schema settings", type="primary", use_container_width=True):
        for doc_type, selected in pending_selections.items():
            schema_cfg.set_selected_fields(doc_type, selected)
        schema_cfg.save()
        st.success("Schema settings saved. New extractions will use the updated field selection.")
        st.rerun()


def _table_name_for(doc_type: str) -> str:
    from src.doc_ai.schema_config import TABLE_NAMES
    return TABLE_NAMES.get(doc_type, doc_type)


if __name__ == "__main__":
    main()
