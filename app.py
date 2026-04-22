from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.doc_ai.config import get_settings
from src.doc_ai.logging_config import setup_logging
from src.doc_ai.pipeline import DocumentPipeline


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
    "total_amount",
    "currency",
]

LLM_PROVIDER_OPTIONS = ["openai", "groq", "openrouter", "ollama"]

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
        if field in {"subtotal", "tax", "total_amount"}:
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


def render_review_form(
    pipeline: DocumentPipeline,
    result,
    extraction_mode: str,
    learn_from_upload: bool,
) -> None:
    st.subheader("Review and correct fields")
    st.caption("If extraction missed anything, update the values below and save the reviewed result.")

    defaults = {field: result.extracted_data.get(field) for field in FIELD_ORDER}
    with st.form("review_form", clear_on_submit=False):
        form_values: dict[str, str] = {}
        col_left, col_right = st.columns(2)
        for index, field in enumerate(FIELD_ORDER):
            target_col = col_left if index % 2 == 0 else col_right
            current = defaults.get(field)
            with target_col:
                form_values[field] = st.text_input(
                    field.replace("_", " ").title(),
                    value="" if current is None else str(current),
                )

        approve_for_future_matching = st.checkbox(
            "Approve this reviewed result for future matching",
            value=True,
            help="If checked, the reviewed values will be saved as a stronger template for similar future documents.",
        )

        submitted = st.form_submit_button("Save reviewed result", type="primary")

    if submitted:
        corrected_data = coerce_form_data(result.source_file, form_values)
        reviewed_result = pipeline.finalize_review(
            source_file=result.source_file,
            upload_path=result.upload_path,
            parsed_text=result.parsed_text,
            corrected_data=corrected_data,
            extraction_mode=extraction_mode,
            learn_from_upload=learn_from_upload,
            approve_for_future_matching=approve_for_future_matching,
        )
        st.session_state["last_result"] = reviewed_result
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
        )
        st.session_state["last_result"] = reviewed_result
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


def render_bulk_summary(bulk_results: list[dict]) -> None:
    st.subheader("Bulk upload summary")
    df = pd.DataFrame(bulk_results)
    total = len(df)
    succeeded = int((df["status"] == "success").sum())
    needs_review_count = int(df["needs_review"].sum())
    failed = int((df["status"] == "failed").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("Succeeded", succeeded)
    c3.metric("Needs Review", needs_review_count)
    c4.metric("Failed", failed)

    display_cols = ["filename", "status", "extraction_mode", "validation_fails", "needs_review", "errors"]
    st.dataframe(df[display_cols], use_container_width=True)

    st.download_button(
        "Export summary as CSV",
        data=df.to_csv(index=False),
        file_name="bulk_upload_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if failed:
        st.error(f"{failed} file(s) failed to process.")
        for row in df[df["status"] == "failed"].itertuples():
            st.write(f"- **{row.filename}**: {row.errors}")

    if needs_review_count:
        st.warning(f"{needs_review_count} file(s) flagged for manual review — check validation failures or missing template matches.")


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
    completed = 0
    status_lines: list[str] = []
    bulk_results: list[dict] = []

    def process_one(item: dict):
        return pipeline.process_bytes(
            item["name"],
            item["bytes"],
            extraction_mode=extraction_mode,
            learn_from_upload=learn_from_upload,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_one, item): item for item in file_items}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            completed += 1
            try:
                result = future.result()
                status = "needs_review" if result.needs_review else "success"
                bulk_results.append({
                    "filename": item["name"],
                    "content_hash": result.content_hash,
                    "status": status,
                    "extraction_mode": result.summary.get("extraction_mode", extraction_mode),
                    "validation_fails": result.summary.get("validation_fails", 0),
                    "needs_review": result.needs_review,
                    "errors": "; ".join(result.errors) if result.errors else "",
                })
                icon = "✓" if status == "success" else "⚠"
                status_lines.append(f"{icon} {item['name']}" + (" — needs review" if result.needs_review else ""))
                logger.info("SUCCESS file=%s hash=%s needs_review=%s", item["name"], result.content_hash, result.needs_review)
            except Exception as exc:
                bulk_results.append({
                    "filename": item["name"],
                    "content_hash": item["hash"],
                    "status": "failed",
                    "extraction_mode": extraction_mode,
                    "validation_fails": 0,
                    "needs_review": True,
                    "errors": str(exc),
                })
                status_lines.append(f"✗ {item['name']}: {exc}")
                logger.error("FAILED file=%s error=%s", item["name"], exc)

            progress_bar.progress(completed / total, text=f"{completed} / {total} processed — {item['name']}")
            status_placeholder.text("\n".join(status_lines[-8:]))

    progress_bar.progress(1.0, text=f"Complete — {total} file(s) processed")
    return bulk_results


def render_bulk_mode(
    pipeline: DocumentPipeline,
    uploaded_files: list,
    extraction_mode: str,
    learn_from_upload: bool,
    logger: logging.Logger,
) -> None:
    st.subheader(f"Bulk upload — {len(uploaded_files)} files selected")

    file_items: list[dict] = []
    seen_hashes: dict[str, str] = {}
    for f in uploaded_files:
        fb = f.getvalue()
        h = compute_upload_signature(fb)
        if h in seen_hashes:
            file_items.append({"name": f.name, "bytes": fb, "hash": h, "skip_reason": f"duplicate of '{seen_hashes[h]}' in this batch"})
        elif pipeline.is_already_processed(h):
            file_items.append({"name": f.name, "bytes": fb, "hash": h, "skip_reason": "already in database"})
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
        if st.session_state.get("bulk_results"):
            render_bulk_summary(st.session_state["bulk_results"])
        return

    max_workers = st.slider("Parallel workers", min_value=1, max_value=8, value=min(4, os.cpu_count() or 2))

    if st.button(f"Process {len(to_process)} document(s)", type="primary", use_container_width=True):
        bulk_results = _run_bulk_processing(
            pipeline, to_process, extraction_mode, learn_from_upload, max_workers, logger
        )
        st.session_state["bulk_results"] = bulk_results

    if st.session_state.get("bulk_results"):
        render_bulk_summary(st.session_state["bulk_results"])


def main() -> None:
    settings = get_settings()
    logger = setup_logging(settings.data_dir / "logs")

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
            if st.button("Clear API key", use_container_width=True):
                st.session_state["ui_openai_api_key"] = ""
                st.rerun()
            st.caption("Your API key is used only for the current Streamlit session.")
        st.caption(
            "Approved corrections improve template memory and extraction behavior. They do not fine-tune the foundation model."
        )

    runtime_settings = resolve_runtime_settings(settings)
    pipeline = DocumentPipeline(runtime_settings)

    if extraction_mode == "llm-assisted" and runtime_settings.llm_provider != "ollama" and not runtime_settings.openai_api_key:
        st.warning("No API key is active for the selected provider. `llm-assisted` mode will fall back to adaptive local extraction.")

    uploaded_files = st.file_uploader(
        "Upload one or more invoices or similar unstructured business documents",
        type=["pdf", "txt", "md", "json"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more files to run the pipeline.")
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

    if len(uploaded_files) > 1:
        render_bulk_mode(pipeline, uploaded_files, extraction_mode, learn_from_upload, logger)
        return

    # --- Single file mode ---
    uploaded_file = uploaded_files[0]
    uploaded_bytes = uploaded_file.getvalue()
    current_upload_signature = compute_upload_signature(uploaded_bytes)
    previous_upload_signature = st.session_state.get("current_upload_signature")
    if previous_upload_signature != current_upload_signature:
        st.session_state["current_upload_signature"] = current_upload_signature
        if st.session_state.get("last_processed_signature") != current_upload_signature:
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_uploaded_name", None)

    if st.button("Process document", type="primary"):
        try:
            with st.spinner("Running parsing, extraction, validation, and storage..."):
                result = pipeline.process_upload(
                    uploaded_file,
                    extraction_mode=extraction_mode,
                    learn_from_upload=learn_from_upload,
                )
            logger.info("SUCCESS file=%s hash=%s needs_review=%s", uploaded_file.name, result.content_hash, result.needs_review)
        except Exception as exc:
            logger.error("FAILED file=%s error=%s", uploaded_file.name, exc)
            st.error(f"Pipeline failed unexpectedly: {exc}")
            return
        st.session_state["last_result"] = result
        st.session_state["last_uploaded_name"] = uploaded_file.name
        st.session_state["last_processed_signature"] = current_upload_signature

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
        st.write(f"{label}: `{path}`")
        if path.suffix == ".json" and path.exists():
            st.download_button(
                label=f"Download {path.name}",
                data=json.dumps(result.extracted_data, indent=2),
                file_name=path.name,
                mime="application/json",
            )


if __name__ == "__main__":
    main()
