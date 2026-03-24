from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.doc_ai.config import get_settings
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


def render_pdf_preview(upload_path: str) -> None:
    path = Path(upload_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        st.info("PDF preview is only available for uploaded PDF files.")
        return

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    pdf_html = f"""
    <iframe
        src="data:application/pdf;base64,{encoded}#toolbar=1&navpanes=0&scrollbar=1"
        width="100%"
        height="720"
        style="border: 1px solid #d9d9d9; border-radius: 8px;"
    ></iframe>
    """
    components.html(pdf_html, height=740, scrolling=True)


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
        )
        st.session_state["last_result"] = reviewed_result
        st.success("Reviewed values saved. The outputs and validation report have been updated.")
        st.rerun()


def main() -> None:
    settings = get_settings()
    pipeline = DocumentPipeline(settings)

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
        if extraction_mode == "llm-assisted" and not settings.openai_api_key:
            st.warning("`OPENAI_API_KEY` is not set. `llm-assisted` mode will fall back to adaptive local extraction.")
        st.caption(
            "Approved corrections improve template memory and extraction behavior. They do not fine-tune the foundation model."
        )

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

    if st.button("Process document", type="primary"):
        try:
            with st.spinner("Running parsing, extraction, validation, and storage..."):
                result = pipeline.process_upload(
                    uploaded_file,
                    extraction_mode=extraction_mode,
                    learn_from_upload=learn_from_upload,
                )
        except Exception as exc:
            st.error(f"Pipeline failed unexpectedly: {exc}")
            st.stop()
        st.session_state["last_result"] = result
        st.session_state["last_uploaded_name"] = uploaded_file.name

    result = st.session_state.get("last_result")
    if not result:
        return

    if st.session_state.get("last_uploaded_name") != uploaded_file.name:
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
