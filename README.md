# AI Document Extraction and Data Quality Platform

This repository contains a proof-of-concept document processing pipeline based on the architecture in the project plan:

`Upload document -> Parse document -> Extract fields -> Validate data -> Store output -> Display results`

## What this MVP includes

- A Streamlit demo UI for uploading a document
- Local file storage for uploads and outputs
- A parsing layer with open-source PDF readers and optional OCR hooks
- An adaptive local extraction layer with rule-based invoice extraction, learned template reuse, and a visible agent trace
- A human review step that lets users inspect the source PDF, copy parsed text, and correct fields before saving
- Validation rules for required fields, dates, and totals
- Persistence to JSON, CSV, and SQLite

## Project layout

- `app.py` - Streamlit UI
- `src/doc_ai/config.py` - environment-aware settings
- `src/doc_ai/pipeline.py` - end-to-end orchestration
- `src/doc_ai/parsers.py` - document parsing
- `src/doc_ai/extractors.py` - invoice extraction strategies
- `src/doc_ai/validators.py` - data quality checks
- `src/doc_ai/storage.py` - JSON, CSV, and SQLite output writers
- `src/doc_ai/schemas.py` - shared data structures
- `src/doc_ai/template_memory.py` - learned template signatures and anchors

## Quick start

1. Install Python 3.11+.
2. Create and activate a virtual environment.
3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env`.
5. Run the app:

```powershell
python -m streamlit run app.py
```

## Supported document flow

The first milestone from the plan is implemented as an invoice-focused pipeline.

- Best experience: upload a text-rich invoice PDF
- Also supported: `.txt`, `.md`, and `.json`
- If PDF tooling is unavailable, the app will explain what is missing and still supports text files

## Extraction modes

The app supports three extraction modes:

- `adaptive-local (default)` - uses learned templates first, then rule and heuristic fallback
- `template-only` - only uses previously learned templates
- `rule-based` - regex and label matching baseline

## Review workflow

After processing a document, the app shows:

- an embedded PDF preview for uploaded PDFs
- a copyable parsed text area
- editable extracted fields

If the extractor misses values or gets them wrong, a user can correct the fields and save the reviewed result. The reviewed data is re-validated and can be used for future template learning.

## Environment variables

- `APP_DATA_DIR` - overrides the local storage directory
- `ENABLE_TEMPLATE_LEARNING` - enables local learning from successful uploads
- `MIN_LEARNING_PASS_RATIO` - minimum validation pass ratio required before a new template is stored

## Notes

- This is intentionally a proof of concept, not a production pipeline.
- The code is structured so the team can add a second document type later, such as NDAs or agreements.
- The adaptive layer improves by saving learned anchors and signatures from validated documents. It does not retrain a base model.
