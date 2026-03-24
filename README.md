# AI Document Extraction and Data Quality Platform

This repository contains a proof-of-concept system for extracting structured data from business documents such as invoices. The app accepts uploaded documents, parses them into text, extracts key fields, validates the results, lets a user correct anything that was missed, and stores the approved output for future reuse.

At a high level, the solution follows this flow:

`Upload document -> Parse document -> Extract fields -> Validate data -> Review corrections -> Store output -> Learn from approved results`

## What This Solution Is

This project is an open-source document extraction and data quality workflow built for demos, capstones, and early-stage prototypes. It is designed to show how a team can:

- ingest business documents through a simple UI
- parse PDFs and text files into usable content
- extract structured invoice fields
- validate data quality with explicit business rules
- support human-in-the-loop correction when extraction is incomplete
- improve future results by learning reusable document patterns

The current implementation is focused on invoices, but the code is structured so the team can later add more document types such as contracts, agreements, or NDAs.

## Core Capabilities

- Streamlit web app for upload, review, and results
- Open-source PDF/text parsing pipeline
- Adaptive extraction with template memory and rule-based fallback
- Validation for required fields, dates, and totals
- Human review workflow with PDF preview and copyable parsed text
- Output persistence to JSON, CSV, SQLite, and extraction trace logs
- Colab notebook version for portable demos and collaboration

## Architecture Overview

The solution is made up of five main layers:

1. Presentation Layer
   - Streamlit UI for uploading documents
   - Displays extracted fields, validation results, PDF preview, parsed text, and manual correction form

2. Ingestion and Parsing Layer
   - Saves uploads locally
   - Parses document content using open-source libraries such as `unstructured`, `pypdf`, and `pdfplumber`

3. Extraction Layer
   - Builds a document signature from text structure
   - Checks template memory for similar previously approved documents
   - Reuses learned anchors when a match is found
   - Falls back to rule-based and heuristic extraction when no match exists

4. Validation and Review Layer
   - Validates required fields, dates, and totals
   - Allows a user to inspect the source document and manually correct values
   - Re-runs validation on reviewed data

5. Persistence and Learning Layer
   - Saves outputs to JSON, CSV, SQLite, and trace files
   - Stores learned templates from high-quality approved runs
   - Improves future extraction for similar document structures

## High-Level Architecture

```text
User
  |
  v
Streamlit UI
  |
  v
Local Upload Storage
  |
  v
Document Parser
  |
  v
Adaptive Extraction Agent
  |----> Template Memory
  |----> Rule-Based Extraction
  |----> Heuristic Inference
  |
  v
Structured Output
  |
  v
Validation Engine
  |
  v
Human Review and Corrections
  |
  v
JSON / CSV / SQLite / Trace Output
  |
  v
Learning Update into Template Memory
```

## Learning Model Architecture

The learning part of this solution does not retrain a large language model. Instead, it learns from approved document runs using adaptive memory.

The system learns in three ways:

- document signature learning
  - remembers top-line structure and key keywords from prior documents

- template anchor learning
  - remembers which labels and line patterns correspond to fields like vendor name, invoice date, and total amount

- human-reviewed feedback
  - if a user corrects extracted values, the approved output can be used to improve future matching and extraction behavior

This means the solution gets better over time without requiring expensive model training infrastructure.

## Human Review Workflow

After processing a document, the app provides:

- an embedded PDF preview for uploaded PDFs
- copyable parsed text
- extracted field values
- validation results
- a correction form for manual edits

If no reliable match is found, or if extraction is incomplete, the user can enter the correct data directly. The approved data is then saved and can contribute to future template learning when validation quality is high enough.

## Project Layout

- `app.py` - Streamlit UI and review workflow
- `src/doc_ai/config.py` - app settings and local paths
- `src/doc_ai/pipeline.py` - end-to-end orchestration
- `src/doc_ai/parsers.py` - PDF and text parsing
- `src/doc_ai/extractors.py` - adaptive, template-based, and rule-based extraction
- `src/doc_ai/validators.py` - data quality rules
- `src/doc_ai/storage.py` - JSON, CSV, SQLite, and trace persistence
- `src/doc_ai/schemas.py` - shared data structures
- `src/doc_ai/template_memory.py` - learned template signatures and anchors
- `AI_Document_Extraction_Colab.ipynb` - Google Colab version of the project

## Quick Start

1. Install Python 3.11 or newer.
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

For environment-specific runs:

```powershell
$env:APP_ENV="dev"
python -m streamlit run app.py
```

## Supported Inputs

- PDF invoices
- `.txt` files
- `.md` files
- `.json` files

Best results come from text-based PDFs. If a PDF does not contain a readable text layer, additional OCR support may be needed.

## Extraction Modes

The app supports three extraction modes:

- `adaptive-local`
  - tries learned templates first, then falls back to rule-based and heuristic extraction

- `template-only`
  - only uses previously learned templates

- `rule-based`
  - uses regex and label-based extraction without adaptive memory

## Environment Variables

- `APP_ENV`
  - identifies the running environment, such as `dev`, `test`, or `prod`

- `APP_DATA_ROOT`
  - base folder used to create isolated environment data directories

- `APP_DATA_DIR`
  - overrides the local data directory

- `ENABLE_TEMPLATE_LEARNING`
  - enables learning from successful reviewed or validated runs

- `MIN_LEARNING_PASS_RATIO`
  - sets the minimum validation pass ratio required before a new template is saved

## Output Artifacts

For each processed document, the system can produce:

- structured JSON output
- CSV output
- SQLite records
- extraction trace logs

These outputs support demos, auditing, troubleshooting, and future expansion.

## CI/CD and Environment Strategy

This repo is set up for a three-environment flow:

- `dev`
- `test`
- `main` as production

GitHub Actions included in this repo:

- `.github/workflows/ci.yml`
  - runs dependency install, import smoke checks, and tests on pull requests and pushes

- `.github/workflows/release.yml`
  - runs a protected production smoke check on `main`

- `.github/workflows/promote-learning.yml`
  - manually promotes approved learning artifacts from `dev` or `test` into a higher environment

The key policy is that learning artifacts must be isolated by environment and promoted intentionally. Development and test learning should never automatically overwrite production memory.

More detail is documented in [docs/BRANCHING_AND_LEARNING_STRATEGY.md](C:/Users/colli/OneDrive/Documents/GitHub/AiOcr/docs/BRANCHING_AND_LEARNING_STRATEGY.md).

## Notes

- This is a proof of concept, not a production-hardened system.
- The current extraction logic is invoice-focused.
- The adaptive learning layer is based on reusable memory, not full model retraining.
- The design is intentionally modular so more document types and smarter extraction strategies can be added later.
