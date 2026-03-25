# AI-Powered Data Quality Platform for Unstructured Data

This repository contains a proof-of-concept AI-driven data quality platform for unstructured data. The app ingests files such as PDFs and text documents, converts them into structured records, validates the outputs, supports human review and correction, and stores the approved results for downstream analytics and business use.

At a high level, the solution follows this flow:

`Upload document -> Parse unstructured content -> Extract and structure fields -> Validate data quality -> Review corrections -> Store output -> Learn from approved results`

## What This Solution Is

This project is a graduate capstone prototype designed around the sponsor brief for an AI-powered data quality platform. It is intended to show how a team can:

- ingest business documents through a simple UI
- parse PDFs and text files into usable content
- convert unstructured inputs into structured datasets
- validate data quality with explicit quality rules and profiling checks
- support human-in-the-loop correction when extraction is incomplete
- improve future results by learning reusable document patterns and approved corrections

The current prototype focuses on invoice-like documents as the first milestone, but the code is structured so the team can later expand to other unstructured sources such as reports, agreements, and domain-specific records.

## Core Capabilities

- Streamlit web app for upload, review, and results
- Open-source PDF/text parsing pipeline
- Adaptive extraction with template memory, rule-based fallback, and optional LLM assistance for unfamiliar layouts
- Validation for required fields, dates, totals, and data-quality style checks
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
   - Uses an optional LLM reasoning layer when the document format is new or weakly matched

4. Validation and Review Layer
   - Validates required fields, dates, totals, and structured output quality
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
  |----> Optional LLM Reasoning Layer for Unfamiliar Formats
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

## LLM and Learning Architecture

The current implementation can use an LLM-assisted extraction path for unfamiliar invoice formats while still keeping human review in the loop. The system does not retrain or fine-tune the foundation model. Instead, it learns from approved document runs using adaptive memory and human-reviewed corrections.

The LLM position in the target architecture is:

- the LLM acts as a reasoning layer for tasks such as schema inference, extraction help, anomaly explanation, and metadata enrichment
- the system first checks whether a document layout looks familiar
- for familiar layouts, it uses learned templates or local extraction logic
- for unfamiliar layouts, the `llm-assisted` mode can call an LLM to interpret the document text and return structured JSON
- after that, the same validation, review, and persistence layers still apply

This means:

- the LLM helps accelerate extraction for new invoice formats
- the reviewer still decides whether the extracted elements are correct
- approved corrections improve future behavior through template memory
- the current project does **not** train the base LLM itself
- repeated runs become more stable once a user approves a correct result for future matching
- unreviewed `llm-assisted` runs do not automatically update template memory

The system learns in three ways:

- document signature learning
  - remembers top-line structure and key keywords from prior documents

- template anchor learning
  - remembers which labels and line patterns correspond to structured fields

- human-reviewed feedback
  - if a user corrects extracted values, the approved output can be used to improve future matching and extraction behavior

This means the solution gets better over time without requiring model fine-tuning infrastructure during the capstone phase.

## Human Review Workflow

After processing a document, the app provides:

- an embedded PDF preview for uploaded PDFs
- copyable parsed text
- extracted field values
- validation results
- a correction form for manual edits

If no reliable match is found, or if extraction is incomplete, the user can enter the correct data directly. The approved data is then saved and can contribute to future template learning when validation quality is high enough.

For stronger consistency on repeated uploads of the same format:

- the LLM now uses a stricter JSON-only extraction request
- the app reuses learned templates sooner when the document layout is similar, but falls back to the LLM if the template result is too incomplete
- a user can explicitly approve either the current extracted result or a reviewed correction for future matching

## Project Layout

- `app.py` - Streamlit UI and review workflow
- `src/doc_ai/config.py` - app settings and local paths
- `src/doc_ai/pipeline.py` - end-to-end orchestration
- `src/doc_ai/parsers.py` - PDF and text parsing
- `src/doc_ai/extractors.py` - adaptive, template-based, and rule-based extraction logic
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

If you want to use `llm-assisted` mode, you can either:

- enter your provider API key and choose a model directly in the Streamlit sidebar, or
- set `LLM_PROVIDER`, `OPENAI_API_KEY`, and `OPENAI_MODEL` in your local environment or `.env`

## Supported Inputs

- PDF invoices and similar unstructured business documents
- `.txt` files
- `.md` files
- `.json` files

Best results come from text-based PDFs. If a PDF does not contain a readable text layer, additional OCR support may be needed.

## Extraction Modes

The app supports three extraction modes:

- `llm-assisted`
  - uses the LLM reasoning layer when the document format looks unfamiliar, then applies the normal validation and review flow

- `adaptive-local`
  - tries learned templates first, then falls back to rule-based and heuristic extraction

- `template-only`
  - only uses previously learned templates

- `rule-based`
  - uses regex and label-based extraction without adaptive memory

These modes represent the current implementation. The recommended choice for new invoice layouts is `llm-assisted`, while `adaptive-local` remains the lowest-cost option for repeated or well-known formats.

When `llm-assisted` is selected, the UI lets a user:

- choose a provider such as OpenAI, Groq, OpenRouter, or Ollama
- paste an API key for the current session when the provider needs one
- choose a recommended model from the sidebar
- enter a custom model id when needed
- override the base URL for a compatible hosted or local endpoint

The API key entered in the UI is kept only in the active Streamlit session. It is not written to repository files, output artifacts, or the learned template store.

For free testing, the easiest options are usually:

- `Groq`
  - fast hosted inference with a free developer tier

- `OpenRouter`
  - good for trying free-model variants behind one API

- `Ollama`
  - fully local testing with no per-call cost if you have the hardware

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

- `LLM_PROVIDER`
  - chooses the default LLM provider, such as `openai`, `groq`, `openrouter`, or `ollama`

- `LLM_BASE_URL`
  - optionally overrides the provider default endpoint for a compatible hosted or local API

- `OPENAI_API_KEY`
  - optionally enables the LLM-assisted extraction path when you do not want to enter the key in the UI

- `OPENAI_MODEL`
  - sets the default model used for `llm-assisted` extraction when a different model is not selected in the UI

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
- The current extraction logic is invoice-focused because that is the first milestone and easiest demoable structured document type.
- The adaptive learning layer is based on reusable memory and approved corrections, not full model retraining.
- LLM-assisted extraction is best used for unseen or weakly matched invoice formats, with a reviewer confirming the extracted elements before they are trusted.
- The design is intentionally modular so more document types, richer data-quality rules, OCR, and an LLM reasoning layer can be added later.
