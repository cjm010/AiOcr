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

The platform now supports five document types across healthcare, legal, and business domains, with a configurable database schema, per-field confidence scoring, and an admin dashboard for session monitoring and pipeline configuration.

## Core Capabilities

- Streamlit web app for upload, review, and results
- Open-source PDF/text parsing pipeline (unstructured, pypdf, pdfplumber, Tesseract OCR fallback)
- Adaptive extraction with template memory, rule-based fallback, and optional LLM assistance for unfamiliar layouts
- **Five document types**: invoices, medical discharge summaries, NDAs, lab reports, and business documents
- Validation for required fields, dates, totals, and data-quality rules per document type
- **Per-field confidence scores** — each extracted field shows a color-coded confidence badge (green ≥ 85%, orange ≥ 60%, red below that) in the review form
- **Extraction completeness indicator** — progress bar showing filled vs. expected fields with contextual guidance (Good / Partial / Low)
- **Admin dashboard** — live session KPIs (docs processed, corrections, approvals, review rate), configurable template match threshold, extraction confidence threshold, output format selector, and auto-approve toggle
- **Database schema settings** — per-type flat SQLite tables with field-level checkbox configuration; DDL preview before saving
- Human review workflow with PDF preview and copyable parsed text
- Output persistence to JSON, CSV, SQLite, and extraction trace logs
- Multi-provider LLM support: OpenAI, Groq, OpenRouter, Ollama, and Gemini
- Colab notebook version for portable demos and collaboration

## Architecture Overview

The solution is made up of five main layers:

1. Presentation Layer
   - Streamlit UI for uploading documents
   - Displays extracted fields, extraction completeness %, validation results, PDF preview, parsed text, and manual correction form

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
- **extraction completeness score** — a percentage metric and progress bar showing how many of the 9 expected fields were filled, with a contextual status label:
  - **Good** (≥ 80%) — most fields extracted successfully
  - **Partial** (50–79%) — review and fill in missing fields
  - **Low** (< 50%) — many fields missing, manual review needed
- validation results
- a correction form for manual edits

If no reliable match is found, or if extraction is incomplete, the user can enter the correct data directly. The approved data is then saved and can contribute to future template learning when validation quality is high enough.

## Admin Dashboard

The admin dashboard (visible below the main tabs) provides:

- **Session KPIs** — docs processed, manual corrections, approvals, review rate, and active extraction mode
- **Template Match Threshold** — slider to control the minimum similarity score before a learned template is applied (default 0.55)
- **Extraction Confidence Threshold** — slider to set the minimum field confidence for auto-approval (default 0.80)
- **Output Format** — selector to limit downloads to JSON, CSV, or both
- **Auto-approve high confidence results** — when enabled, documents where all present fields meet the confidence threshold are approved without manual review in both single-file and bulk flows
- **System Health** — validation pass/fail/warn counts and bar chart for the last processed document

## Database Schema Settings

The Schema Settings tab lets users control which extracted fields are saved to the per-type SQLite tables:

- Required fields are always included and cannot be deselected
- Optional fields can be toggled on or off per document type
- A DDL preview shows the exact `CREATE TABLE` statement before saving
- Changes take effect for documents processed after saving

The five per-type tables are: `invoices`, `discharge_summaries`, `ndas`, `lab_reports`, `business_docs`.

For stronger consistency on repeated uploads of the same format:

- the LLM now uses a stricter JSON-only extraction request
- the app reuses learned templates sooner when the document layout is similar, but falls back to the LLM if the template result is too incomplete
- a user can explicitly approve either the current extracted result or a reviewed correction for future matching

## Project Layout

- `app.py` — Streamlit UI, admin dashboard, review workflow
- `src/doc_ai/config.py` — app settings and local paths
- `src/doc_ai/pipeline.py` — end-to-end orchestration and field confidence scoring
- `src/doc_ai/parsers.py` — PDF and text parsing with OCR fallback
- `src/doc_ai/extractors.py` — adaptive, template-based, and rule-based extraction for all five document types
- `src/doc_ai/validators.py` — data quality rules per document type
- `src/doc_ai/storage.py` — JSON, CSV, SQLite persistence (document-level and per-type flat tables)
- `src/doc_ai/schemas.py` — shared data structures
- `src/doc_ai/schema_config.py` — per-type field catalog, DB table names, and schema settings
- `src/doc_ai/template_memory.py` — learned template signatures and anchors
- `requirements.txt` — full runtime dependencies
- `requirements-test.txt` — lean CI dependencies (no PDF/OCR/LLM stack)
- `AI_Document_Extraction_Colab.ipynb` — Google Colab version of the project

## Quick Start

### Option A — Run locally (no Docker)

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

---

### Option B — Run with Docker (portable, includes Ollama)

Docker packages the entire app and a local Ollama LLM server into one command. No Python install needed on the target machine — just Docker.

**Prerequisites:**

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows / Mac / Linux)

**Steps:**

1. Copy `.env.example` to `.env` and fill in any API keys you want to use (optional — Ollama works with no keys).

2. Start everything:

```bash
docker compose up --build
```

3. Open your browser at `http://localhost:8501`.

That's it. The app and Ollama start together. The first startup takes a few minutes while Docker builds the image.

**Pull a model for Ollama (first time only):**

In a separate terminal while the containers are running:

```bash
docker exec -it aiocr-ollama-1 ollama pull qwen2.5:7b
```

Swap `qwen2.5:7b` for any model from [ollama.com/library](https://ollama.com/library). Good starting points:

| Model | Size | Good for |
|---|---|---|
| `qwen2.5:7b` | ~5 GB | Fast, solid extraction |
| `qwen2.5:14b` | ~9 GB | Better accuracy |
| `llama3.2:3b` | ~2 GB | Lightweight, quick |
| `llama3.3:70b` | ~40 GB | Best quality, needs GPU |

Then in the app sidebar: set **LLM provider** to `ollama` and **model** to the name you pulled (e.g. `qwen2.5:7b`).

**GPU support:**

By default Ollama runs on CPU. To enable GPU acceleration, open `docker-compose.yml` and uncomment the GPU block that matches your hardware (NVIDIA or AMD). On CPU, smaller models (3B–7B) run fine; larger models (14B+) will be slow.

**Stop the app:**

```bash
docker compose down
```

Your data and downloaded models are preserved in Docker volumes and will be there next time you run `docker compose up`.

**Deploy to a cloud server:**

The same `docker compose up` command works on any Linux server. For GPU-accelerated Ollama on AWS:

1. Launch an EC2 instance — `g4dn.xlarge` (NVIDIA T4, ~$0.53/hr) is a good starting point
2. Install Docker: `curl -fsSL https://get.docker.com | sh`
3. Clone this repo and copy your `.env`
4. Uncomment the NVIDIA GPU block in `docker-compose.yml`
5. `docker compose up --build -d`

The app will be accessible at `http://<your-ec2-ip>:8501`.

## Supported Document Types

| Type | Key Extracted Fields |
|---|---|
| Invoice | vendor, invoice number, date, due date, line items, subtotal, tax, total |
| Medical Discharge Summary | patient, dates, diagnoses, medications, discharge instructions, follow-up |
| NDA | parties, agreement type, effective date, term, governing law, confidentiality scope |
| Lab Report | patient, ordering physician, lab name, test panels with values and units, clinical interpretation |
| Business Document | company, document subtype, report period, KPIs, executive summary, recommendations |

## Supported File Formats

- `.pdf` — text-based and scanned (OCR fallback via Tesseract)
- `.txt`
- `.md`
- `.json`

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

- choose a provider: OpenAI, Groq, OpenRouter, Ollama, or Gemini
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
  - runs on Python 3.11 and 3.12 with Node.js 24
  - installs lean test dependencies (`requirements-test.txt`) rather than the full PDF/OCR stack
  - runs import smoke checks, pipeline tests, and UI tests on pull requests and pushes

- `.github/workflows/release.yml`
  - runs a protected production smoke check on `main` with full `requirements.txt`

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
