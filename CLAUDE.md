# AI Coding Instructions

This file is read by AI coding assistants at the start of every session. Follow these instructions for all work in this repository.

## Start Here

Before writing any code, read these two files:

- [README.md](README.md) — architecture, document types, extraction modes, project layout, CI/CD strategy
- [CONTRIBUTING.md](CONTRIBUTING.md) — branch flow, commit policy, PR rules, test requirements, learning data safety

Do not skip this step. The README explains the five-layer architecture and the five supported document types. The CONTRIBUTING guide explains the `feature/* → dev → test → main` promotion flow and what CI must pass before merging.

---

## Project Architecture Summary

The pipeline has five layers. Understand all of them before making changes:

1. **Presentation** — `app.py` (Streamlit UI, four tabs: Single Document, Bulk Upload, Schema Settings, Admin Dashboard)
2. **Parsing** — `src/doc_ai/parsers.py` (unstructured → pypdf → pdfplumber → Tesseract OCR, all lazy imports)
3. **Extraction** — `src/doc_ai/extractors.py` (template memory → rule-based → LLM; one extractor class per document type)
4. **Validation** — `src/doc_ai/validators.py` (one validator class per document type, routed by `get_validator()`)
5. **Persistence** — `src/doc_ai/storage.py` (JSON, CSV, SQLite — both the legacy `document_results` table and per-type flat tables)

Supporting modules:

- `src/doc_ai/pipeline.py` — orchestrates parsing → extraction → validation → storage → template learning; owns `_compute_field_confidence()` and `_build_field_sources()` (both run on every document)
- `src/doc_ai/schema_config.py` — `FIELD_CATALOG`, `TABLE_NAMES`, `SchemaConfig` for per-type DB field selection
- `src/doc_ai/schemas.py` — shared dataclasses (`PipelineResult`, `ParsedDocument`, `ValidationCheck`); `PipelineResult` carries both `field_confidence: dict[str, float]` and `field_sources: dict[str, str]` (source method per field)
- `src/doc_ai/template_memory.py` — learned template signatures and anchors (`TemplateMemory`); also exports `BadPatternStore` (persists bad field values learned from human corrections to `bad_patterns.json`)
- `src/doc_ai/config.py` — `Settings` dataclass (frozen); use `dataclasses.replace()` to override at runtime

## Supported Document Types

| Type key | Extractor class | Validator class | DB table |
|---|---|---|---|
| `invoice` | `RuleBasedInvoiceExtractor` | `InvoiceValidator` | `invoices` |
| `medical_discharge` | `RuleBasedMedicalDischargeExtractor` | `MedicalDischargeValidator` | `discharge_summaries` |
| `nda` | `RuleBasedNDAExtractor` | `NDAValidator` | `ndas` |
| `lab_report` | `RuleBasedLabReportExtractor` | `LabReportValidator` | `lab_reports` |
| `business_doc` | `RuleBasedBusinessDocExtractor` | `BusinessDocValidator` | `business_docs` |

Adding a new document type requires changes in: `extractors.py`, `validators.py`, `schema_config.py`, `storage.py`, `app.py` (FIELDS_BY_TYPE, LIST_FIELDS_BY_TYPE), all test files, and the fixture set (see [Test Fixtures](#test-fixtures) below).

---

## Testing Rules

**Every change must include tests. No exceptions.**

- Bug fixed → add a regression test that would have caught it
- New feature → add tests covering the happy path and key edge cases
- New document type → add extraction tests, validation tests, and SchemaConfig tests
- UI change → add AppTest tests verifying the new behaviour

### Where tests live

| What changed | Test file |
|---|---|
| Pipeline, extractors, validators, schema config, confidence scoring | `tests/test_pipeline.py` |
| UI layout, widgets, tab content, session state, admin controls | `tests/test_ui.py` |
| App config, settings, OCR library/binary availability | `tests/test_config.py` |
| Metrics dashboard helpers (`get_metrics`, `llm_usage_daily`, etc.) | `tests/test_metrics_dashboard.py` |
| Browser-level end-to-end upload and processing flows | `tests/test_e2e.py` |

### Running tests

```bash
# Full suite (excludes E2E — no browser required)
pytest tests/ -v

# By file
pytest tests/test_pipeline.py -v
pytest tests/test_ui.py -v

# Single class
pytest tests/test_ui.py::TestAdminDashboard -v

# E2E only (requires: pip install -r requirements-e2e.txt && playwright install chromium)
pytest tests/test_e2e.py -m e2e -v
```

Tests must pass on Python 3.11 and 3.12. CI uses `requirements-test.txt` (lean — no PDF/OCR/LLM stack). Heavy imports in `parsers.py` and `extractors.py` are all lazy (`try/except ImportError`) so they are safe to omit in CI.

E2E tests are excluded from CI by default (they require a running browser and Streamlit server). Run them locally before releasing user-visible UI changes.

### Streamlit AppTest patterns

The UI tests use `streamlit.testing.v1.AppTest`. Key rules:

```python
# Correct — AppTest SafeSessionState does NOT support .get()
value = at.session_state["admin_confidence_threshold"]  # KeyError if missing

# Wrong — this raises AttributeError in AppTest
value = at.session_state.get("admin_confidence_threshold", 0.80)

# Correct — check membership before access
with pytest.raises((KeyError, AttributeError)):
    _ = at.session_state["key_that_should_not_exist"]

# Correct — tab content IS rendered even without clicking the tab
# All st.tabs() content renders in a single pass; AppTest does not require tab switching

# Correct — iterate widgets, not session state
slider_labels = [s.label.lower() for s in at.slider]

# Wrong — list(at.session_state) raises errors
keys = list(at.session_state)  # do not do this
```

Helper functions in `tests/test_ui.py` — use them:

```python
_app(tmp_path)          # creates an AppTest wired to a throwaway data dir
_single_uploader(at)    # returns the single-file file_uploader widget
_bulk_uploader(at)      # returns the multi-file file_uploader widget
_process_btn(at)        # returns the first button with "process" in its label
_all_text(at)           # joins all title/header/markdown/text/caption/code values
_txt_upload(name, content)  # builds a (name, bytes, mime) tuple for text uploads
_upload(path)           # builds a (name, bytes, mime) tuple from a fixture PDF path
```

Fixture PDFs live in `tests/fixtures/`. Use `pytest.skip()` when a fixture is missing rather than failing.

---

## Test Fixtures

`tests/fixtures/` contains 45 purpose-built PDFs — 9 per document type — covering all test scenarios. Do not add numbered generic fixtures (old `invoice_001.pdf` style is gone).

### Fixture naming convention

```
{type}_format_a_full.pdf       # complete data, Format A layout (primary fixture)
{type}_format_b_full.pdf       # complete data, Format B layout (alternate layout)
{type}_format_a_similar.pdf    # complete data, same layout as A (template matching pair)
{type}_format_a_missing.pdf    # missing optional fields, Format A
{type}_format_b_missing.pdf    # missing optional fields, Format B
{type}_no_text_full.pdf        # image-only PDF, complete data (requires Tesseract)
{type}_no_text_missing.pdf     # image-only PDF, missing fields (requires Tesseract)
{type}_format_a_full_dup.pdf   # byte-identical copy of format_a_full (dedup testing)
{type}_no_text_dup.pdf         # byte-identical copy of no_text_full (dedup testing)
```

`{type}` is one of: `invoice`, `medical_discharge`, `nda`, `lab_report`, `business_doc`.

### Ground-truth JSON

`tests/fixtures/truth_data/{type}_truth.json` holds expected field values for the five searchable fixtures (full A, similar A, missing A, full B, missing B). The module-level `_TRUTH_DATA` dict in `test_pipeline.py` loads all truth files at import time.

When adding a new document type, add a corresponding `{type}_truth.json` with entries for all five searchable scenarios.

### OCR / Tesseract guard

`test_pipeline.py` exposes `_TESSERACT_AVAILABLE` (bool, set at module load). Tests that require Tesseract must guard themselves:

```python
if not _TESSERACT_AVAILABLE:
    pytest.skip("Tesseract not installed — OCR unavailable")
```

No-text dup tests guard on `"no_text" in fixture_name and not _TESSERACT_AVAILABLE` — image-only PDFs can't produce a content hash without Tesseract, so duplicate detection never fires.

### Test classes in `test_pipeline.py` covering fixtures

| Class | What it tests |
|---|---|
| `TestFixtureGroundTruth` | Document type ID and field extraction vs truth data (parametrized) |
| `TestFixtureMissingData` | Null truth fields are `None`; missing fixtures flagged `needs_review` |
| `TestFixtureDuplicateDetection` | Content-hash dedup: dup after original → `duplicate=True`; dup alone → not flagged |
| `TestFixtureTemplateMatching` | `format_a_similar` reuses template learned from `format_a_full`; field source is `Template` |
| `TestFixtureOCR` | Image-only PDFs produce non-empty parsed text and correct doc type (requires Tesseract) |
| `TestFixtureCorrectionFlow` | `finalize_review()` saves corrections; `approve_for_future_matching=True` triggers learning |

---

## Code Patterns

### Adding a new document type extractor

1. Add detection signals to `_DOC_TYPE_SIGNALS` in `extractors.py`
2. Add an `_empty_<type>()` function returning the skeleton dict with all fields as `None`
3. Add a `RuleBased<Type>Extractor` class that inherits from `BaseExtractor` with an `extract(parsed_document)` method
4. Add routing in `RuleBasedInvoiceExtractor.extract()` (it routes all types)
5. Register in `build_extractor()` if needed
6. Add a `<Type>Validator` class in `validators.py` and register in `get_validator()`
7. Add fields to `FIELD_CATALOG` and `TABLE_NAMES` in `schema_config.py`
8. Add `FIELDS_BY_TYPE` and `LIST_FIELDS_BY_TYPE` entries in `app.py`
9. Add 9 fixture PDFs to `tests/fixtures/` following the naming convention above
10. Add `tests/fixtures/truth_data/{type}_truth.json` with expected field values for the 5 searchable fixtures

### Settings overrides at runtime

`Settings` is a frozen dataclass. Override fields without mutating:

```python
from dataclasses import replace
updated = replace(runtime_settings, min_learning_pass_ratio=0.75)
```

### Admin dashboard config values

Admin widgets write to session state via `key=` params. Other parts of `main()` read them before the tabs render:

```python
confidence_threshold = st.session_state.get("admin_confidence_threshold", 0.80)
```

This means on first load the default applies; after the user visits Admin Dashboard and changes a value, every subsequent rerun picks up the new value. Do not read them after the tab is declared.

### Confidence scoring

`DocumentPipeline._compute_field_confidence()` returns a `{field: float}` dict with scores from 0.0 to 1.0. Scores are rounded to 3 decimal places. When writing tests, account for rounding:

```python
assert conf["field"] == pytest.approx(0.75 * 0.75, abs=0.001)  # not exact float math
```

Per-source confidence baselines used as starting points (defined in `pipeline.py`):

| Source | Baseline |
|---|---|
| `Cross-validated` | 0.97 |
| `Manual` | 0.95 |
| `LLM` | 0.88 |
| `Template` | 0.82 |
| `Spatial` | 0.78 |
| `Rule-based` | 0.72 |
| `Inferred` | 0.65 |

`_build_field_sources()` populates `PipelineResult.field_sources` by parsing the extraction trace. Both methods run together after every extraction.

### Heavy dependencies are lazy

`unstructured`, `pypdf`, `pdfplumber`, `pypdfium2`, `pytesseract`, `Pillow`, and `openai` are all imported inside `try/except ImportError` blocks. Never add them as top-level imports. This keeps the test suite runnable without the full dependency stack.

---

## Things to Watch Out For

**Dead code after `return` in Python**: A `def func():` at column-0 that appears textually inside another function's body is module-level, not nested. Code after a `return` statement is unreachable but Python will not warn you. If tab rendering code or function bodies seem to have no effect, check whether they ended up after a `return` in an enclosing function.

**Streamlit tab rendering order**: `st.tabs()` declares the tab bar. Anything rendered between `st.tabs()` and the first `with tab_x:` block appears outside all tabs (always visible below the tab bar). All tab content must be inside a `with tab_x:` block.

**`docs_processed_total` must be incremented in both flows**: Single-file increments after a successful `process_upload()` call. Bulk increments by the count of non-failed, non-duplicate results after `_run_bulk_processing()`. If you add a new processing path, add the increment there too.

**`render_bulk_summary` is dead code** — it is defined but the live bulk flow uses `_render_bulk_results()`. Do not call `render_bulk_summary` for new features.

**Template learning is skipped for `llm-assisted` mode** until a user explicitly approves the result. `allow_automatic_learning=False` is passed in that case. Do not remove this guard.

**`finalize_review()` uses `get_validator()`**, not a `self._validator` attribute. There is no instance-level validator on `DocumentPipeline`. Always route through `get_validator(doc_type)`.

**LLM result is merged per-field, not as a wholesale overwrite**: `_merge_llm_result()` in `extractors.py` applies the LLM value only when the LLM returned one; the prior extracted value is restored when the LLM returns null. When both the LLM and the prior method independently produced the same value, the field is recorded as `Cross-validated` and its confidence is boosted to 0.97. Do not reintroduce a wholesale overwrite for new extraction paths.

**`_resolve_schema_fields()` in `app.py`** is the single source of truth for resolving field order, list fields, and optional fields from `SchemaConfig`/`FIELD_CATALOG`. Use it instead of duplicating that logic.

---

## CI and Workflow

```
feature/* → dev → test → main
```

- Work on a `feature/*` branch, open a PR into `dev`
- CI runs `tests/test_pipeline.py` and `tests/test_ui.py` on Python 3.11 and 3.12
- CI installs `requirements-test.txt` (lean); production smoke check on `main` uses `requirements.txt` (full)
- All three workflows pin Node.js 24 via `actions/setup-node@v4`
- Learning artifacts are environment-scoped (`data/dev`, `data/test`, `data/prod`) and must be promoted intentionally via the `promote-learning` workflow — never copy raw dev learning to prod

## Commit Style

```
feat: short description
fix: short description
test: short description
docs: short description
ci: short description
refactor: short description
```

One purpose per commit. No secrets, no `__pycache__`, no `.venv`, no `data/` folders.

## Before Opening a PR

- [ ] `pytest tests/ -v` passes with no failures
- [ ] New behaviour has test coverage
- [ ] README updated if the feature is user-visible
- [ ] No secrets or local data files staged
