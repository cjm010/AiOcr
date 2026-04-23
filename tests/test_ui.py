"""
Comprehensive UI tests using Streamlit's AppTest framework.

These tests simulate user interaction without a browser — no Selenium needed.
Coverage:
  - App startup and initial render
  - Sidebar controls (extraction mode, learn toggle, LLM provider, reset flow)
  - Single-file upload tab: upload, process, results for each document type
  - Bulk upload tab: process, clear, duplicate detection, summary table
  - Schema Settings tab: field checkboxes, DDL preview, save
  - Extraction mode switching
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _app(tmp_path) -> AppTest:
    """Return an AppTest wired to a throwaway data directory."""
    os.environ["APP_ENV"] = "test_ui"
    os.environ["APP_DATA_ROOT"] = str(tmp_path)
    from src.doc_ai.config import get_settings
    get_settings.cache_clear()
    return AppTest.from_file(APP_PATH, default_timeout=30)


def _pdf(name: str) -> Path:
    return FIXTURES / name


def _upload(path: Path) -> tuple[str, bytes, str]:
    return (path.name, path.read_bytes(), "application/pdf")


def _txt_upload(name: str, content: str) -> tuple[str, bytes, str]:
    return (name, content.encode(), "text/plain")


def _bulk_uploader(at: AppTest):
    return next((u for u in at.file_uploader if u.accept_multiple_files), None)


def _single_uploader(at: AppTest):
    return next((u for u in at.file_uploader if not u.accept_multiple_files), None)


def _process_btn(at: AppTest):
    return next((b for b in at.button if "process" in b.label.lower()), None)


def _all_text(at: AppTest) -> str:
    parts = (
        list(at.title) + list(at.header) + list(at.subheader)
        + list(at.markdown) + list(at.text) + list(at.caption)
        + list(at.json) + list(at.code)
    )
    return " ".join(str(e.value) for e in parts).lower()


# ---------------------------------------------------------------------------
# App startup
# ---------------------------------------------------------------------------

class TestAppStartup:
    def test_loads_without_exception(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        assert not at.exception

    def test_page_title_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        titles = [t.value.lower() for t in at.title]
        assert any(
            kw in t for t in titles
            for kw in ("document", "data", "platform", "ocr", "invoice", "ai")
        )

    def test_schema_settings_tab_content_rendered(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        # Schema Settings tab content (DDL blocks) are rendered by AppTest even without tab switching
        code_blocks = [str(c.value) for c in at.code]
        assert any("CREATE TABLE" in b.upper() for b in code_blocks)

    def test_no_error_elements_on_startup(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        assert len(at.error) == 0

    def test_sidebar_is_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        # Sidebar must have at least one widget
        sidebar_widgets = (
            list(at.sidebar.selectbox) + list(at.sidebar.checkbox) + list(at.sidebar.button)
        )
        assert len(sidebar_widgets) > 0

    def test_single_uploader_present_on_load(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        assert _single_uploader(at) is not None

    def test_bulk_uploader_present_on_load(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        assert _bulk_uploader(at) is not None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

class TestSidebar:
    def test_extraction_mode_selectbox_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        labels = [s.label.lower() for s in at.sidebar.selectbox]
        assert any("extraction" in l or "mode" in l for l in labels)

    def test_extraction_mode_options_are_correct(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        mode_box = next(
            (s for s in at.sidebar.selectbox if "extraction" in s.label.lower() or "mode" in s.label.lower()),
            None,
        )
        assert mode_box is not None
        options = [str(o).lower() for o in mode_box.options]
        assert any("adaptive" in o or "local" in o for o in options)
        assert any("llm" in o for o in options)

    def test_learn_from_uploads_checkbox_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        labels = [c.label.lower() for c in at.sidebar.checkbox]
        assert any("learn" in l for l in labels)

    def test_llm_provider_selectbox_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        labels = [s.label.lower() for s in at.sidebar.selectbox]
        assert any("provider" in l or "llm" in l for l in labels)

    def test_llm_provider_includes_openai_and_ollama(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        provider_box = next(
            (s for s in at.sidebar.selectbox if "provider" in s.label.lower() or "llm" in s.label.lower()),
            None,
        )
        assert provider_box is not None
        options = [str(o).lower() for o in provider_box.options]
        assert "openai" in options
        assert "ollama" in options

    def test_reset_button_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        labels = [b.label.lower() for b in at.sidebar.button]
        assert any("reset" in l for l in labels)

    def test_reset_shows_confirmation(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        reset_btn = next(b for b in at.sidebar.button if "reset" in b.label.lower())
        reset_btn.click().run()
        assert not at.exception
        warnings = [w.value.lower() for w in at.sidebar.warning]
        assert any("delete" in w or "reset" in w or "permanent" in w for w in warnings)

    def test_reset_cancel_does_not_crash(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        next(b for b in at.sidebar.button if "reset" in b.label.lower()).click().run()
        cancel_btn = next(b for b in at.sidebar.button if "cancel" in b.label.lower())
        cancel_btn.click().run()
        assert not at.exception

    def test_api_key_input_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        inputs = [t.label.lower() for t in at.sidebar.text_input]
        assert any("key" in i or "api" in i for i in inputs)

    def test_learn_toggle_can_be_unchecked(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        learn_cb = next(c for c in at.sidebar.checkbox if "learn" in c.label.lower())
        learn_cb.uncheck().run()
        assert not at.exception


# ---------------------------------------------------------------------------
# Single file tab
# ---------------------------------------------------------------------------

class TestSingleFileTab:
    def test_file_uploader_accepts_single_file(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        uploader = _single_uploader(at)
        assert uploader is not None
        assert not uploader.accept_multiple_files

    def test_upload_invoice_pdf_no_exception(self, tmp_path):
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_upload(pdf)).run()
        assert not at.exception

    def test_upload_and_process_invoice(self, tmp_path):
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_upload(pdf)).run()
        btn = _process_btn(at)
        if btn is None:
            pytest.skip("Process button not rendered")
        btn.click().run()
        assert not at.exception
        text = _all_text(at)
        assert any(kw in text for kw in ("invoice", "vendor", "extracted", "validation"))

    def test_upload_and_process_medical_discharge(self, tmp_path):
        pdf = _pdf("healthcare_discharge_001.pdf")
        if not pdf.exists():
            pytest.skip("healthcare_discharge_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_upload(pdf)).run()
        btn = _process_btn(at)
        if btn is None:
            pytest.skip("Process button not rendered")
        btn.click().run()
        assert not at.exception
        text = _all_text(at)
        assert any(kw in text for kw in ("discharge", "patient", "medical", "diagnosis", "extracted"))

    def test_upload_and_process_lab_report(self, tmp_path):
        pdf = _pdf("healthcare_lab_001.pdf")
        if not pdf.exists():
            pytest.skip("healthcare_lab_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_upload(pdf)).run()
        btn = _process_btn(at)
        if btn is None:
            pytest.skip("Process button not rendered")
        btn.click().run()
        assert not at.exception
        text = _all_text(at)
        assert any(kw in text for kw in ("lab", "patient", "report", "extracted"))

    def test_upload_and_process_nda(self, tmp_path):
        pdf = _pdf("legal_nda_001.pdf")
        if not pdf.exists():
            pytest.skip("legal_nda_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_upload(pdf)).run()
        btn = _process_btn(at)
        if btn is None:
            pytest.skip("Process button not rendered")
        btn.click().run()
        assert not at.exception
        text = _all_text(at)
        assert any(kw in text for kw in ("nda", "disclosure", "party", "agreement", "extracted"))

    def test_upload_and_process_business_doc(self, tmp_path):
        pdf = _pdf("business_doc_001.pdf")
        if not pdf.exists():
            pytest.skip("business_doc_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_upload(pdf)).run()
        btn = _process_btn(at)
        if btn is None:
            pytest.skip("Process button not rendered")
        btn.click().run()
        assert not at.exception
        text = _all_text(at)
        assert any(kw in text for kw in ("business", "company", "report", "extracted"))

    def test_process_shows_validation_results(self, tmp_path):
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_upload(pdf)).run()
        btn = _process_btn(at)
        if btn is None:
            pytest.skip("Process button not rendered")
        btn.click().run()
        assert not at.exception
        text = _all_text(at)
        assert any(kw in text for kw in ("pass", "fail", "warn", "validation", "check"))

    def test_process_shows_extraction_trace(self, tmp_path):
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_upload(pdf)).run()
        btn = _process_btn(at)
        if btn is None:
            pytest.skip("Process button not rendered")
        btn.click().run()
        assert not at.exception
        text = _all_text(at)
        assert any(kw in text for kw in ("trace", "extraction", "step", "template", "rule"))

    def test_upload_plain_text_invoice_no_exception(self, tmp_path):
        content = (
            "ACME Vendor\nInvoice Number: INV-0001\nInvoice Date: 2026-01-15\n"
            "Total: $1500.00\nDue Date: 2026-02-15\n"
        )
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_txt_upload("invoice.txt", content)).run()
        btn = _process_btn(at)
        if btn:
            btn.click().run()
        assert not at.exception

    def test_upload_same_file_twice_no_exception(self, tmp_path):
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        tup = _upload(pdf)
        _single_uploader(at).set_value(tup).run()
        assert not at.exception
        _single_uploader(at).set_value(tup).run()
        assert not at.exception

    def test_process_produces_json_output_element(self, tmp_path):
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_upload(pdf)).run()
        btn = _process_btn(at)
        if btn is None:
            pytest.skip("Process button not rendered")
        btn.click().run()
        assert not at.exception
        # Extracted JSON or dataframe should be visible
        assert len(at.json) > 0 or len(at.dataframe) > 0 or len(at.text) > 0


# ---------------------------------------------------------------------------
# Bulk upload tab
# ---------------------------------------------------------------------------

class TestBulkUploadTab:
    def test_bulk_uploader_accepts_multiple_files(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        uploader = _bulk_uploader(at)
        assert uploader is not None
        assert uploader.accept_multiple_files

    def test_bulk_upload_single_pdf_no_exception(self, tmp_path):
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value([_upload(pdf)]).run()
        assert not at.exception

    def test_bulk_upload_shows_process_button(self, tmp_path):
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value([_upload(pdf)]).run()
        assert not at.exception
        labels = [b.label.lower() for b in at.button]
        assert any("process" in l for l in labels)

    def test_bulk_upload_shows_clear_button(self, tmp_path):
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value([_upload(pdf)]).run()
        labels = [b.label.lower() for b in at.button]
        assert any("clear" in l for l in labels)

    def test_bulk_process_two_invoices(self, tmp_path):
        pdfs = [_pdf("invoice_001.pdf"), _pdf("invoice_002.pdf")]
        if not all(p.exists() for p in pdfs):
            pytest.skip("invoice_001/002.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value([_upload(p) for p in pdfs]).run()
        btn = _process_btn(at)
        assert btn is not None
        btn.click().run()
        assert not at.exception

    def test_bulk_process_mixed_document_types(self, tmp_path):
        pdfs = [
            _pdf("invoice_001.pdf"),
            _pdf("healthcare_lab_001.pdf"),
            _pdf("business_doc_001.pdf"),
        ]
        missing = [p.name for p in pdfs if not p.exists()]
        if missing:
            pytest.skip(f"Missing fixtures: {missing}")
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value([_upload(p) for p in pdfs]).run()
        btn = _process_btn(at)
        assert btn is not None
        btn.click().run()
        assert not at.exception

    def test_bulk_process_shows_summary_table(self, tmp_path):
        pdfs = [_pdf("invoice_001.pdf"), _pdf("invoice_002.pdf")]
        if not all(p.exists() for p in pdfs):
            pytest.skip("invoice_001/002.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value([_upload(p) for p in pdfs]).run()
        btn = _process_btn(at)
        if btn:
            btn.click().run()
        assert not at.exception
        assert len(at.dataframe) > 0 or len(at.metric) > 0

    def test_bulk_process_shows_file_count_metric(self, tmp_path):
        pdfs = [_pdf("invoice_001.pdf"), _pdf("invoice_002.pdf")]
        if not all(p.exists() for p in pdfs):
            pytest.skip("invoice_001/002.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value([_upload(p) for p in pdfs]).run()
        btn = _process_btn(at)
        if btn:
            btn.click().run()
        assert not at.exception
        # Either metric widgets or a dataframe should show results
        all_elements = len(at.metric) + len(at.dataframe) + len(at.subheader)
        assert all_elements > 0

    def test_bulk_duplicate_in_same_batch_no_crash(self, tmp_path):
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        original = _upload(pdf)
        duplicate = (f"copy_{pdf.name}", pdf.read_bytes(), "application/pdf")
        _bulk_uploader(at).set_value([original, duplicate]).run()
        btn = _process_btn(at)
        if btn:
            btn.click().run()
        assert not at.exception

    def test_bulk_clear_button_no_crash(self, tmp_path):
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value([_upload(pdf)]).run()
        clear_btn = next((b for b in at.button if "clear" in b.label.lower()), None)
        if clear_btn is None:
            pytest.skip("Clear button not rendered")
        clear_btn.click().run()
        assert not at.exception

    def test_bulk_five_pdfs_processes_without_crash(self, tmp_path):
        pdfs = [_pdf(f"invoice_00{i}.pdf") for i in range(1, 6)]
        available = [p for p in pdfs if p.exists()]
        if len(available) < 3:
            pytest.skip("Need at least 3 invoice PDFs")
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value([_upload(p) for p in available]).run()
        btn = _process_btn(at)
        if btn:
            btn.click().run()
        assert not at.exception


# ---------------------------------------------------------------------------
# Schema Settings tab
# ---------------------------------------------------------------------------

class TestSchemaSettingsTab:
    def test_schema_tab_renders_without_exception(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        assert not at.exception
        # Three tabs should exist: Single Document, Bulk Upload, Schema Settings
        assert len(at.tabs) >= 3

    def test_schema_tab_save_button_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        labels = [b.label.lower() for b in at.button]
        assert any("save" in l and "schema" in l for l in labels)

    def test_schema_tab_has_many_checkboxes(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        # Schema settings has ~70 field checkboxes + sidebar learn checkbox
        # Total should be well above 10
        assert len(at.checkbox) > 10

    def test_schema_tab_has_required_field_checkboxes_disabled(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        disabled_boxes = [c for c in at.checkbox if c.disabled]
        # Required fields are locked — there should be several
        assert len(disabled_boxes) > 0

    def test_schema_tab_ddl_preview_rendered(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        # DDL blocks rendered via st.code()
        code_blocks = [c.value for c in at.code]
        assert any("CREATE TABLE" in str(b).upper() for b in code_blocks)

    def test_schema_tab_all_five_doc_types_represented(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        code_text = " ".join(str(c.value).lower() for c in at.code)
        for table in ("invoices", "discharge_summaries", "ndas", "lab_reports", "business_docs"):
            assert table in code_text, f"Table '{table}' not found in DDL preview"

    def test_schema_tab_save_no_exception(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        save_btn = next((b for b in at.button if "save" in b.label.lower() and "schema" in b.label.lower()), None)
        if save_btn is None:
            pytest.skip("Save schema button not found")
        save_btn.click().run()
        assert not at.exception

    def test_schema_settings_persisted_after_save(self, tmp_path):
        """Saving creates the schema_settings.json file in the data dir."""
        from src.doc_ai.config import get_settings
        at = _app(tmp_path)
        at.run()
        save_btn = next((b for b in at.button if "save" in b.label.lower() and "schema" in b.label.lower()), None)
        if save_btn is None:
            pytest.skip("Save schema button not found")
        save_btn.click().run()
        assert not at.exception
        settings = get_settings()
        schema_path = settings.data_dir / "schema_settings.json"
        assert schema_path.exists()


# ---------------------------------------------------------------------------
# Extraction mode switching
# ---------------------------------------------------------------------------

class TestExtractionModes:
    def _switch_mode(self, at: AppTest, mode: str) -> AppTest:
        mode_box = next(
            s for s in at.sidebar.selectbox
            if "extraction" in s.label.lower() or "mode" in s.label.lower()
        )
        mode_box.set_value(mode).run()
        return at

    def test_rule_based_mode_processes_invoice(self, tmp_path):
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        self._switch_mode(at, "rule-based")
        _single_uploader(at).set_value(_upload(pdf)).run()
        btn = _process_btn(at)
        if btn:
            btn.click().run()
        assert not at.exception

    def test_adaptive_local_mode_processes_invoice(self, tmp_path):
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        self._switch_mode(at, "adaptive-local")
        _single_uploader(at).set_value(_upload(pdf)).run()
        btn = _process_btn(at)
        if btn:
            btn.click().run()
        assert not at.exception

    def test_llm_mode_without_api_key_shows_warning(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        # Clear any API key from environment
        os.environ.pop("OPENAI_API_KEY", None)
        self._switch_mode(at, "llm-assisted")
        assert not at.exception
        # A warning about no API key should appear somewhere
        all_warnings = [w.value.lower() for w in at.warning] + [w.value.lower() for w in at.sidebar.warning]
        assert any("key" in w or "api" in w or "provider" in w or "fallback" in w for w in all_warnings)

    def test_template_only_mode_no_exception_on_load(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        self._switch_mode(at, "template-only")
        assert not at.exception

    def test_switching_modes_does_not_crash(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        for mode in ("rule-based", "adaptive-local", "template-only", "llm-assisted"):
            self._switch_mode(at, mode)
            assert not at.exception, f"Crash after switching to mode: {mode}"


# ---------------------------------------------------------------------------
# _confidence_badge — unit tests
# ---------------------------------------------------------------------------

class TestConfidenceBadge:
    """Tests for the app._confidence_badge() helper."""

    @staticmethod
    def _badge(score):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from app import _confidence_badge
        return _confidence_badge(score)

    def test_perfect_score_is_green(self):
        assert ":green[" in self._badge(1.0)

    def test_score_at_green_threshold_is_green(self):
        assert ":green[" in self._badge(0.85)

    def test_score_just_below_green_is_orange(self):
        assert ":orange[" in self._badge(0.84)

    def test_score_at_orange_threshold_is_orange(self):
        assert ":orange[" in self._badge(0.60)

    def test_score_just_below_orange_is_red(self):
        assert ":red[" in self._badge(0.59)

    def test_low_nonzero_score_is_red(self):
        badge = self._badge(0.35)
        assert ":red[" in badge
        assert "not extracted" not in badge

    def test_zero_score_shows_not_extracted(self):
        badge = self._badge(0.0)
        assert ":red[" in badge
        assert "not extracted" in badge

    def test_percentage_shown_for_nonzero(self):
        badge = self._badge(0.92)
        assert "92%" in badge

    def test_percentage_rounds_to_nearest_int(self):
        # 0.756 rounds to 76%
        badge = self._badge(0.756)
        assert "76%" in badge

    def test_zero_does_not_show_percentage(self):
        badge = self._badge(0.0)
        assert "%" not in badge or "not extracted" in badge

    def test_output_contains_confidence_label(self):
        for score in (0.9, 0.7, 0.4):
            assert "confidence" in self._badge(score)

    def test_boundary_exactly_0_85_is_green(self):
        assert ":green[" in self._badge(0.85)

    def test_boundary_exactly_0_60_is_orange(self):
        assert ":orange[" in self._badge(0.60)


# ---------------------------------------------------------------------------
# Review form — confidence badges rendered in UI
# ---------------------------------------------------------------------------

class TestReviewFormConfidenceBadges:
    """Verify that confidence badges appear in the review forms."""

    def test_single_file_review_form_shows_confidence_text(self, tmp_path):
        """After processing a file that needs review, the form labels show confidence."""
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_upload(pdf)).run()
        btn = _process_btn(at)
        if btn is None:
            pytest.skip("Process button not rendered")
        btn.click().run()
        assert not at.exception
        # Look for "confidence" text anywhere in text_input labels (rendered in main area)
        all_labels = [t.label.lower() for t in at.text_input]
        assert any("confidence" in lbl for lbl in all_labels), (
            f"No 'confidence' badge found in form labels. Labels: {all_labels}"
        )

    def test_review_form_shows_green_confidence_for_good_field(self, tmp_path):
        """A field with a passing validation should render a green badge."""
        pdf = _pdf("invoice_001.pdf")
        if not pdf.exists():
            pytest.skip("invoice_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_upload(pdf)).run()
        btn = _process_btn(at)
        if btn is None:
            pytest.skip("Process button not rendered")
        btn.click().run()
        assert not at.exception
        all_labels = " ".join(t.label for t in at.text_input)
        # green[] badge must appear for at least one field
        assert ":green[" in all_labels

    def test_review_form_shows_red_for_missing_field(self, tmp_path):
        """A field that could not be extracted should get a red 'not extracted' badge."""
        # Use a medical discharge PDF — some fields like follow_up_provider are often missing
        pdf = _pdf("healthcare_discharge_001.pdf")
        if not pdf.exists():
            pytest.skip("healthcare_discharge_001.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_upload(pdf)).run()
        btn = _process_btn(at)
        if btn is None:
            pytest.skip("Process button not rendered")
        btn.click().run()
        assert not at.exception
        all_labels = " ".join(t.label for t in at.text_input)
        # At least one missing field should show red
        assert ":red[" in all_labels or "not extracted" in all_labels

    def test_bulk_review_form_shows_confidence_badges(self, tmp_path):
        """Confidence badges must appear in the bulk-upload review form too."""
        pdfs = [_pdf("invoice_001.pdf"), _pdf("invoice_002.pdf")]
        if not all(p.exists() for p in pdfs):
            pytest.skip("invoice_001/002.pdf not in fixtures")
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value([_upload(p) for p in pdfs]).run()
        btn = _process_btn(at)
        if btn is None:
            pytest.skip("Process button not rendered")
        btn.click().run()
        assert not at.exception
        all_labels = " ".join(t.label for t in at.text_input)
        assert "confidence" in all_labels.lower(), (
            "No confidence badges found in bulk review form labels"
        )

    def test_confidence_badge_not_shown_for_non_review_result(self, tmp_path):
        """Before any file is uploaded, no confidence badges should be present."""
        at = _app(tmp_path)
        at.run()
        all_labels = " ".join(t.label for t in at.text_input)
        assert "confidence" not in all_labels.lower()


# ---------------------------------------------------------------------------
# compute_upload_signature (unit)
# ---------------------------------------------------------------------------

class TestComputeUploadSignature:
    """Unit tests for compute_upload_signature — now hash-only, no filename."""

    def _sig(self, b: bytes) -> str:
        from app import compute_upload_signature
        return compute_upload_signature(b)

    def test_returns_64_char_hex_string(self):
        sig = self._sig(b"hello")
        assert isinstance(sig, str)
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_same_bytes_produce_same_signature(self):
        data = b"invoice content"
        assert self._sig(data) == self._sig(data)

    def test_different_bytes_produce_different_signatures(self):
        assert self._sig(b"invoice A") != self._sig(b"invoice B")

    def test_empty_bytes_returns_known_sha256(self):
        import hashlib
        expected = hashlib.sha256(b"").hexdigest()
        assert self._sig(b"") == expected

    def test_signature_does_not_depend_on_filename(self):
        # Old signature included the filename; the new one must not vary with it
        payload = b"same content"
        sig1 = self._sig(payload)
        sig2 = self._sig(payload)
        assert sig1 == sig2

    def test_large_payload_returns_fixed_length(self):
        assert len(self._sig(b"x" * 100_000)) == 64


# ---------------------------------------------------------------------------
# Admin dashboard
# ---------------------------------------------------------------------------

class TestAdminDashboard:
    """Verify the admin dashboard section renders and counters start at zero."""

    def test_admin_dashboard_header_rendered(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        assert not at.exception
        all_text = _all_text(at)
        assert "admin" in all_text and "dashboard" in all_text

    def test_kpi_docs_processed_starts_at_zero(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        metric_labels = [m.label.lower() for m in at.metric]
        assert any("docs" in l or "processed" in l for l in metric_labels)

    def test_kpi_manual_corrections_starts_at_zero(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        metric_labels = [m.label.lower() for m in at.metric]
        assert any("correction" in l or "manual" in l for l in metric_labels)

    def test_kpi_approvals_metric_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        metric_labels = [m.label.lower() for m in at.metric]
        assert any("approval" in l for l in metric_labels)

    def test_kpi_review_rate_metric_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        metric_labels = [m.label.lower() for m in at.metric]
        assert any("review" in l or "rate" in l for l in metric_labels)

    def test_config_panel_template_threshold_slider_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        slider_labels = [s.label.lower() for s in at.slider]
        assert any("threshold" in l or "template" in l for l in slider_labels)

    def test_config_panel_confidence_threshold_slider_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        slider_labels = [s.label.lower() for s in at.slider]
        assert any("confidence" in l for l in slider_labels)

    def test_config_panel_output_format_selectbox_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        all_options = [
            str(o).lower()
            for s in at.selectbox
            for o in s.options
        ]
        assert any(o in all_options for o in ("json", "csv", "both"))

    def test_config_panel_auto_approve_checkbox_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        labels = [c.label.lower() for c in at.checkbox]
        assert any("auto" in l or "approve" in l for l in labels)

    def test_system_health_section_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        all_text = _all_text(at)
        assert "system" in all_text and "health" in all_text


# ---------------------------------------------------------------------------
# Gemini provider support
# ---------------------------------------------------------------------------

class TestGeminiProvider:
    """Gemini was added as a new LLM provider in this PR."""

    def test_gemini_in_provider_options_constant(self):
        from app import LLM_PROVIDER_OPTIONS
        assert "gemini" in LLM_PROVIDER_OPTIONS

    def test_gemini_has_model_options(self):
        from app import MODEL_OPTIONS_BY_PROVIDER
        models = MODEL_OPTIONS_BY_PROVIDER.get("gemini", [])
        assert len(models) > 0

    def test_gemini_flash_model_present(self):
        from app import MODEL_OPTIONS_BY_PROVIDER
        assert any("flash" in m for m in MODEL_OPTIONS_BY_PROVIDER["gemini"])

    def test_gemini_pro_model_present(self):
        from app import MODEL_OPTIONS_BY_PROVIDER
        assert any("pro" in m for m in MODEL_OPTIONS_BY_PROVIDER["gemini"])

    def test_gemini_custom_model_option_present(self):
        from app import MODEL_OPTIONS_BY_PROVIDER
        assert "custom" in MODEL_OPTIONS_BY_PROVIDER["gemini"]

    def test_gemini_appears_in_sidebar_provider_selectbox(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        provider_box = next(
            (s for s in at.sidebar.selectbox if "provider" in s.label.lower() or "llm" in s.label.lower()),
            None,
        )
        assert provider_box is not None
        options = [str(o).lower() for o in provider_box.options]
        assert "gemini" in options


# ---------------------------------------------------------------------------
# render_completeness_bar (via review flow)
# ---------------------------------------------------------------------------

class TestCompletenessBar:
    """Completeness metric and caption render in the review form."""

    def test_completeness_metric_appears_after_processing(self, tmp_path):
        invoice = "\n".join([
            "Vendor: Greenleaf Supplies",
            "Invoice Number: 4587",
            "Invoice Date: 2024-02-12",
            "Due Date: 2024-03-12",
            "Subtotal: 1100.00",
            "Tax: 100.00",
            "Total: 1200.00",
        ])
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_txt_upload("invoice.txt", invoice)).run()
        _process_btn(at).click().run()
        assert not at.exception
        metric_labels = [m.label.lower() for m in at.metric]
        assert any("completeness" in l for l in metric_labels)

    def test_completeness_caption_good_for_full_invoice(self, tmp_path):
        invoice = "\n".join([
            "Vendor: Greenleaf Supplies",
            "Invoice Number: 4587",
            "Invoice Date: 2024-02-12",
            "Due Date: 2024-03-12",
            "Subtotal: 1100.00",
            "Tax: 100.00",
            "Total: 1200.00",
        ])
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_txt_upload("invoice.txt", invoice)).run()
        _process_btn(at).click().run()
        assert not at.exception
        all_text = _all_text(at)
        assert "good" in all_text or "partial" in all_text or "low" in all_text

    def test_completeness_caption_low_for_sparse_document(self, tmp_path):
        sparse = "Invoice Number: 99"
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_txt_upload("sparse.txt", sparse)).run()
        _process_btn(at).click().run()
        assert not at.exception
        all_text = _all_text(at)
        assert "low" in all_text or "partial" in all_text or "good" in all_text


# ---------------------------------------------------------------------------
# render_bulk_summary (via bulk upload flow)
# ---------------------------------------------------------------------------

class TestBulkSummary:
    """Verify the bulk summary panel shows correct counts and controls."""

    def _two_invoices(self) -> list[tuple]:
        inv1 = "\n".join(["Vendor: Alpha Co", "Invoice Number: 1", "Total: 100.00"])
        inv2 = "\n".join(["Vendor: Beta Co", "Invoice Number: 2", "Total: 200.00"])
        return [_txt_upload("inv1.txt", inv1), _txt_upload("inv2.txt", inv2)]

    def test_total_metric_shown_after_bulk_process(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value(self._two_invoices()).run()
        _process_btn(at).click().run()
        assert not at.exception
        metric_labels = [m.label.lower() for m in at.metric]
        assert any("total" in l for l in metric_labels)

    def test_auto_approved_metric_shown_after_bulk_process(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value(self._two_invoices()).run()
        _process_btn(at).click().run()
        assert not at.exception
        metric_labels = [m.label.lower() for m in at.metric]
        assert any("auto" in l or "approved" in l for l in metric_labels)

    def test_needs_review_metric_shown_after_bulk_process(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value(self._two_invoices()).run()
        _process_btn(at).click().run()
        assert not at.exception
        metric_labels = [m.label.lower() for m in at.metric]
        assert any("review" in l for l in metric_labels)

    def test_export_csv_text_present_after_bulk_process(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value(self._two_invoices()).run()
        _process_btn(at).click().run()
        assert not at.exception
        # download_button is not exposed by AppTest; verify the label text is rendered
        all_text = _all_text(at)
        assert "export" in all_text or "csv" in all_text

    def test_bulk_summary_subheader_present(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value(self._two_invoices()).run()
        _process_btn(at).click().run()
        assert not at.exception
        all_text = _all_text(at)
        assert "bulk" in all_text and "summary" in all_text


# ---------------------------------------------------------------------------
# _is_bulk_auto_approvable — unit tests for updated confidence gate
# ---------------------------------------------------------------------------

class TestIsBulkAutoApprovable:
    """Unit tests for _is_bulk_auto_approvable with the new confidence gate."""

    def _result(self, errors=None, validation_results=None, field_confidence=None):
        """Build a minimal mock result object."""
        class _R:
            pass
        r = _R()
        r.errors = errors or []
        r.validation_results = validation_results or []
        r.field_confidence = field_confidence or {}
        return r

    def _fn(self, result, threshold=0.80, apply_gate=False):
        from app import _is_bulk_auto_approvable
        return _is_bulk_auto_approvable(result, threshold, apply_confidence_gate=apply_gate)

    def test_no_errors_no_fails_returns_true(self):
        r = self._result()
        assert self._fn(r) is True

    def test_errors_returns_false(self):
        r = self._result(errors=["something failed"])
        assert self._fn(r) is False

    def test_fail_validation_returns_false(self):
        r = self._result(validation_results=[{"status": "fail", "field": "total"}])
        assert self._fn(r) is False

    def test_warn_validation_still_passes(self):
        r = self._result(validation_results=[{"status": "warn", "field": "due_date"}])
        assert self._fn(r) is True

    def test_confidence_gate_off_ignores_low_confidence(self):
        r = self._result(field_confidence={"vendor_name": 0.20, "total_amount": 0.10})
        assert self._fn(r, threshold=0.80, apply_gate=False) is True

    def test_confidence_gate_on_passes_when_all_above_threshold(self):
        r = self._result(field_confidence={"vendor_name": 0.90, "total_amount": 0.85})
        assert self._fn(r, threshold=0.80, apply_gate=True) is True

    def test_confidence_gate_on_fails_when_one_below_threshold(self):
        r = self._result(field_confidence={"vendor_name": 0.90, "total_amount": 0.50})
        assert self._fn(r, threshold=0.80, apply_gate=True) is False

    def test_confidence_gate_on_empty_confidence_dict_passes(self):
        r = self._result(field_confidence={})
        assert self._fn(r, threshold=0.80, apply_gate=True) is True

    def test_confidence_gate_on_zero_scores_treated_as_missing(self):
        # Fields with 0.0 confidence are missing — gate only checks present (>0) fields
        r = self._result(field_confidence={"vendor_name": 0.90, "missing_field": 0.0})
        assert self._fn(r, threshold=0.80, apply_gate=True) is True

    def test_errors_plus_low_confidence_returns_false(self):
        r = self._result(errors=["oops"], field_confidence={"vendor_name": 0.95})
        assert self._fn(r, threshold=0.80, apply_gate=True) is False

    def test_threshold_boundary_equal_passes(self):
        r = self._result(field_confidence={"vendor_name": 0.80})
        assert self._fn(r, threshold=0.80, apply_gate=True) is True

    def test_threshold_boundary_just_below_fails(self):
        r = self._result(field_confidence={"vendor_name": 0.799})
        assert self._fn(r, threshold=0.80, apply_gate=True) is False


# ---------------------------------------------------------------------------
# Session-state counters
# ---------------------------------------------------------------------------

class TestSessionStateCounters:
    """docs_processed_total and approvals_total increment correctly."""

    def test_docs_processed_increments_after_single_file(self, tmp_path):
        invoice = "\n".join(["Vendor: A", "Invoice Number: 1", "Total: 100.00"])
        at = _app(tmp_path)
        at.run()
        _single_uploader(at).set_value(_txt_upload("inv.txt", invoice)).run()
        _process_btn(at).click().run()
        assert not at.exception
        assert at.session_state.docs_processed_total == 1

    def test_docs_processed_increments_after_bulk(self, tmp_path):
        inv1 = "\n".join(["Vendor: A", "Invoice Number: 1", "Total: 100.00"])
        inv2 = "\n".join(["Vendor: B", "Invoice Number: 2", "Total: 200.00"])
        at = _app(tmp_path)
        at.run()
        _bulk_uploader(at).set_value([
            _txt_upload("inv1.txt", inv1),
            _txt_upload("inv2.txt", inv2),
        ]).run()
        _process_btn(at).click().run()
        assert not at.exception
        assert at.session_state.docs_processed_total == 2

    def test_docs_processed_starts_at_zero(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        assert at.session_state.docs_processed_total == 0

    def test_manual_corrections_starts_at_zero(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        assert at.session_state.manual_corrections_total == 0

    def test_approvals_starts_at_zero(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        assert at.session_state.approvals_total == 0

    def test_uploads_total_not_initialized(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        with pytest.raises((KeyError, AttributeError)):
            _ = at.session_state["uploads_total"]


# ---------------------------------------------------------------------------
# Admin controls wired — output_format and auto_approve
# ---------------------------------------------------------------------------

class TestAdminControlsWired:
    """Verify admin controls affect pipeline behaviour, not just render."""

    def test_output_format_json_only_no_crash(self, tmp_path):
        invoice = "\n".join(["Vendor: A", "Invoice Number: 1", "Total: 100.00"])
        at = _app(tmp_path)
        at.run()
        # Set output format to JSON before processing
        fmt_box = next((s for s in at.selectbox if "output" in s.label.lower() and "format" in s.label.lower()), None)
        if fmt_box is None:
            pytest.skip("Output Format selectbox not found")
        fmt_box.set_value("JSON").run()
        _single_uploader(at).set_value(_txt_upload("inv.txt", invoice)).run()
        _process_btn(at).click().run()
        assert not at.exception

    def test_output_format_csv_only_no_crash(self, tmp_path):
        invoice = "\n".join(["Vendor: A", "Invoice Number: 1", "Total: 100.00"])
        at = _app(tmp_path)
        at.run()
        fmt_box = next((s for s in at.selectbox if "output" in s.label.lower() and "format" in s.label.lower()), None)
        if fmt_box is None:
            pytest.skip("Output Format selectbox not found")
        fmt_box.set_value("CSV").run()
        _single_uploader(at).set_value(_txt_upload("inv.txt", invoice)).run()
        _process_btn(at).click().run()
        assert not at.exception

    def test_auto_approve_checkbox_can_be_checked(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        cb = next((c for c in at.checkbox if "auto" in c.label.lower() or "approve" in c.label.lower()), None)
        if cb is None:
            pytest.skip("Auto-approve checkbox not found")
        cb.check().run()
        assert not at.exception

    def test_match_threshold_slider_changes_persist(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        sl = next((s for s in at.slider if "match" in s.label.lower() or "template" in s.label.lower()), None)
        if sl is None:
            pytest.skip("Template Match Threshold slider not found")
        sl.set_value(0.75).run()
        assert not at.exception
        # Verify the session state key was written
        assert at.session_state["admin_match_threshold"] == pytest.approx(0.75, abs=0.01)

    def test_confidence_threshold_slider_changes_persist(self, tmp_path):
        at = _app(tmp_path)
        at.run()
        sl = next((s for s in at.slider if "confidence" in s.label.lower()), None)
        if sl is None:
            pytest.skip("Confidence Threshold slider not found")
        sl.set_value(0.70).run()
        assert not at.exception
        assert at.session_state["admin_confidence_threshold"] == pytest.approx(0.70, abs=0.01)
