"""Tests for src/doc_ai/metrics_dashboard.py.

Place at: tests/test_metrics_dashboard.py

These tests build a throwaway SQLite database with the exact schema that
``ResultStore`` produces, write representative rows + traces, and assert that
each helper returns the expected count. No Streamlit, no PDF/OCR/LLM stack —
runs against ``requirements-test.txt``.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.doc_ai.metrics_dashboard import (
    get_metrics,
    llm_fallback_count,
    llm_usage_daily,
    manual_corrections_count,
    records_created,
    template_passed_count,
    total_documents_processed,
)
from src.doc_ai.schema_config import TABLE_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_schema(conn: sqlite3.Connection) -> None:
    """Create the subset of tables metrics_dashboard reads from."""
    conn.executescript(
        """
        CREATE TABLE document_results (
            source_file TEXT,
            original_filename TEXT,
            content_hash TEXT,
            document_type TEXT
        );
        CREATE TABLE extraction_traces (
            source_file TEXT,
            step_number INTEGER,
            message TEXT
        );
        """
    )
    for table in TABLE_NAMES.values():
        conn.execute(
            f"CREATE TABLE {table} ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " source_file TEXT,"
            " processed_at TEXT"
            ")"
        )


def _add_doc(
    conn: sqlite3.Connection,
    source_file: str,
    document_type: str,
    trace_messages: list[str],
    processed_at: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO document_results (source_file, original_filename, content_hash, document_type)"
        " VALUES (?, ?, ?, ?)",
        (source_file, source_file, f"hash_{source_file}", document_type),
    )
    for i, msg in enumerate(trace_messages, start=1):
        conn.execute(
            "INSERT INTO extraction_traces (source_file, step_number, message) VALUES (?, ?, ?)",
            (source_file, i, msg),
        )
    table = TABLE_NAMES.get(document_type)
    if table:
        ts = processed_at or datetime.utcnow().isoformat(sep=" ")
        conn.execute(
            f"INSERT INTO {table} (source_file, processed_at) VALUES (?, ?)",
            (source_file, ts),
        )


@pytest.fixture
def populated_db(tmp_path: Path) -> Path:
    """Build a DB with a known mix of template / LLM / manual / mixed docs."""
    db_path = tmp_path / "metrics_test.db"
    conn = sqlite3.connect(str(db_path))
    try:
        _create_schema(conn)

        today = datetime.utcnow().date()

        # Doc 1 — template hit, invoice
        _add_doc(
            conn,
            "doc_template_1.pdf",
            "invoice",
            ["Applied learned template `acme_v1` from prior approved upload."],
            processed_at=str(today),
        )
        # Doc 2 — LLM fallback, NDA (no template matched)
        _add_doc(
            conn,
            "doc_llm_1.pdf",
            "nda",
            ["Used the LLM reasoning layer for an unseen or weakly matched document format."],
            processed_at=str(today),
        )
        # Doc 3 — LLM fallback, lab report, yesterday (template insufficient)
        _add_doc(
            conn,
            "doc_llm_2.pdf",
            "lab_report",
            ["Used the LLM reasoning layer after incomplete template extraction."],
            processed_at=str(today - timedelta(days=1)),
        )
        # Doc 4 — manually corrected (review form)
        _add_doc(
            conn,
            "doc_manual_1.pdf",
            "invoice",
            ["Used human-reviewed corrections from the UI."],
            processed_at=str(today),
        )
        # Doc 5 — LLM + then manually corrected. Counts in BOTH llm and manual.
        _add_doc(
            conn,
            "doc_llm_then_manual.pdf",
            "business_doc",
            [
                "Used the LLM reasoning layer after incomplete template extraction.",
                "Used human-reviewed corrections from the UI.",
            ],
            processed_at=str(today),
        )
        # Doc 6 — pure rule-based, no template, no LLM, no manual review
        _add_doc(
            conn,
            "doc_rule_1.pdf",
            "invoice",
            ["Used `RuleBasedInvoiceExtractor` extraction."],
            processed_at=str(today),
        )
        # Doc 7 — explicitly approved for future matching (approval only, NOT a manual correction)
        _add_doc(
            conn,
            "doc_approved_1.pdf",
            "medical_discharge",
            ["User explicitly approved this result for future matching."],
            processed_at=str(today),
        )

        conn.commit()
    finally:
        conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Scalar metric tests
# ---------------------------------------------------------------------------


class TestScalarMetrics:
    def test_total_documents_processed(self, populated_db: Path) -> None:
        conn = sqlite3.connect(str(populated_db))
        try:
            assert total_documents_processed(conn) == 7
        finally:
            conn.close()

    def test_template_passed_count_only_counts_template_hits(self, populated_db: Path) -> None:
        # Only doc_template_1 has a "learned template ... applied" message.
        conn = sqlite3.connect(str(populated_db))
        try:
            assert template_passed_count(conn) == 1
        finally:
            conn.close()

    def test_llm_fallback_counts_distinct_documents(self, populated_db: Path) -> None:
        # doc_llm_1, doc_llm_2, doc_llm_then_manual = 3 distinct documents
        conn = sqlite3.connect(str(populated_db))
        try:
            assert llm_fallback_count(conn) == 3
        finally:
            conn.close()

    def test_manual_corrections_excludes_pure_approvals(self, populated_db: Path) -> None:
        # doc_manual_1, doc_llm_then_manual = 2 distinct corrections
        # doc_approved_1 has only an approval trace — must NOT be counted
        conn = sqlite3.connect(str(populated_db))
        try:
            assert manual_corrections_count(conn) == 2
        finally:
            conn.close()

    def test_records_created_sums_per_type_tables(self, populated_db: Path) -> None:
        # 3 invoices + 1 nda + 1 lab_report + 1 business_doc + 1 medical_discharge = 7
        conn = sqlite3.connect(str(populated_db))
        try:
            assert records_created(conn) == 7
        finally:
            conn.close()

    def test_get_metrics_returns_full_dict(self, populated_db: Path) -> None:
        m = get_metrics(populated_db)
        assert m == {
            "total_documents_processed": 7,
            "template_passed": 1,
            "llm_fallback": 3,
            "manually_corrected": 2,
            "records_created": 7,
        }


# ---------------------------------------------------------------------------
# Empty-state behaviour
# ---------------------------------------------------------------------------


class TestEmptyState:
    def test_get_metrics_on_missing_db(self, tmp_path: Path) -> None:
        m = get_metrics(tmp_path / "does_not_exist.db")
        assert all(v == 0 for v in m.values())

    def test_helpers_return_zero_when_tables_missing(self, tmp_path: Path) -> None:
        db_path = tmp_path / "blank.db"
        sqlite3.connect(str(db_path)).close()  # creates an empty db
        conn = sqlite3.connect(str(db_path))
        try:
            assert total_documents_processed(conn) == 0
            assert template_passed_count(conn) == 0
            assert llm_fallback_count(conn) == 0
            assert manual_corrections_count(conn) == 0
            assert records_created(conn) == 0
        finally:
            conn.close()

    def test_llm_usage_daily_returns_backfilled_frame_when_empty(self, tmp_path: Path) -> None:
        db_path = tmp_path / "blank.db"
        conn = sqlite3.connect(str(db_path))
        try:
            _create_schema(conn)
            df = llm_usage_daily(conn, days=14)
        finally:
            conn.close()
        assert list(df.columns) == ["date", "llm_documents"]
        assert len(df) == 14
        assert df["llm_documents"].sum() == 0


# ---------------------------------------------------------------------------
# Time-series tests
# ---------------------------------------------------------------------------


class TestLlmUsageDaily:
    def test_daily_counts_distinct_llm_docs(self, populated_db: Path) -> None:
        conn = sqlite3.connect(str(populated_db))
        try:
            df = llm_usage_daily(conn, days=7)
        finally:
            conn.close()

        assert list(df.columns) == ["date", "llm_documents"]
        assert len(df) == 7
        # Two LLM docs today (doc_llm_1 + doc_llm_then_manual), one yesterday.
        # Use UTC date to match the fixture (datetime.utcnow) and SQLite's date('now').
        df["date"] = pd.to_datetime(df["date"]).dt.date
        today = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)
        today_count = int(df.loc[df["date"] == today, "llm_documents"].sum())
        yesterday_count = int(df.loc[df["date"] == yesterday, "llm_documents"].sum())
        assert today_count == 2
        assert yesterday_count == 1
        assert df["llm_documents"].sum() == 3

    def test_invalid_days_raises(self, populated_db: Path) -> None:
        conn = sqlite3.connect(str(populated_db))
        try:
            with pytest.raises(ValueError):
                llm_usage_daily(conn, days=0)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Integration test — real pipeline output → metrics
# ---------------------------------------------------------------------------

_FIXTURES_PATH = Path(__file__).parent / "fixtures"
_FORMAT_A_FULL = [
    _FIXTURES_PATH / f"{doc_type}_format_a_full.pdf"
    for doc_type in ("invoice", "medical_discharge", "nda", "lab_report", "business_doc")
]
_PDF_LIBS_AVAILABLE = any(
    __import__("importlib").util.find_spec(lib) is not None
    for lib in ("pypdf", "pdfplumber", "unstructured")
)


class TestMetricsWithRealFixtures:
    """Process one format_a_full fixture per doc type, then assert metric counts."""

    def _make_pipeline(self, tmp_path):
        import os
        from src.doc_ai.config import get_settings
        from src.doc_ai.pipeline import DocumentPipeline

        get_settings.cache_clear()
        os.environ["APP_ENV"] = "test"
        os.environ["APP_DATA_ROOT"] = str(tmp_path)
        get_settings.cache_clear()
        return DocumentPipeline(get_settings())

    def test_total_documents_processed_matches_fixture_count(self, tmp_path):
        if not _PDF_LIBS_AVAILABLE:
            pytest.skip("PDF libraries not installed")
        available = [f for f in _FORMAT_A_FULL if f.exists()]
        if not available:
            pytest.skip("No format_a_full fixtures found")

        pipeline = self._make_pipeline(tmp_path)
        for fixture in available:
            pipeline.process_bytes(fixture.name, fixture.read_bytes())

        from src.doc_ai.config import get_settings
        db_path = get_settings().database_path
        assert db_path.exists(), f"Database was not created at {db_path}"
        conn = sqlite3.connect(str(db_path))
        try:
            count = total_documents_processed(conn)
        finally:
            conn.close()
        assert count == len(available), (
            f"Expected {len(available)} processed documents, got {count}"
        )

    def test_approval_without_changes_not_counted_as_manual_correction(self, tmp_path):
        """Approving a result without editing any fields must not increment manually_corrected.

        Uses a corrected_data dict that simulates coerce_form_data output — numeric strings
        like "3,445.00" become floats like 3445.0. That type difference must NOT be treated
        as a user correction.
        """
        if not _PDF_LIBS_AVAILABLE:
            pytest.skip("PDF libraries not installed")
        fixture = next((f for f in _FORMAT_A_FULL if f.exists()), None)
        if fixture is None:
            pytest.skip("No format_a_full fixture found")

        pipeline = self._make_pipeline(tmp_path)
        r = pipeline.process_bytes(fixture.name, fixture.read_bytes())

        # Simulate coerce_form_data: convert numeric-looking string values to floats
        coerced = {}
        for k, v in r.extracted_data.items():
            if isinstance(v, str) and k in {"subtotal", "tax_amount", "total_amount",
                                             "tax", "shipping_handling"}:
                try:
                    coerced[k] = float(v.replace(",", "").replace("$", ""))
                except (ValueError, AttributeError):
                    coerced[k] = v
            else:
                coerced[k] = v

        pipeline.finalize_review(
            source_file=r.source_file,
            upload_path=r.upload_path,
            parsed_text=r.parsed_text,
            corrected_data=coerced,
            extraction_mode="adaptive-local",
            learn_from_upload=True,
            approve_for_future_matching=True,
            content_hash=r.content_hash,
            original_extracted=r.extracted_data,
        )

        from src.doc_ai.config import get_settings
        db_path = get_settings().database_path
        conn = sqlite3.connect(str(db_path))
        try:
            count = manual_corrections_count(conn)
        finally:
            conn.close()
        assert count == 0, (
            f"Expected 0 manual corrections after a no-change approval, got {count}"
        )

    def test_records_created_matches_fixture_count(self, tmp_path):
        if not _PDF_LIBS_AVAILABLE:
            pytest.skip("PDF libraries not installed")
        available = [f for f in _FORMAT_A_FULL if f.exists()]
        if not available:
            pytest.skip("No format_a_full fixtures found")

        pipeline = self._make_pipeline(tmp_path)
        for fixture in available:
            pipeline.process_bytes(fixture.name, fixture.read_bytes())

        from src.doc_ai.config import get_settings
        db_path = get_settings().database_path
        conn = sqlite3.connect(str(db_path))
        try:
            count = records_created(conn)
        finally:
            conn.close()
        assert count == len(available), (
            f"Expected {len(available)} per-type records, got {count}"
        )


# ---------------------------------------------------------------------------
# field_stats table — write path tests
# ---------------------------------------------------------------------------


class TestFieldStatsMetrics:
    """Tests for field_stats table write path and query functions."""

    def _make_store(self, tmp_path: Path):
        from src.doc_ai.config import get_settings
        from src.doc_ai.storage import ResultStore
        import dataclasses
        get_settings.cache_clear()
        s = get_settings()
        s = dataclasses.replace(
            s,
            data_dir=tmp_path,
            database_path=tmp_path / "test.db",
            output_dir=tmp_path,
        )
        return ResultStore(s)

    def test_field_stats_written_on_persist(self, tmp_path):
        """persist() with field_sources/field_confidence writes one row per field."""
        from src.doc_ai.schemas import ValidationCheck
        store = self._make_store(tmp_path)
        check = ValidationCheck(field="vendor_name", status="pass", message="ok")
        store.persist(
            source_file_name="inv.pdf",
            extracted_data={
                "document_type": "invoice",
                "vendor_name": "Acme",
                "invoice_number": None,
                "invoice_date": "2025-01-01",
                "total_amount": "100.00",
            },
            validation_checks=[check],
            extraction_trace=["step 1"],
            content_hash="abc",
            original_filename="inv.pdf",
            field_sources={"vendor_name": "Template", "invoice_date": "Rule-based"},
            field_confidence={"vendor_name": 0.92, "invoice_date": 0.72},
        )
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        rows = conn.execute(
            "SELECT field_name, is_null, extraction_source, confidence "
            "FROM field_stats WHERE source_file='inv.pdf' ORDER BY field_name"
        ).fetchall()
        conn.close()
        by_field = {r[0]: r for r in rows}
        assert "vendor_name" in by_field
        assert by_field["vendor_name"][1] == 0         # is_null=0 (extracted)
        assert by_field["vendor_name"][2] == "Template"
        assert abs(by_field["vendor_name"][3] - 0.92) < 0.001
        assert "invoice_number" in by_field
        assert by_field["invoice_number"][1] == 1      # is_null=1 (missing)
        assert by_field["invoice_number"][3] is None   # confidence NULL when missing
        assert by_field["invoice_number"][2] is None   # extraction_source NULL when missing

    def test_field_stats_replaced_on_repersist(self, tmp_path):
        """Re-persisting the same source_file replaces field_stats rows, not duplicates them."""
        from src.doc_ai.schemas import ValidationCheck
        store = self._make_store(tmp_path)
        check = ValidationCheck(field="vendor_name", status="pass", message="ok")
        for _ in range(2):
            store.persist(
                source_file_name="inv.pdf",
                extracted_data={"document_type": "invoice", "vendor_name": "Acme"},
                validation_checks=[check],
                extraction_trace=["step 1"],
                content_hash="abc",
                original_filename="inv.pdf",
            )
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        count = conn.execute(
            "SELECT COUNT(*) FROM field_stats WHERE source_file='inv.pdf'"
        ).fetchone()[0]
        conn.close()
        from src.doc_ai.schema_config import FIELD_CATALOG
        expected = len(FIELD_CATALOG["invoice"])
        assert count == expected, f"Expected {expected} rows, got {count} (duplicate rows written)"
