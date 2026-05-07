"""Metrics dashboard for the AI Document Extraction pipeline.

Reads from the same SQLite database that ``ResultStore`` writes to
(``document_results``, ``extraction_traces``, plus the per-type tables in
``TABLE_NAMES``). Exposes a small set of pure helpers that can be unit-tested
without Streamlit, and a ``render_metrics_dashboard()`` function that draws the
tab UI.

Drop-in location:    src/doc_ai/metrics_dashboard.py
Wire-up in app.py:   see the snippet at the bottom of this file.

Definitions
-----------
* Total documents processed
    Row count of ``document_results``.
* Passed by template
    Distinct documents whose extraction trace records a learned template being
    matched / applied / used (i.e. the rule-based / LLM fallback was avoided).
* Fell back to LLM
    Distinct documents whose trace mentions an LLM provider keyword
    (llm / openai / groq / openrouter / ollama / gemini). Mirrors the
    same heuristic ``DocumentPipeline._compute_field_confidence`` uses.
* Manually corrected
    Distinct documents finalized through ``DocumentPipeline.finalize_review``
    (trace contains "human-reviewed corrections" or "user explicitly approved").
* Records created
    Sum of row counts across the five per-type tables (invoices,
    discharge_summaries, ndas, lab_reports, business_docs).
* LLM usage over time
    Daily count of LLM-assisted documents over a configurable window. Joins
    ``extraction_traces`` (provider keyword) to a UNION across the per-type
    tables (which carry ``processed_at``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from .schema_config import TABLE_NAMES

# ---------------------------------------------------------------------------
# Trace-message classifiers
# ---------------------------------------------------------------------------

# Mirrors the keyword set in DocumentPipeline._compute_field_confidence so a
# document that scored as "LLM-baselined" is also counted as LLM here.
# These phrases appear only when an LLM was actually invoked for extraction.
# Deliberately excludes "llm-assisted" (the mode name) and "LLM fallback" (intent,
# not invocation) so that documents where the LLM was configured but not called
# are not counted.
_LLM_KEYWORDS: tuple[str, ...] = (
    "llm reasoning layer",
    "used openai",
    "used groq",
    "used openrouter",
    "used ollama",
    "used gemini",
)

_TEMPLATE_HIT_VERBS: tuple[str, ...] = ("matched", "applied", "used")

_MANUAL_CORRECTION_KEYWORDS: tuple[str, ...] = (
    "human-reviewed corrections",
)


# ---------------------------------------------------------------------------
# SQL building blocks
# ---------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


def _like_clause(column: str, keywords: Iterable[str]) -> str:
    """Return ``(LOWER(column) LIKE '%kw1%' OR ...)`` — keywords are static."""
    parts = [f"LOWER({column}) LIKE '%{kw}%'" for kw in keywords]
    return "(" + " OR ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def total_documents_processed(conn: sqlite3.Connection) -> int:
    if not _table_exists(conn, "document_results"):
        return 0
    return int(conn.execute("SELECT COUNT(*) FROM document_results").fetchone()[0])


def template_passed_count(conn: sqlite3.Connection) -> int:
    """Distinct source_files whose trace records a template hit."""
    if not _table_exists(conn, "extraction_traces"):
        return 0
    verb_clause = _like_clause("message", _TEMPLATE_HIT_VERBS)
    sql = (
        "SELECT COUNT(DISTINCT source_file) FROM extraction_traces "
        "WHERE LOWER(message) LIKE '%learned template%' "
        f"AND {verb_clause}"
    )
    return int(conn.execute(sql).fetchone()[0])


def llm_fallback_count(conn: sqlite3.Connection) -> int:
    """Distinct source_files whose trace mentions an LLM provider."""
    if not _table_exists(conn, "extraction_traces"):
        return 0
    sql = (
        "SELECT COUNT(DISTINCT source_file) FROM extraction_traces "
        f"WHERE {_like_clause('message', _LLM_KEYWORDS)}"
    )
    return int(conn.execute(sql).fetchone()[0])


def manual_corrections_count(conn: sqlite3.Connection) -> int:
    if not _table_exists(conn, "extraction_traces"):
        return 0
    sql = (
        "SELECT COUNT(DISTINCT source_file) FROM extraction_traces "
        f"WHERE {_like_clause('message', _MANUAL_CORRECTION_KEYWORDS)}"
    )
    return int(conn.execute(sql).fetchone()[0])


def records_created(conn: sqlite3.Connection) -> int:
    """Sum of rows across the per-type structured tables."""
    total = 0
    for table in TABLE_NAMES.values():
        if _table_exists(conn, table):
            total += int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return total


def get_metrics(db_path: str | Path) -> dict[str, int]:
    """Return all scalar KPIs in a single dict; safe to call before any DB rows exist."""
    db = Path(db_path)
    if not db.exists():
        return {
            "total_documents_processed": 0,
            "template_passed": 0,
            "llm_fallback": 0,
            "manually_corrected": 0,
            "records_created": 0,
        }
    conn = sqlite3.connect(str(db))
    try:
        return {
            "total_documents_processed": total_documents_processed(conn),
            "template_passed": template_passed_count(conn),
            "llm_fallback": llm_fallback_count(conn),
            "manually_corrected": manual_corrections_count(conn),
            "records_created": records_created(conn),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# LLM usage time series
# ---------------------------------------------------------------------------


def llm_usage_daily(conn: sqlite3.Connection, days: int = 30) -> pd.DataFrame:
    """Return a DataFrame ``[date, llm_documents]`` for the last *days* days.

    Missing days are back-filled with zero so the line chart is continuous.
    """
    if days <= 0:
        raise ValueError("days must be positive")

    if not _table_exists(conn, "extraction_traces"):
        return _empty_daily_frame(days)

    union_parts: list[str] = [
        f"SELECT source_file, processed_at FROM {table}"
        for table in TABLE_NAMES.values()
        if _table_exists(conn, table)
    ]
    if not union_parts:
        return _empty_daily_frame(days)

    union_sql = " UNION ALL ".join(union_parts)
    sql = (
        f"WITH all_docs AS ({union_sql}) "
        "SELECT date(d.processed_at) AS date, "
        "       COUNT(DISTINCT d.source_file) AS llm_documents "
        "FROM all_docs d "
        "JOIN extraction_traces t ON t.source_file = d.source_file "
        f"WHERE {_like_clause('t.message', _LLM_KEYWORDS)} "
        "  AND date(d.processed_at) >= date('now', ?) "
        "GROUP BY date(d.processed_at) "
        "ORDER BY date"
    )
    df = pd.read_sql_query(sql, conn, params=(f"-{days - 1} day",))
    return _backfill_daily(df, days)


def _utc_today_naive() -> pd.Timestamp:
    """Today at 00:00 UTC as a tz-naive Timestamp (matches SQLite date('now'))."""
    # tz_convert('UTC').tz_localize(None) yields a naive timestamp anchored to UTC,
    # which lines up with the strings produced by SQLite's date() function.
    return pd.Timestamp.utcnow().tz_convert("UTC").tz_localize(None).normalize()


def _empty_daily_frame(days: int) -> pd.DataFrame:
    idx = pd.date_range(end=_utc_today_naive(), periods=days, freq="D")
    return pd.DataFrame({"date": idx, "llm_documents": [0] * len(idx)})


def _backfill_daily(df: pd.DataFrame, days: int) -> pd.DataFrame:
    full_idx = pd.date_range(end=_utc_today_naive(), periods=days, freq="D")
    if df.empty:
        return pd.DataFrame({"date": full_idx, "llm_documents": [0] * len(full_idx)})
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.set_index("date")
        .reindex(full_idx, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )
    return df


# ---------------------------------------------------------------------------
# Per-type field stats query functions
# ---------------------------------------------------------------------------


def doc_types_with_field_stats(conn: sqlite3.Connection) -> list[str]:
    """Return document types that have at least one row in field_stats."""
    if not _table_exists(conn, "field_stats"):
        return []
    rows = conn.execute(
        "SELECT DISTINCT document_type FROM field_stats ORDER BY document_type"
    ).fetchall()
    return [r[0] for r in rows]


def field_null_rates(conn: sqlite3.Connection, doc_type: str) -> pd.DataFrame:
    """Columns: field, total_docs, extracted, null_rate_pct (0–100)."""
    if not _table_exists(conn, "field_stats"):
        return pd.DataFrame(columns=["field", "total_docs", "extracted", "null_rate_pct"])
    sql = """
        SELECT field_name AS field,
               COUNT(*) AS total_docs,
               SUM(CASE WHEN is_null=0 THEN 1 ELSE 0 END) AS extracted,
               ROUND(SUM(is_null) * 100.0 / COUNT(*), 1) AS null_rate_pct
        FROM field_stats
        WHERE document_type = ?
        GROUP BY field_name
    """
    return pd.read_sql_query(sql, conn, params=(doc_type,))


def field_extraction_sources(conn: sqlite3.Connection, doc_type: str) -> pd.DataFrame:
    """Columns: field, top_source, source_breakdown (e.g. 'Template (7), Rule-based (2)')."""
    if not _table_exists(conn, "field_stats"):
        return pd.DataFrame(columns=["field", "top_source", "source_breakdown"])
    sql = """
        SELECT field_name AS field,
               extraction_source,
               COUNT(*) AS cnt
        FROM field_stats
        WHERE document_type = ? AND is_null = 0 AND extraction_source IS NOT NULL
        GROUP BY field_name, extraction_source
        ORDER BY field_name, cnt DESC
    """
    raw = pd.read_sql_query(sql, conn, params=(doc_type,))
    if raw.empty:
        return pd.DataFrame(columns=["field", "top_source", "source_breakdown"])
    rows = []
    for field, grp in raw.groupby("field", sort=False):
        top = grp.iloc[0]["extraction_source"]
        breakdown = ", ".join(
            f"{r['extraction_source']} ({r['cnt']})" for _, r in grp.iterrows()
        )
        rows.append({"field": field, "top_source": top, "source_breakdown": breakdown})
    return pd.DataFrame(rows)


def field_avg_confidence(conn: sqlite3.Connection, doc_type: str) -> pd.DataFrame:
    """Columns: field, avg_confidence, min_confidence, max_confidence (non-null extractions only)."""
    if not _table_exists(conn, "field_stats"):
        return pd.DataFrame(
            columns=["field", "avg_confidence", "min_confidence", "max_confidence"]
        )
    sql = """
        SELECT field_name AS field,
               ROUND(AVG(confidence), 3) AS avg_confidence,
               ROUND(MIN(confidence), 3) AS min_confidence,
               ROUND(MAX(confidence), 3) AS max_confidence
        FROM field_stats
        WHERE document_type = ? AND is_null = 0 AND confidence IS NOT NULL
        GROUP BY field_name
    """
    return pd.read_sql_query(sql, conn, params=(doc_type,))


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def render_metrics_dashboard(settings) -> None:
    """Render the Metrics Dashboard tab.

    *settings* is a ``doc_ai.config.Settings`` instance — the same object that
    the rest of ``app.py`` already passes around.
    """
    import streamlit as st  # local import keeps the module testable headlessly

    st.header("Metrics Dashboard")
    st.caption(
        "Pipeline-wide totals from the persisted SQLite store. "
        "Counts are over all time unless otherwise noted."
    )

    db_path = Path(settings.database_path)
    if not db_path.exists():
        st.info(
            "No documents have been processed yet — the database hasn't been created. "
            "Upload a document on the Single Document or Bulk Upload tab to populate this view."
        )
        return

    conn = sqlite3.connect(str(db_path))
    try:
        total = total_documents_processed(conn)
        template = template_passed_count(conn)
        llm = llm_fallback_count(conn)
        manual = manual_corrections_count(conn)
        records = records_created(conn)

        cols = st.columns(5)
        cols[0].metric("Total Documents Processed", f"{total:,}")
        cols[1].metric(
            "Passed by Template",
            f"{template:,}",
            help="Subset of total — a document can appear here AND in 'Fell Back to LLM'. Does not add up to total.",
        )
        cols[2].metric(
            "Fell Back to LLM",
            f"{llm:,}",
            help="Subset of total — a document can appear here AND in 'Passed by Template'. Does not add up to total.",
        )
        cols[3].metric("Manually Corrected", f"{manual:,}")
        cols[4].metric(
            "Records Created",
            f"{records:,}",
            help="Rows across all per-type structured tables (invoices, discharge_summaries, ndas, lab_reports, business_docs).",
        )

        st.caption(
            ":information_source: **Passed by Template** and **Fell Back to LLM** are independent "
            "subsets of the total — one document can be counted in both. They will not add up to "
            "the total."
        )

        # Helpful coverage hints — these counts are a *subset* of total.
        if total > 0:
            template_pct = template / total * 100
            llm_pct = llm / total * 100
            manual_pct = manual / total * 100
            st.caption(
                f"Template coverage: **{template_pct:.1f}%** · "
                f"LLM fallback rate: **{llm_pct:.1f}%** · "
                f"Manual review rate: **{manual_pct:.1f}%**"
            )

        st.divider()

        # ---- LLM usage over time -------------------------------------------------
        st.subheader("LLM Usage Over Time")
        st.caption("Distinct documents whose extraction trace mentions an LLM provider, by day.")

        window = st.selectbox(
            "Window",
            options=[7, 14, 30, 90],
            index=2,
            format_func=lambda d: f"Last {d} days",
            key="metrics_llm_window",
        )
        daily = llm_usage_daily(conn, days=int(window))

        if daily["llm_documents"].sum() == 0:
            st.info(
                f"No LLM-assisted runs have been recorded in the last {window} days. "
                "If you expect some, check that you ran with extraction mode `llm-assisted`."
            )
        else:
            chart_df = daily.copy()
            chart_df["date"] = pd.to_datetime(chart_df["date"])
            st.line_chart(chart_df.set_index("date")["llm_documents"])

        with st.expander("Daily breakdown"):
            st.dataframe(daily, use_container_width=True, hide_index=True)

        st.divider()

        # ---- Records by type -----------------------------------------------------
        st.subheader("Records by Document Type")
        breakdown_rows: list[dict[str, object]] = []
        for doc_type, table in TABLE_NAMES.items():
            count = (
                int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                if _table_exists(conn, table)
                else 0
            )
            breakdown_rows.append({"document_type": doc_type, "table": table, "records": count})
        breakdown = pd.DataFrame(breakdown_rows)
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

        st.caption(
            "Definitions — *Passed by Template*: trace records a learned template being "
            "matched / applied / used. *Fell Back to LLM*: trace mentions llm / openai / "
            "groq / openrouter / ollama / gemini. *Manually Corrected*: trace contains "
            "“human-reviewed corrections” or “user explicitly approved”. The five "
            "subset metrics are not mutually exclusive — a document can be both LLM-assisted "
            "and manually corrected. *Manually Corrected* counts only documents where "
            "field values were actually changed during review, not plain approvals."
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# How to wire this into app.py
# ---------------------------------------------------------------------------
#
# Replace the existing tab declaration in main() with a five-tab version:
#
#     tab_single, tab_bulk, tab_schema, tab_admin, tab_metrics = st.tabs(
#         ["Single Document", "Bulk Upload", "Schema Settings",
#          "Admin Dashboard", "Metrics Dashboard"]
#     )
#
# And add the new `with tab_metrics:` block alongside the others:
#
#     with tab_metrics:
#         from src.doc_ai.metrics_dashboard import render_metrics_dashboard
#         render_metrics_dashboard(runtime_settings)
#
# No changes to storage.py, pipeline.py, or extractors.py are required — this
# tab reads what is already persisted.
