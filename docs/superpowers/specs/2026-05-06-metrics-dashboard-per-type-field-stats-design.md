# Metrics Dashboard — Per-Type Field Stats Design

> **For agentic workers:** Use `superpowers:writing-plans` to turn this spec into an implementation plan before writing any code.

**Goal:** Add a "Per Document Type Analysis" section to the Metrics Dashboard that shows per-field metrics (null rate, extraction method breakdown, average confidence) for each document type, with per-session checkboxes to toggle which metric columns are visible.

**Architecture:** Persist per-field stats to a new `field_stats` SQLite table written alongside `document_results` on every `persist()` call. Three new pure query functions in `metrics_dashboard.py` read from this table. A new UI section in `render_metrics_dashboard()` joins their results into a single table per document type, displayed in Streamlit tabs.

**Tech Stack:** Python, SQLite, pandas, Streamlit (AppTest for UI tests).

---

## 1. Data Layer — `field_stats` Table

### Schema

```sql
CREATE TABLE IF NOT EXISTS field_stats (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file       TEXT NOT NULL,
    document_type     TEXT NOT NULL,
    field_name        TEXT NOT NULL,
    is_null           INTEGER NOT NULL,   -- 1 = missing/empty, 0 = extracted
    extraction_source TEXT,               -- 'Template', 'Rule-based', 'LLM', 'Spatial', 'Manual', 'Cross-validated', 'Inferred'
    confidence        REAL                -- 0.0–1.0; NULL when is_null=1
);
```

One row per field per document. Fields iterated from `FIELD_CATALOG[doc_type]`. A field is `is_null=1` when its value in `extracted_data` is `None`, `""`, or `[]`.

### Migration

`ResultStore._migrate_schema()` creates `field_stats` with `CREATE TABLE IF NOT EXISTS` — safe to run on existing DBs. Existing documents will simply have no rows in `field_stats`; the UI shows an info message for types with no data.

### Write Path — `storage.py`

`persist()` gains two new optional parameters:

```python
def persist(
    self,
    source_file_name: str,
    extracted_data: dict[str, Any],
    validation_checks: list[ValidationCheck],
    extraction_trace: list[str],
    content_hash: str = "",
    original_filename: str = "",
    semantic_fingerprint: str = "",
    field_sources: dict[str, str] = {},     # NEW
    field_confidence: dict[str, float] = {}, # NEW
) -> dict[str, str]:
```

Both are threaded through `_write_sqlite()` → `_write_sqlite_locked()`.

Inside `_write_sqlite_locked()`, after the `doc_df.to_sql()` write, iterate `FIELD_CATALOG[doc_type]` and insert one row per field:

```python
doc_type = extracted_data.get("document_type", "invoice")
fields = FIELD_CATALOG.get(doc_type, [])
stats_rows = []
for f in fields:
    val = extracted_data.get(f["key"])
    is_null = 1 if val in (None, "", []) else 0
    stats_rows.append({
        "source_file": source_file_name,
        "document_type": doc_type,
        "field_name": f["key"],
        "is_null": is_null,
        "extraction_source": field_sources.get(f["key"]),
        "confidence": field_confidence.get(f["key"]) if not is_null else None,
    })
pd.DataFrame(stats_rows).to_sql("field_stats", conn, if_exists="append", index=False)
```

Also add a `DELETE FROM field_stats WHERE source_file = ?` to the pre-write cleanup block (same pattern as `document_results`, `validation_results`, `extraction_traces`) so `finalize_review()` replaces rather than duplicates.

### Pipeline Call Site — `pipeline.py`

The existing `persist()` call in `process_bytes()` already has `result.field_sources` and `result.field_confidence` available. Update the call to pass them:

```python
output_files = self._store.persist(
    saved_path.name,
    extracted_data,
    validation_checks,
    extraction_trace,
    content_hash=content_hash,
    original_filename=file_name,
    semantic_fingerprint=semantic_fingerprint,
    field_sources=field_sources,       # NEW
    field_confidence=field_confidence, # NEW
)
```

`field_sources` and `field_confidence` come from:
```python
field_confidence = self._compute_field_confidence(extracted_data, extraction_trace)
field_sources = self._build_field_sources(extracted_data, extraction_trace)
```
Both are already called in `process_bytes()` before `persist()` to populate `PipelineResult`.

The semantic-fingerprint early-return path does **not** call `persist()`, so `field_stats` is never written for duplicates — correct by design.

---

## 2. Query Layer — `metrics_dashboard.py`

Four new pure functions (no Streamlit imports — unit-testable headlessly).

### `doc_types_with_field_stats(conn) → list[str]`

```python
def doc_types_with_field_stats(conn: sqlite3.Connection) -> list[str]:
    if not _table_exists(conn, "field_stats"):
        return []
    rows = conn.execute(
        "SELECT DISTINCT document_type FROM field_stats ORDER BY document_type"
    ).fetchall()
    return [r[0] for r in rows]
```

Used to suppress tabs for types with no data.

### `field_null_rates(conn, doc_type) → pd.DataFrame`

Columns: `field`, `total_docs`, `extracted`, `null_rate_pct` (float, 0–100).

```python
def field_null_rates(conn: sqlite3.Connection, doc_type: str) -> pd.DataFrame:
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
```

### `field_extraction_sources(conn, doc_type) → pd.DataFrame`

Columns: `field`, `top_source`, `source_breakdown` (e.g. `"Template (7), Rule-based (2)"`).

```python
def field_extraction_sources(conn: sqlite3.Connection, doc_type: str) -> pd.DataFrame:
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
        breakdown = ", ".join(f"{r['extraction_source']} ({r['cnt']})" for _, r in grp.iterrows())
        rows.append({"field": field, "top_source": top, "source_breakdown": breakdown})
    return pd.DataFrame(rows)
```

### `field_avg_confidence(conn, doc_type) → pd.DataFrame`

Columns: `field`, `avg_confidence`, `min_confidence`, `max_confidence`. Only includes non-null extractions.

```python
def field_avg_confidence(conn: sqlite3.Connection, doc_type: str) -> pd.DataFrame:
    if not _table_exists(conn, "field_stats"):
        return pd.DataFrame(columns=["field", "avg_confidence", "min_confidence", "max_confidence"])
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
```

---

## 3. UI — `render_metrics_dashboard()`

New section appended after the existing "Records by Document Type" table, below a divider.

### Checkboxes (session state, all default True)

```python
st.subheader("Per Document Type Analysis")
col1, col2, col3 = st.columns(3)
show_null = col1.checkbox("Null Rate", value=True, key="fstats_show_null")
show_method = col2.checkbox("Extraction Method", value=True, key="fstats_show_method")
show_conf = col3.checkbox("Avg Confidence", value=True, key="fstats_show_conf")
```

If all three are unchecked, show `st.info("Select at least one column to display.")` and return.

### Tabs — one per type with data

```python
available_types = doc_types_with_field_stats(conn)
if not available_types:
    st.info("No field-level stats yet — process a document to populate this section.")
else:
    tabs = st.tabs([t.replace("_", " ").title() for t in available_types])
    for tab, doc_type in zip(tabs, available_types):
        with tab:
            _render_field_stats_tab(conn, doc_type, show_null, show_method, show_conf)
```

### `_render_field_stats_tab()` helper

1. Load only the query results for checked boxes.
2. Start from the `FIELD_CATALOG` field order for the type as the canonical row list.
3. Left-join the query results onto the catalog order by `field_name` so fields with no data show as `—`.
4. Build a single display DataFrame with columns: `Field` (always), then conditionally `Null Rate`, `Extraction Method`, `Avg Confidence`.
5. Render with `st.dataframe(..., use_container_width=True)`.
6. Apply a pandas Styler background gradient on the `Null Rate` column (green=0%, red=100%) and on `Avg Confidence` (green=1.0, red=0.0) when those columns are present.

---

## 4. Testing — `tests/test_metrics_dashboard.py`

New class `TestFieldStatsMetrics`:

| Test | What it verifies |
|---|---|
| `test_field_stats_written_on_persist` | `store.persist()` with known `field_sources` / `field_confidence` → correct rows in `field_stats` |
| `test_field_null_rates_correct` | Field A present both docs → 0%; field B missing one → 50% |
| `test_field_extraction_sources_breakdown` | 3 docs with mixed sources → breakdown string contains all counts |
| `test_field_avg_confidence_excludes_nulls` | `is_null=1` rows excluded from avg calculation |
| `test_doc_types_with_field_stats_filters_empty` | Only types with rows are returned |
| `test_field_stats_empty_on_fresh_db` | All three query functions return empty DataFrames when table absent |
| `test_field_stats_not_written_for_semantic_duplicate` | Semantic-fingerprint dedup path does not write to `field_stats` |

---

## File Changes Summary

| File | Change |
|---|---|
| `src/doc_ai/storage.py` | Add `field_stats` table creation in `_migrate_schema`; add `field_sources` / `field_confidence` params to `persist()` / `_write_sqlite()` / `_write_sqlite_locked()`; write `field_stats` rows; add pre-write delete |
| `src/doc_ai/pipeline.py` | Pass `field_sources` and `field_confidence` to `persist()` call |
| `src/doc_ai/metrics_dashboard.py` | Add 4 query functions; add per-type analysis section to `render_metrics_dashboard()` |
| `tests/test_metrics_dashboard.py` | Add `TestFieldStatsMetrics` class (7 tests) |

No changes required to `extractors.py`, `validators.py`, `schema_config.py`, or `app.py`.
