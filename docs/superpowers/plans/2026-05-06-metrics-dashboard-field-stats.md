# Metrics Dashboard — Per-Type Field Stats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist per-field extraction stats (null rate, extraction source, confidence) to a new `field_stats` SQLite table and expose them in the Metrics Dashboard as a toggleable per-type table.

**Architecture:** Four changes in sequence — (1) storage: new table + write path; (2) pipeline: pass field_sources/confidence to persist before the call; (3) metrics_dashboard: four query functions + UI section; (4) finalize_review: same pipeline wiring for the review path.

**Tech Stack:** Python 3.11/3.12, SQLite, pandas, Streamlit (AppTest for UI tests).

---

## File Map

| File | Change |
|---|---|
| `src/doc_ai/storage.py` | Add `field_stats` table; add `field_sources`/`field_confidence` params; write rows; pre-write delete |
| `src/doc_ai/pipeline.py` | Move `_build_field_sources`/`_compute_field_confidence` before `persist()` in both `process_bytes()` and `finalize_review()` |
| `src/doc_ai/metrics_dashboard.py` | Add 4 query functions + `_render_field_stats_tab` helper + new section in `render_metrics_dashboard()` |
| `tests/test_metrics_dashboard.py` | Add `TestFieldStatsMetrics` class (7 tests) |

---

## Task 1: `field_stats` Table — Storage Write Path

**Files:**
- Modify: `src/doc_ai/storage.py`
- Test: `tests/test_metrics_dashboard.py`

Context: `ResultStore` in `storage.py` writes every processed document to `document_results`, `validation_results`, and `extraction_traces`. We need a fourth table, `field_stats`, written alongside those. The `_SYSTEM_COLUMNS` sentinel is NOT used for `field_stats` — it has its own fixed schema.

The `FIELD_CATALOG` dict (imported from `schema_config`) maps `doc_type → list[{key, label, ...}]` and is the canonical field list per type.

- [ ] **Step 1: Write the failing tests**

Add a new class at the bottom of `tests/test_metrics_dashboard.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_metrics_dashboard.py::TestFieldStatsMetrics::test_field_stats_written_on_persist tests/test_metrics_dashboard.py::TestFieldStatsMetrics::test_field_stats_replaced_on_repersist -v
```

Expected: FAIL — `persist()` does not accept `field_sources`/`field_confidence` yet.

- [ ] **Step 3: Update the import in `storage.py`**

Find the existing import line (line ~13):
```python
from .schema_config import SchemaConfig, TABLE_NAMES
```
Replace with:
```python
from .schema_config import FIELD_CATALOG, SchemaConfig, TABLE_NAMES
```

- [ ] **Step 4: Add `field_stats` table creation to `_migrate_schema` in `storage.py`**

Find the `_migrate_schema` method. Inside the `try` block that runs `conn.executescript(...)` for `pdf_uploads` and `error_log`, add `field_stats` to the same script:

```python
conn.executescript("""
    CREATE TABLE IF NOT EXISTS pdf_uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_filename TEXT NOT NULL,
        upload_path TEXT,
        file_size_bytes INTEGER,
        processed_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS error_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        error_type TEXT NOT NULL,
        source TEXT,
        severity TEXT DEFAULT 'error',
        message TEXT,
        logged_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS field_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_file TEXT NOT NULL,
        document_type TEXT NOT NULL,
        field_name TEXT NOT NULL,
        is_null INTEGER NOT NULL,
        extraction_source TEXT,
        confidence REAL
    );
""")
```

Do the same replacement in `_ensure_tables` (which has an identical `executescript` call — update both occurrences).

- [ ] **Step 5: Add `field_sources` and `field_confidence` params to `persist()`, `_write_sqlite()`, `_write_sqlite_locked()`**

In `persist()`, change the signature from:
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
) -> dict[str, str]:
```
To:
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
    field_sources: dict[str, str] | None = None,
    field_confidence: dict[str, float] | None = None,
) -> dict[str, str]:
```

Thread them through to `_write_sqlite`:
```python
self._write_sqlite(
    source_file_name,
    extracted_data,
    validation_checks,
    extraction_trace,
    content_hash=content_hash,
    original_filename=original_filename,
    semantic_fingerprint=semantic_fingerprint,
    field_sources=field_sources or {},
    field_confidence=field_confidence or {},
)
```

Update `_write_sqlite()` signature the same way and thread through to `_write_sqlite_locked()`:
```python
def _write_sqlite(
    self,
    source_file_name: str,
    extracted_data: dict[str, Any],
    validation_checks: list[ValidationCheck],
    extraction_trace: list[str],
    content_hash: str = "",
    original_filename: str = "",
    semantic_fingerprint: str = "",
    field_sources: dict[str, str] | None = None,
    field_confidence: dict[str, float] | None = None,
) -> None:
    with _DB_WRITE_LOCK:
        self._write_sqlite_locked(
            source_file_name, extracted_data, validation_checks,
            extraction_trace, content_hash, original_filename,
            semantic_fingerprint=semantic_fingerprint,
            field_sources=field_sources or {},
            field_confidence=field_confidence or {},
        )
```

Update `_write_sqlite_locked()` signature the same way.

- [ ] **Step 6: Add pre-write delete and field_stats rows in `_write_sqlite_locked()`**

Find the block that deletes prior rows (around line 336):
```python
for tbl in ("document_results", "validation_results", "extraction_traces"):
    try:
        conn.execute(f"DELETE FROM {tbl} WHERE source_file = ?", (source_file_name,))
    except Exception:
        pass
```
Add `"field_stats"` to the tuple:
```python
for tbl in ("document_results", "validation_results", "extraction_traces", "field_stats"):
    try:
        conn.execute(f"DELETE FROM {tbl} WHERE source_file = ?", (source_file_name,))
    except Exception:
        pass
```

Then, after the `pd.DataFrame([type_row]).to_sql(type_table, ...)` call at the end of `_write_sqlite_locked`, add the `field_stats` write:

```python
# Write per-field stats
doc_type_for_stats = extracted_data.get("document_type", "invoice")
catalog_fields = FIELD_CATALOG.get(doc_type_for_stats, [])
if catalog_fields:
    stats_rows = []
    for f in catalog_fields:
        key = f["key"]
        val = extracted_data.get(key)
        is_null = 1 if val in (None, "", []) else 0
        stats_rows.append({
            "source_file": source_file_name,
            "document_type": doc_type_for_stats,
            "field_name": key,
            "is_null": is_null,
            "extraction_source": (field_sources or {}).get(key) if not is_null else None,
            "confidence": (field_confidence or {}).get(key) if not is_null else None,
        })
    pd.DataFrame(stats_rows).to_sql("field_stats", conn, if_exists="append", index=False)
```

- [ ] **Step 7: Run the tests to verify they pass**

```
pytest tests/test_metrics_dashboard.py::TestFieldStatsMetrics::test_field_stats_written_on_persist tests/test_metrics_dashboard.py::TestFieldStatsMetrics::test_field_stats_replaced_on_repersist -v
```

Expected: PASS

- [ ] **Step 8: Run the full metrics dashboard test suite to confirm no regressions**

```
pytest tests/test_metrics_dashboard.py -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 9: Commit**

```bash
git add src/doc_ai/storage.py tests/test_metrics_dashboard.py
git commit -m "feat: add field_stats table to storage — persist per-field null/source/confidence"
```

---

## Task 2: Pipeline Wiring — `process_bytes()`

**Files:**
- Modify: `src/doc_ai/pipeline.py`
- Test: `tests/test_metrics_dashboard.py`

Context: In `process_bytes()`, `_build_field_sources()` and `_compute_field_confidence()` are currently called **after** `persist()` (lines 312–315 in the current file). We need to move them **before** the `persist()` call so their results can be passed in. The semantic-fingerprint early-return path must NOT write to `field_stats` — which is already guaranteed because that path returns before `persist()` is ever called.

- [ ] **Step 1: Write the failing test**

Add to `TestFieldStatsMetrics` in `tests/test_metrics_dashboard.py`:

```python
def test_field_stats_not_written_for_semantic_duplicate(self, tmp_path):
    """Semantic-fingerprint dedup early return must not write any field_stats rows."""
    from src.doc_ai.config import get_settings
    from src.doc_ai.pipeline import DocumentPipeline
    import dataclasses

    get_settings.cache_clear()
    s = get_settings()
    s = dataclasses.replace(s, data_dir=tmp_path)
    pipeline = DocumentPipeline(s)

    text = (
        "Acme Corp\nINVOICE\n"
        "Invoice Number: INV-001\nInvoice Date: 2025-01-15\nTotal: $100.00\n"
    )
    # First upload — stored normally
    pipeline.process_bytes("inv.txt", text.encode())
    # Second upload — same key fields, slightly different bytes → semantic dup
    pipeline.process_bytes("inv_copy.txt", (text + "\n").encode())

    import sqlite3
    conn = sqlite3.connect(str(s.database_path))
    count = conn.execute(
        "SELECT COUNT(*) FROM field_stats WHERE source_file='inv_copy.txt'"
    ).fetchone()[0]
    conn.close()
    assert count == 0, "Semantic duplicate must not write field_stats rows"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_metrics_dashboard.py::TestFieldStatsMetrics::test_field_stats_not_written_for_semantic_duplicate -v
```

Expected: FAIL — `persist()` not yet receiving field_sources/confidence (but the test may also fail because field_stats doesn't exist for `inv_copy.txt` — either way it must fail or pass trivially; running confirms the test is live).

- [ ] **Step 3: Move field computation before `persist()` in `process_bytes()`**

In `src/doc_ai/pipeline.py`, find the `process_bytes()` method. Currently the order around line 288 is:

```python
output_files = self._store.persist(
    saved_path.name,
    extracted_data,
    validation_checks,
    extraction_trace,
    content_hash=content_hash,
    original_filename=file_name,
    semantic_fingerprint=semantic_fingerprint,
)

summary = { ... }

field_sources = self._build_field_sources(extracted_data, extraction_trace)
field_confidence = self._compute_field_confidence(
    extracted_data, validation_checks, extraction_trace, field_sources
)
```

Reorder so `field_sources` and `field_confidence` are computed first, then passed to `persist()`:

```python
field_sources = self._build_field_sources(extracted_data, extraction_trace)
field_confidence = self._compute_field_confidence(
    extracted_data, validation_checks, extraction_trace, field_sources
)

output_files = self._store.persist(
    saved_path.name,
    extracted_data,
    validation_checks,
    extraction_trace,
    content_hash=content_hash,
    original_filename=file_name,
    semantic_fingerprint=semantic_fingerprint,
    field_sources=field_sources,
    field_confidence=field_confidence,
)

summary = { ... }
```

The `PipelineResult(...)` construction at the end already uses `field_sources` and `field_confidence` — no change needed there since the variables are now simply defined earlier.

- [ ] **Step 4: Run the tests**

```
pytest tests/test_metrics_dashboard.py::TestFieldStatsMetrics -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Run the full pipeline test suite to confirm no regressions**

```
pytest tests/test_pipeline.py -q
```

Expected: same pass/skip/xfail counts as before (322 passed, 54 skipped, 44 xfailed).

- [ ] **Step 6: Commit**

```bash
git add src/doc_ai/pipeline.py
git commit -m "feat: pass field_sources/confidence to persist() in process_bytes"
```

---

## Task 3: Pipeline Wiring — `finalize_review()`

**Files:**
- Modify: `src/doc_ai/pipeline.py`

Context: `finalize_review()` also calls `self._store.persist()` (around line 398). It computes `field_sources` and `field_confidence` **after** that call (lines 414–428). Same fix needed: move computation before the `persist()` call. No new tests needed — the existing correction-flow fixture tests (`TestFixtureCorrectionFlow`) cover `finalize_review()` end-to-end.

- [ ] **Step 1: Move field computation before `persist()` in `finalize_review()`**

In `src/doc_ai/pipeline.py`, find `finalize_review()`. Currently:

```python
output_files = self._store.persist(source_file, corrected_data, validation_checks, extraction_trace, content_hash=content_hash)
summary = { ... }

# Rebuild field_sources ...
original_sources = self._build_field_sources(
    original_extracted or corrected_data, prior_trace
)
field_sources = {}
for k, v in corrected_data.items():
    if k in ("document_type", "source_file") or v in (None, "", [], {}):
        continue
    orig_val = (original_extracted or {}).get(k)
    if _has_changes and not _field_values_equivalent(v, orig_val):
        field_sources[k] = "Manual"
    else:
        field_sources[k] = original_sources.get(k, "Rule-based")
field_confidence = self._compute_field_confidence(
    corrected_data, validation_checks, extraction_trace, field_sources
)
```

Reorder to compute `field_sources` and `field_confidence` before `persist()`:

```python
# Rebuild field_sources: only mark fields as "Manual" when the user actually
# changed them. Unchanged fields keep their original source attribution.
original_sources = self._build_field_sources(
    original_extracted or corrected_data, prior_trace
)
field_sources = {}
for k, v in corrected_data.items():
    if k in ("document_type", "source_file") or v in (None, "", [], {}):
        continue
    orig_val = (original_extracted or {}).get(k)
    if _has_changes and not _field_values_equivalent(v, orig_val):
        field_sources[k] = "Manual"
    else:
        field_sources[k] = original_sources.get(k, "Rule-based")
field_confidence = self._compute_field_confidence(
    corrected_data, validation_checks, extraction_trace, field_sources
)

output_files = self._store.persist(
    source_file,
    corrected_data,
    validation_checks,
    extraction_trace,
    content_hash=content_hash,
    field_sources=field_sources,
    field_confidence=field_confidence,
)
summary = { ... }
```

The `return PipelineResult(...)` at the end already uses `field_sources` and `field_confidence` — no change needed there.

- [ ] **Step 2: Run the full pipeline test suite**

```
pytest tests/test_pipeline.py -q
```

Expected: same counts as before (322 passed, 54 skipped, 44 xfailed).

- [ ] **Step 3: Commit**

```bash
git add src/doc_ai/pipeline.py
git commit -m "feat: pass field_sources/confidence to persist() in finalize_review"
```

---

## Task 4: Query Functions — `metrics_dashboard.py`

**Files:**
- Modify: `src/doc_ai/metrics_dashboard.py`
- Test: `tests/test_metrics_dashboard.py`

Context: Four new pure functions that read from `field_stats`. They must return empty DataFrames (not raise) when the table doesn't exist — matching the existing pattern of `_table_exists()` guards used by all other query functions in this file.

- [ ] **Step 1: Write the failing tests**

Add to `TestFieldStatsMetrics` in `tests/test_metrics_dashboard.py`:

```python
def _make_field_stats_db(self, tmp_path: Path) -> Path:
    """Build a DB with field_stats rows for two invoice docs."""
    db_path = tmp_path / "fstats.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE field_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            document_type TEXT NOT NULL,
            field_name TEXT NOT NULL,
            is_null INTEGER NOT NULL,
            extraction_source TEXT,
            confidence REAL
        )
    """)
    rows = [
        # doc1: vendor_name extracted via Template, invoice_number missing
        ("doc1.pdf", "invoice", "vendor_name",    0, "Template",   0.92),
        ("doc1.pdf", "invoice", "invoice_number", 1, None,         None),
        ("doc1.pdf", "invoice", "invoice_date",   0, "Rule-based", 0.72),
        # doc2: vendor_name extracted via LLM, invoice_number extracted via Rule-based
        ("doc2.pdf", "invoice", "vendor_name",    0, "LLM",        0.88),
        ("doc2.pdf", "invoice", "invoice_number", 0, "Rule-based", 0.71),
        ("doc2.pdf", "invoice", "invoice_date",   0, "Template",   0.95),
    ]
    conn.executemany(
        "INSERT INTO field_stats (source_file, document_type, field_name, is_null, extraction_source, confidence) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path

def test_field_null_rates_correct(self, tmp_path):
    from src.doc_ai.metrics_dashboard import field_null_rates
    db_path = self._make_field_stats_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    df = field_null_rates(conn, "invoice")
    conn.close()
    by_field = df.set_index("field")
    assert by_field.loc["vendor_name", "null_rate_pct"] == 0.0
    assert by_field.loc["invoice_number", "null_rate_pct"] == 50.0
    assert by_field.loc["invoice_date", "null_rate_pct"] == 0.0

def test_field_extraction_sources_breakdown(self, tmp_path):
    from src.doc_ai.metrics_dashboard import field_extraction_sources
    db_path = self._make_field_stats_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    df = field_extraction_sources(conn, "invoice")
    conn.close()
    by_field = df.set_index("field")
    # vendor_name: Template (1), LLM (1) — top_source is whichever comes first
    vn_breakdown = by_field.loc["vendor_name", "source_breakdown"]
    assert "Template" in vn_breakdown
    assert "LLM" in vn_breakdown
    # invoice_date: Template (1), Rule-based (1)
    id_breakdown = by_field.loc["invoice_date", "source_breakdown"]
    assert "Template" in id_breakdown
    assert "Rule-based" in id_breakdown

def test_field_avg_confidence_excludes_nulls(self, tmp_path):
    from src.doc_ai.metrics_dashboard import field_avg_confidence
    db_path = self._make_field_stats_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    df = field_avg_confidence(conn, "invoice")
    conn.close()
    by_field = df.set_index("field")
    # vendor_name: (0.92 + 0.88) / 2 = 0.9
    assert abs(by_field.loc["vendor_name", "avg_confidence"] - 0.900) < 0.001
    # invoice_number: doc1 is null (excluded), doc2 = 0.71
    assert abs(by_field.loc["invoice_number", "avg_confidence"] - 0.710) < 0.001

def test_doc_types_with_field_stats_filters_correctly(self, tmp_path):
    from src.doc_ai.metrics_dashboard import doc_types_with_field_stats
    db_path = self._make_field_stats_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    types = doc_types_with_field_stats(conn)
    conn.close()
    assert types == ["invoice"]  # only invoice rows were inserted

def test_field_stats_query_functions_empty_on_no_table(self, tmp_path):
    from src.doc_ai.metrics_dashboard import (
        doc_types_with_field_stats,
        field_avg_confidence,
        field_extraction_sources,
        field_null_rates,
    )
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db_path))  # no tables created
    assert doc_types_with_field_stats(conn) == []
    assert field_null_rates(conn, "invoice").empty
    assert field_extraction_sources(conn, "invoice").empty
    assert field_avg_confidence(conn, "invoice").empty
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_metrics_dashboard.py::TestFieldStatsMetrics::test_field_null_rates_correct tests/test_metrics_dashboard.py::TestFieldStatsMetrics::test_field_extraction_sources_breakdown tests/test_metrics_dashboard.py::TestFieldStatsMetrics::test_field_avg_confidence_excludes_nulls tests/test_metrics_dashboard.py::TestFieldStatsMetrics::test_doc_types_with_field_stats_filters_correctly tests/test_metrics_dashboard.py::TestFieldStatsMetrics::test_field_stats_query_functions_empty_on_no_table -v
```

Expected: FAIL — functions not yet defined.

- [ ] **Step 3: Add the four query functions to `metrics_dashboard.py`**

Add after the `records_created()` function and before `get_metrics()`:

```python
# ---------------------------------------------------------------------------
# Per-type field stats
# ---------------------------------------------------------------------------


def doc_types_with_field_stats(conn: sqlite3.Connection) -> list[str]:
    """Return distinct document_type values present in field_stats, alphabetically."""
    if not _table_exists(conn, "field_stats"):
        return []
    rows = conn.execute(
        "SELECT DISTINCT document_type FROM field_stats ORDER BY document_type"
    ).fetchall()
    return [r[0] for r in rows]


def field_null_rates(conn: sqlite3.Connection, doc_type: str) -> pd.DataFrame:
    """Return null rate per field for *doc_type*.

    Columns: field, total_docs, extracted, null_rate_pct (float 0-100).
    """
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
    """Return extraction source breakdown per field for *doc_type*.

    Columns: field, top_source, source_breakdown (e.g. 'Template (7), Rule-based (2)').
    Only counts rows where is_null=0 (field was actually extracted).
    """
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
    result_rows = []
    for field, grp in raw.groupby("field", sort=False):
        top = grp.iloc[0]["extraction_source"]
        breakdown = ", ".join(
            f"{r['extraction_source']} ({r['cnt']})" for _, r in grp.iterrows()
        )
        result_rows.append({"field": field, "top_source": top, "source_breakdown": breakdown})
    return pd.DataFrame(result_rows)


def field_avg_confidence(conn: sqlite3.Connection, doc_type: str) -> pd.DataFrame:
    """Return average confidence per field for *doc_type*.

    Columns: field, avg_confidence, min_confidence, max_confidence.
    Only counts rows where is_null=0 and confidence IS NOT NULL.
    Values rounded to 3 decimal places.
    """
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

- [ ] **Step 4: Run the tests to verify they pass**

```
pytest tests/test_metrics_dashboard.py::TestFieldStatsMetrics -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Run the full metrics dashboard test suite**

```
pytest tests/test_metrics_dashboard.py -v
```

Expected: all previously passing tests still pass plus the 7 new ones.

- [ ] **Step 6: Commit**

```bash
git add src/doc_ai/metrics_dashboard.py tests/test_metrics_dashboard.py
git commit -m "feat: add field_stats query functions to metrics_dashboard"
```

---

## Task 5: UI Section — Per Document Type Analysis

**Files:**
- Modify: `src/doc_ai/metrics_dashboard.py`

Context: Add a new section at the bottom of `render_metrics_dashboard()`. Uses only the query functions added in Task 4. No new Streamlit session state keys beyond the three checkboxes. The `FIELD_CATALOG` import at the top of the file is needed to order fields correctly — add it alongside the existing import.

- [ ] **Step 1: Add `FIELD_CATALOG` to the import in `metrics_dashboard.py`**

Find the existing import (line ~43):
```python
from .schema_config import TABLE_NAMES
```
Replace with:
```python
from .schema_config import FIELD_CATALOG, TABLE_NAMES
```

- [ ] **Step 2: Add the `_render_field_stats_tab()` helper function**

Add this function to `metrics_dashboard.py` just before `render_metrics_dashboard()`:

```python
def _render_field_stats_tab(
    conn: sqlite3.Connection,
    doc_type: str,
    show_null: bool,
    show_method: bool,
    show_conf: bool,
) -> None:
    """Render the per-field stats table for one document type tab."""
    import streamlit as st

    # Canonical field order from FIELD_CATALOG
    catalog_fields = [f["key"] for f in FIELD_CATALOG.get(doc_type, [])]
    catalog_labels = {f["key"]: f["label"] for f in FIELD_CATALOG.get(doc_type, [])}

    # Build base DataFrame from catalog order
    display = pd.DataFrame({"field": catalog_fields})
    display["Field"] = display["field"].map(catalog_labels)

    if show_null:
        null_df = field_null_rates(conn, doc_type)[["field", "null_rate_pct"]]
        display = display.merge(null_df, on="field", how="left")
        display = display.rename(columns={"null_rate_pct": "Null Rate %"})

    if show_method:
        src_df = field_extraction_sources(conn, doc_type)[["field", "source_breakdown"]]
        display = display.merge(src_df, on="field", how="left")
        display = display.rename(columns={"source_breakdown": "Extraction Method"})

    if show_conf:
        conf_df = field_avg_confidence(conn, doc_type)[["field", "avg_confidence"]]
        display = display.merge(conf_df, on="field", how="left")
        display = display.rename(columns={"avg_confidence": "Avg Confidence"})

    display = display.drop(columns=["field"]).fillna("—")

    # Apply colour gradient to numeric metric columns
    styler = display.style
    if show_null and "Null Rate %" in display.columns:
        numeric_null = pd.to_numeric(display["Null Rate %"], errors="coerce")
        styler = styler.background_gradient(
            subset=["Null Rate %"],
            cmap="RdYlGn_r",
            gmap=numeric_null,
            vmin=0,
            vmax=100,
        )
    if show_conf and "Avg Confidence" in display.columns:
        numeric_conf = pd.to_numeric(display["Avg Confidence"], errors="coerce")
        styler = styler.background_gradient(
            subset=["Avg Confidence"],
            cmap="RdYlGn",
            gmap=numeric_conf,
            vmin=0.0,
            vmax=1.0,
        )

    st.dataframe(styler, use_container_width=True, hide_index=True)
```

- [ ] **Step 3: Add the new section to `render_metrics_dashboard()`**

At the end of `render_metrics_dashboard()`, just before the `finally: conn.close()` block, add:

```python
        st.divider()

        # ---- Per document type analysis ------------------------------------------
        st.subheader("Per Document Type Analysis")
        st.caption(
            "Per-field breakdown for each document type. "
            "Only document types with processed records appear as tabs."
        )

        check_col1, check_col2, check_col3 = st.columns(3)
        show_null = check_col1.checkbox("Null Rate", value=True, key="fstats_show_null")
        show_method = check_col2.checkbox("Extraction Method", value=True, key="fstats_show_method")
        show_conf = check_col3.checkbox("Avg Confidence", value=True, key="fstats_show_conf")

        if not (show_null or show_method or show_conf):
            st.info("Select at least one column above to display.")
        else:
            available_types = doc_types_with_field_stats(conn)
            if not available_types:
                st.info(
                    "No field-level stats yet. Process a document to populate this section. "
                    "Documents processed before this update will not appear here until reprocessed."
                )
            else:
                tab_labels = [t.replace("_", " ").title() for t in available_types]
                type_tabs = st.tabs(tab_labels)
                for type_tab, doc_type in zip(type_tabs, available_types):
                    with type_tab:
                        _render_field_stats_tab(
                            conn, doc_type, show_null, show_method, show_conf
                        )
```

- [ ] **Step 4: Run the full test suite**

```
pytest tests/test_metrics_dashboard.py tests/test_pipeline.py tests/test_ui.py -q
```

Expected: all previously passing tests still pass. The UI is not tested for this new section (AppTest does not exercise the metrics dashboard tab rendering with a real DB) — confirmed by existing test coverage approach.

- [ ] **Step 5: Commit**

```bash
git add src/doc_ai/metrics_dashboard.py
git commit -m "feat: add per-type field stats UI section to metrics dashboard"
```

---

## Task 6: Rebuild Docker and Smoke-Test

**Files:** none (Docker only)

- [ ] **Step 1: Build the Docker image**

```bash
docker build -t aiocr:local .
```

Expected: build succeeds with exit code 0.

- [ ] **Step 2: Restart the container**

```bash
docker stop aiocr && docker rm aiocr && docker run -d --name aiocr -p 8501:8501 -v "c:/Users/colli/AiOcr/data:/app/data" aiocr:local
```

- [ ] **Step 3: Smoke-test in browser**

Open `http://localhost:8501`, navigate to the Metrics Dashboard tab.

Verify:
- If documents have been processed previously: the "Per Document Type Analysis" section shows tabs for each type that has data.
- Unchecking a checkbox removes that column from all type tabs.
- Documents processed before this update show the info message ("will not appear here until reprocessed").
- Process one new document — confirm the type tab appears with its field rows.

- [ ] **Step 4: Commit (if any hotfixes were needed)**

```bash
git add -A
git commit -m "fix: <describe any smoke-test fix>"
```

If no fixes were needed, skip this step.
