from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings
from .schema_config import FIELD_CATALOG, SchemaConfig, TABLE_NAMES
from .schemas import ValidationCheck

_DB_WRITE_LOCK = threading.Lock()


class ResultStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._schema_config = SchemaConfig(settings.data_dir / "schema_settings.json")
        self._ensure_tables()
        self._migrate_schema()

    def _ensure_tables(self) -> None:
        conn = self._connect()
        # Add processed_at to existing per-type tables when missing.
        for type_table in TABLE_NAMES.values():
            try:
                cols = {row[1] for row in conn.execute(f"PRAGMA table_info({type_table})")}
                if cols and "processed_at" not in cols:
                    conn.execute(
                        f"ALTER TABLE {type_table} ADD COLUMN processed_at TEXT"
                    )
            except sqlite3.OperationalError:
                pass
        conn.commit()
        try:
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
            conn.commit()
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._settings.database_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _migrate_schema(self) -> None:
        db = self._settings.database_path
        if not Path(db).exists():
            return
        conn = self._connect()
        # Get existing columns for document_results and add any that are missing.
        try:
            existing = {row[1] for row in conn.execute("PRAGMA table_info(document_results)")}
            new_columns = {
                # Core / invoice
                "content_hash": "TEXT DEFAULT ''",
                "semantic_fingerprint": "TEXT DEFAULT ''",
                "original_filename": "TEXT DEFAULT ''",
                "currency": "TEXT",
                "shipping_handling": "REAL",
                "document_type": "TEXT",
                "vendor_name": "TEXT",
                "invoice_number": "TEXT",
                "invoice_date": "TEXT",
                "due_date": "TEXT",
                "subtotal": "REAL",
                "tax": "REAL",
                "total_amount": "REAL",
                # Business doc
                "company_name": "TEXT",
                "document_subtype": "TEXT",
                "report_period": "TEXT",
                "report_date": "TEXT",
                "report_id": "TEXT",
                "prepared_by": "TEXT",
                "approved_by": "TEXT",
                "classification": "TEXT",
                "executive_summary": "TEXT",
                # Medical discharge
                "facility_name": "TEXT",
                "patient_name": "TEXT",
                "admission_date": "TEXT",
                "discharge_date": "TEXT",
                "primary_diagnosis": "TEXT",
                "attending_physician": "TEXT",
                # NDA
                "disclosing_party": "TEXT",
                "receiving_party": "TEXT",
                "agreement_date": "TEXT",
                "effective_date": "TEXT",
                "expiration_date": "TEXT",
                "agreement_type": "TEXT",
                "governing_law": "TEXT",
                # Lab report
                "lab_name": "TEXT",
                "patient_id": "TEXT",
                "collected_date": "TEXT",
                "reported_date": "TEXT",
                "ordering_physician": "TEXT",
                "clinical_interpretation": "TEXT",
            }
            for col, col_type in new_columns.items():
                if col not in existing:
                    try:
                        conn.execute(f"ALTER TABLE document_results ADD COLUMN {col} {col_type}")
                    except sqlite3.OperationalError:
                        pass
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Table not yet created — fine, will be created on first write
        try:
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
            conn.commit()
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()

    def log_upload(self, original_filename: str, upload_path: str, file_size_bytes: int) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO pdf_uploads (original_filename, upload_path, file_size_bytes) VALUES (?, ?, ?)",
                (original_filename, upload_path, file_size_bytes),
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()

    def log_error(self, error_type: str, source: str, message: str, severity: str = "error") -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO error_log (error_type, source, severity, message) VALUES (?, ?, ?, ?)",
                (error_type, source, severity, message),
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()

    def get_processing_trace(self, source_file: str) -> list[str]:
        """Return the stored trace steps for *source_file*, ordered by step_number."""
        if not Path(self._settings.database_path).exists():
            return []
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT message FROM extraction_traces WHERE source_file = ? ORDER BY step_number",
                (source_file,),
            )
            return [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def get_error_log(self, limit: int = 200) -> list[dict]:
        if not Path(self._settings.database_path).exists():
            return []
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT logged_at, severity, error_type, source, message "
                "FROM error_log ORDER BY logged_at DESC LIMIT ?",
                (limit,),
            )
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def get_upload_log(self, limit: int = 200) -> list[dict]:
        if not Path(self._settings.database_path).exists():
            return []
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT processed_at, original_filename, upload_path, file_size_bytes "
                "FROM pdf_uploads ORDER BY processed_at DESC LIMIT ?",
                (limit,),
            )
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def has_been_processed(self, content_hash: str) -> bool:
        if not Path(self._settings.database_path).exists():
            return False
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT 1 FROM document_results WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            )
            return cursor.fetchone() is not None
        except sqlite3.OperationalError:
            return False
        finally:
            conn.close()

    def has_been_processed_by_fingerprint(self, semantic_fingerprint: str) -> bool:
        """Return True if a document with the same semantic fingerprint was already stored.

        Used as a secondary dedup gate after extraction to catch the same document
        processed in two different formats (e.g. scanned image-only vs. text-based PDF).
        Never matches on empty fingerprint.
        """
        if not semantic_fingerprint or not Path(self._settings.database_path).exists():
            return False
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT 1 FROM document_results "
                "WHERE semantic_fingerprint = ? AND semantic_fingerprint != '' LIMIT 1",
                (semantic_fingerprint,),
            )
            return cursor.fetchone() is not None
        except sqlite3.OperationalError:
            return False
        finally:
            conn.close()

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
        stem = Path(source_file_name).stem
        json_path = self._settings.output_dir / f"{stem}.json"
        csv_path = self._settings.output_dir / f"{stem}.csv"
        trace_path = self._settings.output_dir / f"{stem}_trace.json"

        json_path.write_text(json.dumps(extracted_data, indent=2), encoding="utf-8")
        trace_path.write_text(json.dumps({"trace": extraction_trace}, indent=2), encoding="utf-8")
        pd.DataFrame([extracted_data]).to_csv(csv_path, index=False)
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

        return {
            "json": str(json_path),
            "csv": str(csv_path),
            "trace": str(trace_path),
            "sqlite": str(self._settings.database_path),
        }

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

    def _write_sqlite_locked(
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
        conn = self._connect()
        try:
            # Delete any prior rows for this source_file so that finalize_review()
            # replaces the initial process_bytes() write rather than duplicating it.
            for tbl in ("document_results", "validation_results", "extraction_traces", "field_stats"):
                try:
                    conn.execute(f"DELETE FROM {tbl} WHERE source_file = ?", (source_file_name,))
                except Exception:
                    pass

            serialized = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else v
                for k, v in extracted_data.items()
            }
            doc_df = pd.DataFrame(
                [
                    {
                        "source_file": source_file_name,
                        "original_filename": original_filename or source_file_name,
                        "content_hash": content_hash,
                        "semantic_fingerprint": semantic_fingerprint,
                        **serialized,
                    }
                ]
            )
            validation_df = pd.DataFrame(
                [
                    {
                        "source_file": source_file_name,
                        **check.to_dict(),
                    }
                    for check in validation_checks
                ]
            )
            trace_df = pd.DataFrame(
                [
                    {
                        "source_file": source_file_name,
                        "step_number": index + 1,
                        "message": message,
                    }
                    for index, message in enumerate(extraction_trace)
                ]
            )
            existing_col_rows = conn.execute("PRAGMA table_info(document_results)").fetchall()
            if existing_col_rows:  # table exists — add any new columns before appending
                existing_cols = {row[1] for row in existing_col_rows}
                altered = False
                for col in doc_df.columns:
                    if col not in existing_cols:
                        conn.execute(f"ALTER TABLE document_results ADD COLUMN [{col}] TEXT")
                        altered = True
                if altered:
                    conn.commit()  # schema changes must be committed before to_sql sees them
            doc_df.to_sql("document_results", conn, if_exists="append", index=False)
            validation_df.to_sql("validation_results", conn, if_exists="append", index=False)
            trace_df.to_sql("extraction_traces", conn, if_exists="append", index=False)

            # Write to the per-type table with only the user-selected fields.
            doc_type = extracted_data.get("document_type", "invoice")
            type_table = TABLE_NAMES.get(doc_type)
            if type_table:
                try:
                    conn.execute(f"DELETE FROM {type_table} WHERE source_file = ?", (source_file_name,))
                except Exception:
                    pass
                selected = self._schema_config.get_selected_fields(doc_type)
                type_row: dict[str, Any] = {
                    "source_file": source_file_name,
                    "original_filename": original_filename or source_file_name,
                    "content_hash": content_hash,
                    "processed_at": datetime.utcnow().isoformat(sep=" ", timespec="seconds"),
                }
                for key in selected:
                    raw = extracted_data.get(key)
                    type_row[key] = json.dumps(raw) if isinstance(raw, (list, dict)) else raw
                # If the per-type table predates processed_at, add the column on the fly.
                try:
                    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({type_table})")}
                    if cols and "processed_at" not in cols:
                        conn.execute(f"ALTER TABLE {type_table} ADD COLUMN processed_at TEXT")
                except sqlite3.OperationalError:
                    pass
                pd.DataFrame([type_row]).to_sql(type_table, conn, if_exists="append", index=False)

            # Write per-field stats
            catalog_fields = FIELD_CATALOG.get(doc_type, [])
            if catalog_fields:
                stats_rows = []
                for f in catalog_fields:
                    key = f["key"]
                    val = extracted_data.get(key)
                    is_null = 1 if val in (None, "", []) else 0
                    stats_rows.append({
                        "source_file": source_file_name,
                        "document_type": doc_type,
                        "field_name": key,
                        "is_null": is_null,
                        "extraction_source": (field_sources or {}).get(key) if not is_null else None,
                        "confidence": (field_confidence or {}).get(key) if not is_null else None,
                    })
                try:
                    pd.DataFrame(stats_rows).to_sql("field_stats", conn, if_exists="append", index=False)
                except Exception:
                    pass
        finally:
            conn.close()
