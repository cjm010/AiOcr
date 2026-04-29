from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings
from .schema_config import SchemaConfig, TABLE_NAMES
from .schemas import ValidationCheck

_DB_WRITE_LOCK = threading.Lock()


class ResultStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._schema_config = SchemaConfig(settings.data_dir / "schema_settings.json")
        self._migrate_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._settings.database_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _migrate_schema(self) -> None:
        db = self._settings.database_path
        if not Path(db).exists():
            return
        conn = self._connect()
        try:
            conn.execute("ALTER TABLE document_results ADD COLUMN content_hash TEXT DEFAULT ''")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists or table not yet created — both are fine
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

    def persist(
        self,
        source_file_name: str,
        extracted_data: dict[str, Any],
        validation_checks: list[ValidationCheck],
        extraction_trace: list[str],
        content_hash: str = "",
        original_filename: str = "",
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
    ) -> None:
        with _DB_WRITE_LOCK:
            self._write_sqlite_locked(
                source_file_name, extracted_data, validation_checks,
                extraction_trace, content_hash, original_filename,
            )

    def _write_sqlite_locked(
        self,
        source_file_name: str,
        extracted_data: dict[str, Any],
        validation_checks: list[ValidationCheck],
        extraction_trace: list[str],
        content_hash: str = "",
        original_filename: str = "",
    ) -> None:
        conn = self._connect()
        try:
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
            doc_df.to_sql("document_results", conn, if_exists="append", index=False)
            validation_df.to_sql("validation_results", conn, if_exists="append", index=False)
            trace_df.to_sql("extraction_traces", conn, if_exists="append", index=False)

            # Write to the per-type table with only the user-selected fields.
            doc_type = extracted_data.get("document_type", "invoice")
            type_table = TABLE_NAMES.get(doc_type)
            if type_table:
                selected = self._schema_config.get_selected_fields(doc_type)
                type_row: dict[str, Any] = {
                    "source_file": source_file_name,
                    "original_filename": original_filename or source_file_name,
                    "content_hash": content_hash,
                }
                for key in selected:
                    raw = extracted_data.get(key)
                    type_row[key] = json.dumps(raw) if isinstance(raw, (list, dict)) else raw
                pd.DataFrame([type_row]).to_sql(type_table, conn, if_exists="append", index=False)
        finally:
            conn.close()
