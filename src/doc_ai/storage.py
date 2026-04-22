from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings
from .schemas import ValidationCheck


class ResultStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def has_been_processed(self, content_hash: str) -> bool:
        db = self._settings.database_path
        if not Path(db).exists():
            return False
        conn = sqlite3.connect(db)
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
        conn = sqlite3.connect(self._settings.database_path)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            doc_df = pd.DataFrame(
                [
                    {
                        "source_file": source_file_name,
                        "original_filename": original_filename or source_file_name,
                        "content_hash": content_hash,
                        **extracted_data,
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
        finally:
            conn.close()
