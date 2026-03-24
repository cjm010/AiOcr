from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ValidationCheck:
    field: str
    status: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class ParsedDocument:
    file_name: str
    file_path: Path
    raw_text: str
    sections: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    source_file: str
    upload_path: str
    parsed_text: str
    extracted_data: dict[str, Any]
    validation_results: list[dict[str, str]]
    output_files: dict[str, str]
    summary: dict[str, Any]
    errors: list[str]
    extraction_trace: list[str] = field(default_factory=list)
    processed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
