from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    upload_dir: Path
    output_dir: Path
    database_path: Path
    template_store_path: Path
    enable_template_learning: bool
    min_learning_pass_ratio: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()

    data_dir = Path(os.getenv("APP_DATA_DIR", "data")).resolve()
    upload_dir = data_dir / "uploads"
    output_dir = data_dir / "outputs"

    for directory in (data_dir, upload_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return Settings(
        data_dir=data_dir,
        upload_dir=upload_dir,
        output_dir=output_dir,
        database_path=data_dir / "document_results.db",
        template_store_path=data_dir / "learned_templates.json",
        enable_template_learning=os.getenv("ENABLE_TEMPLATE_LEARNING", "true").lower() == "true",
        min_learning_pass_ratio=float(os.getenv("MIN_LEARNING_PASS_RATIO", "0.6")),
    )
