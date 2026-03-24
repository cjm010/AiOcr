from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    app_env: str
    data_dir: Path
    upload_dir: Path
    output_dir: Path
    database_path: Path
    template_store_path: Path
    promoted_template_store_path: Path
    review_export_dir: Path
    enable_template_learning: bool
    min_learning_pass_ratio: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()

    app_env = os.getenv("APP_ENV", "dev").strip().lower()
    base_data_root = Path(os.getenv("APP_DATA_ROOT", "data"))
    data_dir = Path(os.getenv("APP_DATA_DIR", str(base_data_root / app_env))).resolve()
    upload_dir = data_dir / "uploads"
    output_dir = data_dir / "outputs"
    review_export_dir = data_dir / "review_exports"

    for directory in (data_dir, upload_dir, output_dir, review_export_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return Settings(
        app_env=app_env,
        data_dir=data_dir,
        upload_dir=upload_dir,
        output_dir=output_dir,
        database_path=data_dir / "document_results.db",
        template_store_path=data_dir / "learned_templates.json",
        promoted_template_store_path=data_dir / "promoted_templates.json",
        review_export_dir=review_export_dir,
        enable_template_learning=os.getenv("ENABLE_TEMPLATE_LEARNING", "true").lower() == "true",
        min_learning_pass_ratio=float(os.getenv("MIN_LEARNING_PASS_RATIO", "0.6")),
    )
