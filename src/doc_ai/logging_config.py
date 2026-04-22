from __future__ import annotations

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-8s — %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("doc_ai")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    return logger
