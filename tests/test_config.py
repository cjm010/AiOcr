import pytest
from pathlib import Path

from src.doc_ai.config import get_settings


def test_settings_build_data_dir(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.delenv("APP_DATA_DIR", raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", "data")

    settings = get_settings()

    assert settings.data_dir.parts[-2:] == ("data", "test")
    assert settings.app_env == "test"


# ---------------------------------------------------------------------------
# OCR library dependency checks
# ---------------------------------------------------------------------------

def test_pypdfium2_importable():
    """pypdfium2 is required for rasterizing PDF pages before OCR."""
    pytest.importorskip("pypdfium2", reason="pypdfium2 not installed — OCR unavailable")


def test_pillow_importable():
    """Pillow (PIL) is required to convert pypdfium2 bitmaps to images for Tesseract."""
    pytest.importorskip("PIL", reason="Pillow not installed — OCR unavailable")


def test_pytesseract_importable():
    """pytesseract Python package must be importable."""
    pytest.importorskip("pytesseract", reason="pytesseract not installed")


def test_tesseract_binary_accessible():
    """Tesseract binary must be on PATH (or pytesseract.tesseract_cmd must point to it).

    Install: Windows → winget install UB-Mannheim.TesseractOCR (add to PATH)
             Linux   → apt-get install tesseract-ocr
             macOS   → brew install tesseract
    """
    pytesseract = pytest.importorskip("pytesseract", reason="pytesseract not installed")
    try:
        version = pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as exc:
        pytest.skip(
            f"Tesseract binary not found: {exc} — "
            "install Tesseract and ensure it is on PATH to enable OCR tests"
        )
    assert version is not None


def test_ocr_pipeline_produces_text_from_image_pdf(tmp_path):
    """End-to-end OCR check: an image-only PDF must yield non-empty parsed text."""
    pytesseract = pytest.importorskip("pytesseract", reason="pytesseract not installed")
    pytest.importorskip("pypdfium2", reason="pypdfium2 not installed")
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        pytest.skip("Tesseract binary not installed — skipping OCR pipeline check")

    import os
    import shutil
    from src.doc_ai.parsers import DocumentParser

    os.environ["APP_ENV"] = "test"
    os.environ["APP_DATA_ROOT"] = str(tmp_path)
    get_settings.cache_clear()

    no_text_fixture = Path(__file__).parent / "fixtures" / "invoice_no_text_full.pdf"
    if not no_text_fixture.exists():
        pytest.skip("invoice_no_text_full.pdf fixture not present")

    dest = tmp_path / "invoice_no_text_full.pdf"
    shutil.copy2(no_text_fixture, dest)

    parser = DocumentParser()
    parsed = parser.parse(dest)
    assert parsed.raw_text.strip() != "", (
        "OCR produced no text from an image-only PDF. "
        "Check that the Tesseract binary is installed and the fixture is truly image-only."
    )
