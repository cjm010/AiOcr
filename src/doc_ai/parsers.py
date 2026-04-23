from __future__ import annotations

import json
from pathlib import Path

from .schemas import ParsedDocument

# PDFs with fewer characters than this after text extraction are treated as
# scanned images and sent through Tesseract OCR.
_MIN_TEXT_CHARS = 100


class DocumentParser:
    """Parses uploaded documents into plain text for downstream extraction."""

    def parse(self, file_path: Path) -> ParsedDocument:
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md"}:
            raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".json":
            raw_text = json.dumps(json.loads(file_path.read_text(encoding="utf-8")), indent=2)
        elif suffix == ".pdf":
            raw_text = self._parse_pdf(file_path)
        else:
            raw_text = file_path.read_text(encoding="utf-8", errors="ignore")

        sections = [chunk.strip() for chunk in raw_text.splitlines() if chunk.strip()]
        return ParsedDocument(
            file_name=file_path.name,
            file_path=file_path,
            raw_text=raw_text,
            sections=sections,
            metadata={"suffix": suffix, "section_count": len(sections)},
        )

    def _parse_pdf(self, file_path: Path) -> str:
        text = self._parse_pdf_with_unstructured(file_path)
        if text and len(text) >= _MIN_TEXT_CHARS:
            return text

        text = self._parse_pdf_with_pypdf(file_path)
        if text and len(text) >= _MIN_TEXT_CHARS:
            return text

        text = self._parse_pdf_with_pdfplumber(file_path)
        if text and len(text) >= _MIN_TEXT_CHARS:
            return text

        # All text-based parsers returned too little content — treat as scanned.
        text = self._parse_pdf_with_ocr(file_path)
        if text:
            return text

        raise RuntimeError(
            "Could not extract text from this PDF. The file may be corrupted or an unsupported format."
        )

    def _parse_pdf_with_ocr(self, file_path: Path) -> str:
        """Rasterize each PDF page and run Tesseract OCR on it."""
        try:
            import pypdfium2 as pdfium
            import pytesseract
            from PIL import Image
        except ImportError:
            return ""

        try:
            doc = pdfium.PdfDocument(str(file_path))
        except Exception:
            return ""

        pages: list[str] = []
        try:
            for page in doc:
                try:
                    bitmap = page.render(scale=2.0)  # 2x scale → ~144 dpi, better OCR accuracy
                    pil_image = bitmap.to_pil()
                    page_text = pytesseract.image_to_string(pil_image).strip()
                    if page_text:
                        pages.append(page_text)
                except Exception:
                    continue
        finally:
            doc.close()

        return "\n\n".join(pages).strip()

    def _parse_pdf_with_unstructured(self, file_path: Path) -> str:
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError:
            return ""

        try:
            elements = partition_pdf(filename=str(file_path))
        except Exception:
            return ""

        parts = [str(element).strip() for element in elements if str(element).strip()]
        return "\n".join(parts).strip()

    def _parse_pdf_with_pypdf(self, file_path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            return ""

        reader = PdfReader(str(file_path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        text = "\n\n".join(page for page in pages if page)
        return text.strip()

    def _parse_pdf_with_pdfplumber(self, file_path: Path) -> str:
        try:
            import pdfplumber
        except ImportError:
            return ""

        try:
            with pdfplumber.open(str(file_path)) as pdf:
                pages = [(page.extract_text() or "").strip() for page in pdf.pages]
        except Exception:
            return ""

        return "\n\n".join(page for page in pages if page).strip()
