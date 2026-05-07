"""
Layout-aware PDF field extraction using pdfplumber spatial coordinates.

Strategy
--------
1. Words are extracted from pdfplumber with (x0, top, x1, bottom) bounding
   boxes and grouped into visual rows by Y-centre proximity.
2. Each row is inspected for a label→value pair.  Two layouts are handled:
   - Inline colon  : "Invoice Number: INV-001" on the same row.
   - Column gap    : a large horizontal gap on a row separates a short label
                     (left half) from the value (right half).
3. Raw label strings are normalised and mapped to canonical field names via
   _LABEL_TO_FIELD.
4. When a template is being *learned* we record each found field's normalised
   page position (0-1 range) as a "spatial anchor" so future similar documents
   can be extracted by looking at the right region of the page.
5. When a template already has spatial anchors we can extract values by
   jumping straight to the anchor's page coordinates — accurate even when the
   surrounding text is different.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Label → canonical field name mapping (covers all five document types)
# ---------------------------------------------------------------------------

_LABEL_TO_FIELD: dict[str, str] = {
    # Invoice
    "invoice number": "invoice_number",
    "invoice no": "invoice_number",
    "invoice #": "invoice_number",
    "inv #": "invoice_number",
    "inv no": "invoice_number",
    "invoice date": "invoice_date",
    "date of invoice": "invoice_date",
    "issue date": "invoice_date",
    "date": "invoice_date",
    "due date": "due_date",
    "payment due": "due_date",
    "due": "due_date",
    "net 30": "due_date",
    "vendor": "vendor_name",
    "vendor name": "vendor_name",
    "supplier": "vendor_name",
    "supplier name": "vendor_name",
    "bill from": "vendor_name",
    "sold by": "vendor_name",
    "from": "vendor_name",
    "company": "vendor_name",
    "subtotal": "subtotal",
    "sub total": "subtotal",
    "sub-total": "subtotal",
    "tax": "tax",
    "sales tax": "tax",
    "gst": "tax",
    "vat": "tax",
    "hst": "tax",
    "shipping": "shipping_handling",
    "shipping & handling": "shipping_handling",
    "shipping and handling": "shipping_handling",
    "freight": "shipping_handling",
    "handling": "shipping_handling",
    "total": "total_amount",
    "total amount": "total_amount",
    "amount due": "total_amount",
    "balance due": "total_amount",
    "grand total": "total_amount",
    "total due": "total_amount",
    "currency": "currency",
    "po number": "po_number",
    "purchase order": "po_number",
    "purchase order number": "po_number",
    "bill to": "bill_to",
    "customer": "bill_to",
    "ship to": "ship_to",
    # Medical discharge
    "patient name": "patient_name",
    "patient": "patient_name",
    "patient id": "patient_name",
    "admission date": "admission_date",
    "admitted": "admission_date",
    "date admitted": "admission_date",
    "discharge date": "discharge_date",
    "discharged": "discharge_date",
    "date discharged": "discharge_date",
    "diagnosis": "primary_diagnosis",
    "primary diagnosis": "primary_diagnosis",
    "admitting diagnosis": "primary_diagnosis",
    "date of birth": "date_of_birth",
    "dob": "date_of_birth",
    "physician": "attending_physician",
    "attending physician": "attending_physician",
    "follow up date": "follow_up_date",
    "follow-up date": "follow_up_date",
    # NDA
    "disclosing party": "disclosing_party",
    "receiving party": "receiving_party",
    "agreement date": "agreement_date",
    "effective date": "effective_date",
    "expiration date": "expiration_date",
    "governing law": "governing_law",
    "agreement type": "agreement_type",
    # Lab report
    "collected": "collected_date",
    "collection date": "collected_date",
    "specimen collected": "collected_date",
    "reported": "reported_date",
    "report date": "reported_date",
    "date reported": "reported_date",
    "ordering physician": "ordering_physician",
    # Business doc
    "company name": "company_name",
    "prepared by": "prepared_by",
    "approved by": "approved_by",
    "report period": "report_period",
    "period": "report_period",
    "period ending": "report_period",
    "reporting period": "report_period",
    "report id": "report_id",
    "document type": "document_subtype",
    "document classification": "classification",
    "classification": "classification",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_COLON_ROW = re.compile(r"^(.+?)\s*:\s*(.+)\s*$")
_LABEL_ONLY = re.compile(r"^(.+?)\s*:\s*$")


@dataclass
class SpatialWord:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass
class SpatialRow:
    words: list[SpatialWord] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def top(self) -> float:
        return min(w.top for w in self.words) if self.words else 0.0

    @property
    def x0(self) -> float:
        return self.words[0].x0 if self.words else 0.0

    @property
    def x1(self) -> float:
        return self.words[-1].x1 if self.words else 0.0


@dataclass
class SpatialLayout:
    page_width: float
    page_height: float
    rows: list[SpatialRow]
    page_number: int = 0

    def norm_x(self, x: float) -> float:
        return round(x / self.page_width, 4) if self.page_width else 0.0

    def norm_y(self, y: float) -> float:
        return round(y / self.page_height, 4) if self.page_height else 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _group_into_rows(words: list[SpatialWord], y_tol: float = 3.0) -> list[SpatialRow]:
    """Cluster words into visual rows by Y-centre proximity."""
    rows: list[list[SpatialWord]] = []
    for word in sorted(words, key=lambda w: (w.center_y, w.x0)):
        placed = False
        for row in rows:
            if abs(word.center_y - row[0].center_y) <= y_tol:
                row.append(word)
                placed = True
                break
        if not placed:
            rows.append([word])
    return [SpatialRow(sorted(row, key=lambda w: w.x0)) for row in rows]


def _normalise_label(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip().lower().rstrip(":").strip())


def _row_to_pair(row: SpatialRow) -> tuple[str, str] | None:
    """Return (label, value) if the row contains a colon-separated pair."""
    text = row.text.strip()
    m = _COLON_ROW.match(text)
    if m:
        label = _normalise_label(m.group(1))
        value = m.group(2).strip()
        if label and value:
            return label, value
    return None


def _column_gap_pairs(rows: list[SpatialRow]) -> dict[str, str]:
    """
    Detect two-column label/value layouts: a significant horizontal gap
    within a row separates a short label (≤ 5 words) from its value.
    Only triggers when there is no colon in the row (to avoid double-counting).
    """
    pairs: dict[str, str] = {}
    for row in rows:
        if len(row.words) < 2 or ":" in row.text:
            continue
        for i in range(1, len(row.words)):
            gap = row.words[i].x0 - row.words[i - 1].x1
            if gap < 20:
                continue
            label_words = row.words[:i]
            value_words = row.words[i:]
            if len(label_words) > 5:
                continue
            label = _normalise_label(" ".join(w.text for w in label_words))
            value = " ".join(w.text for w in value_words).strip()
            if label and value and label in _LABEL_TO_FIELD:
                pairs[label] = value
            break
    return pairs


def _stacked_pairs(rows: list[SpatialRow]) -> dict[str, str]:
    """
    Detect stacked label/value pairs where a label-only row (ends with ':')
    is immediately followed by a value row at the same or similar X position.
    """
    pairs: dict[str, str] = {}
    for i, row in enumerate(rows[:-1]):
        m = _LABEL_ONLY.match(row.text.strip())
        if not m:
            continue
        label = _normalise_label(m.group(1))
        if label not in _LABEL_TO_FIELD:
            continue
        next_row = rows[i + 1]
        # Value row should be close in X and directly below
        if abs(next_row.x0 - row.x0) < row.x1 * 0.15:
            value = next_row.text.strip()
            if value and not _LABEL_ONLY.match(value):
                pairs[label] = value
    return pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _layout_from_tesseract(page: Any, page_number: int) -> SpatialLayout | None:
    """
    Build a SpatialLayout from Tesseract OCR word bounding boxes.
    Called when pdfplumber finds no extractable text (scanned PDF page).
    Word pixel coordinates are scaled back to PDF point space.
    """
    try:
        import pytesseract
        pil_image = page.to_image(resolution=150).original
        ocr = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    except Exception:
        return None

    img_w = pil_image.width or 1
    img_h = pil_image.height or 1
    sx = page.width / img_w
    sy = page.height / img_h

    words: list[SpatialWord] = []
    for i, text in enumerate(ocr["text"]):
        text = (text or "").strip()
        if not text:
            continue
        try:
            conf = int(ocr["conf"][i])
        except (ValueError, TypeError):
            conf = 0
        if conf < 30:
            continue
        x = float(ocr["left"][i]) * sx
        y = float(ocr["top"][i]) * sy
        w = float(ocr["width"][i]) * sx
        h = float(ocr["height"][i]) * sy
        words.append(SpatialWord(text=text, x0=x, top=y, x1=x + w, bottom=y + h))

    if not words:
        return None
    return SpatialLayout(
        page_width=float(page.width),
        page_height=float(page.height),
        rows=_group_into_rows(words),
        page_number=page_number,
    )


def extract_spatial_layout(pdf_path: Path) -> list[SpatialLayout]:
    """
    Return per-page SpatialLayout objects for *pdf_path*.

    For native PDFs, pdfplumber word extraction is used.  For scanned pages
    (no extractable text layer), the function falls back to Tesseract OCR
    bounding boxes so that the same spatial fingerprinting works on both.
    """
    try:
        import pdfplumber
    except ImportError:
        return []

    layouts: list[SpatialLayout] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_number, page in enumerate(pdf.pages):
                words_raw = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=True,
                )
                words = [
                    SpatialWord(
                        text=w["text"],
                        x0=float(w["x0"]),
                        top=float(w["top"]),
                        x1=float(w["x1"]),
                        bottom=float(w["bottom"]),
                    )
                    for w in (words_raw or [])
                    if w.get("text", "").strip()
                ]
                if words:
                    rows = _group_into_rows(words)
                    layouts.append(SpatialLayout(
                        page_width=float(page.width),
                        page_height=float(page.height),
                        rows=rows,
                        page_number=page_number,
                    ))
                else:
                    layout = _layout_from_tesseract(page, page_number)
                    if layout:
                        layouts.append(layout)
    except Exception:
        return []
    return layouts


# ---------------------------------------------------------------------------
# Zone-density fingerprint
# ---------------------------------------------------------------------------

_ZONE_COLS = 8
_ZONE_ROWS = 10


def build_zone_density(
    layouts: list[SpatialLayout],
    grid_cols: int = _ZONE_COLS,
    grid_rows: int = _ZONE_ROWS,
) -> list[float]:
    """
    Divide the first page into a grid and measure character density per zone.

    Returns a flat list of grid_cols * grid_rows floats in [0.0, 1.0].
    This fingerprint is content-independent: only *where* text appears on the
    page matters, not what the text says.  Documents with the same layout but
    different company names or values produce nearly identical vectors.
    Works for both native PDFs (pdfplumber) and scanned PDFs (Tesseract).
    """
    n = grid_cols * grid_rows
    if not layouts:
        return [0.0] * n

    layout = layouts[0]
    counts = [0] * n
    for row in layout.rows:
        for word in row.words:
            nx = layout.norm_x(word.x0)
            ny = layout.norm_y(word.top)
            col_idx = min(int(nx * grid_cols), grid_cols - 1)
            row_idx = min(int(ny * grid_rows), grid_rows - 1)
            counts[row_idx * grid_cols + col_idx] += len(word.text)

    peak = max(counts) if any(counts) else 1
    return [round(c / peak, 4) for c in counts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors. No external deps."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return round(dot / (mag_a * mag_b), 4)


def _split_multifield_value(text: str) -> dict[str, str]:
    """
    Parse a text segment that packs multiple 'Label: value' pairs on one line,
    e.g. 'Invoice #: INV-001 Date: Jan 1 2025 PO Number: PO-99'.

    Returns {normalised_label: value} for every recognised label found.
    Returns an empty dict when fewer than two known labels are detected so
    single-pair rows fall through to the existing _row_to_pair logic.
    """
    if not text or text.count(":") < 2:
        return {}

    # Find every position where a known label immediately precedes ':'
    # Sort labels longest-first so 'invoice number' wins over 'invoice'.
    segments: list[tuple[int, int, str]] = []  # (label_start, value_start, norm_label)
    for label in sorted(_LABEL_TO_FIELD, key=len, reverse=True):
        pat = rf"(?:^|(?<=\s)){re.escape(label)}\s*:(?=\s|$)"
        for m in re.finditer(pat, text, re.IGNORECASE):
            if not any(s[0] <= m.start() < s[1] for s in segments):
                segments.append((m.start(), m.end(), _normalise_label(label)))

    if len(segments) < 2:
        return {}

    segments.sort(key=lambda s: s[0])

    result: dict[str, str] = {}
    for i, (_, val_start, norm_label) in enumerate(segments):
        val_end = segments[i + 1][0] if i + 1 < len(segments) else len(text)
        value = text[val_start:val_end].strip()
        # Skip values that contain no alphanumeric characters (e.g. bare "|" separators
        # that appear in OCR output between label columns).
        if value and re.search(r"[a-zA-Z0-9]", value):
            result[norm_label] = value

    return result


def extract_fields_from_layout(layouts: list[SpatialLayout]) -> dict[str, Any]:
    """
    Extract key-value fields from spatial layout data across all pages.
    Returns a dict keyed by canonical field name.
    Unrecognised labels are silently ignored — callers only get clean fields.
    """
    raw_pairs: dict[str, str] = {}

    for layout in layouts:
        for row in layout.rows:
            multi = _split_multifield_value(row.text.strip())
            if multi:
                for lbl, val in multi.items():
                    raw_pairs.setdefault(lbl, val)
            else:
                pair = _row_to_pair(row)
                if pair:
                    raw_pairs.setdefault(pair[0], pair[1])

        for label, value in _column_gap_pairs(layout.rows).items():
            raw_pairs.setdefault(label, value)

        for label, value in _stacked_pairs(layout.rows).items():
            raw_pairs.setdefault(label, value)

    canonical: dict[str, Any] = {}
    for raw_label, value in raw_pairs.items():
        field_name = _LABEL_TO_FIELD.get(raw_label)
        if field_name:
            canonical[field_name] = value

    return canonical


_SKIP_ANCHOR_FIELDS = {"document_type", "source_file"}
_MIN_ANCHOR_VALUE_LEN = 3


def build_spatial_anchors(
    layouts: list[SpatialLayout],
    extracted: dict[str, Any],
) -> list[dict]:
    """
    For each field successfully extracted in *extracted*, record its
    normalised page position so future documents can look at the right
    region.  Only anchors from the first page are stored.

    Pass 1 — label-based: rows that match _LABEL_TO_FIELD get a named anchor.
    Pass 2 — value-scan: any remaining field whose value text appears verbatim
    in any row gets a position anchor (label_text may be empty).

    Returns a list of anchor dicts ready to embed in the template JSON.
    """
    if not layouts:
        return []

    layout = layouts[0]
    anchors: list[dict] = []
    seen_fields: set[str] = set()

    # Pass 1: label-based anchors
    for row in layout.rows:
        # Multi-field rows (e.g. "Invoice #: INV-001 Date: Jan 1 PO Number: PO-9")
        # need each sub-label handled individually; single-pair rows fall through.
        row_pairs = _split_multifield_value(row.text.strip())
        if not row_pairs:
            pair = _row_to_pair(row)
            if pair:
                row_pairs = {pair[0]: pair[1]}
        for raw_label, value in row_pairs.items():
            field_name = _LABEL_TO_FIELD.get(raw_label)
            if not field_name or field_name in seen_fields:
                continue
            if field_name not in extracted:
                continue
            ext_val = str(extracted[field_name]).strip()
            if not ext_val or (ext_val.lower() not in value.lower() and value.lower() not in ext_val.lower()):
                continue
            anchors.append({
                "field": field_name,
                "label_text": raw_label,
                "norm_x": layout.norm_x(row.x0),
                "norm_y": layout.norm_y(row.top),
            })
            seen_fields.add(field_name)

    # Pass 2: value-scan for fields that weren't anchored by a label
    for field, value in extracted.items():
        if field in seen_fields or field in _SKIP_ANCHOR_FIELDS:
            continue
        if value in (None, "", []):
            continue
        value_text = str(value).strip()
        if len(value_text) < _MIN_ANCHOR_VALUE_LEN:
            continue
        for row in layout.rows:
            if value_text.lower() not in row.text.lower():
                continue
            pair = _row_to_pair(row)
            row_label = pair[0] if pair else ""
            # If this row's label maps to a DIFFERENT field, don't store the label —
            # recording a foreign label would cause extract_by_spatial_anchors to
            # pull the wrong value on future documents.
            mapped_field = _LABEL_TO_FIELD.get(row_label, "")
            label_to_store = row_label if (not mapped_field or mapped_field == field) else ""
            # Skip position-only anchors — without a verifiable label they will
            # match random text at the same position in unrelated documents.
            if not label_to_store:
                break
            anchors.append({
                "field": field,
                "label_text": label_to_store,
                "norm_x": layout.norm_x(row.x0),
                "norm_y": layout.norm_y(row.top),
            })
            seen_fields.add(field)
            break

    return anchors


def extract_by_spatial_anchors(
    layouts: list[SpatialLayout],
    anchors: list[dict],
    x_tol: float = 0.10,
    y_tol: float = 0.04,
) -> dict[str, Any]:
    """
    Given stored spatial anchors, find the corresponding rows in *layouts*
    and extract their values by position proximity.

    Tolerances are in normalised coordinates (0–1 range).
    Falls back to full-layout extraction for anchors that cannot be placed.
    """
    if not layouts or not anchors:
        return {}

    layout = layouts[0]
    extracted: dict[str, Any] = {}

    for anchor in anchors:
        ax = anchor.get("norm_x", 0.0)
        ay = anchor.get("norm_y", 0.0)
        label_text = anchor.get("label_text", "")
        field_name = anchor.get("field", "")
        if not field_name:
            continue

        for row in layout.rows:
            nx = layout.norm_x(row.x0)
            ny = layout.norm_y(row.top)
            if abs(nx - ax) > x_tol or abs(ny - ay) > y_tol:
                continue
            # Build per-label value map; handles rows with multiple "Label: v" pairs.
            all_pairs = _split_multifield_value(row.text.strip())
            if not all_pairs:
                single = _row_to_pair(row)
                if single:
                    all_pairs = {single[0]: single[1]}
            # Exact label match
            if label_text and label_text in all_pairs:
                extracted[field_name] = all_pairs[label_text]
                break
            # Label text appears anywhere in the row (case-insensitive)
            if label_text and label_text.lower() in row.text.lower():
                val = all_pairs.get(label_text) or (next(iter(all_pairs.values())) if all_pairs else row.text.strip())
                extracted[field_name] = val
                break
            # Value-based anchor (no label): return first pair value or raw row text.
            if not label_text:
                extracted[field_name] = next(iter(all_pairs.values())) if all_pairs else row.text.strip()
                break

    return extracted
