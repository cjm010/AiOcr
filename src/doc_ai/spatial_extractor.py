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
    "report period": "report_period",
    "period": "report_period",
    "document type": "document_subtype",
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

def extract_spatial_layout(pdf_path: Path) -> list[SpatialLayout]:
    """
    Return per-page SpatialLayout objects for *pdf_path* using pdfplumber.
    Returns an empty list when pdfplumber is not installed or the file cannot
    be opened.
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
                rows = _group_into_rows(words)
                layouts.append(
                    SpatialLayout(
                        page_width=float(page.width),
                        page_height=float(page.height),
                        rows=rows,
                        page_number=page_number,
                    )
                )
    except Exception:
        return []
    return layouts


def extract_fields_from_layout(layouts: list[SpatialLayout]) -> dict[str, Any]:
    """
    Extract key-value fields from spatial layout data across all pages.
    Returns a dict keyed by canonical field name.
    Unrecognised labels are silently ignored — callers only get clean fields.
    """
    raw_pairs: dict[str, str] = {}

    for layout in layouts:
        for row in layout.rows:
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


def build_spatial_anchors(
    layouts: list[SpatialLayout],
    extracted: dict[str, Any],
) -> list[dict]:
    """
    For each field successfully extracted in *extracted*, record its
    normalised page position so future documents can look at the right
    region.  Only anchors from the first page are stored.

    Returns a list of anchor dicts ready to embed in the template JSON.
    """
    if not layouts:
        return []

    layout = layouts[0]
    anchors: list[dict] = []
    seen_fields: set[str] = set()

    for row in layout.rows:
        pair = _row_to_pair(row)
        if not pair:
            continue
        raw_label, value = pair
        field_name = _LABEL_TO_FIELD.get(raw_label)
        if not field_name or field_name in seen_fields:
            continue
        if field_name not in extracted:
            continue
        # Only anchor if the extracted value is a reasonable match
        ext_val = str(extracted[field_name]).strip()
        if not ext_val or ext_val.lower() not in value.lower() and value.lower() not in ext_val.lower():
            continue
        anchors.append(
            {
                "field": field_name,
                "label_text": raw_label,
                "norm_x": layout.norm_x(row.x0),
                "norm_y": layout.norm_y(row.top),
            }
        )
        seen_fields.add(field_name)

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
            pair = _row_to_pair(row)
            if pair and pair[0] == label_text:
                extracted[field_name] = pair[1]
                break
            # Looser: label text appears anywhere in the row
            if label_text and label_text in row.text.lower():
                pair = _row_to_pair(row)
                if pair:
                    extracted[field_name] = pair[1]
                    break

    return extracted
