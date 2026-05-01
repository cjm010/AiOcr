from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TEMPLATE_WRITE_LOCK = threading.Lock()
_BAD_PATTERN_WRITE_LOCK = threading.Lock()
_MAX_TEMPLATES_PER_TYPE = 20
_SKIP_BAD_PATTERN_FIELDS = {"document_type", "source_file"}


def _derive_bad_pattern(value: str) -> str:
    """Derive a generalizable regex from a rejected field value.

    Values consisting of a single repeated character (e.g. "___", "---") are
    generalised to match any length of that character.  Everything else becomes
    a case-insensitive exact match so we don't accidentally reject valid values.
    """
    stripped = value.strip()
    if stripped and len(set(stripped)) == 1:
        char = re.escape(stripped[0])
        return rf"^\s*{char}+\s*$"
    return rf"^\s*{re.escape(stripped)}\s*$"


@dataclass
class TemplateMatch:
    template: dict[str, Any]
    score: float


class TemplateMemory:
    def __init__(self, store_path: Path) -> None:
        self._store_path = store_path
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._store_path.exists():
            self._store_path.write_text("[]", encoding="utf-8")

    def load_templates(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self._store_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def find_best_match(
        self,
        signature: dict[str, Any],
        document_type: str | None = None,
    ) -> TemplateMatch | None:
        best_match: TemplateMatch | None = None
        for template in self.load_templates():
            if document_type and template.get("document_type") != document_type:
                continue
            score = self._score_signature(signature, template.get("signature", {}))
            if best_match is None or score > best_match.score:
                best_match = TemplateMatch(template=template, score=score)
        return best_match

    def learn_template(
        self,
        source_file: str,
        signature: dict[str, Any],
        extracted_data: dict[str, Any],
        lines: list[str],
        spatial_anchors: list[dict] | None = None,
    ) -> dict[str, Any]:
        with _TEMPLATE_WRITE_LOCK:
            templates = self.load_templates()
            anchors = self._build_anchors(extracted_data, lines)

            template: dict[str, Any] = {
                "template_name": Path(source_file).stem,
                "document_type": extracted_data.get("document_type", "invoice"),
                "signature": signature,
                "anchors": anchors,
            }
            if spatial_anchors:
                template["spatial_anchors"] = spatial_anchors

            doc_type = extracted_data.get("document_type")
            best_match = self.find_best_match(signature, document_type=doc_type)
            if best_match and best_match.score >= 0.85:
                template["template_name"] = best_match.template.get("template_name", template["template_name"])
                existing = best_match.template.get("spatial_anchors", [])
                if spatial_anchors:
                    new_fields = {a["field"] for a in spatial_anchors}
                    merged = [a for a in existing if a.get("field") not in new_fields] + spatial_anchors
                    template["spatial_anchors"] = merged
                elif existing:
                    template["spatial_anchors"] = existing
                templates = [
                    template if item.get("template_name") == template["template_name"] else item
                    for item in templates
                ]
            else:
                # Enforce per-type cap before adding a new template.
                type_templates = [t for t in templates if t.get("document_type") == doc_type]
                if len(type_templates) >= _MAX_TEMPLATES_PER_TYPE:
                    # Replace the existing template most similar to the new one.
                    most_similar = max(
                        type_templates,
                        key=lambda t: self._score_signature(signature, t.get("signature", {})),
                    )
                    templates = [
                        template if t.get("template_name") == most_similar.get("template_name") else t
                        for t in templates
                    ]
                else:
                    templates.append(template)

            self._store_path.write_text(json.dumps(templates, indent=2), encoding="utf-8")
            return template

    def _build_anchors(self, extracted_data: dict[str, Any], lines: list[str]) -> dict[str, dict[str, str]]:
        anchors: dict[str, dict[str, str]] = {}
        used_patterns: set[str] = set()
        for field, value in extracted_data.items():
            if field in {"document_type", "source_file"} or value in (None, "", []):
                continue
            value_text = str(value).strip()
            for line in lines:
                line_text = line.strip()
                if not line_text or value_text.lower() not in line_text.lower():
                    continue
                label = self._extract_label(line_text)
                if not label:
                    continue
                pattern = rf"{re.escape(label)}\s*:\s*(?P<value>.+)"
                if pattern in used_patterns:
                    continue
                # Verify the pattern captures exactly the expected value on this line.
                # If it captures much more (e.g. the value is buried in a longer string),
                # the anchor would extract wrong data on future documents.
                m = re.search(pattern, line_text, re.IGNORECASE)
                if not m:
                    continue
                captured = m.group("value").strip()
                if captured.lower() != value_text.lower():
                    continue
                used_patterns.add(pattern)
                anchors[field] = {"label": label, "pattern": pattern}
                break
        return anchors

    @staticmethod
    def build_signature(lines: list[str], layouts: list | None = None) -> dict[str, Any]:
        zone_density: list[float] = []
        if layouts:
            try:
                from .spatial_extractor import build_zone_density
                zone_density = build_zone_density(layouts)
            except Exception:
                pass
        return {"zone_density": zone_density}

    @staticmethod
    def _extract_label(line: str) -> str:
        if ":" in line:
            return line.split(":", 1)[0].strip()
        return ""

    @staticmethod
    def _score_signature(left: dict[str, Any], right: dict[str, Any]) -> float:
        ld = left.get("zone_density", [])
        rd = right.get("zone_density", [])
        if ld and rd:
            from .spatial_extractor import cosine_similarity
            return cosine_similarity(ld, rd)
        return 0.0


class BadPatternStore:
    """Persists per-field bad-value patterns learned from human corrections.

    Patterns are scoped to ``{doc_type}.{field}`` so they don't bleed across
    document types.  A value cleared by a reviewer during ``finalize_review``
    is generalised into a regex and stored here; future extractions that
    produce a matching value have that field silently nulled before validation.
    """

    def __init__(self, store_path: Path) -> None:
        self._path = store_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("{}", encoding="utf-8")

    def load(self) -> dict[str, list[str]]:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def add_pattern(self, doc_type: str, field: str, rejected_value: str) -> str | None:
        """Derive and store a bad-value pattern. Returns the pattern, or None if skipped."""
        if not rejected_value or not str(rejected_value).strip():
            return None
        pattern = _derive_bad_pattern(str(rejected_value))
        key = f"{doc_type}.{field}"
        with _BAD_PATTERN_WRITE_LOCK:
            patterns = self.load()
            existing = patterns.get(key, [])
            if pattern not in existing:
                existing.append(pattern)
                patterns[key] = existing
                self._path.write_text(json.dumps(patterns, indent=2), encoding="utf-8")
        return pattern

    def apply(self, extracted: dict[str, Any]) -> list[str]:
        """Null out extracted values that match stored bad patterns.

        Returns the list of field names that were cleared.
        """
        doc_type = extracted.get("document_type", "invoice")
        patterns = self.load()
        cleared: list[str] = []
        for field, value in list(extracted.items()):
            if field in _SKIP_BAD_PATTERN_FIELDS or value in (None, "", []):
                continue
            key = f"{doc_type}.{field}"
            field_patterns = patterns.get(key, [])
            if not field_patterns:
                continue
            value_str = str(value).strip()
            for pat in field_patterns:
                try:
                    if re.search(pat, value_str, re.IGNORECASE):
                        extracted[field] = None
                        cleared.append(field)
                        break
                except re.error:
                    pass
        return cleared
