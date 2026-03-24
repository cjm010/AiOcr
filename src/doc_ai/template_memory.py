from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


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

    def find_best_match(self, signature: dict[str, Any]) -> TemplateMatch | None:
        best_match: TemplateMatch | None = None
        for template in self.load_templates():
            template_signature = template.get("signature", {})
            score = self._score_signature(signature, template_signature)
            if best_match is None or score > best_match.score:
                best_match = TemplateMatch(template=template, score=score)
        return best_match

    def learn_template(
        self,
        source_file: str,
        signature: dict[str, Any],
        extracted_data: dict[str, Any],
        lines: list[str],
    ) -> dict[str, Any]:
        templates = self.load_templates()
        anchors = self._build_anchors(extracted_data, lines)

        template = {
            "template_name": Path(source_file).stem,
            "document_type": extracted_data.get("document_type", "invoice"),
            "signature": signature,
            "anchors": anchors,
            "example_fields": {
                key: value
                for key, value in extracted_data.items()
                if key not in {"source_file"} and value not in (None, "", [])
            },
        }

        best_match = self.find_best_match(signature)
        if best_match and best_match.score >= 0.92:
            template["template_name"] = best_match.template.get("template_name", template["template_name"])
            templates = [
                template if item.get("template_name") == template["template_name"] else item
                for item in templates
            ]
        else:
            templates.append(template)

        self._store_path.write_text(json.dumps(templates, indent=2), encoding="utf-8")
        return template

    def _build_anchors(self, extracted_data: dict[str, Any], lines: list[str]) -> dict[str, dict[str, str]]:
        anchors: dict[str, dict[str, str]] = {}
        for field, value in extracted_data.items():
            if field in {"document_type", "source_file"} or value in (None, "", []):
                continue

            value_text = str(value).strip()
            for line in lines:
                line_text = line.strip()
                if not line_text or value_text.lower() not in line_text.lower():
                    continue
                label = self._extract_label(line_text)
                if label:
                    anchors[field] = {
                        "label": label,
                        "pattern": rf"{re.escape(label)}\s*:\s*(?P<value>.+)",
                    }
                else:
                    anchors[field] = {"label": "", "pattern": re.escape(value_text)}
                break
        return anchors

    @staticmethod
    def build_signature(lines: list[str]) -> dict[str, Any]:
        top_lines = [TemplateMemory._normalize(line) for line in lines[:8]]
        joined = " ".join(lines[:20]).lower()
        keywords = sorted(
            {
                keyword
                for keyword in ("invoice", "vendor", "supplier", "total", "amount due", "tax", "due date")
                if keyword in joined
            }
        )
        return {"top_lines": top_lines, "keywords": keywords}

    @staticmethod
    def _extract_label(line: str) -> str:
        if ":" in line:
            return line.split(":", 1)[0].strip()
        return ""

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"\d", "#", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _score_signature(left: dict[str, Any], right: dict[str, Any]) -> float:
        left_text = " | ".join(left.get("top_lines", []))
        right_text = " | ".join(right.get("top_lines", []))
        text_score = SequenceMatcher(None, left_text, right_text).ratio()
        left_keywords = set(left.get("keywords", []))
        right_keywords = set(right.get("keywords", []))
        if not left_keywords and not right_keywords:
            keyword_score = 1.0
        else:
            keyword_score = len(left_keywords & right_keywords) / max(len(left_keywords | right_keywords), 1)
        return round((text_score * 0.7) + (keyword_score * 0.3), 4)
