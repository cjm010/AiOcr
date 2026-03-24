from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def merge_templates(source_templates: list[dict[str, Any]], target_templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {
        item.get("template_name", f"template-{index}"): item for index, item in enumerate(target_templates)
    }
    for index, template in enumerate(source_templates):
        name = template.get("template_name", f"template-{index}")
        merged[name] = template
    return list(merged.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote reviewed learning artifacts from one environment store into another."
    )
    parser.add_argument("--source", required=True, help="Source learned_templates.json path")
    parser.add_argument("--target", required=True, help="Target promoted_templates.json or learned_templates.json path")
    args = parser.parse_args()

    source_path = Path(args.source)
    target_path = Path(args.target)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    source_templates = load_json(source_path, [])
    target_templates = load_json(target_path, [])
    merged_templates = merge_templates(source_templates, target_templates)

    target_path.write_text(json.dumps(merged_templates, indent=2), encoding="utf-8")
    print(f"Promoted {len(source_templates)} templates into {target_path}")


if __name__ == "__main__":
    main()
