#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path


def resolve(task_dir: Path) -> str | None:
    for name in ("task-source.json", "tmax-source.json"):
        path = task_dir / name
        if path.is_file():
            value = str(
                json.loads(path.read_text(encoding="utf-8")).get("source_image") or ""
            )
            if value:
                return value
    config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    value = str((config.get("environment") or {}).get("docker_image") or "")
    return value or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve a prebuilt source image declared by one Harbor task"
    )
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()
    selected = resolve(args.task_dir.expanduser().resolve())
    if selected:
        print(selected)


if __name__ == "__main__":
    main()
