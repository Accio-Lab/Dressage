#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_task_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def prepare(
    *,
    task_ids_file: Path,
    source_root: Path,
    namespace: str,
    benchmark: str,
    agent: str,
    selection_manifest: Path,
    prompt_manifest: Path,
    validation_profile: str = "standard",
) -> None:
    if validation_profile not in {"standard", "trusted-security"}:
        raise ValueError(f"unsupported fixed workload profile: {validation_profile}")
    task_ids = read_task_ids(task_ids_file)
    tasks = [
        {"task_id": task_id, "task_ref": f"{namespace}/{task_id}"}
        for task_id in task_ids
    ]
    selection = {
        "schema_version": "fixed-harbor-workload/v0",
        "validation_profile": validation_profile,
        "namespace": namespace,
        "task_count": len(tasks),
        "tasks": tasks,
    }
    _atomic_json(selection_manifest, selection)

    prompt_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = prompt_manifest.with_name(f".{prompt_manifest.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for task in tasks:
                task_id = task["task_id"]
                prompt = (source_root / task_id / "instruction.md").read_text(
                    encoding="utf-8"
                )
                stream.write(
                    json.dumps(
                        {
                            "prompt": prompt.strip(),
                            "label": None,
                            "metadata": {
                                "harbor_task_ref": task["task_ref"],
                                "benchmark": benchmark,
                                "task_id": task_id,
                                "agent": agent,
                                "validation_profile": validation_profile,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        os.replace(temporary, prompt_manifest)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create prompt/selection manifests from an explicit Harbor task list"
    )
    parser.add_argument("--task-ids-file", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--agent", default="claude-code")
    parser.add_argument(
        "--validation-profile",
        choices=("standard", "trusted-security"),
        default="standard",
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    args = parser.parse_args()
    prepare(
        task_ids_file=args.task_ids_file.expanduser().resolve(),
        source_root=args.source_root.expanduser().resolve(),
        namespace=args.namespace,
        benchmark=args.benchmark,
        agent=args.agent,
        validation_profile=args.validation_profile,
        selection_manifest=args.selection_manifest.expanduser().resolve(),
        prompt_manifest=args.prompt_manifest.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
