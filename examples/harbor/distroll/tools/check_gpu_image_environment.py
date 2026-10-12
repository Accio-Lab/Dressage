#!/usr/bin/env python3
"""Validate GPU image dependencies without depending on project source."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("schema_version") != "dressage-distroll-gpu-runtime-lock/v1":
        raise RuntimeError(f"unsupported GPU runtime lock: {args.lock}")

    versions: dict[str, str] = {}
    for distribution, expected in lock["packages"].items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise RuntimeError(
                f"package version mismatch for {distribution}: "
                f"expected {expected}, got {actual}"
            )
        versions[distribution] = actual

    print(
        json.dumps(
            {
                "schema_version": "dressage-distroll-gpu-image-environment/v1",
                "packages": versions,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
