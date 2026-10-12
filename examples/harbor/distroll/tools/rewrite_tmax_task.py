#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from harbor.models.task.config import TaskConfig

try:
    from tools.harbor_task_toml import remove_toml_value, set_toml_value
except ModuleNotFoundError:  # direct execution adds tools/, not project root
    from harbor_task_toml import remove_toml_value, set_toml_value


def apply_public_network_policy(text: str) -> str:
    rewritten = remove_toml_value(text, "environment", "allow_internet")
    rewritten = remove_toml_value(rewritten, "environment", "allowed_hosts")
    for section in ("agent", "verifier"):
        rewritten = remove_toml_value(rewritten, section, "network_mode")
        rewritten = remove_toml_value(rewritten, section, "allowed_hosts")
    return set_toml_value(
        rewritten, "environment", "network_mode", json.dumps("public")
    )


def rewrite(
    source: Path,
    output: Path,
    *,
    admitted_image: str,
    source_image: str,
    source_image_digest: str,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to replace admitted TMAX task: {output}")
    shutil.copytree(source, output)
    config_path = output / "task.toml"
    original = config_path.read_text(encoding="utf-8")
    (output / "task.source.toml").write_text(original, encoding="utf-8")
    rewritten = apply_public_network_policy(original)
    for section, key, value in (
        ("environment", "docker_image", json.dumps(admitted_image)),
        ("metadata", "dressage_tmax_source_image", json.dumps(source_image)),
        (
            "metadata",
            "dressage_tmax_source_image_digest",
            json.dumps(source_image_digest),
        ),
        ("metadata", "dressage_admitted_image", json.dumps(admitted_image)),
    ):
        rewritten = set_toml_value(rewritten, section, key, value)
    config_path.write_text(rewritten, encoding="utf-8")
    validated = TaskConfig.model_validate_toml(rewritten)
    if validated.environment.docker_image != admitted_image:
        raise ValueError("rewritten TMAX task did not retain admitted image alias")
    if validated.environment.network_mode.value != "public":
        raise ValueError("rewritten TMAX environment network mode is not public")
    if validated.agent.network_mode is not None:
        raise ValueError("rewritten TMAX agent must inherit the public baseline")
    if validated.verifier.network_mode is not None:
        raise ValueError("rewritten TMAX verifier must inherit the public baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize one admitted TMAX task")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admitted-image", required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--source-image-digest", required=True)
    args = parser.parse_args()
    rewrite(
        args.source.resolve(),
        args.output.resolve(),
        admitted_image=args.admitted_image,
        source_image=args.source_image,
        source_image_digest=args.source_image_digest,
    )


if __name__ == "__main__":
    main()
