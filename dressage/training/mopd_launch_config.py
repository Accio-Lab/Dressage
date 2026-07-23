"""Resolve and validate the MOPD JSON for the shell launcher."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dressage.rollout.mopd import MOPDConfig, load_mopd_config


def resolve_launch_values(
    config_path: str, *, validate_paths: bool = True
) -> tuple[str, str, str, str, str, str, str]:
    config: MOPDConfig = load_mopd_config(config_path)
    if validate_paths:
        for teacher in config.teachers.values():
            checkpoint = Path(teacher.load)
            if not checkpoint.is_dir():
                raise ValueError(
                    f"MOPD teacher {teacher.teacher_id!r} checkpoint is not a directory: "
                    f"{checkpoint}"
                )
        for dataset in config.datasets:
            if not Path(dataset.path).is_file():
                raise ValueError(
                    f"MOPD dataset {dataset.name!r} is not a file: {dataset.path}"
                )
        if config.base_model is not None and not Path(config.base_model).is_dir():
            raise ValueError(f"MOPD base_model is not a directory: {config.base_model}")

    primary = next(iter(config.teachers.values()))
    first_dataset = config.datasets[0].path if config.datasets else ""
    modes = sorted({dataset.agent_mode for dataset in config.datasets})
    return (
        first_dataset,
        ",".join(modes),
        ",".join(config.runtime_env_keys),
        ",".join(config.reward_modules),
        config.base_model or "",
        primary.load,
        "" if primary.ckpt_step is None else str(primary.ckpt_step),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--skip-path-validation",
        action="store_true",
        default=os.environ.get("MOPD_SKIP_PATH_VALIDATION", "0") == "1",
    )
    args = parser.parse_args()
    for value in resolve_launch_values(
        args.config, validate_paths=not args.skip_path_validation
    ):
        print(value)


if __name__ == "__main__":
    main()
