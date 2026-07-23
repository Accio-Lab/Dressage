"""Metadata routing for CPU-backed, rotating Megatron MOPD teachers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

_BLACKBOX_GENERATE_PATH = "dressage.rollout.generate.blackbox_dispatch.generate"
_ROLLOUT_MODES = frozenset({"blackbox", "whitebox"})


def _optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _rollout_config(raw: dict[str, Any], *, owner: str) -> tuple[str, str | None]:
    mode = _optional_string(raw.get("agent_mode"))
    if mode is None:
        raise ValueError(f"MOPD {owner} requires agent_mode")
    mode = mode.lower()
    if mode not in _ROLLOUT_MODES:
        expected = "|".join(sorted(_ROLLOUT_MODES))
        raise ValueError(f"MOPD {owner} agent_mode must be {expected}, got {mode!r}")
    function_path = _optional_string(raw.get("generate_function_path"))
    if function_path is None and mode == "blackbox":
        function_path = _BLACKBOX_GENERATE_PATH
    if function_path is None and mode == "whitebox":
        raise ValueError(
            f"MOPD {owner} uses whitebox rollout and requires generate_function_path"
        )
    return mode, function_path


@dataclass(frozen=True)
class MOPDTeacher:
    teacher_id: str
    load: str
    ckpt_step: int | None


@dataclass(frozen=True)
class MOPDDataset:
    name: str
    path: str
    teacher_id: str
    weight: float
    metadata: dict[str, Any]
    agent_mode: str
    generate_function_path: str | None


@dataclass(frozen=True)
class MOPDConfig:
    teachers: dict[str, MOPDTeacher]
    datasets: tuple[MOPDDataset, ...] = ()
    reward_modules: tuple[str, ...] = ()
    runtime_env_keys: tuple[str, ...] = ()
    base_model: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MOPDConfig":
        if not isinstance(raw, dict):
            raise ValueError("MOPD config must be a JSON object")
        raw_teachers = raw.get("teachers")
        if not isinstance(raw_teachers, dict) or not raw_teachers:
            raise ValueError("MOPD config requires a non-empty 'teachers' object")

        teachers: dict[str, MOPDTeacher] = {}
        for raw_id, value in raw_teachers.items():
            teacher_id = str(raw_id).strip()
            if not teacher_id or not isinstance(value, dict):
                raise ValueError(f"invalid MOPD teacher entry: {raw_id!r}")
            load = str(value.get("load") or "").strip()
            if not load:
                raise ValueError(
                    f"MOPD teacher {teacher_id!r} requires a Megatron checkpoint 'load'"
                )
            raw_step = value.get("ckpt_step")
            ckpt_step = None if raw_step in (None, "") else int(raw_step)
            if ckpt_step is not None and ckpt_step < 0:
                raise ValueError(
                    f"MOPD teacher {teacher_id!r} ckpt_step must be non-negative"
                )
            teachers[teacher_id] = MOPDTeacher(teacher_id, load, ckpt_step)

        datasets = cls._parse_datasets(raw.get("datasets") or [], teachers=teachers)
        return cls(
            teachers=teachers,
            datasets=datasets,
            reward_modules=cls._string_tuple(
                raw.get("reward_modules") or [], "reward_modules"
            ),
            runtime_env_keys=cls._string_tuple(
                raw.get("runtime_env_keys") or [], "runtime_env_keys"
            ),
            base_model=_optional_string(raw.get("base_model")),
        )

    @staticmethod
    def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise ValueError(f"MOPD {field} must be a list")
        return tuple(str(item).strip() for item in value if str(item).strip())

    @staticmethod
    def _parse_datasets(
        raw_datasets: Any,
        *,
        teachers: dict[str, MOPDTeacher],
    ) -> tuple[MOPDDataset, ...]:
        if not isinstance(raw_datasets, list):
            raise ValueError("MOPD datasets must be a list")
        datasets: list[MOPDDataset] = []
        names: set[str] = set()
        for position, value in enumerate(raw_datasets):
            if not isinstance(value, dict):
                raise ValueError(
                    f"MOPD dataset at position {position} must be an object"
                )
            path = str(value.get("path") or "").strip()
            if not path:
                raise ValueError(f"MOPD dataset at position {position} requires path")
            name = str(value.get("name") or Path(path).stem).strip()
            if not name or name in names:
                raise ValueError(
                    f"MOPD dataset name must be unique and non-empty, got {name!r}"
                )
            names.add(name)

            teacher_id = _optional_string(value.get("teacher_id"))
            if teacher_id is None:
                raise ValueError(f"MOPD dataset {name!r} requires teacher_id")
            if teacher_id not in teachers:
                raise ValueError(
                    f"MOPD dataset {name!r} routes to unknown teacher {teacher_id!r}"
                )
            weight = float(value.get("weight", 1.0))
            if weight <= 0:
                raise ValueError(f"MOPD dataset {name!r} weight must be positive")
            metadata = value.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise ValueError(f"MOPD dataset {name!r} metadata must be an object")
            routed_metadata = dict(metadata)
            routed_metadata["teacher_id"] = teacher_id
            agent_mode, generate_function_path = _rollout_config(
                value, owner=f"dataset {name!r}"
            )
            datasets.append(
                MOPDDataset(
                    name=name,
                    path=path,
                    teacher_id=teacher_id,
                    weight=weight,
                    metadata=routed_metadata,
                    agent_mode=agent_mode,
                    generate_function_path=generate_function_path,
                )
            )
        return tuple(datasets)


@lru_cache(maxsize=16)
def load_mopd_config(path: str | os.PathLike[str]) -> MOPDConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"MOPD config does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid MOPD config JSON at {config_path}: {exc}") from exc
    config = MOPDConfig.from_dict(raw)
    teachers = {
        teacher_id: replace(
            teacher,
            load=str(
                (config_path.parent / teacher.load).resolve()
                if not Path(teacher.load).expanduser().is_absolute()
                else Path(teacher.load).expanduser().resolve()
            ),
        )
        for teacher_id, teacher in config.teachers.items()
    }
    datasets = tuple(
        replace(
            dataset,
            path=str(
                (config_path.parent / dataset.path).resolve()
                if not Path(dataset.path).expanduser().is_absolute()
                else Path(dataset.path).expanduser().resolve()
            ),
        )
        for dataset in config.datasets
    )
    base_model = config.base_model
    if base_model is not None:
        base_path = Path(base_model).expanduser()
        base_model = str(
            base_path.resolve()
            if base_path.is_absolute()
            else (config_path.parent / base_path).resolve()
        )
    return replace(config, teachers=teachers, datasets=datasets, base_model=base_model)


def route_mopd_teacher(metadata: Any, config: MOPDConfig) -> MOPDTeacher:
    if not isinstance(metadata, dict):
        raise ValueError("MOPD sample metadata requires teacher_id")
    teacher_id = _optional_string(metadata.get("teacher_id"))
    if teacher_id is None:
        raise ValueError("MOPD sample metadata requires teacher_id")
    if teacher_id not in config.teachers:
        raise ValueError(f"unknown MOPD teacher_id {teacher_id!r}")
    return config.teachers[teacher_id]


def collect_mopd_teacher_ids(samples: list[Any], config: MOPDConfig) -> list[str]:
    """Validate routes and return one teacher id per flattened train sample."""
    parent_routes: dict[str, str] = {}
    teacher_ids: list[str] = []
    for position, sample in enumerate(samples):
        teacher = route_mopd_teacher(getattr(sample, "metadata", None), config)
        parent_id = str(
            sample.metadata.get("parent_traj_id")
            or getattr(sample, "session_id", None)
            or f"sample:{getattr(sample, 'index', position)}"
        )
        previous = parent_routes.setdefault(parent_id, teacher.teacher_id)
        if previous != teacher.teacher_id:
            raise ValueError(
                f"MOPD sibling routing conflict for parent_traj_id={parent_id!r}: "
                f"{previous!r} != {teacher.teacher_id!r}"
            )
        teacher_ids.append(teacher.teacher_id)
    return teacher_ids
