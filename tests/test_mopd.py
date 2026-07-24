from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from dressage.rollout.data_source import DressageDataSource
from dressage.rollout.mopd import (
    MOPDConfig,
    collect_mopd_teacher_ids,
    route_mopd_teacher,
)


@dataclass
class SampleLike:
    index: int = 0
    session_id: str | None = None
    metadata: dict = field(default_factory=dict)


def _raw_config(**extra):
    return {
        "teachers": {
            "a": {"load": "/checkpoint/a", "ckpt_step": 3},
            "b": {"load": "/checkpoint/b"},
        },
        **extra,
    }


def _write_config(tmp_path, raw=None):
    path = tmp_path / "mopd.json"
    path.write_text(json.dumps(raw or _raw_config()), encoding="utf-8")
    return path


def test_config_uses_direct_dataset_teacher_ids():
    config = MOPDConfig.from_dict(
        _raw_config(
            datasets=[
                {
                    "name": "alpha",
                    "path": "/data/alpha.jsonl",
                    "teacher_id": "a",
                    "weight": 2,
                    "agent_mode": "blackbox",
                }
            ]
        )
    )
    dataset = config.datasets[0]
    assert dataset.teacher_id == "a"
    assert dataset.metadata["teacher_id"] == "a"
    assert dataset.agent_mode == "blackbox"
    assert dataset.generate_function_path == (
        "dressage.rollout.generate.blackbox_dispatch.generate"
    )
    assert config.teachers["a"].ckpt_step == 3


def test_config_rejects_missing_or_unknown_teacher_id():
    with pytest.raises(ValueError, match="requires teacher_id"):
        MOPDConfig.from_dict(_raw_config(datasets=[{"name": "x", "path": "/x"}]))
    with pytest.raises(ValueError, match="unknown teacher"):
        MOPDConfig.from_dict(
            _raw_config(datasets=[{"name": "x", "path": "/x", "teacher_id": "missing"}])
        )
    with pytest.raises(ValueError, match="requires agent_mode"):
        MOPDConfig.from_dict(
            _raw_config(datasets=[{"name": "x", "path": "/x", "teacher_id": "a"}])
        )


def test_data_source_stamps_rollout_fields(tmp_path):
    alpha = tmp_path / "alpha.jsonl"
    beta = tmp_path / "beta.jsonl"
    alpha.write_text('{"prompt":"a","label":"1"}\n', encoding="utf-8")
    beta.write_text('{"prompt":"b","label":"2"}\n', encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        _raw_config(
            datasets=[
                {
                    "name": "alpha",
                    "path": str(alpha),
                    "teacher_id": "a",
                    "agent_mode": "blackbox",
                },
                {
                    "name": "beta",
                    "path": str(beta),
                    "teacher_id": "b",
                    "agent_mode": "whitebox",
                    "generate_function_path": "pkg.beta.generate",
                },
            ]
        ),
    )
    args = SimpleNamespace(
        mopd_teacher_config=str(config_path),
        prompt_data=str(alpha),
        input_key="prompt",
        label_key="label",
        metadata_key="metadata",
        multimodal_keys=None,
        apply_chat_template=False,
        rollout_shuffle=False,
        rollout_seed=42,
        n_samples_per_prompt=1,
    )
    first, second = [group[0] for group in DressageDataSource(args).get_samples(2)]
    assert [first.metadata["teacher_id"], second.metadata["teacher_id"]] == ["a", "b"]
    assert first.generate_function_path == (
        "dressage.rollout.generate.blackbox_dispatch.generate"
    )
    assert second.generate_function_path == "pkg.beta.generate"


def test_collect_routes_validates_siblings():
    config = MOPDConfig.from_dict(_raw_config())
    samples = [
        SampleLike(index=0, metadata={"parent_traj_id": "p", "teacher_id": "a"}),
        SampleLike(index=1, metadata={"parent_traj_id": "p", "teacher_id": "a"}),
        SampleLike(index=2, metadata={"parent_traj_id": "q", "teacher_id": "b"}),
    ]
    assert collect_mopd_teacher_ids(samples, config) == ["a", "a", "b"]
    assert route_mopd_teacher(samples[2].metadata, config).teacher_id == "b"

    samples[1].metadata["teacher_id"] = "b"
    with pytest.raises(ValueError, match="sibling routing conflict"):
        collect_mopd_teacher_ids(samples, config)
