from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import dressage.training.mopd_megatron_actor as mopd_actor_module
from dressage.rollout.log_rollout import log_rollout_data
from dressage.training.mopd_megatron_actor import (
    MOPDMegatronTrainRayActor,
    _train_aggregation_mean_contribution,
)


def test_train_aggregation_mean_matches_fractional_rollout_ownership(monkeypatch):
    monkeypatch.setattr(
        mopd_actor_module.mpu,
        "get_context_parallel_world_size",
        lambda: 1,
    )
    rollout_data = {
        "loss_masks": [
            torch.tensor([1, 1]),
            torch.tensor([1, 1, 1]),
            torch.tensor([1]),
        ],
        "total_lengths": [2, 3, 1],
        "response_lengths": [2, 3, 1],
        "rollout_mask_sums": torch.tensor([5.0, 5.0, 1.0]),
    }
    values = [
        torch.tensor([1.0, 1.0]),
        torch.tensor([3.0, 3.0, 3.0]),
        torch.tensor([5.0]),
    ]

    numerator, count = _train_aggregation_mean_contribution(
        values,
        [0, 1, 2],
        rollout_data,
    )

    assert numerator == pytest.approx(7.2)
    assert count == pytest.approx(2.0)
    assert numerator / count == pytest.approx(3.6)


def test_mopd_postprocess_logs_only_per_teacher_token_k1(monkeypatch):
    monkeypatch.setattr(
        mopd_actor_module.mpu,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        mopd_actor_module.mpu,
        "is_pipeline_last_stage",
        lambda: True,
    )
    monkeypatch.setattr(
        mopd_actor_module.mpu,
        "get_context_parallel_world_size",
        lambda: 1,
    )
    captured = {}

    def fake_gather(metric_name, args, rollout_id, log_dict):
        assert metric_name == "rollout"
        captured.update(log_dict)

    monkeypatch.setattr(mopd_actor_module, "gather_log_data", fake_gather)
    actor = object.__new__(MOPDMegatronTrainRayActor)
    actor._active_mopd_teacher_ids = ["a", "a", "b"]
    actor._base_rollout_data_postprocess = None
    actor.mopd_teacher_metric_components = {"a": "a", "b": "b"}
    rollout_data = {
        "loss_masks": [
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([1]),
        ],
        "total_lengths": [1, 1, 1],
        "response_lengths": [1, 1, 1],
        "rollout_mask_sums": torch.tensor([1.0, 0.0, 1.0]),
        "opd_reverse_kl": [
            torch.tensor([0.5]),
            torch.tensor([0.5]),
            torch.tensor([0.5]),
        ],
    }

    actor._postprocess_mopd_metrics(SimpleNamespace(), 0, rollout_data)

    assert set(captured) == {
        "mopd/opd_reverse_kl_train_aggregation_mean/a",
        "mopd/opd_reverse_kl_train_aggregation_mean/b",
    }
    assert captured["mopd/opd_reverse_kl_train_aggregation_mean/a"] == (
        pytest.approx(0.5),
        pytest.approx(1.0),
    )
    assert captured["mopd/opd_reverse_kl_train_aggregation_mean/b"] == (
        pytest.approx(0.5),
        pytest.approx(1.0),
    )


def test_rollout_logs_per_teacher_trainable_trajectory_reward():
    def sample(teacher_id, session_id, reward, loss_mask=(1,)):
        return SimpleNamespace(
            metadata={"teacher_id": teacher_id, "session_id": session_id},
            remove_sample=False,
            response_length=1,
            loss_mask=loss_mask,
            reward=reward,
        )

    samples = [
        sample("alfworld", "a-1", 1.0),
        sample("alfworld", "a-2", 0.0),
        sample("hotpotqa", "h-1", 0.25),
        sample("hotpotqa", "h-dropped", 1.0, loss_mask=(0,)),
    ]
    extra_metrics = {}

    assert (
        log_rollout_data(
            rollout_id=1,
            args=SimpleNamespace(use_wandb=False),
            samples=samples,
            extra_metrics=extra_metrics,
            rollout_time=1.0,
        )
        is False
    )
    assert extra_metrics == {
        "rollout/mopd/raw_reward_trainable_trajectory_mean/alfworld": 0.5,
        "rollout/mopd/raw_reward_trainable_trajectory_mean/hotpotqa": 0.25,
    }
