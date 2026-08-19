from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dressage.training.mopd_loss import (
    MOPD_ADVANTAGE_FUNCTION_PATH,
    MOPD_LOSS_FUNCTION_PATH,
    compute_mopd_advantages,
    mopd_policy_loss,
    validate_pure_mopd_args,
)


def _pure_args(**overrides):
    values = {
        "advantage_estimator": "mopd",
        "loss_type": "custom_loss",
        "custom_advantage_function_path": MOPD_ADVANTAGE_FUNCTION_PATH,
        "custom_loss_function_path": MOPD_LOSS_FUNCTION_PATH,
        "use_opd": False,
        "kl_coef": 0.0,
        "use_kl_loss": False,
        "kl_loss_coef": 0.0,
        "entropy_coef": 0.0,
        "normalize_advantages": False,
        "rewards_normalization": False,
        "use_critic": False,
        "compute_advantages_and_returns": True,
        "use_rollout_logprobs": False,
        "use_tis": False,
        "use_opsm": False,
        "num_steps_per_rollout": 1,
        "n_samples_per_prompt": 1,
        "mopd_advantage_clip": 5.0,
        "rollout_top_p": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_mopd_advantage_is_teacher_minus_student_clipped_and_reward_free():
    args = _pure_args(mopd_advantage_clip=2.0)
    first = {
        "log_probs": [torch.tensor([-2.0, -1.0, -3.0])],
        "teacher_log_probs": [torch.tensor([-1.5, -4.0, 1.0])],
        "rewards": [12345.0],
    }
    second = {
        "log_probs": [first["log_probs"][0].clone()],
        "teacher_log_probs": [first["teacher_log_probs"][0].clone()],
        "rewards": [-99999.0],
    }

    compute_mopd_advantages(args, first)
    compute_mopd_advantages(args, second)

    expected = torch.tensor([0.5, -2.0, 2.0])
    torch.testing.assert_close(first["advantages"][0], expected)
    torch.testing.assert_close(second["advantages"][0], expected)
    torch.testing.assert_close(
        first["opd_reverse_kl"][0], torch.tensor([-0.5, 3.0, -4.0])
    )
    assert not first["advantages"][0].requires_grad


def test_mopd_loss_is_direct_logprob_loss_without_ppo_ratio(monkeypatch):
    current_log_probs = torch.tensor([-1.2, -2.5], requires_grad=True)
    fake_loss_module = types.ModuleType("slime.backends.megatron_utils.loss")
    fake_loss_module.get_log_probs_and_entropy = lambda *args, **kwargs: (
        None,
        {
            "log_probs": [current_log_probs],
            "entropy": [torch.tensor([0.7, 0.9])],
        },
    )
    fake_loss_module.get_rollout_top_p_logprob_kwargs = lambda *args, **kwargs: {}
    monkeypatch.setitem(
        sys.modules, "slime.backends.megatron_utils.loss", fake_loss_module
    )

    advantages = torch.tensor([0.5, -1.5])
    batch = {
        "advantages": [advantages],
        "log_probs": [torch.tensor([-1.0, -2.0])],
        "teacher_log_probs": [torch.tensor([-0.5, -3.5])],
        "unconcat_tokens": [],
        "total_lengths": [],
        "response_lengths": [],
    }
    loss, metrics = mopd_policy_loss(
        _pure_args(), batch, torch.empty(0), lambda tensor: tensor.mean()
    )

    expected = -(advantages * current_log_probs).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(current_log_probs.grad, -advantages / 2)
    assert "pg_clipfrac" not in metrics


def test_pure_mopd_validation_rejects_incompatible_options():
    validate_pure_mopd_args(_pure_args())

    with pytest.raises(ValueError, match="--use-opd must be disabled"):
        validate_pure_mopd_args(_pure_args(use_opd=True))
    with pytest.raises(ValueError, match="--n-samples-per-prompt must be 1"):
        validate_pure_mopd_args(_pure_args(n_samples_per_prompt=4))
    with pytest.raises(ValueError, match="reward normalization must be disabled"):
        validate_pure_mopd_args(_pure_args(rewards_normalization=True))


def test_public_launcher_selects_only_the_pure_objective():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "examples/scripts/run_mopd_qwen3.5_sync.sh"
    ).read_text(encoding="utf-8")

    assert "dressage.training.mopd_loss.compute_mopd_advantages" in launcher
    assert "dressage.training.mopd_loss.mopd_policy_loss" in launcher
    assert 'N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"' in launcher
    assert 'GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"' in launcher
    assert '"${GLOBAL_BATCH_SIZE}" != "${ROLLOUT_BATCH_SIZE}"' in launcher
    assert "--use-opd" not in launcher
    assert "--opd-kl-coef" not in launcher
    assert "--advantage-estimator grpo" not in launcher
    assert "--eps-clip" not in launcher
