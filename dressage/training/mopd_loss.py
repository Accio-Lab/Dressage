"""Paper-standard on-policy distillation objective for Dressage MOPD.

For student-sampled response tokens, the stopped-gradient advantage is

    A_t = clip(log pi_teacher(y_t) - log pi_student_old(y_t), -c, c)

and the actor minimizes

    L = -mean(A_t * log pi_student(y_t)).

Environment rewards are deliberately absent from both equations. They may
still be present in rollout data for monitoring and dataset diagnostics.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch

MOPD_ADVANTAGE_FUNCTION_PATH = "dressage.training.mopd_loss.compute_mopd_advantages"
MOPD_LOSS_FUNCTION_PATH = "dressage.training.mopd_loss.mopd_policy_loss"


def _require_aligned_log_probs(
    rollout_data: dict[str, Any],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    student_log_probs = rollout_data.get("log_probs")
    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if not student_log_probs:
        raise ValueError("pure MOPD requires train-side student log_probs")
    if not teacher_log_probs:
        raise ValueError("pure MOPD requires routed teacher_log_probs")
    if len(student_log_probs) != len(teacher_log_probs):
        raise ValueError(
            "MOPD log-prob sample count mismatch: "
            f"student={len(student_log_probs)} teacher={len(teacher_log_probs)}"
        )
    for index, (student, teacher) in enumerate(
        zip(student_log_probs, teacher_log_probs, strict=True)
    ):
        if student.shape != teacher.shape:
            raise ValueError(
                f"MOPD log-prob shape mismatch at sample {index}: "
                f"student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
            )
    return student_log_probs, teacher_log_probs


def compute_mopd_advantages(args: Namespace, rollout_data: dict[str, Any]) -> None:
    """Populate pure MOPD advantages without reading environment rewards."""
    student_log_probs, teacher_log_probs = _require_aligned_log_probs(rollout_data)
    clip = float(args.mopd_advantage_clip)
    if clip <= 0:
        raise ValueError(f"--mopd-advantage-clip must be positive; got {clip}")

    advantages: list[torch.Tensor] = []
    reverse_kls: list[torch.Tensor] = []
    for student, teacher in zip(student_log_probs, teacher_log_probs, strict=True):
        teacher = teacher.to(device=student.device, dtype=student.dtype)
        reverse_kl = (student - teacher).detach()
        advantages.append((-reverse_kl).clamp(min=-clip, max=clip).detach())
        reverse_kls.append(reverse_kl)

    rollout_data["advantages"] = advantages
    # MOPD has no value target. Slime expects returns to exist, so retain a
    # detached copy of the policy-gradient signal rather than an RL return.
    rollout_data["returns"] = [advantage.clone() for advantage in advantages]
    rollout_data["opd_reverse_kl"] = reverse_kls


def mopd_policy_loss(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute ``-stopgrad(A_mopd) * log pi_student`` exactly."""
    # Import lazily so the objective remains unit-testable without initializing
    # Megatron's model stack.
    from slime.backends.megatron_utils.loss import (
        get_log_probs_and_entropy,
        get_rollout_top_p_logprob_kwargs,
    )

    advantages = torch.cat(batch["advantages"], dim=0).detach()
    old_log_probs = torch.cat(batch["log_probs"], dim=0).detach()
    teacher_log_probs = torch.cat(batch["teacher_log_probs"], dim=0).detach()

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=True,
        **get_rollout_top_p_logprob_kwargs(args, batch),
    )
    log_probs = torch.cat(log_probs_and_entropy["log_probs"], dim=0)
    entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)

    if not (
        advantages.shape
        == old_log_probs.shape
        == teacher_log_probs.shape
        == log_probs.shape
    ):
        raise ValueError(
            "MOPD loss tensors must be response-token aligned: "
            f"advantage={tuple(advantages.shape)} old={tuple(old_log_probs.shape)} "
            f"teacher={tuple(teacher_log_probs.shape)} current={tuple(log_probs.shape)}"
        )

    token_loss = -advantages * log_probs
    loss = sum_of_sample_mean(token_loss)
    if log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

    raw_teacher_gap = teacher_log_probs - old_log_probs
    clip = float(args.mopd_advantage_clip)
    metrics = {
        "loss": loss.detach().clone(),
        "mopd_loss": loss.detach().clone(),
        "entropy": sum_of_sample_mean(entropy).detach().clone(),
        # Sampled-token reverse-KL estimator: log pi_student - log pi_teacher.
        "mopd_reverse_kl": sum_of_sample_mean(-raw_teacher_gap).detach().clone(),
        "mopd_advantage": sum_of_sample_mean(advantages).detach().clone(),
        "mopd_advantage_abs": sum_of_sample_mean(advantages.abs()).detach().clone(),
        "mopd_advantage_clipfrac": sum_of_sample_mean(
            (raw_teacher_gap.abs() > clip).to(dtype=log_probs.dtype)
        )
        .detach()
        .clone(),
        # Optimizer drift within this update, not teacher KL.
        "mopd_update_kl": sum_of_sample_mean(old_log_probs - log_probs)
        .detach()
        .clone(),
    }
    return loss, metrics


def validate_pure_mopd_args(args: Namespace) -> None:
    """Reject options that are incompatible with the dedicated MOPD path."""
    errors: list[str] = []
    if args.advantage_estimator != "mopd":
        errors.append("the MOPD entrypoint must select the pure mopd estimator")
    if args.loss_type != "custom_loss":
        errors.append("--loss-type must be custom_loss")
    if args.custom_advantage_function_path != MOPD_ADVANTAGE_FUNCTION_PATH:
        errors.append(
            "--custom-advantage-function-path must select the Dressage pure-MOPD function"
        )
    if args.custom_loss_function_path != MOPD_LOSS_FUNCTION_PATH:
        errors.append(
            "--custom-loss-function-path must select the Dressage pure-MOPD loss"
        )
    if bool(args.use_opd):
        errors.append("--use-opd must be disabled for the dedicated MOPD objective")
    if float(args.kl_coef) != 0.0:
        errors.append("--kl-coef must be 0")
    if bool(args.use_kl_loss) or float(args.kl_loss_coef) != 0.0:
        errors.append("auxiliary KL loss must be disabled")
    if float(args.entropy_coef) != 0.0:
        errors.append("--entropy-coef must be 0")
    if bool(args.normalize_advantages):
        errors.append("--normalize-advantages must be disabled")
    if bool(args.rewards_normalization):
        errors.append("reward normalization must be disabled")
    if bool(args.use_critic):
        errors.append("critic/value training must be disabled")
    if not bool(args.compute_advantages_and_returns):
        errors.append("advantage computation must be enabled")
    if bool(args.use_rollout_logprobs):
        errors.append("--use-rollout-logprobs must be disabled")
    if bool(args.use_tis) or bool(args.use_opsm):
        errors.append("TIS/OPSM off-policy correction must be disabled")
    if int(args.num_steps_per_rollout) != 1:
        errors.append("--num-steps-per-rollout must be 1")
    if int(args.n_samples_per_prompt) != 1:
        errors.append("--n-samples-per-prompt must be 1 for paper-standard MOPD")
    if float(args.mopd_advantage_clip) <= 0:
        errors.append("--mopd-advantage-clip must be positive")
    if errors:
        raise ValueError("invalid pure MOPD configuration:\n- " + "\n- ".join(errors))
