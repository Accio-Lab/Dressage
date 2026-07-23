"""Dressage-owned multi-teacher extension for slime's Megatron OPD actor.

The upstream actor already knows how to load a Megatron teacher, switch model
weights, compute response-token log-probabilities, and apply the OPD loss.  This
subclass only adds the missing multi-teacher policy:

* load the configured frozen teachers into the existing pinned-CPU weight backuper;
* partition each local DP batch by its routed teacher;
* score only that teacher's samples; and
* scatter the response-aligned log-probabilities back to batch order.

Keeping this class in Dressage prevents slime from depending on Dressage's
configuration schema or hard-coding MOPD route names.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from megatron.core import mpu
from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
from slime.backends.megatron_utils.data import gather_log_data, get_data_iterator
from slime.utils.timer import Timer

from dressage.rollout.mopd import MOPDTeacher, load_mopd_config

_SCORER_FIELDS = (
    "tokens",
    "loss_masks",
    "multimodal_train_inputs",
    "total_lengths",
    "response_lengths",
    "rollout_top_p_token_ids",
    "rollout_top_p_token_offsets",
)


def _metric_component(value: str) -> str:
    """Return a stable W&B-safe path component for a configured teacher ID."""
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return component or "unnamed"


def _train_aggregation_mean_contribution(
    values: list[torch.Tensor],
    selected_indices: list[int],
    rollout_data: dict[str, Any],
) -> tuple[float, float]:
    """Return a distributed sum/count pair using the policy-loss aggregation."""
    rollout_mask_sums = rollout_data.get("rollout_mask_sums")
    if rollout_mask_sums is None:
        raise ValueError("MOPD train-aggregation metrics require rollout_mask_sums")

    selected_indices = [
        index
        for index in selected_indices
        if rollout_data["loss_masks"][index].sum().item() > 0
        and rollout_mask_sums[index].item() > 0
    ]
    if not selected_indices:
        return 0.0, 0.0

    selected_values = [values[index] for index in selected_indices]
    total_lengths = [rollout_data["total_lengths"][index] for index in selected_indices]
    response_lengths = [
        rollout_data["response_lengths"][index] for index in selected_indices
    ]
    loss_masks = [rollout_data["loss_masks"][index] for index in selected_indices]
    sample_denoms = rollout_mask_sums[selected_indices]
    reducer = get_sum_of_sample_mean(
        total_lengths,
        response_lengths,
        loss_masks,
        sample_denoms=sample_denoms,
    )
    local_sum = reducer(torch.cat(selected_values, dim=0)).item()

    # Full masks/denominators are replicated on CP ranks. Fractional ownership
    # makes flattened sibling segments add to the effective loss count.
    local_count = (
        sum(
            rollout_data["loss_masks"][index].sum().item()
            / rollout_mask_sums[index].item()
            for index in selected_indices
        )
        / mpu.get_context_parallel_world_size()
    )
    return local_sum, local_count


def build_teacher_subset(
    rollout_data: dict[str, Any],
    teacher_ids: list[str],
    teacher_id: str,
) -> tuple[dict[str, Any], list[int]]:
    """Build a compact dynamic-batch view for one teacher.

    Existing dynamic microbatches may contain samples routed to different
    teachers.  Filter each microbatch and remap its indices into a compact
    teacher-local sample array so slime's normal ``forward_only`` ordering
    restoration remains valid.
    """
    sample_count = len(teacher_ids)
    selected_indices = [
        position
        for position, routed_id in enumerate(teacher_ids)
        if routed_id == teacher_id
    ]
    old_to_new = {old: new for new, old in enumerate(selected_indices)}
    compact_microbatches: list[list[int]] = []
    for microbatch in rollout_data["micro_batch_indices"]:
        compact = [old_to_new[index] for index in microbatch if index in old_to_new]
        if compact:
            compact_microbatches.append(compact)
    if not compact_microbatches:
        raise ValueError(f"MOPD teacher {teacher_id!r} has no trainable microbatches")

    subset: dict[str, Any] = {"micro_batch_indices": compact_microbatches}
    for key in _SCORER_FIELDS:
        values = rollout_data.get(key)
        if values is None:
            continue
        if len(values) != sample_count:
            raise ValueError(
                f"MOPD scorer field {key!r} has {len(values)} values; expected {sample_count}"
            )
        subset[key] = [values[index] for index in selected_indices]
    return subset, selected_indices


class MOPDMegatronTrainRayActor(MegatronTrainRayActor):
    """Megatron actor that extends stock single-teacher OPD to routed teachers."""

    def init(
        self,
        args: Any,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        # Let stock slime initialize the actor/ref and all training machinery,
        # but suppress its one global teacher.  This subclass owns teachers.
        start_rollout_id = super().init(
            args,
            role,
            with_ref=with_ref,
            with_opd_teacher=False,
        )
        if args.debug_rollout_only or role != "actor" or not with_opd_teacher:
            return start_rollout_id

        # Slime invokes this hook after OPD advantages are available and before
        # its normal rollout-data logging. Chain any caller-provided hook.
        self._base_rollout_data_postprocess = self.rollout_data_postprocess
        self.rollout_data_postprocess = self._postprocess_mopd_metrics
        self._active_mopd_teacher_ids: list[str] | None = None

        config_path = getattr(args, "mopd_teacher_config", None)
        if not config_path:
            import os

            config_path = os.environ.get("DRESSAGE_MOPD_TEACHER_CONFIG")
        if not config_path:
            raise ValueError(
                "Dressage MOPD actor requires DRESSAGE_MOPD_TEACHER_CONFIG"
            )

        # The base init starts train_wait as a deferred timer.  Teacher loading
        # is initialization, not rollout wait time, so fence it out here.
        Timer().end("train_wait")
        self.mopd_teacher_tags: dict[str, str] = {}
        self.mopd_teacher_metric_components: dict[str, str] = {}
        try:
            if self.args.offload_train:
                self.wake_up()
            for teacher_id, teacher in load_mopd_config(config_path).teachers.items():
                component = _metric_component(teacher_id)
                if component in self.mopd_teacher_metric_components.values():
                    raise ValueError(
                        "MOPD teacher IDs must have distinct metric-safe names; "
                        f"{teacher_id!r} collides at {component!r}"
                    )
                tag = f"teacher:{teacher_id}"
                self._load_mopd_teacher(tag, teacher)
                self.mopd_teacher_tags[teacher_id] = tag
                self.mopd_teacher_metric_components[teacher_id] = component
            self._switch_model("actor")
        finally:
            if self.args.offload_train:
                self.sleep()
            Timer().start("train_wait")
        return start_rollout_id

    def _load_mopd_teacher(self, tag: str, teacher: MOPDTeacher) -> None:
        # Stock load_other_checkpoint supports a per-step override only for its
        # single literal "teacher" tag.  Temporarily set ckpt_step around the
        # generic loader so no slime change is needed for named teacher tags.
        old_step = self.args.ckpt_step
        try:
            self.args.ckpt_step = teacher.ckpt_step
            super().load_other_checkpoint(tag, teacher.load)
        finally:
            self.args.ckpt_step = old_step

    def _score_routed_teachers(self, rollout_data: dict[str, Any]) -> None:
        # Slime DP-partitions the train-side ``prompt`` field. Dressage uses
        # that otherwise-unused channel for one teacher ID per sample, then
        # removes it before stock logging attempts numeric reduction.
        teacher_ids = rollout_data.pop("prompt")
        if len(teacher_ids) != len(rollout_data["tokens"]):
            raise ValueError("MOPD teacher route count does not match sample count")
        if self.args.use_routing_replay:
            raise ValueError("Dressage routed MOPD does not yet support routing replay")

        routed_values: list[Any | None] = [None] * len(teacher_ids)
        produced_values = False
        try:
            for teacher_id in dict.fromkeys(teacher_ids):
                subset, selected_indices = build_teacher_subset(
                    rollout_data,
                    teacher_ids,
                    teacher_id,
                )
                self._switch_model(self.mopd_teacher_tags[teacher_id])
                output = self.compute_log_prob(
                    get_data_iterator(subset),
                    [len(subset["micro_batch_indices"])],
                    store_prefix="teacher_",
                )
                values = output.get("teacher_log_probs")
                if values is None:
                    # Non-last pipeline stages do not own response log-probs.
                    continue
                if len(values) != len(selected_indices):
                    raise ValueError(
                        f"MOPD teacher {teacher_id!r} returned {len(values)} samples; "
                        f"expected {len(selected_indices)}"
                    )
                produced_values = True
                for original_index, value in zip(selected_indices, values, strict=True):
                    routed_values[original_index] = value
        finally:
            self._switch_model("old_actor" if self.args.keep_old_actor else "actor")

        if produced_values:
            missing = [
                index for index, value in enumerate(routed_values) if value is None
            ]
            if missing:
                raise ValueError(
                    f"MOPD scorer did not produce values for positions {missing}"
                )
            rollout_data["teacher_log_probs"] = routed_values

        self._active_mopd_teacher_ids = teacher_ids

    def _postprocess_mopd_metrics(
        self,
        args: Any,
        rollout_id: int,
        rollout_data: dict[str, Any],
    ) -> None:
        """Log sampled-token reverse KL split by routed teacher."""
        teacher_ids = self._active_mopd_teacher_ids
        if not teacher_ids:
            if self._base_rollout_data_postprocess is not None:
                self._base_rollout_data_postprocess(args, rollout_id, rollout_data)
            return
        if not (
            mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage()
        ):
            if self._base_rollout_data_postprocess is not None:
                self._base_rollout_data_postprocess(args, rollout_id, rollout_data)
            return

        reverse_kls = rollout_data.get("opd_reverse_kl")
        if reverse_kls is None or len(reverse_kls) != len(teacher_ids):
            raise ValueError(
                "MOPD opd_reverse_kl must contain one tensor per routed sample"
            )

        log_dict: dict[str, float | tuple[float, float]] = {}
        for teacher_id, component in self.mopd_teacher_metric_components.items():
            selected_indices = [
                index
                for index, routed_teacher_id in enumerate(teacher_ids)
                if routed_teacher_id == teacher_id
            ]
            log_dict[f"mopd/opd_reverse_kl_train_aggregation_mean/{component}"] = (
                _train_aggregation_mean_contribution(
                    reverse_kls,
                    selected_indices,
                    rollout_data,
                )
            )

        # Slime maps rollout/* to rollout/step in W&B.
        gather_log_data("rollout", args, rollout_id, log_dict)
        if self._base_rollout_data_postprocess is not None:
            self._base_rollout_data_postprocess(args, rollout_id, rollout_data)

    def train_actor(
        self, rollout_id: int, rollout_data: dict[str, Any], external_data=None
    ) -> None:
        # Preserve slime's train/train_wait timing semantics even though the
        # custom scoring phase executes immediately before the stock method.
        Timer().end("train_wait")
        Timer().start("train")
        try:
            self._score_routed_teachers(rollout_data)
        finally:
            Timer().end("train")
            Timer().start("train_wait")
        try:
            return super().train_actor(
                rollout_id, rollout_data, external_data=external_data
            )
        finally:
            self._active_mopd_teacher_ids = None
