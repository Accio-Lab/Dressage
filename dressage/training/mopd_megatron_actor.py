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

from typing import Any

from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from slime.backends.megatron_utils.data import get_data_iterator
from slime.utils.timer import Timer

from dressage.rollout.mopd import (
    MOPDTeacher,
    load_mopd_config,
    pop_mopd_teacher_ids_from_rollout_data,
)

_SCORER_FIELDS = (
    "tokens",
    "loss_masks",
    "multimodal_train_inputs",
    "total_lengths",
    "response_lengths",
    "max_seq_lens",
    "rollout_top_p_token_ids",
    "rollout_top_p_token_offsets",
)


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
    if not selected_indices:
        raise ValueError(f"MOPD teacher {teacher_id!r} has no local samples")

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
        try:
            if self.args.offload_train:
                self.wake_up()
            for teacher_id, teacher in load_mopd_config(config_path).teachers.items():
                tag = f"teacher:{teacher_id}"
                self._load_mopd_teacher(tag, teacher)
                self.mopd_teacher_tags[teacher_id] = tag
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
        # Slime natively forwards the train-side ``prompt`` field through its
        # DP splitter.  Dressage uses that otherwise-unused channel for a
        # versioned route payload, then removes it before stock slime logging
        # attempts to reduce every remaining rollout-data field numerically.
        teacher_ids = pop_mopd_teacher_ids_from_rollout_data(rollout_data)
        if self.args.use_routing_replay:
            raise ValueError("Dressage routed MOPD does not yet support routing replay")

        unknown = sorted(set(teacher_ids) - set(self.mopd_teacher_tags))
        if unknown:
            raise ValueError(f"unknown MOPD teacher id(s): {unknown}")

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
        return super().train_actor(
            rollout_id, rollout_data, external_data=external_data
        )
