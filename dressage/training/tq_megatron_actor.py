"""Megatron actor that hydrates selected fields directly from TransferQueue."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch
from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

from dressage.training.tq_hydration import (
    FULL_LOGPROBS_FIELD,
    ROUTED_EXPERTS_FIELD,
    clear_requests_from_layouts,
    hydrate_training_layouts,
    remote_fields_from_layouts,
)
from dressage.transport import normalize_transfer_params
from dressage.transport.store import TransferQueueStore

logger = logging.getLogger(__name__)


def _configured_remote_fields() -> set[str]:
    params = set(
        normalize_transfer_params(os.environ.get("DRESSAGE_TRANSFER_PARAMS"))
    )
    fields = set()
    if "logprobs" in params:
        fields.add(FULL_LOGPROBS_FIELD)
    if "routed_experts" in params:
        fields.add(ROUTED_EXPERTS_FIELD)
    return fields


class TQMegatronTrainRayActor(MegatronTrainRayActor):
    """Load a DP shard's remote fields before Slime builds data iterators."""

    def _batch_get(self, **kwargs: Any) -> Any:
        if not hasattr(self, "_tq_api"):
            try:
                import transfer_queue as tq
            except ImportError as exc:
                raise RuntimeError(
                    "TransferQueue is required by the TQ training entry"
                ) from exc
            tq.init()
            self._tq_api = tq
        started_at = time.perf_counter()
        result = self._tq_api.kv_batch_get(**kwargs)
        elapsed = time.perf_counter() - started_at
        payload_bytes = TransferQueueStore.estimate_bytes(result)
        logger.info(
            "TransferQueue read: keys=%s field=%s bytes=%s seconds=%.6f",
            len(kwargs["keys"]),
            kwargs["select_fields"],
            payload_bytes,
            elapsed,
        )
        return result

    def _get_rollout_data(self, rollout_data_ref):
        rollout_data = super()._get_rollout_data(rollout_data_ref)
        raw_layouts = rollout_data.pop("prompt", None)
        if raw_layouts is None:
            return rollout_data

        remote_fields = remote_fields_from_layouts(raw_layouts)
        if not remote_fields:
            remote_fields = _configured_remote_fields()
        device = torch.cuda.current_device()

        def transform_logprobs(values, total_length, response_length):
            return slice_log_prob_with_cp(
                values,
                total_length,
                response_length,
            ).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

        hydrate_training_layouts(
            self.args,
            rollout_data,
            raw_layouts,
            remote_fields=remote_fields,
            batch_get=self._batch_get,
            logprob_transform=transform_logprobs,
        )
        consumed_fields = set(remote_fields)
        if not getattr(self.args, "use_rollout_routing_replay", False):
            consumed_fields.discard(ROUTED_EXPERTS_FIELD)
        self._tq_clear_requests = clear_requests_from_layouts(
            raw_layouts,
            consumed_fields,
        )
        from megatron.core import parallel_state as mpu

        rank_functions = (
            "get_tensor_model_parallel_rank",
            "get_pipeline_model_parallel_rank",
            "get_context_parallel_rank",
            "get_expert_model_parallel_rank",
        )
        self._tq_is_shard_leader = all(
            not hasattr(mpu, name) or getattr(mpu, name)() == 0
            for name in rank_functions
        )
        return rollout_data

    def train(self, rollout_id, rollout_data_ref, external_data=None):
        self._tq_clear_requests = {}
        self._tq_is_shard_leader = False
        result = super().train(
            rollout_id,
            rollout_data_ref,
            external_data=external_data,
        )
        requests = self._tq_clear_requests
        is_shard_leader = self._tq_is_shard_leader
        self._tq_clear_requests = {}
        self._tq_is_shard_leader = False
        if not requests:
            return result

        if is_shard_leader:
            try:
                for partition, keys in requests.items():
                    self._tq_api.kv_clear(keys=keys, partition_id=partition)
            except Exception:
                logger.exception(
                    "TransferQueue cleanup failed; retention will reclaim the data"
                )
        return result
