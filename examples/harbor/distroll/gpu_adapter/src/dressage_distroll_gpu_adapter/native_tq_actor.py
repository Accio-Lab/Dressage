"""Native Dressage hydration; release shared source keys after all ranks finish."""

import resource

import numpy as np
import torch
from dressage.training.tq_hydration import (
    decode_layout_bundle,
    hydrate_training_layouts,
    remote_fields_from_layouts,
)
from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

from .grad_sync import fenced_grad_sync


def hydrate_checked(args, data, layouts, batch_get, transform):
    required = {}
    for bundle in layouts:
        for field, layout in decode_layout_bundle(bundle).items():
            if field != "full_logprobs":
                raise ValueError("DistRoll native delivery supports logprobs only")
            for f in layout.fragments:
                key = (f.ref.partition, f.ref.key, f.ref.field)
                required[key] = max(required.get(key, 0), f.source_start + f.length)

    def read(**kwargs):
        result = batch_get(**kwargs)
        field = kwargs["select_fields"]
        column = result[field]
        if len(column) != len(kwargs["keys"]):
            raise ValueError("incomplete TQ field batch")
        tensors = []
        for key, row in zip(kwargs["keys"], column, strict=True):
            # Never unwrap numpy.data into an unsupported '=f' memoryview.
            if not isinstance(row, (list, tuple, np.ndarray, torch.Tensor)):
                row = getattr(row, "data", row)
            value = torch.as_tensor(row)
            minimum = required[(kwargs["partition_id"], key, field)]
            if (
                value.ndim != 1
                or not value.dtype.is_floating_point
                or value.numel() < minimum
                or not torch.isfinite(value).all()
            ):
                raise ValueError("TQ logprobs missing, nonnumeric, short, or nonfinite")
            tensors.append(value.to(dtype=torch.float32))
        return {field: tensors}

    hydrate_training_layouts(
        args,
        data,
        layouts,
        remote_fields=remote_fields_from_layouts(layouts),
        batch_get=read,
        logprob_transform=transform,
    )


class NativeTQTrainRayActor(MegatronTrainRayActor):
    """Keep native layout/CP/loss, but no early per-DP key deletion."""

    def train_actor(self, rollout_id, rollout_data, external_data=None):
        import torch.distributed as dist
        from slime.utils.distributed_utils import get_gloo_group

        # Scope to this private trainer, never the node or Docker daemon.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if self.args.actor_num_nodes == 1:
            return super().train_actor(rollout_id, rollout_data, external_data)
        if (
            self.args.overlap_grad_reduce
            or self.args.overlap_param_gather
            or self.args.pipeline_model_parallel_size != 1
        ):
            raise ValueError(
                "gradient phase fences require non-overlapped PP1 synchronous training"
            )

        def rendezvous():
            torch.cuda.synchronize()
            # The existing CPU group includes all training ranks, not inference.
            dist.barrier(group=get_gloo_group())

        originals = []
        try:
            for chunk in self.model:
                original = chunk.finish_grad_sync
                originals.append((chunk, original))
                chunk.finish_grad_sync = fenced_grad_sync(original, rendezvous)
            return super().train_actor(rollout_id, rollout_data, external_data)
        finally:
            for chunk, original in originals:
                chunk.finish_grad_sync = original

    def _get_rollout_data(self, rollout_data_ref):
        data = super()._get_rollout_data(rollout_data_ref)
        layouts = data.pop("prompt", None)
        error = None
        device = torch.cuda.current_device()
        try:
            if layouts is not None:
                import ray
                import transfer_queue as tq

                ray.get_actor("TransferQueueController", namespace="transfer_queue")
                hydrate_checked(
                    self.args,
                    data,
                    layouts,
                    tq.kv_batch_get,
                    lambda x, total, response: slice_log_prob_with_cp(
                        x, total, response
                    ).to(device=device, dtype=torch.float32, non_blocking=True),
                )
            if not data.get("rollout_log_probs") or any(
                not torch.isfinite(x).all() for x in data["rollout_log_probs"]
            ):
                raise ValueError("training requires finite rollout logprobs")
        except Exception as exc:  # noqa: BLE001 - every rank must enter the failure barrier
            error = exc
        # All ranks must hydrate successfully before any optimizer is allowed.
        import torch.distributed as dist

        failed = torch.tensor(int(error is not None), device=device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError(
                "a training rank failed trajectory hydration; no optimizer allowed"
            ) from error
        return data
