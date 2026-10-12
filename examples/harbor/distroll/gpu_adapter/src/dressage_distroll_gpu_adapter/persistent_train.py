from __future__ import annotations

import logging
import math
import os
from collections import Counter
from typing import Any


def _rollout_manager_options(args: Any, env_vars: dict[str, str]) -> dict[str, Any]:
    resource_name = os.environ.get(
        "ROLLOUT_GATEWAY_RESOURCE_NAME", "rollout_gateway"
    ).strip()
    resource_amount = float(os.environ.get("ROLLOUT_GATEWAY_RESOURCE_AMOUNT", "0.01"))
    if not resource_name or not math.isfinite(resource_amount) or resource_amount <= 0:
        raise ValueError(
            "rollout gateway resource name must be set and amount must be finite and positive"
        )
    options: dict[str, Any] = {
        "num_cpus": 1,
        "num_gpus": 0,
        "resources": {resource_name: resource_amount},
        "runtime_env": {"env_vars": env_vars},
    }
    if getattr(args, "rollout_data_transport", "object-store") == "nixl":
        options["enable_tensor_transport"] = True
    return options


def _create_gateway_rollout_manager(args: Any, placement_group: Any):
    import ray
    from slime.ray.rollout import RolloutManager
    from slime.ray.utils import add_default_ray_env_vars

    options = _rollout_manager_options(args, add_default_ray_env_vars())
    rollout_manager = RolloutManager.options(**options).remote(args, placement_group)
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(
            rollout_manager.get_num_rollout_per_epoch.remote()
        )
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        if args.num_rollout <= 0:
            raise ValueError("calculated num_rollout must be positive")
    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))
    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())
    return rollout_manager, num_rollout_per_epoch


def _require_node_layout(
    actor_ips: list[str], rollout_ips: list[str], *, actors: list[str], rollout: str
) -> None:
    expected = Counter({ip: 8 for ip in actors})
    if (
        not actors
        or any(not ip for ip in actors)
        or len(set(actors)) != len(actors)
        or rollout in actors
        or Counter(actor_ips) != expected
        or rollout_ips != [rollout] * 8
    ):
        raise RuntimeError(
            "non-colocated GPU placement mismatch: "
            f"actor={actor_ips} expected={actors} (8 GPUs each); "
            f"rollout={rollout_ips} expected={rollout}"
        )


def _partition_node_layout(ips, indices, gpu_ids, *, actors, rollout):
    """Keep TP/CP ranks node-local; do not infer roles from IP sort order."""
    rows = list(zip(ips, indices, gpu_ids, strict=True))
    actor_rows = [row for ip in actors for row in rows if row[0] == ip]
    rollout_rows = [row for row in rows if row[0] == rollout]
    _require_node_layout(
        [row[0] for row in actor_rows],
        [row[0] for row in rollout_rows],
        actors=actors,
        rollout=rollout,
    )
    if len(actor_rows) + len(rollout_rows) != len(rows):
        raise RuntimeError("GPU placement includes an unapproved node")
    return {
        role: ([row[1] for row in selected], [row[2] for row in selected])
        for role, selected in (("actor", actor_rows), ("rollout", rollout_rows))
    }


def _check_node_layout(args: Any, placement_groups: dict) -> None:
    """Partition actual Ray bundles before starting Megatron or SGLang."""
    actor_ips = os.environ.get("DRESSAGE_ACTOR_NODE_IPS")
    rollout_ip = os.environ.get("DRESSAGE_ROLLOUT_NODE_IP")
    if not actor_ips and not rollout_ip:
        return
    if not actor_ips or not rollout_ip:
        raise ValueError("both DRESSAGE GPU node IPs must be set")
    actors = [ip.strip() for ip in actor_ips.split(",")]
    if (
        args.actor_num_nodes != len(actors)
        or args.actor_num_gpus_per_node != 8
        or args.rollout_num_gpus != 8
    ):
        raise ValueError("GPU layout requires 8 GPUs per actor node and 8 rollout GPUs")

    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    @ray.remote(num_cpus=0.01)
    def node_ip() -> str:
        import ray

        return ray.util.get_node_ip_address()

    # Upstream's actor entry initially contains ALL bundles, ordered by IP;
    # its rollout entry is a suffix. Repartition that one shared PG in place.
    pg, indices, gpu_ids = placement_groups["actor"]
    refs = [
        node_ip.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=index
            )
        ).remote()
        for index in indices
    ]
    ips = ray.get(refs, timeout=60)
    layout = _partition_node_layout(
        ips, indices, gpu_ids, actors=actors, rollout=rollout_ip
    )
    for role, (role_indices, role_gpu_ids) in layout.items():
        placement_groups[role] = (pg, role_indices, role_gpu_ids)
    logging.getLogger(__name__).info(
        "GPU role layout: actor_nodes=%s rollout_node=%s bundles=%s",
        actors,
        rollout_ip,
        {role: values[0] for role, values in layout.items()},
    )


def train(args) -> None:
    from .checkpoints import CheckpointWindow

    if args.save is None:
        raise ValueError("synchronous training requires a checkpoint save directory")
    if getattr(args, "save_hf", None) is not None:
        raise ValueError("separate HF checkpoint export is not covered by retention")
    with CheckpointWindow(
        args.save,
        os.environ["GPU_SOURCE_ROOT"],
        os.environ["DRESSAGE_DISTROLL_RUN_ID"],
    ) as checkpoints:
        _train(args, checkpoints)


def _train(args, checkpoints) -> None:
    import ray
    from dressage.training.tq_hydration import validate_tq_training_config
    from slime.ray.placement_group import (
        create_placement_groups,
        create_training_models,
    )
    from slime.utils.logging_utils import (
        configure_logger,
        finish_tracking,
        init_tracking,
    )

    from .native_tq_actor import NativeTQTrainRayActor
    from .trajectory_delivery import release_training_trajectories

    validate_tq_training_config(args)
    if os.environ.get("TRAIN_RESUME_FROM") and (
        args.no_load_optim or args.no_load_rng or args.finetune
    ):
        raise ValueError("resume requires native optimizer, scheduler and RNG loading")
    if args.custom_convert_samples_to_train_data_path != (
        "dressage.rollout.convert_samples.convert_samples_to_train_data"
    ):
        raise ValueError("native delivery requires the original Dressage converter")
    if args.use_rollout_routing_replay:
        raise ValueError(
            "DistRoll native delivery currently supports logprobs, not expert replay"
        )
    if args.colocate:
        raise ValueError("persistent train entry forbids --colocate")
    if args.offload_train or args.offload_rollout or args.release_train:
        raise ValueError("persistent train entry forbids offload/release")
    if args.update_weight_transport != "nccl":
        raise ValueError("persistent train entry requires NCCL weight transport")
    if getattr(args, "use_critic", False):
        raise ValueError("persistent v1 entry supports GRPO without a critic only")
    if getattr(args, "eval_interval", None) is not None:
        raise ValueError("this synchronous GRPO run does not schedule evaluation")

    configure_logger()
    placement_groups = create_placement_groups(args)
    _check_node_layout(args, placement_groups)
    init_tracking(args)
    rollout_manager, _ = _create_gateway_rollout_manager(
        args, placement_groups["rollout"]
    )
    actor_model, _ = create_training_models(
        args,
        placement_groups,
        rollout_manager,
        actor_cls=NativeTQTrainRayActor,
    )

    # Initial and subsequent updates are in-memory NCCL tensor publication.
    actor_model.update_weights()
    try:
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            release_training_trajectories(rollout_id)

            final_step = rollout_id == args.num_rollout - 1
            interval_due = bool(
                args.save_interval
                and args.save_interval > 0
                and (rollout_id + 1) % args.save_interval == 0
            )
            if args.save is not None and (final_step or interval_due):
                checkpoints.save(actor_model, rollout_id)
                # Metadata-only source cursor; never a trajectory/batch backup.
                ray.get(rollout_manager.save.remote(rollout_id))

            actor_model.clear_memory()
            if not final_step:
                actor_model.update_weights()
    finally:
        ray.get(rollout_manager.dispose.remote())
        finish_tracking(args)


def _parse_train_args():
    from slime.backends.sglang_utils.arguments import (
        validate_args as upstream_sglang_validate_args,
    )
    from slime.utils import arguments as slime_arguments

    def ensure_sglang_parallel_aliases(args):
        # The pinned parser omits these fields at their neutral default.
        for current, legacy in (
            ("sglang_dp_size", "sglang_data_parallel_size"),
            ("sglang_pp_size", "sglang_pipeline_parallel_size"),
            ("sglang_ep_size", "sglang_expert_parallel_size"),
        ):
            value = getattr(args, current, getattr(args, legacy, 1))
            setattr(args, current, value)
            setattr(args, legacy, value)
        return args

    def validate_with_pinned_sglang_compat(args):
        upstream_sglang_validate_args(ensure_sglang_parallel_aliases(args))

    slime_arguments.sglang_validate_args = validate_with_pinned_sglang_compat
    return slime_arguments.parse_args()


if __name__ == "__main__":
    train(_parse_train_args())
