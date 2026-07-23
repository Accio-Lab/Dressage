"""Slime train loop using its native ``actor_cls`` factory hook for MOPD.

The loop intentionally mirrors upstream ``slime/train.py``. The only semantic
delta is passing ``MOPDMegatronTrainRayActor`` to
``create_training_models(..., actor_cls=...)``. No Slime module is patched.
"""

from __future__ import annotations

import ray

from slime.ray.placement_group import (
    create_placement_groups,
    create_rollout_manager,
    create_training_models,
)
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action

from dressage.training.mopd_megatron_actor import MOPDMegatronTrainRayActor


def train(args) -> None:
    configure_logger()
    release_train = args.release_train

    pgs = create_placement_groups(args)
    init_tracking(args)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(
        args, pgs["rollout"]
    )
    actor_model, critic_model = create_training_models(
        args,
        pgs,
        rollout_manager,
        actor_cls=MOPDMegatronTrainRayActor,
    )

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())
    actor_model.update_weights()
    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))
    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if (
            args.eval_interval is not None
            and rollout_id == 0
            and not args.skip_eval_before_train
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())
        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(
                    actor_model.async_train(
                        rollout_id,
                        rollout_data_ref,
                        external_data=value_refs,
                    )
                )
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if release_train or should_run_periodic_action(
            rollout_id,
            args.save_interval,
            num_rollout_per_epoch,
            args.num_rollout,
        ):
            force_sync = release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        offload_train(actor_trains)
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())
        if should_run_periodic_action(
            rollout_id,
            args.eval_interval,
            num_rollout_per_epoch,
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
