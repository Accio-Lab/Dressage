# Metadata-routed multi-teacher OPD

Dressage MOPD trains one Megatron student from multiple frozen Megatron
teachers on the same actor GPUs. Every teacher and the student must have the
same architecture, tokenizer, vocabulary, and token IDs.

Teachers are not separate services. During actor initialization, each teacher
checkpoint is loaded once and copied into Slime's existing pinned-CPU
`TensorBackuper`. For every local training batch the actor groups samples by
`teacher_id`, restores one teacher onto the shared GPU model buffers, scores
that teacher's subset, and then restores the student before training. GPU model
memory is reused; CPU memory grows with the number of teachers.

## Native Slime boundaries

The implementation uses:

- `Sample.generate_function_path` for per-dataset rollout dispatch;
- Dressage's existing custom train-data converter to place `teacher_id` in
  Slime's native train-side `prompt` passthrough field;
- upstream `create_training_models(..., actor_cls=...)` to install the
  Dressage-owned rotating actor without monkey-patching a Slime module;
- Slime's `TensorBackuper`, checkpoint loader, `compute_log_prob`, DP
  partitioning, and OPD loss.

The small `dressage.training.mopd_train` driver mirrors upstream `train.py`
because upstream exposes `actor_cls` at the model factory but not yet on the
stock CLI. Compatibility tests detect drift in this factory contract.

There is no MOPD code under `dressage/rollout/generate` and no Slime source
patch.

## Configuration

Start from
`examples/data/mopd/mopd_alfworld_hotpotqa.example.json`. Important fields:

- `teachers.<id>.load`: frozen Megatron checkpoint root;
- `teachers.<id>.ckpt_step`: optional checkpoint iteration;
- `datasets[].teacher_id`: authoritative teacher for that dataset;
- `datasets[].weight`: smooth weighted-round-robin sampling weight;
- `datasets[].agent_mode`: `blackbox` or `whitebox`;
- `datasets[].generate_function_path`: required for whitebox data and optional
  for a specialized blackbox implementation;
- `reward_modules`: task reward registration modules;
- `runtime_env_keys`: task-specific environment variables copied to Ray.

There is no domain router or default teacher. The data source writes direct
`metadata["teacher_id"]` and the native `Sample.generate_function_path`.
Conversion validates the route and fails before training if multi-segment
siblings disagree.

## Per-step execution

For every DP-local batch:

1. Decode one teacher ID per sample from the native train-data passthrough.
2. Build compact dynamic microbatches for the first distinct teacher.
3. Restore that teacher from pinned CPU memory to the shared model buffers.
4. Compute response-token log-probabilities only for its routed samples.
5. Repeat for other distinct teachers and scatter results to original order.
6. Restore the student/old actor.
7. Let stock Slime compute student log-probs, OPD advantages, backward, and the
   optimizer step.

The launcher intentionally uses:

```text
--use-opd --opd-type megatron --opd-teacher-load <first-teacher>
```

The first teacher path satisfies Slime's stock argument validation and tells
the actor factory that an OPD teacher is needed. `MOPDMegatronTrainRayActor`
suppresses the stock single-teacher load and loads all configured named
teachers instead.

## Launch

```bash
export DRESSAGE_MOPD_TEACHER_CONFIG=/path/to/mopd.json

TP_SIZE=4 \
CP_SIZE=1 \
ROLLOUT_BATCH_SIZE=16 \
N_SAMPLES_PER_PROMPT=8 \
GLOBAL_BATCH_SIZE=128 \
bash examples/scripts/run_mopd_qwen3.5_sync.sh
```

No teacher process is started separately. Checkpoint paths are validated by
the launcher, and all teacher loading happens inside the student actor group.
