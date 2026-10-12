# Preparing a frozen TMAX workload

Training consumes prepared inputs. It does not discover tasks, rebuild images or
install Agent dependencies during rollout. Keep the raw data and publication
workspace outside the source repository.

The publication tool accepts a frozen list, a selection manifest and a Harbor
task tree. It adds the pinned Claude runtime, publishes immutable image digests,
and writes the catalog, resource index and prompt JSONL used by training.
It does not download the TMAX dataset or choose the sample list for you.

## Required inputs

Each selected task directory must contain `task.toml`, `instruction.md`,
`tmax-source.json` and its task files. `tmax-source.json` must resolve to the
task's original image through `tools/resolve_harbor_task_source.py`.

`task-ids-1500.txt` contains one task ID per line. `source/selection.json` uses
`schema_version: tmax-selection/v0`, `sample_size: 1500` and an ordered `tasks`
array with each task's `task_id` and `tree_digest`. IDs and order must match the
list. The digest is the deterministic file-tree digest implemented by
`tools/prepare_tmax_parallel.py`; do not substitute an unchecked file listing.

The tool uses the explicit `trusted-security` publication profile. It does not
run gold solutions or verifiers while building images, and records that fact in
the publication manifest. Use this profile only for approved task inputs. Actual
rollouts still run the task's verifier; publication is not a claim that the task
is solvable or that its verifier is correct.

## First publication

Use a dedicated or drained CPU build host with Docker/buildx and the installed
CPU Python environment. Set real Registry repositories and an immutable Node
runtime image before running. Do not run image building concurrently with
training on the same resource-constrained host.

```bash
cd /path/to/distroll
export PREP_ROOT=/path/to/tmax-preparation
export CPU_HEAD_INTERNAL_IP='<Registry host internal IP>'
export NODE_RUNTIME_IMAGE='<repository@sha256:digest>'
.venv-cpu/bin/python tools/prepare_tmax_parallel.py \
  --project-root "$PWD" --validation-profile trusted-security --ack-trusted-security \
  --task-list "$PREP_ROOT/task-ids-1500.txt" \
  --selection-manifest "$PREP_ROOT/source/selection.json" \
  --task-root "$PREP_ROOT/raw/tmax-15k" \
  --output-root "$PREP_ROOT/release" --work-root "$PREP_ROOT/work" \
  --release-id tmax1500-claude21207 --expected-task-count 1500 --jobs 4 \
  --output-repository "$CPU_HEAD_INTERNAL_IP:5000/dressage-distroll/tmax-claude-only" \
  --claude-runtime-repository "$CPU_HEAD_INTERNAL_IP:5000/dressage-distroll/tmax-claude-runtime" \
  --source-mirror-repository "$CPU_HEAD_INTERNAL_IP:5000/dressage-distroll/tmax-source" \
  --node-runtime-image "$NODE_RUNTIME_IMAGE" --claude-code-version 2.1.207
```

Add `--check-inputs-only` for input validation without image building. Add
`--resume` only for an existing publication with exactly matching frozen inputs
and configuration. Model-training attempt retries are unrelated to publication
build retries.

For later publications with a running but drained Control service, the wrapper
`scripts/data/prepare_tmax1500_parallel.sh` also verifies zero allocated CPU
capacity. Set `TMAX1500_ROOT`, `CPU_HEAD_INTERNAL_IP`,
`TMAX1500_NODE_RUNTIME_IMAGE` and `ACK_TRUSTED_SECURITY_TASKS=1` explicitly.
Its Control check deliberately refuses a disconnected service; use the direct
first-publication procedure before a new service has a resource index.

## Install the published inputs

After publication succeeds, review its summary and manifests. The release contains:

- `tasks/`: the Harbor catalog, including fixed task-image references;
- `task-resource-index-v2.json`: resource requirements and immutable execution images;
- `tmax1500.jsonl`: prompt text and stable task references;
- `admission.json` and `fixed-tasks.json`: publication and task identity records.

On both CPU nodes, install the same catalog under `DISTROLL_ROOT/tasks/` and the
same index at `DISTROLL_ROOT/configs/task-resource-index-v2.json`. Retain publication
records with the dataset. On the GPU shared mount, install the JSONL at
`DISTROLL_ROOT/data/tmax1500.jsonl` or set `PROMPT_DATA` to its frozen location.

These artifacts are deployment data and must not be committed to the example.
The Registry remains the persistent image source; Worker image caches can be
reclaimed without rebuilding or discarding the published images.
