# Synchronous GRPO training runbook

This recipe uses two eight-device actor nodes and one eight-device inference node,
plus a CPU Head and two CPU Workers. The CPU Head may also host a Worker.
Addresses, model files, data and output storage are deployment inputs.

Commands are run manually on the named role. Use allocated GPU devices and
project-owned services only. The example does not configure host networking.
CPU service configuration requires a root shell or explicit sudo; GPU commands
run inside the allocated project container.
The current source-layout changes need a real single-iteration acceptance before
a long run; the historical curves are not that acceptance.

## 1. Prerequisites and paths

- GPU nodes share the same Dressage checkout, model mount, runtime sources and output paths.
- CPU nodes provide Docker Compose/buildx, cgroup v2 memory/PID accounting and the configured per-container storage quota support. Install these prerequisites separately; do not change a working Docker daemon from the recipe.
- The two Heads can reach each other on a trusted network. Control and Policy do not provide a public multi-tenant authentication layer.
- CPU Workers reach the CPU Head over their internal network. GPU ranks and inference engines also use their cluster's internal addresses.

Use three distinct roots:

```text
DRESSAGE_ROOT  = outer official checkout
DISTROLL_ROOT  = DRESSAGE_ROOT/examples/harbor/distroll
GPU_SOURCE_ROOT / PROJECT_ROOT = DISTROLL_ROOT
```

The container layout supplied by the scripts is `/workspace/Dressage`, with
the example below `examples/harbor/distroll`. Models mount at `/models`.
`outputs/` contains checkpoints, TensorBoard and control state; preserve it on
durable storage. Ray's temporary directory stays on local storage.

## 2. Frozen inputs

Prepare these before starting Workers:

| Input | Installation location |
| --- | --- |
| Harbor task catalog | `DISTROLL_ROOT/tasks/` on both CPU nodes |
| Resource index | `DISTROLL_ROOT/configs/task-resource-index-v2.json` on both CPU nodes |
| Prompt JSONL | `DISTROLL_ROOT/data/tmax1500.jsonl` on the GPU shared mount |
| Task images | Accessible Registry, bound by immutable digest |
| Initial HF model | `/models/Qwen3.5-9B` in GPU containers |
| Initial Megatron weights | `DISTROLL_ROOT/outputs/models/Qwen3.5-9B_torch_dist` |

See [data preparation](DATA.md). The rollout service never builds task images or
installs Claude during training. Both Workers must use the same inventory,
catalog and resource-index contents. The Registry must already be trusted by
their Docker installations; this runbook does not rewrite Docker configuration.

## 3. CPU installation and startup

Copy this example to each CPU node; the outer Dressage checkout is not required
there. Run the installer from that local copy after installing `uv`, Docker and
Python prerequisites:

```bash
cd /path/to/distroll
sudo bash scripts/cpu/install_local.sh
```

It installs the Python packages, project units and `dressage-distroll-rollout.slice`. It creates
missing configuration files under `/etc/dressage-distroll/`, preserving existing
files, and does not start the services. It resolves the catalog/index path
placeholders from this copy's actual location.

All CPU units, including the optional dashboard, use the installed `.venv-cpu`
entry points. The supplied units use the root-owned configuration model; the
dashboard retains its read-only filesystem restrictions and resource limits.

Fill in:

| File | Required values |
| --- | --- |
| `inventory.yaml` | CPU Head internal Policy URL; each Worker's internal address/relay URL; verify catalog and index locations |
| `control.env` on Head | Control bind address: the Head's actual internal IP |
| `worker.env` on each Worker | Its `cpu-01` or `cpu-02` ID, internal bind address, Head internal Control URL |
| `network-gateway.yaml` on Head | Internal listener and the reachable GPU Head Policy address |
| `dashboard.yaml`, optional | Internal dashboard bind address, Control and observer URLs |

Public/reachable addresses and local bind addresses are different inputs. Do not
bind to a NAT address that is not assigned to the host. All blank addresses are
intentional placeholders, not defaults. Render identical inventories for both
Workers; if their source paths differ, use consistent deployed catalog paths.

For an existing installation, stop new submissions and reconcile its exact
executions and retained TQ first. On the CPU Head:

```bash
set -a; source /etc/dressage-distroll/control.env; set +a
docker compose -f deploy/docker-compose.control.yaml up -d --wait
.venv-cpu/bin/python tools/check_execution_upgrade.py
sudo systemctl start dressage-distroll-control.service dressage-distroll-network-gateway.service
```

On each Worker:

```bash
sudo systemctl start dressage-distroll-image-cleaner.service dressage-distroll-worker.service
sudo systemctl is-active dressage-distroll-image-cleaner.service dressage-distroll-worker.service
```

Check `GET /v4/capacity` on Control: both Workers must be READY with matching
inventory/index, no old allocated charge, and sufficient disk headroom. Worker
Policy connectivity can be checked after GPU services are ready with
`scripts/cpu/check_worker_network_local.sh`.

The optional Registry launcher requires explicit inputs and preserves an existing
matching container:

```bash
sudo env DRESSAGE_REGISTRY_LISTEN_IP='<CPU Head internal IP>' \
  DRESSAGE_REGISTRY_DATA_ROOT='/path/to/persistent/registry' \
  bash scripts/cpu/start_registry.sh
```

Do not remove its persistent storage when draining training. To monitor that
filesystem, set `DRESSAGE_REGISTRY_DATA_ROOT` in the Head observer's environment.

## 4. GPU image and runtime sources

Before allocating the runtime on each GPU node, inspect utilization and existing
compute processes. If another workload is using the requested devices, stop the
setup and arrange an allocation; do not terminate that workload:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

Read [dependencies](DEPENDENCIES.md). Build the small derivative of the pinned
official image on an approved build machine, then publish an immutable RepoDigest
or distribute the same saved image to all nodes:

```bash
cd /path/to/Dressage/examples/harbor/distroll
bash scripts/gpu/build_runtime_image.sh
GPU_RUNTIME_PUSH_REF='<registry/repository:tag>' bash scripts/gpu/publish_runtime_image.sh
```

For a locally loaded image, `ALLOW_LOCAL_GPU_IMAGE=1` permits a local image name
at container creation; compare its image ID on every node. Otherwise use the
published `repository@sha256:...` as `GPU_RUNTIME_IMAGE`.

Prepare pinned runtime sources once on the shared mount. The temporary container
below does not request GPUs and runs only the source-extraction script:

```bash
export GPU_RUNTIME_IMAGE='<published immutable image>'
export DRESSAGE_ROOT='/path/to/Dressage'
docker run --rm --network none --entrypoint bash \
  --mount "type=bind,src=$DRESSAGE_ROOT,dst=/workspace/Dressage" \
  "$GPU_RUNTIME_IMAGE" \
  /workspace/Dressage/examples/harbor/distroll/scripts/gpu/bootstrap_runtime_sources.sh
```

It reads pinned objects from the image's `/root/slime` and `/root/Megatron-LM`.
If those objects are unavailable, provide reviewed repositories through
`SLIME_SOURCE_ROOT` and `MEGATRON_SOURCE_ROOT`; do not substitute their HEADs.

Create `data/`, `outputs/` and a GPU configuration directory on the shared mount.
Place a copy of `configs/gpu.training.env.example` in that directory as `gpu.env`
and fill in all five node-address fields: `GPU_HEAD_PUBLIC_IP`,
`CPU_HEAD_PUBLIC_IP`, `GPU_NODE_IP`, `DRESSAGE_ACTOR_NODE_IPS` and
`DRESSAGE_ROLLOUT_NODE_IP`. The first two are reachable endpoint addresses; the
last three specify internal node roles. Input blanks must be replaced, not deleted.

On each GPU node, from its example directory:

```bash
export GPU_RUNTIME_IMAGE='<same published immutable image>'
bash scripts/gpu/prepull_runtime_image.sh
export GPU_MODEL_DIR='/path/to/models'
export GPU_CONFIG_DIR='/path/to/shared/gpu-config'
export GPU_DATA_DIR="$PWD/data"
export GPU_OUTPUT_DIR="$PWD/outputs"
bash scripts/gpu/start_runtime_container.sh
```

The container mounts official and example source read-only, and data read-only;
only outputs are writable. Existing containers are not replaced automatically.

Inside the GPU Head container, prepare the two required private runtime entries:

```bash
cd /workspace/Dressage/examples/harbor/distroll
export GPU_ENV_FILE=/etc/dressage-distroll/gpu.env
source scripts/common.sh
source scripts/gpu/runtime.sh
export TQ_SOURCE_ROOT="$PWD/outputs/runtime-deps/tq-linger-zero"
TQ_PACKAGE=$("$GPU_PYTHON" -c 'import transfer_queue; from pathlib import Path; print(Path(transfer_queue.__file__).parent)')
"$GPU_PYTHON" tools/prepare_tq_linger_overlay.py --source "$TQ_PACKAGE" --target "$TQ_SOURCE_ROOT"
"$GPU_PYTHON" tools/prepare_tito_proxy_overlay.py \
  --source "$DRESSAGE_ROOT/dressage/proxy/server.py" \
  --target "$PWD/outputs/runtime-deps/tito-production/server.py"
```

Prepare these before starting new TQ readers. On subsequent shells, export the
same `GPU_ENV_FILE` and `TQ_SOURCE_ROOT` before any runtime script. If an existing
output differs, use a new reviewed path; do not overwrite a live dependency.

If needed, convert the HF model once, inside an allocated idle GPU node:

```bash
HF_CHECKPOINT=/models/Qwen3.5-9B bash scripts/gpu/convert_qwen3_5_9b_to_megatron.sh
```

The converter refuses to overwrite existing output. It consumes GPU resources;
do not run it alongside unrelated workloads.

## 5. Ray, services and training

On GPU Head, inside its container:

```bash
cd /workspace/Dressage/examples/harbor/distroll
export GPU_ENV_FILE=/etc/dressage-distroll/gpu.env
export TQ_SOURCE_ROOT="$PWD/outputs/runtime-deps/tq-linger-zero"
bash scripts/gpu/start_ray_head.sh
```

On the other training node and the inference node, use their actual internal IPs:

```bash
cd /workspace/Dressage/examples/harbor/distroll
export TQ_SOURCE_ROOT="$PWD/outputs/runtime-deps/tq-linger-zero"
GPU_NODE_IP='<this node internal IP>' RAY_HEAD_INTERNAL_IP='<Head internal IP>' \
  bash scripts/gpu/start_ray_worker.sh
```

On Head, with the same environment:

```bash
bash scripts/gpu/doctor_runtime_cluster.sh
bash scripts/gpu/start_services.sh
bash scripts/gpu/check_cross_cluster.sh
export RUN_NAME="distroll-grpo-$(date -u +%Y%m%dT%H%M%SZ)"
export REUSE_RAY=1
bash scripts/gpu/start_training.sh
```

For a first acceptance, set `NUM_ROLLOUT=1` before `start_training.sh`. A complete
acceptance includes real Agent traffic, a completed optimizer update, finite
loss/grad norm, all training ranks returned, and source TQ keys released.
Ray RUNNING alone does not establish success.

That single-step setting applies to a fresh run. `NUM_ROLLOUT` is the exclusive
upper bound of the absolute rollout index, not the number of additional steps
after a checkpoint. To test one update after checkpoint iteration `k`, use
`NUM_ROLLOUT=k+2`; reset it to the intended training limit before a longer resume.

Do not reuse a run ID or a populated checkpoint output directory. The normal
profile runs up to 1000 rollout/update steps and can be stopped cooperatively.

## 6. Recipe parameters

| Category | Values |
| --- | --- |
| Model/data | Qwen3.5-9B; 1500 frozen TMAX tasks |
| Training | Synchronous, non-colocated; 16 prompt groups × 8 slots; up to 32 candidates |
| Selection | Full groups first; partial groups need 6/8 successes and nonzero reward variance; fewer qualified groups yield a smaller update; zero qualified groups stop |
| Concurrency | RL limit 128; two Workers with 64 CPU units and 196608 MiB virtual capacity each; actual admission uses both CPU and memory charges |
| Retry/early stop | Failed-attempt retry 0; transport retries remain enabled; early stop off unless explicitly enabled |
| Agent | Claude Code 2.1.207; 40 turns/1200 s; only explicit max-turns exit is accepted for subsequent verification |
| Context | 65472 tokens; response 8192; autocompact 80%; Read output limit 1024 |
| Actor parallelism | Two eight-device nodes; TP4 × CP2 × PP1 × DP2; sequence parallel; full recompute; optimizer CPU offload |
| Inference | Eight single-device SGLang engines, concurrency 16 each; static memory fraction 0.55; CUDA graphs disabled |
| Loss/optimizer | GRPO with TIS; clipping 0.2/0.28; per-token loss; no second advantage normalization; LR 1e-6 constant; KL and entropy coefficients 0; Adam β=(0.9,0.98), weight decay 0.1 |
| Token packing | 32736 tokens per GPU × CP2 = 65472; do not lower below the context requirement |
| Reporting | TensorBoard, Ray/service logs and CPU trial/verifier artifacts; no full training-trajectory export; no evaluation schedule |
| Checkpoints | Every 10 steps and the normal last step; at most five directories per run; native optimizer/RNG and matching sampler cursor |
| Image cache | Scan 60 s; idle TTL 3600 s; pressure minimum idle 300 s; high/low watermarks 75/70%; at most 50 deletions; pull floor 50 GiB |
| Pulls | Two concurrent pulls per Worker, 900 s timeout, at most three attempts |
| Sandbox protection | Enforce; PID limit 2048, configured storage 10240 MiB; repeated OOM kills plus memory pressure trigger exact cleanup |
| Execution deadline | Fixed 3900 s from claim, plus 3900 s settlement budget; Worker heartbeat 30 s, timeout 90 s |

Virtual capacity is not additional physical memory. Size it for actual task
working sets and peaks. Safety controls and cleanup checks remain active when
failed-attempt retries are disabled.

## 7. Logs, stop and resume

Use `scripts/gpu/status_training.sh` to inspect Ray status. Durable Ray output
is under `outputs/logs/$RUN_NAME/`; TensorBoard is under
`outputs/tensorboard/$RUN_NAME/`. The job's control metadata lives under
`outputs/rl-state/$RUN_NAME/rollout/`. CPU trials and resource-guard records are
stored in the configured Worker state directory. These records are not a
recoverable partial training batch.

For the exact active, uncommitted step:

```bash
RUN_NAME='<active run>' STEP='<active step>' bash scripts/gpu/stop_training.sh
```

Wait for CPU execution/ACK/outbox cleanup and GPU generation/TQ release before
stopping project services. A hung cleanup gate requires investigation; do not
mark it complete manually or force-stop a whole Ray cluster to bypass it.

Resume only from a complete native checkpoint with its `source_cursor.json`:

```bash
export TRAIN_RESUME_FROM="$PWD/outputs/checkpoints/<previous-run>"
export RUN_NAME="distroll-resume-$(date -u +%Y%m%dT%H%M%SZ)"
export REUSE_RAY=1
bash scripts/gpu/start_training.sh
```

Resume restores model, optimizer, scheduler, RNG and the frozen-input cursor.
It uses a new output directory and does not replay partial trajectories or an
ambiguous optimizer update. Keep old failed-run readers stopped and ownership
settled before starting a new run.
