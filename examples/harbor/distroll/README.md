# Dressage DistRoll

[Design and results](docs/PROJECT.md) · [中文介绍](docs/PROJECT.zh-CN.md) · [Training runbook](docs/TRAINING.md) · [Dependencies](docs/DEPENDENCIES.md) · [Data preparation](docs/DATA.md)

DistRoll is a self-contained example for running Harbor tasks on a pool of
user-managed CPU servers, independently of a Dressage GPU training cluster.
The GPU environment handles inference and training; CPU Workers run Docker
sandboxes, Claude Code and verifiers. Only the two Heads communicate across clusters.

Dressage already supports remote sandboxes through E2B. This example adds a
service for users who want to manage their own worker capacity, image cache,
task cleanup and result delivery. It does not establish a cost or model-quality
advantage over the E2B path.

The RL side queues candidates and reserves capacity before preparing a model
context. CPU services manage individual executions and never schedule GRPO groups.
Stable request identities, bounded retries, response deduplication and a durable
result outbox handle recoverable transport failures. Native TITO/TQ references
remain valid until all training ranks finish consuming them.

## Layout

| Directory | Purpose |
| --- | --- |
| `contract/` | Lightweight models and HTTP clients shared by CPU and GPU |
| `gpu_adapter/` | Rollout orchestration, Policy, trajectory delivery and synchronous training |
| `rollout_service/` | Control, Workers, Harbor execution, image cache and sandbox protection |
| `configs/`, `deploy/` | Explicit deployment inputs and optional service/container definitions |
| `scripts/`, `tools/` | Runtime preparation, data publication, launch and controlled stopping |
| `docs/` | Design, measured results and operating instructions |

These packages are installed explicitly. The example does not add CPU service
dependencies to Dressage's default installation. The CPU service requires
Python 3.12 and Harbor 0.22; the GPU recipe uses the dependencies and source
revisions in `configs/gpu-runtime-lock.json`. Read the dependency instructions
before launching: the pinned Proxy and TransferQueue still need the scoped
adaptations described there.

## Run

Follow [the runbook](docs/TRAINING.md) in this order:

1. Prepare immutable task images, a task catalog, resource index and prompt file.
2. Configure and start the CPU Head and Workers.
3. Prepare the GPU image and pinned runtime sources on a shared mount.
4. Configure node roles, start Ray and the Policy/Proxy services, then launch training.

All deployment addresses are operator inputs. Direct connectivity must already
exist on a trusted network; this example does not create tunnels, firewall rules
or public authentication. GPU processes must use devices allocated to this job.

The recipe uses 16 prompt groups × 8 slots, up to 32 candidates, 128 concurrent
executions, no failed-attempt retry and default full-group-first selection.
Early stop is optional. The implementation is synchronous; it does not provide
partial-rollout recovery or fully asynchronous training.

## Validation boundary

Historical training results are described separately from the current example's
implementation. A successful local regression is not a GPU training acceptance.
After any runtime or source-layout change, verify a complete optimizer iteration
and both execution and TQ cleanup before a long run.

The six small regression cases cover lost-response retry identity, success quorum,
zero reward variance, multi-segment counting and explicit failed-slot placeholders.
They need no GPU, Docker daemon or remote service. From a Python 3.12 development
environment with `pytest`, `pytest-asyncio`, `httpx`, `pydantic` and `ruff` installed:

```bash
cd examples/harbor/distroll
CHECK_PYTHON=python3.12 RUFF_BIN=ruff bash scripts/check_static.sh
PYTHONPATH="$PWD/contract/src:$PWD/gpu_adapter/src" python3.12 -m pytest -q tests
```

This example is distributed under the parent project's license. Third-party
dependencies and downloaded task data retain their own licenses.
