# Harbor Integration

Harbor-managed agent rollouts and synchronous training through the Dressage Gateway and Proxy.

[Back to the main README](../README.md) · [Harbor examples](../examples/harbor)

## Preliminary Terminal-Bench 2.1 result

We evaluated Qwen3.6-35B-A3B with OpenCode 1.3.13 through the Dressage Harbor integration rollout on **E2B** sandbox. The experiment is conducted on a single node with 8× NVIDIA H200 GPUs (140 GB memory per GPU).

| Item | Value |
|---|---|
| Dataset | Terminal-Bench 2.1, 89 tasks |
| Total attempts | 5 per task, 445 trials total |
| Raw positive verifier-reward rate | 124 / 445 = **27.87%** |

For reference, the public Terminal-Bench 2.0 leaderboard reports [little-coder with Qwen3.6-35B-A3B at **24.6% ± 3.2**](https://www.tbench.ai/leaderboard/terminal-bench/2.0?models=Qwen3.6-35B-A3B). [Terminal-Bench 2.1](https://www.tbench.ai/news/terminal-bench-2-1) revises the same 89-task benchmark by fixing issues in 28 tasks, including external dependency drift, resource mismatches, and instruction/test misspecification. The release reports that most representative agent-model pairs improve on 2.1.

Our raw positive verifier-reward rate of **27.87%** is therefore directionally plausible: it is 3.27 percentage points above 24.6% and still within the leaderboard's displayed 24.6% ± 3.2 range (21.4%–27.8%). The Job was configured to align with the official evaluation semantics where possible: `n_attempts=5`, `max_retries=3`, the original task timeouts, and no timeout multiplier.

## How it works

Harbor owns Dataset resolution, Environments, Agents, Verifiers, trial retries, and rewards. Dressage routes every model request through its Gateway and Proxy, records trainable token-level data, and combines it with Harbor's verifier result. slime can consume the resulting trajectories to update the model.

### Rollout

```text
Harbor Job -> Agent <-> Environment -> Gateway -> Proxy -> SGLang
           -> Verifier reward + trajectory artifacts
```

### Training

```text
Harbor rollout -> trainable trajectory + reward
               -> slime update -> refreshed model
```

The public runner is deliberately named *Harbor rollout*, not *Harbor evaluation* or *Harbor benchmark*. Evaluation normally consumes the final reward or success rate; a rollout also preserves the model-environment interaction as a trainable trajectory.

> By capturing trainable trajectories and verifier rewards—not just benchmark scores—the Harbor rollout path can also serve as a building block for online RL workflows.

This is a capability statement, not a claim that the integration continuously learns from live production traffic.

## Requirements

All commands below assume the default Dressage image, whose working directory is `/root/Dressage` and whose Python version is already 3.12.

- `harbor==0.18.0` and `dressage-blackbox-server==1.1.0`, installed by the Harbor extra.
- For E2B jobs: a valid `E2B_API_KEY`.
- For bwrap jobs: Linux with `bwrap` and the local Agent dependencies.

```bash
cd /root/Dressage
python -m pip install -e '.[harbor]'
harbor plugins list
```

`harbor plugins list` must include the `dressage` plugin. See the [slime quick start](../slime/docs/en/get_started/quick_start.md) for training environment and checkpoint preparation.

## Dressage Integration Config

`DRESSAGE_HARBOR_INTEGRATION_CONFIG` selects how Dressage integrates with a Harbor Job. The repository provides six profiles:

| Profile | Environment | Routing | Use |
|---|---|---|---|
| `rollout-native-local.yaml` | native | `configure_only` | Local rollout |
| `rollout-native-remote.yaml` | native | `configure_only` | Remote/E2B rollout |
| `rollout-bwrap.yaml` | bwrap | `enforced` | Isolated local rollout |
| `training-native-local.yaml` | native | `configure_only` | Local training |
| `training-native-remote.yaml` | native | `configure_only` | Remote/E2B training |
| `training-bwrap.yaml` | bwrap | `enforced` | Isolated local training |

The top-level Integration Config modules have the following roles:

| Module | Role |
|---|---|
| `schema_version` | Validated Dressage Harbor schema. |
| `execution_mode` | Direct `rollout` or slime `training`. |
| `environment` | Harbor-native provider or local bwrap. |
| `gateway` | Listener, advertised Agent URL, logging, and limits. |
| `backend` | Dressage Proxy routing, credentials, and TLS. |
| `security` | Routing guarantee, TLS, model listing, and egress. |
| `trajectory` | Agent steps, sampling temperature, and token requirements. |
| `artifacts` | Storage mode, location, durability, and permissions. |
| `agent_protocol_overrides` | Per-Agent OpenAI/Anthropic protocol override. |
| `training` | Reward, sampling, failure, and weight-version rules. |

Within `gateway`, `listen_host` and `listen_port` are the local bind address. `advertise_url` is the address injected into the Harbor Agent. The latter must be reachable from the Agent; changing it does not create a listener, TLS certificate, reverse proxy, firewall rule, or tunnel.

`security.routing_guarantee: configure_only` configures authenticated routing but permits public-network tasks with a warning. `enforced` rejects incompatible public Agent network policies before issuing route credentials. bwrap profiles always use `enforced`.

`trajectory` determines whether the captured rollout is trainable. `training` is required only by training profiles and controls how slime accepts those trajectories.

## Remote Gateway setup

> [!WARNING]
> `https://replace-me.invalid` is an intentionally invalid placeholder. Before using either native-remote profile, manually replace `gateway.advertise_url` with the real public HTTPS address and port reachable from the Harbor Agent.

Rollout and training use separate native-remote profiles; update whichever profile you select. You may edit the example or maintain your own copy. Dressage does not prescribe or automate that choice.

`advertise_url` only tells the Agent where to connect. It does not create DNS, TLS, a reverse proxy, port forwarding, or a tunnel. You must make that public endpoint forward to the Dressage Gateway on port `39100`. Complete this setup before starting an E2B Job.

Only the Gateway should be exposed. Do not expose the Dressage Proxy on `8800`, the SGLang routers on `30000`/`8000`, or an SGLang worker directly. The `host.docker.internal` address in native-local profiles is for local containers and is not reachable from E2B.

## DAPO

DAPO is the one bundled Dataset that requires an explicit conversion step. The converter validates the source and writes a content-addressed Harbor Dataset and JobConfig outside the repository.

### Rollout

```bash
export DRESSAGE_HARBOR_JOB_CONFIG="$(
  python examples/harbor/dataset_tools/dapo/prepare_dataset.py \
    --input examples/data/dressage_dapo_prompts.jsonl \
    --cache-root /root/dressage-harbor/datasets \
    --limit all
)"
export DRESSAGE_HARBOR_INTEGRATION_CONFIG=/root/Dressage/examples/harbor/dressage_profiles/rollout-bwrap.yaml
examples/harbor/run_harbor_rollout_qwen3.5_4b.sh
```

### Training

```bash
export DRESSAGE_HARBOR_JOB_CONFIG="$(
  python examples/harbor/dataset_tools/dapo/prepare_dataset.py \
    --input examples/data/dressage_dapo_prompts.jsonl \
    --cache-root /root/dressage-harbor/datasets \
    --limit all
)"
export DRESSAGE_HARBOR_INTEGRATION_CONFIG=/root/Dressage/examples/harbor/dressage_profiles/training-bwrap.yaml
examples/harbor/run_harbor_training_qwen3.5_4b.sh
```

## Terminal-Bench 2

Terminal-Bench uses the official Harbor Registry Dataset directly. No local Dataset preparation is required.

### Rollout

```bash
export E2B_API_KEY='<your-e2b-api-key>'
export DRESSAGE_HARBOR_JOB_CONFIG=/root/Dressage/examples/harbor/harbor_job_configs/terminal-bench-2-e2b.yaml
export DRESSAGE_HARBOR_INTEGRATION_CONFIG=/root/Dressage/examples/harbor/dressage_profiles/rollout-native-remote.yaml
examples/harbor/run_harbor_rollout_qwen3.5_4b.sh
```

### Training

```bash
export E2B_API_KEY='<your-e2b-api-key>'
export DRESSAGE_HARBOR_JOB_CONFIG=/root/Dressage/examples/harbor/harbor_job_configs/terminal-bench-2-e2b.yaml
export DRESSAGE_HARBOR_INTEGRATION_CONFIG=/root/Dressage/examples/harbor/dressage_profiles/training-native-remote.yaml
examples/harbor/run_harbor_training_qwen3.5_4b.sh
```

## τ³-bench

τ³-bench also uses its official Harbor Registry Dataset and the same runners.

### Rollout

```bash
export E2B_API_KEY='<your-e2b-api-key>'
export DRESSAGE_HARBOR_JOB_CONFIG=/root/Dressage/examples/harbor/harbor_job_configs/tau3-bench-e2b.yaml
export DRESSAGE_HARBOR_INTEGRATION_CONFIG=/root/Dressage/examples/harbor/dressage_profiles/rollout-native-remote.yaml
examples/harbor/run_harbor_rollout_qwen3.5_4b.sh
```

### Training

```bash
export E2B_API_KEY='<your-e2b-api-key>'
export DRESSAGE_HARBOR_JOB_CONFIG=/root/Dressage/examples/harbor/harbor_job_configs/tau3-bench-e2b.yaml
export DRESSAGE_HARBOR_INTEGRATION_CONFIG=/root/Dressage/examples/harbor/dressage_profiles/training-native-remote.yaml
examples/harbor/run_harbor_training_qwen3.5_4b.sh
```

The committed Terminal-Bench and τ³-bench JobConfigs use `n_tasks: 5` as a smoke/example size. For a full run, make a private JobConfig copy and remove `n_tasks` or set the desired task count. Do not edit the committed example for a one-off run.

## Official Harbor references

The committed Harbor JobConfigs use the standard Harbor schema, so this guide does not duplicate its field reference:

- [Harbor Core Concepts and JobConfig][harbor-concepts]
- [Running Harbor Datasets][harbor-datasets]
- [Terminal-Bench 2 on Harbor Hub][terminal-bench]
- [τ³-bench on Harbor Hub][tau3-bench]

[harbor-concepts]: https://www.harborframework.com/docs/core-concepts
[harbor-datasets]: https://www.harborframework.com/docs/run-jobs/run-evals
[terminal-bench]: https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2
[tau3-bench]: https://hub.harborframework.com/datasets/sierra-research/tau3-bench
