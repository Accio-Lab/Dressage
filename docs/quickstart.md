# Quick Start Guide

**Get Dressage Up and Running in Minutes**

[← Back to Main README](../README.md) · [Prerequisites](#-prerequisites) · [Installation](#1️⃣-installation) · [Verify](#2️⃣-verify-installation) · [Agent Modes](#3️⃣-choose-your-agent-mode) · [First Training](#4️⃣-run-your-first-training) · [Configuration](#️-configuration-reference) · [Troubleshooting](#-troubleshooting)

## 📖 Overview

This guide walks you through installing Dressage, configuring your environment, and running your first agentic RL training job. By the end, you'll have a working training pipeline — either a whitebox agent (Python tool loop) or a blackbox agent (HTTP agent like `opencode`) — producing real RL gradients from agent trajectories.

Dressage is designed to get you from zero to training with minimal configuration. The framework uses environment variables and dotted Python import paths for all slime hooks, so there's no config file authoring required.

## 📋 Prerequisites

 | Requirement | Version | Purpose | Notes |
 | :------------ | :-------- | :-------- | :------ |
 | **Python** | ≥ 3.10 | Runtime | 3.11+ recommended for performance |
 | **pip** | Latest | Package installation | `pip install --upgrade pip` |
 | **Git** | Any | Clone with submodules | Needs `--recurse-submodules` support |
 | **SGLang** | Compatible with slime | Inference router | Must be running before training starts |
 | **GPU** | CUDA-capable | Model inference and training | At least 1 GPU for inference, more for training |

<details>
<summary><b> Optional Requirements</b></summary>
<br>

 | Requirement | Purpose | When Needed |
 | :------------ | :-------- | :------------ |
 | **bubblewrap** | Local sandbox isolation via Linux namespaces | `DRESSAGE_SANDBOX_PROVIDER=local_bwrap` |
 | **opencode** | Blackbox code-editing agent | `DRESSAGE_BLACKBOX_TYPE=opencode` |
 | **openclaw** | Blackbox OpenClaw agent | `DRESSAGE_BLACKBOX_TYPE=openclaw` |
 | **E2B account** | Remote cloud sandboxes | `DRESSAGE_SANDBOX_PROVIDER=e2b` |
 | **Docker** | Reproducible local environment | Optional — for containerized setup |
 | **faiss-cpu** | FAISS similarity search | HotpotQA recipe only |
 | **sentence-transformers** | BGE embeddings | HotpotQA recipe only |

</details>

## 1️⃣ Installation

### Use the slime Base Image

```bash
slimerl/slime:nightly-dev-20260430b
```

### Clone the Repository

```bash
git clone --recurse-submodules https://github.com/Accio-Lab/Dressage.git
cd Dressage
```

> [!TIP]
> The `--recurse-submodules` flag automatically pulls the [slime](https://github.com/THUDM/slime) submodule. If you cloned without it, run `git submodule update --init --recursive` afterwards.

### Install Dressage

```bash
# Install Dressage (core package)
pip install --no-build-isolation -e .

# Install BlackboxServer if you use blackbox agents
pip install --no-build-isolation blackbox_server/

# Install slime (upstream RL framework)
pip install --no-build-isolation -e slime/
```

### Install Optional Dependencies

```bash
# 🧪 For running tests
pip install -e ".[test]"

# 🔍 For HotpotQA recipe (retrieval dependencies)
pip install faiss-cpu sentence-transformers numpy
```

> [!NOTE]
> The root package currently defines only the `test` extra. Ray and E2B support are installed as core dependencies; BlackboxServer is installed from `blackbox_server/` without editable mode because local bwrap sandboxes do not mount the BlackboxServer source directory.

## 2️⃣ Verify Installation

After installation, verify that the CLI entry points and test suite work:

```bash
# ✅ Check CLI entry points
dressage-proxy --help          # Proxy server
blackbox-server --help         # BlackboxServer, if blackbox_server/ was installed

# ✅ Check bwrap and local blackbox CLIs
dressage-local-bwrap-start --help
dressage-local-bwrap-status --help
dressage-local-bwrap-stop --help
dressage-local-blackbox-start --help
dressage-local-blackbox-status --help
dressage-local-blackbox-stop --help
dressage-blackbox-start --help
dressage-blackbox-status --help
dressage-blackbox-stop --help

# ✅ Run tests
pytest tests/ -x -q
```

> [!NOTE]
> If `dressage-proxy` is not found, ensure that your Python environment's `bin/` directory is in your `PATH`. For virtual environments: `source venv/bin/activate`.

To start the proxy manually, provide a tokenizer path:

```bash
dressage-proxy \
  --tokenizer-path /path/to/Qwen3.5-4B \
  --sglang-router-url http://<router-host>:8000
```

> [!IMPORTANT]
> For blackbox rollouts, `DRESSAGE_PROXY_URL` must be reachable from the sandbox. Avoid local-only URLs such as `http://localhost:8800` unless the sandbox shares the same network namespace; set `PROXY_PUBLIC_HOST` or `DRESSAGE_PROXY_URL` to a sandbox-reachable host.

## 3️⃣ Choose Your Agent Mode

Dressage supports two agent paradigms. Choose based on your use case:

### Whitebox — Python Tool Agents

**Best for**: custom tools, retrieval, API calls, lightweight environments. You write the agent logic in Python.

```text
Your Python Agent
  └── self.chat(messages)          → Proxy → SGLang
  └── self.paddock.tool_call(...)  → Sandbox (shell, files)
```

**Advantages:**
- Full control over agent prompting and tool logic
- Easier to debug (Python stack traces)
- Lower latency (no HTTP agent subprocess)
- Direct access to LLM responses for custom processing

**Get started:**

```bash
# ALFWorld recipe (TextWorld navigation)
bash examples/scripts/run_alfworld_whitebox_agent_qwen3.5_4b.sh

# HotpotQA recipe (multi-hop retrieval)
bash examples/scripts/run_hotpotqa_whitebox_agent_qwen3.5_4b.sh
```

### Blackbox — HTTP Agent Rollouts

**Best for**: real-world coding agents, complex environments, production agent frameworks like opencode/openclaw.

```text
BlackboxServer (in sandbox)
  └── Backend Agent (opencode/openclaw)
      └── Agent's LLM calls → In-process Proxy → Dressage Proxy → SGLang
      └── Agent's tools → execution inside sandbox
```

**Advantages:**
- Use real-world agent frameworks without modification
- Agent complexity is handled by the backend (no Python coding needed)
- Full sandbox isolation (agent can't affect the host)
- Closer to production behavior

**Get started:**

```bash
# Local bubblewrap sandbox
bash examples/scripts/run_blackbox_qwen3.5_4b_async_local.sh

# E2B remote sandbox
bash examples/scripts/run_blackbox_qwen3.5_4b_async_remote.sh
```

## 4️⃣ Run Your First Training

### Option A: Whitebox Agent (ALFWorld)

The simplest way to start — no sandbox infrastructure needed:

```bash
# 1️⃣ Set agent mode
export DRESSAGE_PADDOCK_MODE=whitebox
export DRESSAGE_SANDBOX_PROVIDER=local_bwrap
export DRESSAGE_LOCAL_BWRAP_POOL_MODE=command_only

# 2️⃣ Run training
# The example script sources examples/scripts/default/dressage_env_defaults.sh
# and applies its helper defaults internally.
bash examples/scripts/run_alfworld_whitebox_agent_qwen3.5_4b.sh
```

### Option B: Blackbox Agent (Local Bubblewrap)

For real-world coding agents with full sandbox isolation:

```bash
# 1️⃣ Set agent mode
export DRESSAGE_PADDOCK_MODE=blackbox
export DRESSAGE_SANDBOX_PROVIDER=local_bwrap
export DRESSAGE_LOCAL_BWRAP_POOL_MODE=blackbox
export DRESSAGE_BLACKBOX_TYPE=opencode

# 2️⃣ Start the bubblewrap sandbox pool
dressage-local-bwrap-start

# 3️⃣ Verify pool is healthy
dressage-local-bwrap-status

# 4️⃣ Run training
# The example script sources examples/scripts/default/dressage_env_defaults.sh
# and applies its helper defaults internally.
bash examples/scripts/run_blackbox_qwen3.5_4b_async_local.sh

# 5️⃣ Stop pool when done
dressage-local-bwrap-stop
```

### Option C: Docker (All-in-One)

For a reproducible environment with all dependencies pre-installed:

```bash
# Build the Docker image
docker/build.sh

# Run with GPU access
docker/run.sh
# Starts with: --gpus all --network host --ipc host --privileged
```

> [!NOTE]
> `--privileged` is required for bubblewrap inside Docker containers. The image includes bubblewrap, opencode, openclaw, Dressage, BlackboxServer, and all dependencies. See [docker/README.md](../docker/README.md) for details.

## ⚙️ Configuration Reference

All Dressage configuration is via environment variables. Here's the complete reference organized by component:

<details>
<summary><b> Agent & Paddock</b></summary>
<br>

 | Variable | Values | Default | Description |
 | :--------- | :------- | :-------- | :------------ |
 | `DRESSAGE_PADDOCK_MODE` | `blackbox` \| `whitebox` | `blackbox` | Agent interaction paradigm. Determines paddock subclass. |
 | `DRESSAGE_PADDOCK_CLASS` | `module.Class` | — | Custom paddock class override for specialized lifecycle logic. |

</details>

<details>
<summary><b> Sandbox</b></summary>
<br>

 | Variable | Values | Default | Description |
 | :--------- | :------- | :-------- | :------------ |
 | `DRESSAGE_SANDBOX_PROVIDER` | `local_bwrap` \| `e2b` | `local_bwrap` | Sandbox backend. Determines where agent code runs. |
 | `DRESSAGE_LOCAL_BWRAP_POOL_MODE` | `blackbox` \| `command_only` | Auto: `command_only` for whitebox, otherwise `blackbox` | Bwrap pool mode. Must match paddock mode. |
 | `DRESSAGE_LOCAL_BWRAP_AUTO_START` | `0` \| `1` | `1` in example scripts | Auto-start the local bwrap pool from run scripts. |
 | `DRESSAGE_LOCAL_BWRAP_RAY_NAMESPACE` | string | `dressage` | Ray namespace for local bwrap actors. |
 | `DRESSAGE_LOCAL_BWRAP_MANAGER_NAME` | string | `dressage_local_bwrap_manager` | Ray actor name for the bwrap pool manager. |
 | `DRESSAGE_LOCAL_BWRAP_TOTAL_SERVERS` | int | computed by scripts | Total bwrap slots across the Ray cluster. |
 | `DRESSAGE_LOCAL_BWRAP_BASE_PORT` | int | `31000` | Base port for local BlackboxServer slots. |
 | `DRESSAGE_BLACKBOX_SLOTS_PER_NODE` | int | script-specific | Local bwrap slot count per Ray node. |
 | `DRESSAGE_BLACKBOX_RUNNER_MODE` | `bwrap` \| `bubblewrap` | `bwrap` | Local blackbox runner mode. |
 | `DRESSAGE_BLACKBOX_BWRAP_BIN` | path | `bwrap` | Bubblewrap binary used by local slots. |
 | `DRESSAGE_LOCAL_BWRAP_DESTROY_ACTORS_ON_STOP` | `0` \| `1` | `1` | Destroy Ray actors when stopping the pool. |
 | `DRESSAGE_LOCAL_BWRAP_CLEANUP_ON_EXIT` | `0` \| `1` | `1` | Stop the local bwrap pool on example-script exit. |
 | `DRESSAGE_BLACKBOX_PRESERVE_SESSION_ARTIFACTS` | `0` \| `1` | `0` | Preserve sandbox filesystem after sessions for debugging. |

</details>

<details>
<summary><b> Proxy</b></summary>
<br>

 | Variable | Values | Default | Description |
 | :--------- | :------- | :-------- | :------------ |
 | `DRESSAGE_PROXY_URL` | URL | `http://${PROXY_PUBLIC_HOST}:${PROXY_PORT}` in scripts | Proxy server endpoint. Must be reachable from sandboxes. |
 | `TRAJECTORY_BUILD_MODE` | `concat` \| `last_step` | `concat` in scripts | Script helper passed to proxy `--trajectory-build-mode`. |
 | `TITO_MODEL` | string | `qwen3_5` in scripts | Script helper passed to proxy `--tito-model`. |
 | `DRESSAGE_PROXY_MAX_STEPS_PER_SESSION` | int | `0` (unlimited) | Returns HTTP 400 before the next proxy generation once the session already has this many steps. |

</details>

<details>
<summary><b> Partial Rollout</b></summary>
<br>

 | Variable | Values | Default | Description |
 | :--------- | :------- | :-------- | :------------ |
 | `DRESSAGE_PROXY_PAUSE_AROUND_WEIGHT_UPDATE` | `0` \| `1` | `1` | Enable proxy pause/resume around weight updates. |
 | `DRESSAGE_PROXY_PAUSE_REQUIRED` | `0` \| `1` | — | Require pause to succeed (fail if proxy unreachable). |
 | `DRESSAGE_PROXY_PAUSE_TIMEOUT_SEC` | int | `300` | Timeout for pause confirmation in seconds. |

</details>

<details>
<summary><b> Blackbox & BlackboxServer</b></summary>
<br>

 | Variable | Values | Default | Description |
 | :--------- | :------- | :-------- | :------------ |
 | `DRESSAGE_BLACKBOX_TYPE` | `opencode` \| `openclaw` | `opencode` | Backend agent type. |
 | `DRESSAGE_BLACKBOX_MAX_STEPS` | int | — | Positive int forwarded to `backend_options.proxy.max_steps`; set `0` to disable the backend proxy step limit. |
 | `DRESSAGE_BLACKBOX_COMPACT_THRESHOLD` | int | — | Positive value no greater than the context window; controls backend compaction reserve sizing. |
 | `BBS_HOST` | host | `0.0.0.0` | BlackboxServer bind host. |
 | `BBS_PORT` | port | `23456` | BlackboxServer bind port. |
 | `BBS_BACKEND_TIMEOUT` | float | `960.0` | Agent call timeout in seconds (16 min). |
 | `BBS_EXECUTE_CMD_TIMEOUT` | float | `600.0` | Shell command timeout in seconds (10 min). |
 | `BBS_ROUTER_TIMEOUT` | int | `600000` | Timeout for requests from BlackboxServer's in-process proxy to the upstream router. |
 | `BBS_SHUTDOWN_TIMEOUT` | float | `30.0` | Shutdown grace period in seconds. |
 | `BBS_RUNTIME_HEALTH_CHECK_INTERVAL` | float | `10.0` | Interval between backend runtime health checks. |
 | `BBS_RUNTIME_HEALTH_CHECK_RETRIES` | int | `3` | Runtime health-check retry count. |
 | `BBS_RUNTIME_HEALTH_CHECK_RETRY_DELAY` | float | `0.5` | Delay between runtime health-check retries. |
 | `OPENCODE_BIN` | path | `opencode` | Path to the opencode binary. |
 | `OPENCLAW_BIN` | path | `openclaw` | Path to the openclaw binary. |

</details>

<details>
<summary><b> Reward</b></summary>
<br>

 | Variable | Values | Default | Description |
 | :--------- | :------- | :-------- | :------------ |
 | `DRESSAGE_REWARD_MODULES` | comma-separated | — | Reward function modules to load (e.g., `my_project.rewards`). |

</details>

<details>
<summary><b> Async Rollout</b></summary>
<br>

 | Variable | Values | Default | Description |
 | :--------- | :------- | :-------- | :------------ |
 | `DRESSAGE_PARTIAL_ROLLOUT_TARGET_GROUPS` | int | — | Override target groups for partial async mode. |
 | `DRESSAGE_PARTIAL_ROLLOUT_TARGET_SAMPLES` | int | — | Override partial async readiness by sample count. |
 | `DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS` | int | — | Cap active async prompt groups. |
 | `DRESSAGE_ASYNC_OUTPUT_QUEUE_SIZE` | int | `1000` | Async worker output queue size. |
 | `DRESSAGE_ASYNC_WORKER_STOP_TIMEOUT_SEC` | float | `300` | Timeout for stopping async workers. |
 | `DRESSAGE_ROLLOUT_MAX_RETRIES` | int | `2` | Per-group rollout retry limit. |
 | `DRESSAGE_ASYNC_NO_PROGRESS_WARN_SEC` | float | `600` | No-progress warning threshold. |
 | `DRESSAGE_ASYNC_MAX_DROPPED_FAILED_GROUPS` | int | — | Maximum failed groups that can be dropped by async rollout. |
 | `DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH` | `0` \| `1` | `0` | Allow empty train batches after failures. |

</details>

<details>
<summary><b> Logging & Debugging</b></summary>
<br>

 | Variable | Values | Default | Description |
 | :--------- | :------- | :-------- | :------------ |
 | `DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR` | path | — | Directory for trajectory payload logs. |
 | `DRESSAGE_TRAJECTORY_ERROR_LOG_DIR` | path | — | Directory for trajectory error logs. |
 | `DRESSAGE_LOG_WRITE_MODE` | `background` \| `await` | `background` | Log write mode. `background` is faster; `await` ensures logs are flushed before proceeding. |

</details>

## 🔧 Slime Wiring Cheatsheet

Quick-reference examples for common training configurations:

<details>
<summary><b> Blackbox + Sync + GRPO</b></summary>
<br>

The simplest blackbox configuration: synchronous rollout with GRPO advantage estimation.

```bash
python3 -m slime.train \
  --custom-generate-function-path \
    dressage.rollout.generate.blackbox_dispatch.generate \
  --rollout-function-path \
    dressage.rollout.sync_rollout.generate_rollout_sync \
  --custom-convert-samples-to-train-data-path \
    dressage.rollout.convert_samples.convert_samples_to_train_data \
  --custom-reward-post-process-path \
    dressage.training.reward_post_process.reward_post_process \
  --custom-rm-path \
    dressage.reward.custom_rm.custom_rm \
  --data-source-path \
    dressage.rollout.data_source.DressageDataSource \
  --custom-rollout-log-function-path \
    dressage.rollout.log_rollout.log_rollout_data \
  --advantage-estimator grpo
```

</details>

<details>
<summary><b> Whitebox + Fully Async</b></summary>
<br>

Whitebox agents with fully async scheduling use slime's async entry point.

```bash
export DRESSAGE_REWARD_MODULES=my_recipe.reward

cd slime
python3 train_async.py \
  --custom-generate-function-path \
    my_recipe.agent.generate \
  --rollout-function-path \
    dressage.rollout.fully_async_rollout.generate_rollout_fully_async \
  --custom-convert-samples-to-train-data-path \
    dressage.rollout.convert_samples.convert_samples_to_train_data \
  --custom-reward-post-process-path \
    dressage.training.reward_post_process.reward_post_process \
  --custom-rm-path \
    dressage.reward.custom_rm.custom_rm \
  --data-source-path \
    dressage.rollout.data_source.DressageDataSource \
  --advantage-estimator grpo
```

Select a registered reward with `sample.metadata["reward_fn"]`; omitted values fall back to `default`.

</details>

<details>
<summary><b> Whitebox + Sync (Simplest)</b></summary>
<br>

The simplest configuration for development and debugging. Synchronous rollout — all samples complete before training.

```bash
cd slime
python3 train.py \
  --custom-generate-function-path \
    dressage.recipes.alfworld.agent_whitebox.generate \
  --rollout-function-path \
    dressage.rollout.sync_rollout.generate_rollout_sync \
  --custom-convert-samples-to-train-data-path \
    dressage.rollout.convert_samples.convert_samples_to_train_data \
  --custom-rm-path \
    dressage.reward.custom_rm.custom_rm \
  --advantage-estimator grpo
```

For example, set `DRESSAGE_REWARD_MODULES=dressage.recipes.alfworld.reward` so the custom reward registry loads the ALFWorld reward.

</details>

## 🆘 Troubleshooting

Common issues and their solutions:

 | Issue | Symptoms | Solution |
 | :------ | :--------- | :--------- |
 | **Proxy connection refused** | `ConnectionRefusedError` on chat completions | Check `DRESSAGE_PROXY_URL` and ensure the proxy server is running (`dressage-proxy`). |
 | **Bwrap slot unavailable** | Rollout hangs waiting for sandbox | Run `dressage-local-bwrap-status` to check pool health. Ensure enough slots are provisioned. |
 | **Mode mismatch error** | `ValueError` at startup about paddock/pool mode | Ensure `DRESSAGE_PADDOCK_MODE` matches `DRESSAGE_LOCAL_BWRAP_POOL_MODE` (blackbox↔blackbox, whitebox↔command_only). |
 | **Backend not found** | `FileNotFoundError` for opencode/openclaw binary | Set `OPENCODE_BIN` or `OPENCLAW_BIN` to the full path of the binary. |
 | **TITO failure** | Warning about `concat_incremental_tokenization_failed` | Check model compatibility — TITO currently supports `qwen3_5` only. Failure triggers safe segment boundary. |
 | **Pause timeout** | `TimeoutError` during weight update | Increase `DRESSAGE_PROXY_PAUSE_TIMEOUT_SEC`. Default 300s may not be enough for large batches. |
 | **Session desync** | BlackboxServer reports `desynced` state | Agent process may have crashed. Check sandbox logs. Session will be aborted and retried automatically. |
 | **Docker bwrap fails** | Bubblewrap errors inside container | Ensure Docker is running with `--privileged` flag. Required for Linux namespace operations inside containers. |
 | **Zero reward** | All trajectories get reward=0.0 | Check reward function registration. Ensure `DRESSAGE_REWARD_MODULES` includes your module and `@register_reward` decorator is applied. |

> [!TIP]
> Enable trajectory payload logging for debugging: `export DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR=/tmp/dressage_logs`. This saves full trajectory data to disk for offline inspection.

## 📚 Next Steps

After your first successful training run, explore these resources to go deeper:

 | Resource | Description |
 | :--------- | :------------ |
 | [Proxy](./proxy.md) | Deep dive into trajectory recording, TITO, segment boundaries, and routing replay |
 | [Paddock](./paddock.md) | Understand environment interaction, blackbox vs whitebox, factory pattern |
 | [Sandbox](./sandbox.md) | Configure isolation backends — bwrap pools and E2B cloud |
 | [BlackboxServer](./blackbox-server.md) | Set up blackbox agent rollouts, understand the adapter protocol |
 | [Rollout](./rollout.md) | Customize slime integration, async modes, reward registry |
 | [Training](./training.md) | Multi-segment training, prompt-equal scaling, partial rollout |
 | [Recipes](./recipes.md) | Build your own agents with ALFWorld and HotpotQA as templates |

---

[← Recipes](./recipes.md) · [Back to Main README](../README.md)
