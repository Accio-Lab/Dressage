# <img src="assets/dressage_logo.png" alt="Dressage logo" width="96" align="absmiddle" /> Dressage: Scalable RL for Any Agent and Sandbox


**Dressage** is an agentic reinforcement learning training framework built on top of [slime](https://github.com/THUDM/slime). It bridges the gap between policy rollouts, sandboxed tool execution, and training data conversion through a shared proxy and paddock layer.

Dressage lets you train diverse types of LLM agents that use real tools — like code editors, shell commands, file I/O, retrieval APIs — with full RL gradient flow. Both **whitebox** (Python tool loops) and **blackbox** (HTTP agents like `opencode`, `openclaw`, `claude_code`, and `codex`) paradigms are supported through a unified interface.

## Table of Contents

- [News](#-news)
- [Technical Highlights](#-technical-highlights)
- [Architecture Overview](#️-architecture-overview)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Docker](#-docker)
- [Team](#-team)
- [Contributing](#-contributing)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

## 📢 News

- **[2026/07/09]** **Multi-Agent Training.** From Timeline to Lineage, **Lineage-aware TITO** builds correct multi-agent training trajectories by reconstructing context inheritance across main-agent and subagent calls. Read the [technical deep dive](https://zhuanlan.zhihu.com/p/2059302405888537724).
- **[2026/07/04]** **Multi-Harness Training.** Supports [Claude Code](https://github.com/anthropics/claude-code) and [Codex](https://github.com/openai/codex), and you can train multiple harnesses in a single [script](https://github.com/huang3eng/Dressage/blob/main/examples/scripts/run_blackbox_qwen3.5_4b_async_local.sh).
- **[2026/06/30]**  Released [whitebox agent training curves](dressage/recipes/README.md) and [true staleness control](docs/staleness.md).
- **[2026/06/20]**  **Dressage is now open source!**


## 💡 Technical Highlights

Dressage introduces several key innovations for agentic RL:

### Any Agent, Any Sandbox

Dressage separates agent semantics from execution placement. Whitebox Python tool loops, blackbox HTTP agents such as `opencode`, `openclaw`, `claude_code`, and `codex`, local bubblewrap pools, and remote E2B sandboxes all converge on the same proxy-to-training path.

### Token-Wise Control

Dressage records training evidence at token granularity: `token_id`, `logprob`, `loss_mask`, `token_version`, and `token_expert`. **TITO — Token-In-Token-Out** avoids retokenization drift by encoding only each turn's append delta and splicing token IDs incrementally.

The same token-wise model powers Token-Boundary Pause/Resume through `GenerationController`, version-aware masking through `token_version`, and MoE **Routing Replay (R3)** through `token_expert`. See [Training Layer](docs/training.md) for the deep dive.

### Segment-Aware Training

When trajectories split because of history compaction or tool-schema changes, Dressage expands **every** segment into a training sample and broadcasts the anchor segment's terminal advantage to its siblings. Shared `rollout_id` / `parent_traj_id` values keep segments in the same slime training step, while prompt-equal denominators prevent split-heavy trajectories from receiving extra gradient weight, please refer to [Training Layer](docs/training.md).

### Slime-Native Integration

Dressage is built on slime, not maintained as a fork. Rollout generation, reward processing, sample conversion, and training behavior plug into upstream slime through dotted import-path hooks.

### Production-Grade Rollout Safety

Atomic trajectory logging, HTTP error redaction, context overflow detection, session artifact archiving, and abort safety contracts keep failed rollouts diagnosable without corrupting training data.

## 🏗️ Architecture Overview

Dressage's architecture is built on an elegant separation of concerns across three orthogonal axes. The entire system sits on top of slime as a git submodule, extending it through dotted import-path customization hooks — no upstream fork required.

![Dressage architecture overview](assets/artchitecture.png)

Compact mental model:

- **Dressage owns the agentic RL bridge**: rollout hooks create sessions, the proxy records token-level evidence, and the training layer converts trajectories into slime-ready samples.
- **Paddocks own interaction semantics**: whitebox paddocks run Python tool loops; blackbox paddocks delegate to HTTP agents through BlackboxServer.
- **Sandboxes own placement and isolation**: the same rollout path can target local bubblewrap slots or remote E2B sandboxes without changing the agent logic.
- **Slime owns the training substrate**: Megatron training, Ray rollout orchestration, and SGLang inference stay upstream and are customized through import-path hooks.

The key insight: **paddock mode** (whitebox vs blackbox) and **sandbox provider** (local vs remote) are orthogonal axes — mix and match freely.
The sandbox layer now exposes provider-neutral `SandboxSpec`, `SandboxLease`, and service endpoints; paddocks create and terminate leases while blackbox rollouts resolve the `blackbox` service endpoint from the lease.

Both whitebox and blackbox agent scaffolds converge on the same post-rollout path:

```text
proxy.finalize_session → trajectory/read → expand_segments_to_samples → list[Sample]
```

### End-to-End Rollout Flow

```text
1️⃣  Generate hook creates a paddock lease + proxy session
2️⃣  Agent runs (whitebox Python loop or blackbox HTTP agent)
3️⃣  Each LLM call goes through Proxy → SGLang; tokens recorded per step
4️⃣  Session finalizes into one or more trajectory segments
5️⃣  Segments expand to training Samples; reward + loss hooks prepare train_data
```

## 🚀 Quick Start

### Start Docker Environment

```bash
docker pull huang3eng/dressage:latest

docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it huang3eng/dressage:latest /bin/bash
```

For reproducible environments, pin a dated tag such as
`huang3eng/dressage:nightly-dev-20260704a`.

### Prepare Model Checkpoint

```bash
hf download Qwen/Qwen3.5-4B \
  --local-dir /root/Qwen3.5-4B

cd /root/Dressage
git submodule update --init --recursive
git submodule status --recursive
cd slime
source scripts/models/qwen3.5-4B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /root/Qwen3.5-4B \
  --save /root/Qwen3.5-4B_torch_dist
```

### Run a Training Example

```bash
cd /root/Dressage
bash examples/scripts/run_blackbox_qwen3.5_4b_async_local.sh
```

For detailed setup instructions, all configuration options, and troubleshooting, see the **[Quick Start Guide](docs/quickstart.md)**. To prepare and run the bundled ALFWorld and HotpotQA whitebox recipes, see the **[Whitebox Agent Quick Start](docs/whitebox-agent-quickstart.md)**.

## 📁 Project Structure

```text
dressage/
├── proxy/                 # Inference proxy, trajectory recording, TITO
│   └── tito/              # Token-In-Token-Out incremental tokenizer
├── paddock/               # Environment interaction (blackbox | whitebox)
├── sandbox/               # Isolation backends (bwrap, E2B)
│   ├── local/bwrap/       # Ray-managed bubblewrap pool
│   └── remote/e2b/        # E2B provider
├── rollout/               # Generate hooks, multi-segment, async rollout
│   └── generate/          # Blackbox dispatch + whitebox agent scaffolds
├── training/              # Reward post-process, async train with pause
├── reward/                # Sample-oriented reward registry
├── recipes/               # Example agents (alfworld, hotpotqa)
└── config/                # Shared defaults and env vars

blackbox_server/
├── api/                   # FastAPI route handlers
├── adapters/              # Backend adapters (opencode, openclaw, claude_code, codex)
├── core/                  # Server logic, models, monitoring
├── proxy/                 # In-process LLM proxy with session headers
├── store/                 # In-memory session store
└── runtime/               # Path and runtime ID resolution

slime/                     # Git submodule — upstream RL framework (v0.3.0)
examples/
├── scripts/               # Training run scripts (sync, async, partial)
└── data/                  # Sample/demo prompt datasets (DAPO Math)
tests/                     # 30+ test modules
docs/                      # Protocol and implementation docs
docker/                    # Reproducible local environment
```

Core modules map onto the main runtime layers:

### [Inference Proxy](docs/proxy.md): Token-Level Trajectory Recording

OpenAI-compatible HTTP service that sits between agent rollouts and the SGLang inference router. Every LLM call passes through it — every token, logprob, and loss mask is captured for training. Agents never call SGLang directly. Supports per-step recording, auto-segmentation on history rewrites, preemptible generation via `GenerationController`, weight version tracking, and MoE routing replay. The central nervous system of trajectory recording.

### [Paddock](docs/paddock.md): Unified Environment Interaction

Single-class abstraction that manages all environment interaction during rollouts — sandbox lease creation, agent/tool calls, pause/resume lifecycle, and cleanup. The "what-to-do" layer. `BlackboxPaddock` drives HTTP agents (`register_agent` → `call_agent` → `pause/resume`), while `WhiteboxPaddock` exposes `tool_call` for Python agents (shell, file read/write). Factory pattern: `create_paddock_from_env()` wires everything from environment variables.

### [Sandbox Backends](docs/sandbox.md): Pluggable Isolation

Two pluggable isolation backends, swapped via a single environment variable. **Local bubblewrap** (`local_bwrap`): Ray-managed pool of bubblewrap-isolated slots with supervisor health monitoring. **E2B** (`e2b`): elastic cloud-native sandboxes from E2B templates/images. Two local pool modes: `blackbox` (full BlackboxServer service endpoint) or `command_only` (shell/file ops). Local blackbox clusters can also be managed with the `dressage-local-blackbox-*` / `dressage-blackbox-*` scripts.

### [BlackboxServer](docs/blackbox-server.md): Unified HTTP Adapter

Bundled HTTP adapter service that decouples the rollout manager from concrete agentic backends. Sits inside sandboxes, manages exactly one backend agent process and one active session at a time, and transparently proxies all LLM calls back through the Dressage inference proxy. Supports `opencode`, `openclaw`, `claude_code`, and `codex` through a pluggable adapter pattern. Features turn idempotency, register-and-rebind, and background health monitoring.

### [Rollout Hooks](docs/rollout.md): Slime Integration

Bridges slime's RL training loop with Dressage's agentic capabilities through customizable hooks — all specified as dotted import paths, no fork required. Two generate paradigms: `whitebox_agent` (Python tool loops) and `blackbox_dispatch` (BlackboxServer delegation). Three async scheduling modes: sync, fully async (background worker pipelines), and partial async (early return on `global_batch_size`). Includes pluggable reward registry, prompt-equal sample conversion, and trajectory-level logging.

### [Training Layer](docs/training.md): Multi-Segment & TITO

Transforms proxy-recorded trajectories into slime-compatible training data. Multi-segment expansion trains on all segments (not just the last). TITO tokenization eliminates retokenization drift across turns. Prompt-equal gradient aggregation ensures fair scaling for GRPO. Partial rollout resume preserves in-flight generation across weight updates. Reward post-processing broadcasts anchor advantages to sibling segments.

### [Recipes](docs/recipes.md): Example Agents

Complete whitebox agent implementations demonstrating the framework. **ALFWorld**: TextWorld navigation agent with `env_step` tool for household task completion. **HotpotQA**: multi-hop retrieval agent with local FAISS+BGE index for complex question answering. Both include training scripts, reward functions, and a build-your-own guide for custom agents.

For the end-to-end data preparation and launch commands for these recipes, see the **[Whitebox Agent Quick Start](docs/whitebox-agent-quickstart.md)**.

## 📚 Documentation

- **[Proxy](docs/proxy.md)** — Inference proxy architecture, session model, trajectory build modes, routing replay
- **[Paddock](docs/paddock.md)** — Paddock interface, blackbox vs whitebox interaction, factory pattern
- **[Sandbox](docs/sandbox.md)** — Sandbox providers, bwrap pools, E2B integration
- **[BlackboxServer](docs/blackbox-server.md)** — HTTP adapter protocol, backends, in-process LLM proxy, session states
- **[Rollout](docs/rollout.md)** — Generate hooks, async modes, reward registry, slime wiring
- **[Training](docs/training.md)** — Multi-segment, TITO, prompt-equal aggregation, partial rollout
- **[Recipes](docs/recipes.md)** — ALFWorld and HotpotQA example agents, build-your-own guide
- **[Whitebox Agent Quick Start](docs/whitebox-agent-quickstart.md)** — Data preparation and launch commands for ALFWorld and HotpotQA whitebox agents
- **[Quick Start](docs/quickstart.md)** — Step-by-step setup, configuration reference, troubleshooting

## 🐳 Docker

A reproducible local environment is provided via Docker:

```bash
docker/build.sh    # Build the image
docker/run.sh      # Run with --gpus all --network host --ipc host --privileged
```

The image includes bubblewrap, opencode, openclaw, Claude Code, Codex CLI, Dressage, and BlackboxServer. `--privileged` is required for bubblewrap inside containers. See [docker/README.md](docker/README.md) for details.

## 👥 Authors and Contributors

Dressage is developed and maintained by its authors and contributors.

**Author**:
Liangmeng Huang (huang3eng@gmail.com)

**Core Contributors** (alphabetical):
Hao Dong (dh20373967@gmail.com)
Qingchuan Li (lqcustc@gmail.com)
Hongwei Xue (xuehongwe@gmail.com)
Shilin Yan (tattoo.ysl@gmail.com)
For academic collaborations, citations, or technical inquiries, please contact us.

**Contributors** (alphabetical): 
Wenhui Chen, Xueyuan Han, Jing He, Hongyu Li, Junbo Li, Zicheng Liu, Sibo Wei, Senyu Zhang, Guannan Zhang.

To cite Dressage:

```bibtex
@misc{dressage_github,
  author       = {Liangmeng Huang and Qingchuan Li and Hongwei Xue and Shilin Yan and {Dressage Contributors}},
  title        = {{Dressage}: Scalable {RL} for Any Agent and Any Sandbox},
  year         = {2026},
  howpublished = {\url{https://github.com/Accio-Lab/Dressage}}
}
```

## 🤝 Contributing

Contributions are welcome! If you have suggestions for new features, performance tuning, or feedback on user experience, feel free to submit an Issue or PR.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Ensure tests pass (`pytest`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

**Development notes:**
- When bumping the slime submodule, diff `convert_samples.py` against upstream `slime/ray/rollout.py`
- TITO currently supports `qwen3_5` only — contributions for additional model templates welcome
- The `claude_code` adapter runs Claude Code headless through the BlackboxServer Anthropic Messages bridge.
- The `codex` adapter runs `codex exec --json` with an isolated sandbox-local `CODEX_HOME`, points Codex at the in-process Dressage proxy, and does not mount host `~/.codex` credentials.

## 🙏 Acknowledgements

We gratefully acknowledge:

- **[slime](https://github.com/THUDM/slime)** (THUDM, Tsinghua University) — The upstream RL post-training framework that Dressage builds upon. slime provides the foundational Megatron training loop, Ray rollout management, and SGLang inference integration.
- **[SGLang](https://github.com/sgl-project/sglang)** — High-performance inference engine powering the generation backend.
- **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** (NVIDIA) — Distributed training infrastructure used via slime.
- **[bubblewrap](https://github.com/containers/bubblewrap)** — Unprivileged sandboxing for local agent isolation.

## 📄 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

The Docker setup can install `opencode`, `openclaw`, Claude Code, and Codex CLI using their public install scripts. Those tools are not source code distributed by this repository; review their upstream licenses before redistribution.
