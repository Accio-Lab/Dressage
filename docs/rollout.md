# Rollout Hooks & Async Modes

**Slime Integration for Agentic RL Training**

[← Back to Main README](../README.md) · [Overview](#-overview) · [Integration Hooks](#-integration-hooks) · [Generate Functions](#-generate-functions) · [Async Modes](#-async-rollout-modes) · [Data Source](#-data-source) · [Multi-Segment](#-multi-segment-expansion) · [Reward Registry](#-reward-registry) · [Slime Wiring](#-typical-slime-wiring)

## 📖 Overview

Dressage hooks into [slime](https://github.com/THUDM/slime) via CLI paths — **no fork of training code required**. The rollout layer bridges slime's RL training loop with Dressage's agentic capabilities through customizable generate functions, async scheduling modes, data sources, and logging hooks. Current slime resolves hook paths as dotted Python import paths, splitting on the last dot.

This design means Dressage rides the slime upgrade train for free: when slime releases new training optimizations, new gradient estimators, or infrastructure improvements, you get them without merge conflicts. Dressage only touches the rollout-side hooks that slime explicitly exposes for customization.

```text
slime training loop
        │
        ├── --custom-generate-function-path     → Agent rollout (blackbox or whitebox)
        ├── --rollout-function-path             → Async scheduling strategy
        ├── --custom-convert-samples-to-...     → Training data conversion
        ├── --custom-rm-path                    → Reward function (sample-oriented)
        ├── --custom-reward-post-process-path   → Advantage broadcast (multi-segment)
        ├── --data-source-path                  → Prompt data loading
        └── --custom-rollout-log-function-path  → Trajectory-level metrics
```

> [!IMPORTANT]
> All hooks are specified as dotted import paths (e.g., `dressage.rollout.generate.blackbox_dispatch.generate`). No need to modify slime source code, no monkey-patching, no custom forks.

## ✨ Integration Hooks

Dressage provides seven hook points that plug into slime's training loop. Each hook handles a specific aspect of the rollout → training pipeline:

 | Hook | Module Path | Purpose | When It Runs |
 | :----- | :------------ | :-------- | :------------- |
 | **Generate** | `dressage.rollout.generate.blackbox_dispatch.generate` or a whitebox `*.generate` | Agent rollout lifecycle — run agent, capture trajectory, produce Samples | Once per prompt in each rollout batch |
 | **Rollout** | `dressage.rollout.sync_rollout.generate_rollout_sync`, `dressage.rollout.fully_async_rollout.generate_rollout_fully_async`, or `dressage.rollout.partial_async_rollout.generate_rollout_partial_async` | Scheduling — how rollout batches are dispatched and collected | Orchestrates the generate function across the batch |
 | **Convert Samples** | `dressage.rollout.convert_samples.convert_samples_to_train_data` | Transforms Samples into slime's `train_data` format with prompt-equal mask sums | After all samples are collected |
 | **Reward** | `dressage.reward.custom_rm.custom_rm` | Sample-oriented reward computation via pluggable registry | After trajectory finalization |
 | **Reward Post-Process** | `dressage.training.reward_post_process.reward_post_process` | GRPO normalization + anchor advantage broadcast to sibling segments | After reward computation |
 | **Data Source** | `dressage.rollout.data_source.DressageDataSource` | Custom prompt data loading from JSONL files | At training loop startup |
 | **Rollout Log** | `dressage.rollout.log_rollout.log_rollout_data` | Trajectory-level metrics using `parent_traj_id` anchor lookup | After each rollout batch |

## 🤖 Generate Functions

Two paradigms for agent rollouts, both producing the same `list[Sample]` output format. The choice between them determines **how** the agent interacts with the environment, but the post-rollout path (finalize → segments → samples) is identical.

### Whitebox Agent

Python agents that call the LLM via proxy and execute tools directly through the paddock. You write the agent logic as a Python class, with full control over the prompting strategy, tool execution, and multi-turn flow.

```python
from dressage.rollout.generate.whitebox_agent import WhiteboxAgent, make_generate

class MyAgent(WhiteboxAgent):
    name = "my_agent"

    async def rollout(self, sample, sampling_params) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": sample.prompt},
        ]

        response = await self.chat({
            "messages": messages,
            "model": "qwen3.5-4b",
            "tools": self.get_tools(),
        })

        while has_tool_calls(response):
            tool_result = await self.execute_tool(response)
            messages.append(tool_result)
            response = await self.chat({"messages": messages, "model": "qwen3.5-4b"})

        return extract_assistant_content(response)

generate = make_generate(MyAgent)
```

**Agent variants:**
- `WhiteboxAgent` — Pure proxy agent, no sandbox. For lightweight Python tools (API calls, retrieval, computation). Agent calls `self.chat(...)` to interact with the LLM. Tools are Python functions.
- `PaddockWhiteboxAgent` — Adds sandbox lifecycle for shell/file execution. Agent calls `self.paddock.tool_call(self.session_id, tool_id, args)` for isolated command execution. Paddock handles `init/terminate` automatically.

### Blackbox Dispatch

Delegates the entire agent loop to an external HTTP agent running inside a sandbox via BlackboxServer. The generate function manages the lifecycle but doesn't implement any agent logic — that's the backend's job.

```text
1️⃣  paddock.init(traj_id)        → create sandbox lease with BlackboxServer
2️⃣  paddock.register_agent(...)  → start backend process (opencode/openclaw)
3️⃣  paddock.execute_cmd(...)     → run optional setup commands (clone repo, install deps)
4️⃣  paddock.call_agent(...)      → send task prompt, wait for agent completion
5️⃣  proxy.finalize_session(...)  → finalize trajectory → segments
6️⃣  expand_segments_to_samples   → segments → list[Sample]
7️⃣  paddock.terminate(traj_id)   → terminate sandbox lease
```

<details>
<summary><b> Blackbox Customization Options</b></summary>
<br>

 | Customization | Description |
 | :-------------- | :------------ |
 | `DRESSAGE_BLACKBOX_TYPE` | Selects the backend agent: `opencode` (default, code-editing) or `openclaw` (OpenClaw gateway). Determines which adapter the BlackboxServer uses. |
 | `DRESSAGE_BLACKBOX_MAX_STEPS` | Positive int forwarded to `backend_options.proxy.max_steps`; `0` disables the backend proxy step limit. |
 | `DRESSAGE_BLACKBOX_COMPACT_THRESHOLD` | Positive int no greater than the context window; controls backend compaction reserve sizing. |
 | `metadata["blackbox_execute_cmds"]` | Per-sample shell commands to run before/after agent calls. Supports only `before_agent` and `after_agent`; each stage is a list of objects with `name`, `cmd`, boolean `required`, and optional `timeout`. |
 | `DRESSAGE_PADDOCK_CLASS` | Override the default paddock class with `module.Class` for custom lifecycle logic. |

> [!WARNING]
> `blackbox_dispatch` rejects `DRESSAGE_PADDOCK_MODE=whitebox` at startup. Use a `WhiteboxAgent` subclass with the whitebox generate function instead.

</details>

### Shared Post-Rollout Path

Both whitebox and blackbox modes converge on the same training data pipeline after the agent finishes:

```text
proxy.finalize_session(session_id)
    │  closes all open segments, writes to trajectory store
    ▼
proxy.read_trajectory(session_id)
    │  returns list of finalized segments
    ▼
expand_segments_to_samples(segments, template_sample)
    │  one Sample per segment, shared rollout_id
    │  anchor segment gets reward=None (for reward_fn)
    │  non-anchor segments get reward=0.0
    ▼
list[Sample]
    └── ready for convert_samples_to_train_data
```

## 🔄 Async Rollout Modes

Three scheduling strategies for managing rollout batches. The choice determines how much overlap there is between rollout and training, and how latency-tolerant the system is.

### Sync Mode (Default)

The simplest mode — slime's default behavior. All samples in a batch must complete before training begins.

```text
Full batch ─── all samples complete ───▶ Train step
```

- Simplest to understand and debug
- Deterministic batch composition
- Training GPU is idle during the entire rollout phase
- **Best for**: development, debugging, small-scale experiments

### Fully Async Mode

Background worker pipelines keep prompt groups in flight continuously. Completed groups are buffered and assembled into training batches when enough are ready.

```text
Group₁ ─── complete ───▶ ┐
Group₂ ─── complete ───▶ ├─▶ Train step (when enough groups ready)
Group₃ ─── running  ───▶ ┘   (continues in background)
Group₄ ─── running  ───▶     (queued for next train step)
Group₅ ─── queued   ───▶     (waiting for worker)
```

- Hides blackbox agent latency by overlapping rollout with training
- Flattens `list[Sample]` from multi-segment `generate()` automatically
- Better GPU utilization — training can start before all rollouts finish
- More complex scheduling and debugging
- **Best for**: production blackbox rollouts where agent latency varies widely

### Partial Async Mode

Returns early once enough prompt groups fill `global_batch_size`. Remaining groups continue in background for the next training step. Designed for the common case where `rollout_batch_size × n_samples_per_prompt` exceeds what training needs.

```text
Groups ─── global_batch_size filled ───▶ Train step
          (e.g., 8 of 16 groups complete)
          remaining groups continue → next step's buffer
```

- Solves batch-size mismatch efficiently
- No wasted rollout — excess groups carry over to next step
- Natural load balancing across variable-latency rollouts
- Batch composition varies between steps
- **Best for**: large-scale async rollouts with partial rollout (pause/resume)

<details>
<summary><b> Configuration</b></summary>
<br>

```bash
# Sync
--rollout-function-path \
  dressage.rollout.sync_rollout.generate_rollout_sync

# Fully async
--rollout-function-path \
  dressage.rollout.fully_async_rollout.generate_rollout_fully_async

# Partial async
--rollout-function-path \
  dressage.rollout.partial_async_rollout.generate_rollout_partial_async

# Override target readiness for partial async
DRESSAGE_PARTIAL_ROLLOUT_TARGET_GROUPS=<int>
DRESSAGE_PARTIAL_ROLLOUT_TARGET_SAMPLES=<int>
DRESSAGE_ROLLOUT_MAX_RETRIES=2
DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH=0

# Blackbox sandbox prewarming for fully/partial async rollout.
# Defaults to 1 for e2b and 0 for local_bwrap.
DRESSAGE_SANDBOX_PREWARM=1
DRESSAGE_SANDBOX_PREWARM_AHEAD=8
```

> [!TIP]
> Combine async modes with partial rollout via `train_async_with_rollout_pause` for proxy pause/resume around weight updates. This gives you the best of both worlds: continuous rollout without wasting computation on stale weights.

Sandbox prewarming defaults to enabled for `e2b`, where it overlaps cloud
sandbox provisioning, endpoint resolution, and health checks with current
rollout work. It starts `paddock.init()` for upcoming prompt groups, then
transfers the live lease to blackbox dispatch instead of creating a second
sandbox. Unclaimed or cancelled prewarms are terminated during group or worker
cleanup.
`DRESSAGE_SANDBOX_PREWARM_AHEAD` counts prompt groups, so the maximum number of
prefetched leases is approximately `ahead × n_samples_per_prompt`, which may
increase concurrent E2B sandbox cost. `local_bwrap` defaults to disabled because
its Ray manager already allocates from a pre-created slot pool; it can still be
enabled explicitly for deployments where slot acquisition includes meaningful
setup latency.

</details>

Use slime's sync training entry point (`train.py` / `slime.train`) with `generate_rollout_sync`, slime's async training entry point (`train_async.py`) with `generate_rollout_fully_async`, and `python3 -m dressage.training.train_async_with_rollout_pause` only when proxy pause/resume around weight updates is required.

## 📂 Data Source

`DressageDataSource` (`dressage/rollout/data_source.py`) provides custom prompt data loading for slime. It wraps JSONL prompt files and handles prompt-level metadata needed for prompt-equal aggregation — specifically the `instance_id` that ties all samples from the same prompt together.

```bash
--data-source-path dressage.rollout.data_source.DressageDataSource
```

### Prompt Format

Each line in the JSONL file is a prompt object:

```json
{"prompt": "Fix the authentication bug in auth.py", "label": "tests pass", "metadata": {"instance_id": "inst-001", "reward_fn": "contains_label", "blackbox_execute_cmds": {"before_agent": [{"name": "clone_repo", "cmd": "git clone ...", "required": true, "timeout": 120}]}}}
```

### Sample Datasets

Example prompt datasets are provided in `examples/data/`:

 | File | Description | Use Case |
 | :----- | :------------ | :--------- |
 | `dressage_dapo_prompts.jsonl` | DAPO-style coding prompts | Blackbox code-editing agent training |

## 🧬 Multi-Segment Expansion

Long agentic trajectories often split into multiple segments (due to history compaction, tool schema changes, or TITO fallback). Rather than discarding earlier segments, Dressage trains on **all** of them via `expand_segments_to_samples()`.

### How It Works

```text
Session:  [segment 0] → [segment 1] → [segment 2]
              ↓              ↓              ↓
Samples:   Sample₀        Sample₁        Sample₂  (anchor)
           reward=0.0     reward=0.0     reward=None → reward_fn
           adv broadcast  adv broadcast  adv = A ← terminal reward
```

### Expansion Logic

Core logic in `dressage/rollout/multi_segment.py` → `expand_segments_to_samples()`:

1. **Deep-copy** the template sample for each segment
2. **Write** tokens, masks, logprobs via `rollout.artifacts.samples.write_sample_from_segment`
3. **Tag** each sample with `parent_traj_id` = session ID, `segment_index`
4. **Share** the same `rollout_id` across all segments (grouped in one training step)
5. **Anchor** segment gets `reward=None` (filled by reward function); non-anchor segments get `reward=0.0`
6. **Broadcast** anchor's advantage to all siblings via `reward_post_process`

> [!NOTE]
> Single-segment trajectories produce exactly one Sample — zero overhead. The multi-segment path only activates when there are multiple segments.

### Metrics

 | Metric | Description |
 | :------- | :------------ |
 | `rollout/segments_per_trajectory_mean` | Average segments per trajectory (1.0 = no splitting) |
 | `rollout/segments_per_trajectory_max` | Maximum segment count among live trajectories |
 | `rollout/segments_per_trajectory_min` | Minimum segment count among live trajectories |
 | `rollout/num_trajectories` | Total distinct trajectories in the batch |
 | `rollout/num_segments` | Total live segments in the batch |
 | `rollout/raw_reward_trajectory_mean` | Trajectory-level raw reward mean from the rollout log hook |

### Abort Safety

On failure, `mark_aborted_no_grad` ensures clean cleanup:
- Sets `remove_sample=True` (excluded from training)
- Stamps `parent_traj_id` / `instance_id` for tracking
- Clears `session_id` for retry availability
- Preserves `last_failed_session_id` for debugging

## 📊 Sample Conversion

`convert_samples_to_train_data` (`dressage/rollout/convert_samples.py`) transforms Dressage's `list[Sample]` into slime's `train_data` format. It is a near-verbatim copy of slime's `_convert_samples_to_train_data` with one critical addition: **prompt-equal `rollout_mask_sums`** for `grpo` and `reinforce_plus_plus_baseline`.

```text
Samples → convert_samples_to_train_data → train_data
                                              │
                                              ├── prompt-equal denominators (M_P × N_P / gbs)
                                              ├── rollout_mask_sums per sample
                                              └── fair gradient scaling across segments
```

> [!CAUTION]
> When bumping the slime submodule, **always** diff `convert_samples.py` against upstream `slime/ray/rollout.py`. The base implementation may change across slime versions, and our prompt-equal additions need to be re-applied correctly.

## 🎯 Reward Registry

Pluggable reward functions via `dressage/reward/registry.py`. Register reward functions with decorators or dynamic module loading, and the registry dispatches them at evaluation time.

```python
from dressage.reward.registry import register_reward

@register_reward("my_custom_reward")
def my_reward(sample, *, args=None):
    """Compute reward for a single sample.

    Args:
        sample: The completed Sample with trajectory data
        args: Optional training args for context

    Returns:
        float: Reward value
    """
    return compute_reward(sample)
```

### Configuration

- **Module loading**: Reward modules are loaded via `DRESSAGE_REWARD_MODULES` environment variable (comma-separated module paths)
- **Function signature**: `fn(sample, *, args=None) → float` — sample-oriented, receives the complete trajectory
- **Slime entry point**: `dressage.reward.custom_rm.custom_rm` — the registry dispatcher
- **Selection**: `sample.metadata["reward_fn"]` chooses the registered reward name; missing values fall back to `default`

```bash
# Load custom reward modules
export DRESSAGE_REWARD_MODULES=my_project.rewards,my_project.bonus_rewards

# Wire into slime
--custom-rm-path dressage.reward.custom_rm.custom_rm
```

## 📝 Rollout Logging

`log_rollout.py` computes trajectory-level metrics using `parent_traj_id` anchor lookup, aggregating across all segments of each trajectory. This gives you a rollout-level view of training progress.

 | Metric | Description |
 | :------- | :------------ |
 | `rollout/segments_per_trajectory_mean` | Average number of segments per trajectory |
 | `rollout/segments_per_trajectory_max` / `rollout/segments_per_trajectory_min` | Segment-count range across trajectories |
 | `rollout/num_trajectories` | Total distinct trajectories in the batch |
 | `rollout/num_segments` | Total live segments in the batch |
 | `rollout/raw_reward_trajectory_mean` | Mean raw reward after summing sparse anchor rewards per trajectory |
 | `rollout/reward_mean` | Mean anchor reward across trajectories |
 | `rollout/reward_std` | Reward standard deviation |
 | `rollout/reward_max` / `rollout/reward_min` | Reward range |

```bash
--custom-rollout-log-function-path dressage.rollout.log_rollout.log_rollout_data
```

## 🔧 Typical Slime Wiring

A complete example showing how all hooks wire together for a blackbox training run with synchronous rollout and GRPO estimation:

```bash
python3 -m slime.train \
  # Generate function: blackbox agent dispatch
  --custom-generate-function-path \
    dressage.rollout.generate.blackbox_dispatch.generate \
  # Sync scheduling: wait for the full rollout batch before training
  --rollout-function-path \
    dressage.rollout.sync_rollout.generate_rollout_sync \
  # Sample conversion: prompt-equal mask sums
  --custom-convert-samples-to-train-data-path \
    dressage.rollout.convert_samples.convert_samples_to_train_data \
  # Reward post-processing: GRPO normalization + segment broadcast
  --custom-reward-post-process-path \
    dressage.training.reward_post_process.reward_post_process \
  # Reward function: pluggable registry
  --custom-rm-path \
    dressage.reward.custom_rm.custom_rm \
  # Data source: JSONL prompt loading
  --data-source-path \
    dressage.rollout.data_source.DressageDataSource \
  # Rollout logging
  --custom-rollout-log-function-path \
    dressage.rollout.log_rollout.log_rollout_data \
  # GRPO advantage estimation
  --advantage-estimator grpo
```

## 📊 Upgrade Checklist

When bumping the slime submodule to a new version:

- [ ]  Diff `convert_samples.py` against slime `rollout.py` — check for upstream changes
- [ ]  Verify `build_dp_schedule` still uses `rollout_id` for segment grouping
- [ ]  Run `tests/test_convert_samples_multi_segment.py`
- [ ]  Run `tests/test_blackbox_dispatch_multi_segment.py`
- [ ]  Check for new CLI flags or hook points in slime's training loop

## 📁 Package Structure

```text
dressage/rollout/
├── generate/
│   ├── blackbox_dispatch.py     # Blackbox agent rollout orchestration
│   └── whitebox_agent.py        # WhiteboxAgent + PaddockWhiteboxAgent base classes
├── artifacts/
│   └── samples.py               # Sample writing from trajectory segments
├── fully_async_rollout.py       # Fully async scheduling with worker pipelines
├── partial_async_rollout.py     # Partial async with early return on batch fill
├── sync_rollout.py              # Synchronous rollout (slime default)
├── multi_segment.py             # expand_segments_to_samples()
├── convert_samples.py           # Samples → train_data with prompt-equal masks
├── data_source.py               # DressageDataSource for JSONL prompts
└── log_rollout.py               # Trajectory-level rollout metrics

dressage/reward/
├── custom_rm.py                 # Slime reward entry point (registry dispatcher)
├── registry.py                  # @register_reward decorator + module loading
└── helpers.py                   # Reward computation utilities
```

## 🔗 Integration Points

 | Component | Relationship |
 | :---------- | :------------ |
 | [Proxy](./proxy.md) | Generate functions use `ProxyClient` to manage sessions, finalize trajectories, and read segments |
 | [Paddock](./paddock.md) | Generate functions hold a paddock singleton for sandbox lifecycle management |
 | [BlackboxServer](./blackbox-server.md) | `blackbox_dispatch` drives BlackboxServer via paddock HTTP calls |
 | [Training](./training.md) | Training layer consumes Samples produced by rollout hooks |
 | [Recipes](./recipes.md) | Recipe agents are `WhiteboxAgent` subclasses wired via `make_generate` |

---

[← BlackboxServer](./blackbox-server.md) · [Back to Main README](../README.md) · [Next: Training →](./training.md)
