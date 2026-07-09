# Training Layer

**Multi-Segment Training, TITO, and Reward Processing**

[← Back to Main README](../README.md) · [Overview](#-overview) · [Multi-Segment](#-multi-segment-training) · [TITO](#-tito-tokenizer) · [Prompt-Equal](#️-prompt-equal-aggregation) · [Partial Rollout](#️-partial-rollout) · [Reward Post-Processing](#-reward-post-processing) · [Entry Points](#️-training-entry-points)

## 📖 Overview

The training layer transforms proxy-recorded trajectories into slime-compatible training data. It handles **segment expansion**, **token alignment**, **loss aggregation**, **partial rollout resume**, and **custom reward hooks** — ensuring that every token from every trajectory segment contributes to learning.

This layer is the bridge between Dressage's agentic rollout system and slime's Megatron-based training loop. Its core responsibility: take raw trajectory segments (with their token IDs, logprobs, and loss masks) and produce correctly formatted, correctly scaled training data that slime can consume.

```text
Proxy Segments (raw trajectory data)
        │
        ├── Multi-Segment Expansion  → train on ALL segments, not just the last
        ├── TITO Tokenization        → drift-free token alignment across turns
        ├── Prompt-Equal Aggregation → fair gradient scaling for GRPO
        ├── ⏸Partial Rollout Resume  → preserve in-flight generation across weight updates
        └── Reward Post-Process      → broadcast anchor advantage to sibling segments
        │
        ▼
slime train_data (Megatron-compatible)
```

## 🧬 Multi-Segment Training

Long agentic trajectories often split into multiple **segments** due to history compaction, tool-schema changes, or TITO fallback. Naive approaches discard earlier segments and train only on the last one — losing potentially valuable reasoning and tool-use tokens. Dressage trains on **all** segments, recovering this lost signal.

### Why Not Last-Segment-Only?

```text
Session:  [segment 0: 500 tokens] → [segment 1: 300 tokens] → [segment 2: 200 tokens]
                                                                        ↑
                    ❌ naive approach: only train on this → 200 tokens used
                    → 800 tokens of reasoning and tool-use LOST
```

In a typical SWE agent trajectory, segment 0 might contain the initial analysis and first round of code edits. Segment 1 might contain test execution and debugging. Segment 2 might contain the final fix. Discarding segments 0 and 1 means the model never learns from its analysis and debugging behavior — only from the final fix.

### Dressage Approach

```text
Session:  [segment 0: 500 tokens] → [segment 1: 300 tokens] → [segment 2: 200 tokens]
              ↓                          ↓                          ↓
Samples:   Sample₀ (500 tokens)      Sample₁ (300 tokens)      Sample₂ (200 tokens)
           reward = 0.0              reward = 0.0              reward = None → R
           adv = A (broadcast)       adv = A (broadcast)       adv = A ← terminal
           
           All share: rollout_id, parent_traj_id → same training step
```

> [!IMPORTANT]
> Single-segment trajectories produce exactly one Sample — identical to non-multi-segment behavior. The multi-segment path introduces zero overhead for the common case.

### Segment Data Shape

Each finalized segment carries a complete training-ready data bundle:

```python
{
    "segment_index": int,           # position in the session (0, 1, 2, ...)
    "tokens": list[int],            # token IDs (TITO-aligned or snapshot)
    "full_loss_mask": list[int],    # 0 = prompt/tool, 1 = trainable assistant token
    "full_logprobs": list[float],   # per-token log probabilities from rollout
    "messages": list[dict],         # conversation messages for this segment
    "finish_reason": str,           # "stop", "length", "tool_calls", "segment_boundary"
    "extra_info": dict,             # metadata: weight versions, TITO state, etc.
}
```

When segments are written into slime `Sample` objects, `write_sample_from_segment()` caps token arrays at `max_tokens_per_gpu * context_parallel_size` (or `cp_size` when that is the available context-parallel field). Truncated segments get `metadata["truncated"] = True`; samples whose proxy segment has `finish_reason == "length"` are marked `TRUNCATED`.

### `expand_segments_to_samples` — Step by Step

Core logic in `dressage/rollout/multi_segment.py`:

1. **Deep-copy** the template sample for each segment — ensures isolation between samples
2. **Write** tokens, masks, logprobs via `rollout.artifacts.samples.write_sample_from_segment` — maps segment data into slime's Sample format
3. **Tag** with `parent_traj_id = session_id` and `segment_index` — establishes the sibling relationship
4. **Share** `rollout_id` across all segments — slime's `build_dp_schedule` groups them in the same training step
5. **Anchor** assignment: the last segment gets `reward=None` (filled by reward function); earlier segments get `reward=0.0` (filled by advantage broadcast later)
6. **Broadcast**: `reward_post_process` copies the anchor's computed advantage to all siblings

### Training Step Scheduling

All segments share `Sample.rollout_id`. Slime's `build_dp_schedule` (v0.3.0+) uses this field to keep all segments from one trajectory in the **same training step**. This ensures:

- Gradient updates from all segments are applied together
- Prompt-equal scaling works correctly across segments
- Rollout logging can aggregate metrics per trajectory

### Abort Safety

On trajectory failure, `mark_aborted_no_grad` ensures clean handling:

 | Action | Purpose | 
 | :------- | :-------- | 
 | `remove_sample=True` | Excluded from training — no gradient contribution | 
 | Stamp `parent_traj_id` / `instance_id` | Maintains tracking for logging and prompt-equal calculation | 
 | Clear `session_id` | Makes the session ID available for retry by the next rollout | 
 | Preserve `last_failed_session_id` | Enables forensic debugging of failed trajectories | 

### Metrics

 | Metric | Description | 
 | :------- | :------------ | 
 | `rollout/segments_per_trajectory_mean` | Average segments per trajectory (≈1.0 means few splits) | 
 | `rollout/num_trajectories` | Total distinct trajectories in the batch | 
 | `rollout/reward_mean` | Mean reward across anchor segments | 

## 🔤 TITO Tokenizer

**TITO** (Token-In-Token-Out) builds long multi-turn trajectories without **retokenization drift** — the #1 correctness challenge in agentic RL training.

### The Problem: Retokenization Drift

When you re-tokenize the full message list each turn, the tokenizer may produce different token IDs for the same text prefix. This happens because tokenizer behavior depends on context — adding text after a prefix can change how the prefix itself is tokenized (especially around token boundaries, special characters, and BPE merge rules).

```text
Turn 1:  tokenize([system, user₁])                        → ids₁ = [101, 202, 303, 404]
Turn 2:  tokenize([system, user₁, asst₁, tool₁, user₂])   → ids₂ = [101, 202, 305, ...]
                                                                         ↑
                                                              DRIFT! 303 ≠ 305
```

When this happens:
- Logprobs recorded at Turn 1 don't align with token IDs at Turn 2
- Loss masks reference wrong token positions
- Training data is silently corrupted
- The model learns from misaligned supervision signals

### TITO Solution: Incremental Tokenization

TITO renders and encodes only the **append delta** each turn, then concatenates token IDs. The prefix is never re-tokenized.

```text
Turn 1:  encode(system + user₁)                → fragment₁ = [101, 202, 303, 404]
Turn 2:  encode(delta: asst₁ + tool₁ + user₂)  → fragment₂ = [505, 606, 707]
         stitch(fragment₁ + fragment₂)         → [101, 202, 303, 404, 505, 606, 707]
                                                  ✅ prefix [101, 202, 303, 404] intact!
```

### Implementation Details

The proxy records TITO fragments per step in `StepRecord` fields:

 | Field | Description | 
 | :------ | :------------ | 
 | `concat_token_ids` | Concatenated context and response token IDs for this step |
 | `concat_response_logprobs` | Per-token logprobs, with context positions filled by `0.0` |
 | `concat_response_mask` | Loss mask, with context positions set to `0` and generated response positions set to `1` |
 | `concat_versions` | Token weight-version markers |
 | `concat_context_token_count` / `concat_output_token_count` | Context and generated-token counts |
 | `concat_logprobs_invalid` / `tito_incremental_tokenization_failed` | Safety flags for TITO assembly |

At finalize time, the `tito` token build mode stitches all fragments into a single coherent sequence per lineage segment.

### Append-Only Contract

TITO depends on append-only conversation history:

- **Allowed**: New messages append to the previous message prefix without changing earlier content or tool schemas
- **Forbidden**: Rewriting, compacting, reordering earlier messages, or changing the existing message prefix
- **On violation**: Segment boundary triggered — current segment finalizes, new segment starts with fresh TITO state

> [!NOTE]
> On TITO failure (template rendering error, encoding mismatch): marks `concat_incremental_tokenization_failed=True`, starts a new segment. No data is lost — just split into separate segments.

### Key Modules

 | Module | Role | 
 | :------- | :----- | 
 | `dressage/proxy/tito/tito_tokenizer.py` | `Qwen35TITOTokenizer` — implements incremental tokenization for Qwen3.5 | 
 | `dressage/proxy/tito/template_utils.py` | Fixed-template rendering — renders messages using a pinned Jinja template | 
 | `dressage/proxy/tito/templates/qwen3_5_fixed.jinja` | Pinned chat template — ensures consistent rendering across versions | 

<details>
<summary><b> Configuration</b></summary>
<br>

```bash
dressage-proxy \
  --tokenizer-path /path/to/Qwen3.5-4B \
  --token-build-mode tito \
  --token-build-model qwen3_5 \
  --tito-model qwen3_5
```

</details>

### Mode Comparison

 | | `snapshot` | `tito` |
 | :-- | :-----------: | :---------------: |
 | **Token Source** | Last step's `all_token_ids` | Concatenated TITO fragments |
 | **Multi-Turn Context** | Step snapshot | Full trajectory per lineage segment (incremental) |
 | **Prefix Consistency** | Not guaranteed | Guaranteed |
 | **Model Support** | General (any model) | `qwen3_5` with fixed template |
 | **Best For** | Shorter trajectories (1-3 turns) | Long agentic rollouts (10+ turns) |
 | **Overhead** | Lower (single tokenization) | Slightly higher (per-step recording) | 

## ⚖️ Prompt-Equal Aggregation

With multi-segment training, slime's default per-segment `rollout_mask_sums` biases gradient scale toward trajectories that split into more segments. If prompt A produces 3 segments and prompt B produces 1, prompt A gets 3× the gradient contribution — even though it's just one trajectory.

Dressage fixes this with **prompt-equal denominators** that normalize gradient contribution per prompt, not per segment.

### Why Prompt-Equal Is the Default

Prompt-equal scaling matches the natural unbiasedness property of group-based RL. For a group of trajectories that share the same prefix, the shared-prefix tokens have identical policy-gradient terms across the group. Since GRPO-style advantages are centered within the prompt group, those shared-prefix contributions cancel to zero in expectation. In other words, the shared prefix does not receive a biased update just because several continuations branched from it.

This is especially important for multi-segment agent trajectories: splitting one prompt into more samples should not turn the common prefix or the same prompt into extra gradient weight. Prompt-equal denominators preserve the intended unit of averaging: one prompt contributes as one prompt, regardless of how many live segments it produced.

The same choice is supported empirically by ScaleRL in *The Art of Scaling Reinforcement Learning Compute for LLMs* ([arXiv:2510.13786](https://arxiv.org/abs/2510.13786)), which compares sample-average, prompt-average, and token-average loss aggregation and reports prompt-average as the best-performing option for its ScaleRL recipe. For that reason, Dressage uses prompt-equal denominators as the default for supported GRPO-family estimators.

### Formula

```text
Per-sample denominator = M_P × N_P / gbs

Where:
  M_P = total loss-mask sum for prompt P (all live segments combined)
  N_P = number of distinct prompts with live samples in the batch
  gbs = global_batch_size
```

### Key Identifiers

 | ID | Where Set | Purpose | 
 | :--- | :---------- | :-------- | 
 | `metadata["instance_id"]` | Proxy header `X-Instance-Id` | Groups segments by prompt identity | 
 | `metadata["parent_traj_id"]` | Proxy header `X-Session-Id` | Groups segments by trajectory identity | 

<details>
<summary><b> Worked Example</b></summary>
<br>

With `gbs=4`, two prompts in the batch:

 | Prompt | Segments | Mask Sums | M_P | N_P | Per-Sample Denom | 
 | :------- | :--------- | :---------- | :---- | :---- | :----------------: | 
 | prompt-1 | 2 segments | 100 + 150 | 250 | 2 | 250 × 2 / 4 = 125 | 
 | prompt-2 | 1 segment | 200 | 200 | 2 | 200 × 2 / 4 = 100 | 

Both segments of prompt-1 share the same denominator (125), ensuring fair gradient contribution. Without prompt-equal scaling, prompt-1's segments would each get denominators of 100 and 150 respectively — giving prompt-1 disproportionate gradient weight.

> [!TIP]
> Dead samples (`remove_sample=True`) are excluded from both M_P and N_P, so aborted trajectories do not distort gradient scaling.

</details>

### Estimator Support

 | Estimator | `rollout_mask_sums` | 
 | :---------- | :-------------------: | 
 | `grpo` | Prompt-equal (Dressage) | 
 | `reinforce_plus_plus_baseline` | Prompt-equal (Dressage) | 
 | `gspo` and others | Trajectory-equal (slime default) |

## ⏸️ Partial Rollout

Preserves in-flight blackbox generation across weight updates by resuming at token boundaries instead of restarting trajectories. This is what makes continuous training possible — without partial rollout, every weight update would discard all in-progress agent work.

### Flow

```text
Agent is generating tokens (turn 5 of a long trajectory)
        │
Weight update imminent
        │
        ▼
train_async_with_rollout_pause
        │
        ├── POST /v1/rollout/pause → proxy
        │   └── GenerationController aborts SGLang at next token boundary
        │   └── Partial output preserved in current StepRecord
        │
        ├── Weight update executes (Megatron step)
        │
        └── POST /v1/rollout/resume → proxy
            └── GenerationController re-enables generation
            └── Agent's next chat_completions call picks up seamlessly
```

### Token-Level Weight Version Tracking

When one segment spans multiple weight versions (agent was generating, got paused, resumed with new weights):

- `write_sample_from_segment` sets `metadata["dressage_partial_rollout"]=True`
- Records weight version spans: which tokens came from which weight version
- Stores token-version metadata in `metadata["full_versions"]`, `metadata["version_spans"]`, `metadata["dressage_start_token_version"]`, and `metadata["dressage_end_token_version"]`
- Supports `--mask-nonlast-version-tokens`: masks out trainable tokens not generated by the latest weight version in partial rollout.

<details>
<summary><b> Configuration</b></summary>
<br>

```bash
# Enable proxy pause/resume around weight updates
DRESSAGE_PROXY_PAUSE_AROUND_WEIGHT_UPDATE=1
DRESSAGE_PROXY_PAUSE_REQUIRED=1
DRESSAGE_PROXY_PAUSE_TIMEOUT_SEC=300

# Dressage proxy startup flags
--dressage-partial-rollout
--record-token-versions
--mask-nonlast-version-tokens
```

</details>

### BlackboxServer Integration

BlackboxServer's in-process LLM proxy injects `X-Dressage-Partial-Rollout: 1` on proxied chat calls, but token-version tracking is controlled by Dressage proxy startup flags: `--dressage-partial-rollout`, `--record-token-versions`, and `--mask-nonlast-version-tokens`.

### Orthogonal to Multi-Segment

> [!NOTE]
> Partial rollout (resume within a step) and multi-segment (split at history boundaries) are **orthogonal** — they solve different problems and can coexist within the same trajectory:
>
> ```text
>                     Multi-Segment (split at history boundaries)
>                     ─────────────────────────────────────────
> Partial Rollout     │  segment₀  │  segment₁  │  segment₂  │
> (resume within)     │  w₁ → w₂   │  w₂        │  w₂ → w₃   │
>                     └────────────┴────────────┴────────────┘
>
> segment₀ spans weight versions w₁ and w₂ (got paused mid-generation)
> segment₁ is fully within w₂
> segment₂ spans w₂ and w₃ (another pause/resume)
> ```

## 🎯 Reward Post-Processing

The reward post-processing hook handles two responsibilities: GRPO advantage normalization and multi-segment advantage broadcast.

### GRPO Normalization

Groups samples by `group_index` → computes per-group mean/std → normalizes advantages. This is standard GRPO behavior, applied before the broadcast step.

### Anchor Advantage Broadcast

After normalization, the anchor segment's advantage is broadcast to all sibling segments within the same trajectory:

```text
Before broadcast:
  [seg₀: reward=0, adv=0]   [seg₁: reward=0, adv=0]   [seg₂: reward=R, adv=A]
                                                              ↑ anchor

After broadcast:
  [seg₀: reward=0, adv=A]   [seg₁: reward=0, adv=A]   [seg₂: reward=R, adv=A]
                                                              ↑ anchor
```

### Key Design Decisions

- **Raw rewards stay sparse** — only the anchor carries the terminal reward. This preserves correct trajectory-level reward logging.
- **Advantages are broadcast** — non-anchor segments receive the anchor's computed advantage (not raw reward). This ensures all segments contribute meaningfully to training.
- **Non-GRPO path** — For estimators other than GRPO, advantages are still broadcast. Without this, non-anchor segments would have zero advantage while carrying non-zero trainable tokens — effectively wasting those tokens.

> [!TIP]
> The broadcast ensures that if a trajectory gets a high reward, the model learns from all its segments (including early analysis and debugging), not just the final segment that produced the reward.

## 📊 `convert_samples_to_train_data`

This function (`dressage/rollout/convert_samples.py`) transforms Samples into slime's internal `train_data` format. It is a near-verbatim copy of slime's `_convert_samples_to_train_data` with one critical change: **prompt-equal `rollout_mask_sums`** for `grpo` and `reinforce_plus_plus_baseline`.

The function handles:
- Token padding/truncation to `max_sequence_length`
- Loss mask alignment
- Logprob extraction
- Prompt-equal denominator computation
- Routing replay data extraction (when R3 is enabled)

> [!CAUTION]
> When bumping the slime submodule, always diff `convert_samples.py` against upstream `slime/ray/rollout.py` to catch any changes in the base implementation. Our prompt-equal additions need to be re-applied correctly.

## 🏋️ Training Entry Points

Dressage uses the normal slime training entry points unless proxy pause/resume is required around weight updates.

 | Mode | Entry Point | Rollout Function |
 | :--- | :---------- | :--------------- |
 | Sync | `train.py` / `slime.train` | `dressage.rollout.sync_rollout.generate_rollout_sync` |
 | Fully async | `train_async.py` | `dressage.rollout.fully_async_rollout.generate_rollout_fully_async` |
 | Pause/resume async | `python3 -m dressage.training.train_async_with_rollout_pause` | `dressage.rollout.partial_async_rollout.generate_rollout_partial_async` |

Pause/resume wiring example:

```bash
python3 -m dressage.training.train_async_with_rollout_pause \
  # Generate function
  --custom-generate-function-path \
    dressage.rollout.generate.blackbox_dispatch.generate \
  # Sample conversion with prompt-equal scaling
  --custom-convert-samples-to-train-data-path \
    dressage.rollout.convert_samples.convert_samples_to_train_data \
  # Reward post-processing with segment broadcast
  --custom-reward-post-process-path \
    dressage.training.reward_post_process.reward_post_process \
  # Partial async rollout scheduler
  --rollout-function-path \
    dressage.rollout.partial_async_rollout.generate_rollout_partial_async \
  # GRPO advantage estimation
  --advantage-estimator grpo
```

### `train_async_with_rollout_pause`

This module extends slime's async training loop with proxy pause/resume:

1. Before each weight update → `POST /v1/rollout/pause` to proxy
2. Wait for `GenerationController` to confirm all generation paused
3. Execute Megatron training step (weight update)
4. `POST /v1/rollout/resume` to proxy
5. Agent generation resumes with new weights

This is transparent to the agent — it just sees a slightly longer response time for one LLM call.

## 📁 Package Structure

```text
dressage/training/
├── train_async_with_rollout_pause.py  # Async training with proxy pause/resume
├── reward_post_process.py             # GRPO normalization + segment advantage broadcast
└── log_helpers.py                     # Training log formatting utilities

dressage/rollout/artifacts/
├── samples.py                         # Segment-to-Sample writing, token caps, version metadata
└── writer.py                          # Trajectory/sample/error artifact logging

dressage/proxy/tito/
├── tito_tokenizer.py                  # Qwen35TITOTokenizer implementation
├── template_utils.py                  # Fixed-template rendering for TITO
└── templates/
    └── qwen3_5_fixed.jinja            # Pinned Jinja chat template
```

## 🔗 Integration Points

 | Component | Relationship | 
 | :---------- | :------------ | 
 | [Proxy](./proxy.md) | Proxy records the raw trajectory segments that training layer processes | 
 | [Rollout](./rollout.md) | Rollout hooks produce Samples that training layer converts to train_data | 
 | [Paddock](./paddock.md) | Partial rollout pause/resume coordinates with paddock for agent lifecycle | 
 | [Recipes](./recipes.md) | Recipe reward functions feed into reward post-processing | 

---

[← Rollout](./rollout.md) · [Back to Main README](../README.md) · [Next: Recipes →](./recipes.md)
