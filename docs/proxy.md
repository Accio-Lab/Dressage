# Inference Proxy

**Token-Level Trajectory Recording for Agentic RL**

[← Back to Main README](../README.md) · [Overview](#-overview) · [Key Features](#-key-features) · [Core Modules](#-core-modules) · [Session Model](#-session--step-model) · [Build Modes](#-token-build-modes) · [Endpoints](#-http-endpoints) · [Routing Replay](#-routing-replay-r3)

## 📖 Overview

The Dressage **Proxy** is an OpenAI-compatible HTTP service that sits between agent rollouts and the SGLang inference router. It is the central nervous system of trajectory recording — every LLM call passes through it, and every token, logprob, and loss mask is captured for training. The proxy is what makes Dressage's training pipeline possible: without it, there would be no way to faithfully reconstruct the exact token sequences, probabilities, and decision boundaries that the agent produced during rollout.

> [!IMPORTANT]
> Agents never call SGLang directly. The proxy transparently forwards generation requests while building rich, training-ready trajectory data. This design ensures that token-level recording is always active, regardless of whether the agent is a Python whitebox loop or an external HTTP blackbox like `opencode`.

```text
Agent (whitebox or blackbox)
        │  POST /v1/chat/completions
        │  headers/body ids: X-Session-Id, X-SMG-Routing-Key, X-Instance-Id, X-Turn-Id
        ▼
Dressage Proxy
        │  forwards generation to SGLang
        │  records tokens, logprobs, loss masks per step
        │  tracks weight versions, MoE routing IDs
        ▼
SGLang Router  →  Policy Model
```

The proxy runs as a standalone FastAPI service (CLI: `dressage-proxy`) and is designed to handle concurrent sessions from multiple rollout workers. Each session represents one complete agent trajectory, and each call to `/v1/chat/completions` within a session appends a new **step** to the trajectory record.

## ✨ Key Features

- **OpenAI-Compatible API** — Drop-in replacement for `/v1/chat/completions`. Agents don't need any custom integrations — just point your `base_url` at the proxy. Supports streaming and non-streaming modes, tool calls, and all standard OpenAI chat completion parameters.
- **Per-Step Recording** — Every proxy call captures the full request messages, prompt/response token IDs, per-token logprobs, weight version stamps, and computed loss masks. These per-step records form the raw material for training data construction.
- **TITO Support** — When `tito` token build mode is active, the proxy records incremental tokenization data in fields such as `concat_token_ids`, `concat_response_logprobs`, `concat_response_mask`, and `concat_versions`. These fields are later stitched together at finalize time, guaranteeing exact prefix consistency across arbitrarily long multi-turn trajectories. See [TITO Tokenizer](#-tito-deep-dive) below.
- **Auto Segmentation** — The proxy automatically detects when an agent rewrites conversation history (compaction, summarization) or changes the available tool schema mid-trajectory. When this happens, it closes the current segment and starts a new one, preserving clean token boundaries for training. Each segment becomes an independent training sample.
- **Preemptible Generation** — The `GenerationController` can abort active SGLang generation at any token boundary in response to a weight update signal. Partial output is preserved in the step record, and generation continues after `/v1/rollout/resume` when the proxy was started with `--dressage-partial-rollout`. This enables continuous rollout without discarding in-flight computation.
- **Weight Version Tracking** — Every generated token is stamped with the model weight version that produced it. When a trajectory spans multiple weight updates (partial rollout), `--record-token-versions` stores the per-token versions and `--mask-nonlast-version-tokens` marks tokens from older versions for selective loss masking.
- **Routing Replay (R3)** — For Mixture-of-Experts (MoE) models, the proxy captures routed expert IDs per generated token via `--use-rollout-routing-replay`. This data is stored as base64-encoded chunks and forwarded to training for faithful MoE routing replay.
- **Configurable Parsers** — Pluggable tool call and reasoning extraction backends (`local`, `sglang_api`, `hybrid`). Both parser backends default to `sglang_api`; `local` parses model output directly, and `hybrid` tries SGLang first with local fallback. Reasoning parsers extract `<think>` blocks for models like Qwen3.
- **Version and Context Safety** — Non-partial trajectories are rejected if the model weight version or rollout epoch changes mid-trajectory (`trajectory_version_changed`). Proxy-side context checks return stable `context_overflow` payloads and clamp generation to the exact remaining context budget. `--max-output-tokens` is an optional additional per-request hard cap.

## 🧱 Core Modules

The proxy codebase is organized into focused, single-responsibility modules:

 | Module | Responsibility | 
 | :------- | :--------------- | 
 | `server.py` | FastAPI application — chat completions endpoint, session finalize, trajectory read. CLI entry point `dressage-proxy`. Handles request validation, header extraction, and response formatting. | 
 | `session_manager.py` | Per-session step management, turn tracking, and history-rewrite detection. Maintains the ordered list of `StepRecord` objects for each active session. Detects when conversation messages violate the append-only contract and triggers segment boundaries. | 
 | `trajectory_store.py` | Thread-safe in-memory segment store. Finalized segments are written here and can be read back by rollout code via `/trajectory/read`. Supports cleanup by session ID. | 
 | `generation_controller.py` | Preemptible SGLang generation for partial rollout. Wraps SGLang client calls with abort/resume capability. Manages generation state machine (idle → generating → paused → resumed). | 
 | `sglang_client.py` | Low-level SGLang router client with weight-version tracking. Sends generation requests, receives responses with token IDs and logprobs, records which weight version was active. | 
 | `tool_call_parser.py` | Model-specific tool call extraction from assistant responses. Supports multiple backend modes (`local` for direct parsing, `sglang_api` for SGLang-native, `hybrid` for fallback chain). Currently optimized for Qwen3.5 tool call format. |
 | `reasoning_parser.py` | Reasoning-content parsing for models that produce structured thinking blocks (e.g., Qwen3's `<think>...</think>` format). Separates reasoning tokens from action tokens for selective loss masking. |
 | `proxy_client.py` | Async HTTP client used by rollout code to interact with the proxy. Provides typed methods for `chat_completions`, `finalize_session`, and `read_trajectory`. |
 | `tool_call_ids.py` | Deterministic tool call ID generation. Ensures that tool call IDs are reproducible across re-runs, which is important for trajectory consistency. |
 | `last_step/prompt_assistant_mask.py` | Snapshot alignment helper. Builds masks for complete step snapshots. |

## 📋 Session & Step Model

A **session** (`session_id`) represents one complete agent trajectory — from the first user message through the final assistant response. Each call to `/v1/chat/completions` within a session appends a new **step** to the trajectory. Steps are ordered and immutable once recorded.

### What Each Step Records

Every step captures a comprehensive snapshot of one LLM interaction:

- **Request messages** — The full conversation history sent by the agent
- **Prompt token IDs** — Tokenized input with per-token logprobs (when available)
- **Response token IDs** — Generated output tokens with per-token logprobs
- **Weight versions** — Which model weight version generated each token
- **Loss masks** — Binary masks indicating which tokens are trainable
- **TITO fields** — Incremental tokenization data (`concat_token_ids`, `concat_response_logprobs`, `concat_response_mask`, etc.) when tito mode is active
- **Segment markers** — Whether this step triggered a segment boundary
- **MoE routing data** — Routed expert IDs per token (when R3 is enabled)

### Runtime Identifiers

Every `/v1/chat/completions` request must provide these identifiers for proper trajectory attribution. Headers are preferred, but `session_id`, `instance_id`, and `turn_id` body fields are accepted as fallbacks. `X-SMG-Routing-Key` is also accepted as a session routing key when `X-Session-Id` is absent.

 | Header | Purpose | Example | 
 | :------- | :-------- | :-------- | 
 | `X-Session-Id` | Trajectory key — becomes `parent_traj_id` in training samples | `sess-abc123` | 
 | `X-SMG-Routing-Key` | Alternate session key used by sticky routing proxies | `sess-abc123` |
 | `X-Instance-Id` | Prompt / task instance — used for prompt-equal gradient aggregation | `inst-xyz789` | 
 | `X-Turn-Id` | Optional explicit turn identifier for idempotency tracking | `turn-001` | 

> [!TIP]
> The `X-Instance-Id` header is critical for prompt-equal gradient scaling. All samples from the same prompt instance share a gradient denominator, ensuring fair contribution regardless of how many segments each trajectory produces.

## 🧬 Token Build Modes

Dressage supports two modes for converting proxy-recorded steps into training-ready segments. The choice of build mode fundamentally affects how token sequences are constructed and how multi-turn context is handled.

### TITO Mode (Default)

The default and recommended mode for long agentic trajectories. Segments are assembled by concatenating per-step TITO fragments across the full multi-turn context, guaranteeing exact prefix consistency.

```text
Turn 1 → TITO fragment₁ (system + user₁)
Turn 2 → TITO fragment₂ (asst₁ + tool₁ + user₂)
Turn 3 → TITO fragment₃ (asst₂ + tool₂ + user₃)
                    ↓
         stitch(fragment₁ + fragment₂ + fragment₃) → Segment
```

- With `token_build_model=qwen3_5`, infers `model_mask_type=qwen3_5`, `model_tool_call_type=qwen3_5`, `model_reasoning_type=qwen3`, and `tito_model=qwen3_5` in tito mode
- Best for **long agentic trajectories** (SWE tasks, coding agents, multi-step reasoning)
- Avoids retokenization drift — the #1 correctness challenge in agentic RL training
- Each fragment is independently tokenized, then IDs are concatenated (never re-tokenized as a whole)

### Snapshot Mode

A simpler mode where each segment is built from the last assistant step's full message snapshot. The entire conversation is re-tokenized from scratch at finalize time.

```text
Turn 1 → (context, not directly used)
Turn 2 → (context, not directly used)
Turn 3 → Full message list snapshot → tokenize → Segment
```

- Loss masks mark only assistant tokens as trainable
- Best for **shorter trajectories** where retokenization drift is negligible
- More general model support (no model-specific TITO template required)
- Not recommended for long multi-turn rollouts due to prefix inconsistency risk

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

## 🧬 TITO Deep Dive

TITO (Token-In-Token-Out) is the proxy's answer to the retokenization drift problem. In standard multi-turn LLM inference, re-encoding the full message list each turn can produce subtly different token IDs for the same prefix text — breaking the alignment between logprobs recorded at rollout time and the token sequences used during training.

### The Problem

```text
Turn 1:  tokenize("system: ... user: Hello")                          → [101, 202, 303]
Turn 2:  tokenize("system: ... user: Hello assistant: Hi user: How?")  → [101, 202, 304, ...]
                                                                              ↑ DRIFT! 303 ≠ 304
```

### How TITO Fixes It

```text
Turn 1:  encode("system: ... user: Hello")           → fragment₁ = [101, 202, 303]
Turn 2:  encode("assistant: Hi user: How?")           → fragment₂ = [405, 506]
         stitch(fragment₁ + fragment₂)                → [101, 202, 303, 405, 506]  ✅ prefix intact
```

The proxy stores TITO data in `StepRecord` fields:
- `concat_token_ids` — concatenated context and response token IDs for the step
- `concat_response_logprobs` — per-token logprobs, with context positions filled by `0.0`
- `concat_response_mask` — loss mask, with context positions set to `0` and generated response positions set to `1`
- `concat_versions` — token weight-version markers
- `concat_context_token_count` / `concat_output_token_count` — context and generated-token counts
- `concat_logprobs_invalid` / `tito_incremental_tokenization_failed` — safety flags for TITO assembly

### Append-Only Contract

TITO depends on an **append-only contract** on conversation history. If the agent rewrites history, changes the existing message prefix, changes tool schemas, or TITO tokenization fails, the proxy triggers a **segment boundary** — closing the current segment and starting a fresh one with TITO state reset.

> [!NOTE]
> On TITO failure (e.g., template rendering error), the proxy marks `concat_incremental_tokenization_failed=True` on the step and starts a new segment. This is a safe fallback — no data is lost, just split into separate segments.

## ✂️ Segment Boundaries

The proxy automatically splits one session into multiple segments when it detects events that would break token-level consistency. Understanding segment boundaries is important because each segment becomes an independent training sample.

 | Trigger | Detection | What Happens | 
 | :-------- | :---------- | :------------- | 
 | **History Rewrite** | Agent sends messages that don't extend the previous conversation | Current segment finalizes; new segment starts with fresh state | 
 | **Tool Schema Change** | Available tools change between turns | Segment boundary; new tool context starts clean | 
 | **TITO Prefix Mismatch** | The existing message prefix changes in tito mode | Current segment finalizes; new segment starts with fresh state |
 | **TITO Fallback** | Incremental tokenization fails (template error, encoding mismatch) | Marks failure flag; starts new segment with reset TITO state | 

> [!NOTE]
> Each segment becomes an independent training sample, but all segments from one session share the same `parent_traj_id` and `rollout_id`, ensuring they are grouped together during training.

`DRESSAGE_PROXY_MAX_STEPS_PER_SESSION` is a separate guard: once a proxy session already has that many steps, the next generation request returns HTTP 400 before generation. It does not finalize the session automatically.

## 🌐 HTTP Endpoints

The proxy exposes these endpoints for agent interaction and rollout management:

 | Endpoint | Method | Purpose | Details | 
 | :--------- | :------- | :-------- | :-------- | 
 | `/v1/models` | `GET` | Model listing | OpenAI-compatible model list passthrough. |
 | `/v1/chat/completions` | `POST` | Agent inference | OpenAI-compatible. Records step data. Requires session headers. | 
 | `/session/finalize` | `POST` | Finalize session | Closes all open segments, writes to trajectory store. | 
 | `/trajectory/read` | `POST` | Read segments | Returns finalized segments by session ID or trajectory ID. | 
 | `/trajectory/stats` | `GET` | Store stats | Reports in-memory trajectory store statistics. |
 | `/v1/rollout/pause` | `POST` | Pause generation | Signals `GenerationController` to abort at next token boundary. |
 | `/v1/rollout/resume` | `POST` | Resume generation | Re-enables generation after weight update completes. |
 | `/v1/rollout/pause_state` | `GET` | Pause state | Reports `GenerationController` pause/resume state. |
 | `/health` | `GET` | Health check | Returns active session, trajectory store, rollout pause, and proxy config state. |

### Preemptible Generation Flow

The `GenerationController` enables safe interruption of active generation for weight updates during partial rollout. This is critical for continuous training where rollout and training overlap.

```text
1️⃣  Weight update signal arrives
2️⃣  POST /v1/rollout/pause → GenerationController.abort()
3️⃣  Active SGLang request aborts at next token boundary
4️⃣  Partial output preserved in current StepRecord
5️⃣  Weight update completes
6️⃣  POST /v1/rollout/resume → GenerationController.resume()
7️⃣  Next chat_completions call picks up where generation left off
```

> [!TIP]
> The pause/resume mechanism is atomic — there's no window where tokens could be generated with stale weights. The `GenerationController` state machine guarantees clean transitions between `idle → generating → paused → resumed` states.

## 🚀 Usage

### Starting the Proxy

```bash
# With current startup and parser controls
dressage-proxy \
  --tokenizer-path /path/to/Qwen3.5-4B \
  --sglang-router-url http://<sglang-router-host>:<port> \
  --token-build-model qwen3_5 \
  --context-window 32768 \
  --rollout-temperature 1.0 \
  --record-token-versions \
  --mask-nonlast-version-tokens \
  --dressage-partial-rollout \
  --tool-call-parse-backend sglang_api \
  --reasoning-parse-backend sglang_api \
  --model-tool-call-type qwen3_5 \
  --model-reasoning-type qwen3
```

Output limits are resolved independently for each request:

- An Agent-provided `max_tokens` limits that model call.
- `--max-output-tokens`, when explicitly configured, is a Proxy-wide hard cap.
- With dynamic context limiting enabled, `context_window - prompt_tokens` is
  the physical budget. The Proxy uses the full remainder, including a final
  single token, and combines it with any request or Proxy cap using `min(...)`.
- If neither request nor CLI supplies a limit, the remaining context is used.
  If `--context-window` is also absent, the Proxy omits `max_new_tokens` and the
  backend chooses its default ([SGLang 0.5.12 SamplingParams](https://github.com/sgl-project/sglang/blob/v0.5.12.post1/python/sglang/srt/sampling/sampling_params.py)
  defaults to 128).

`--no-dynamic-max-tokens` disables only the context-derived clamp. It does not
change the request or optional CLI cap.

### Using the Proxy Client

```python
from dressage.proxy.proxy_client import ProxyClient

client = ProxyClient(proxy_url="http://localhost:8800")

# Send a chat completion
response = await client.chat_completions(
    {"model": "proxy-model", "messages": [{"role": "user", "content": "Hello!"}]},
    session_id="sess-001",
    instance_id="inst-001",
    turn_id="turn-001",
)

# Finalize the session
await client.finalize_session("sess-001", instance_id="inst-001")

# Read the trajectory
payload = await client.read_trajectory(session_id="sess-001", drain=True)
segments = payload["data"]
```

## 🔀 Routing Replay (R3)

For **Mixture-of-Experts (MoE)** models, the proxy can capture **routed expert IDs** per generated token, enabling faithful routing replay during training. Without R3, training would use random expert routing, potentially diverging from the rollout-time behavior.

```text
Proxy (--use-rollout-routing-replay)
        │
        ├── Requests routed expert IDs from SGLang for each generated token
        ├── Encodes expert ID arrays as base64 chunks for efficient transfer
        ├── Stores in trajectory segment metadata
        └── rollout.artifacts.samples.extract_routed_experts → training data
```

### Data Formats

R3 stores routed expert IDs as base64-encoded int32 payloads. Dressage supports three record shapes:

| Field                   | Description                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------ |
| `routed_experts`        | Direct payload for a single uninterrupted generation.                                |
| `routed_experts_chunks` | Chunked payload for partial or resumed generation.                                   |
| `routed_experts_parts`  | Multi-step wrapper for TITO segments; each part may contain direct data or chunks. |

Enable R3 by setting `--use-rollout-routing-replay` on the proxy.

## 🔧 Configurable Parsers

The proxy supports pluggable backends for tool call and reasoning extraction, accommodating different model architectures and SGLang configurations:

 | Parser Type | Backend | Description | 
 | :------------ | :-------- | :------------ | 
 | Tool Call | `local` | Direct model output parsing using model-specific regex/heuristics | 
 | Tool Call | `sglang_api` | Delegates to SGLang's built-in tool call extraction | 
 | Tool Call | `hybrid` | Tries `sglang_api` first, falls back to `local` on failure | 
 | Reasoning | `local` | Parses `<think>...</think>` blocks from model output | 
 | Reasoning | `sglang_api` | Delegates reasoning extraction to SGLang | 
 | Reasoning | `hybrid` | SGLang-first with local fallback | 

Both `--tool-call-parse-backend` and `--reasoning-parse-backend` default to `sglang_api`.

```bash
dressage-proxy \
  --tokenizer-path /path/to/Qwen3.5-4B \
  --tool-call-parse-backend sglang_api \
  --reasoning-parse-backend sglang_api \
  --model-tool-call-type qwen3_5 \
  --model-reasoning-type qwen3
```

> [!TIP]
> The `hybrid` backend is recommended for production. It leverages SGLang's optimized parsing when available, with graceful fallback to local parsing when SGLang doesn't support the model's format.

## 📊 Data Flow

The complete data flow from agent request to stored trajectory:

```text
┌─────────────┐     ┌──────────────────────────┐     ┌──────────────┐
│    Agent    │────▶│          Proxy           │────▶│   SGLang     │
│             │     │                          │     │   Router     │
│  whitebox   │◀────│  records per-step:       │◀────│              │
│ or blackbox │     │  • token IDs + logprobs  │     │   policy     │
└─────────────┘     │  • loss masks            │     │   model      │
                    │  • weight versions       │     └──────────────┘
                    │  • TITO fragments        │
                    │  • MoE routing IDs       │
                    └──────────┬───────────────┘
                               │ finalize
                    ┌──────────▼───────────────┐
                    │    Trajectory Store      │
                    │                          │
                    │    segments[]            │
                    │    ├── tokens[]          │
                    │    ├── logprobs[]        │
                    │    ├── loss_mask[]       │
                    │    ├── weight_vers[]     │
                    │    └── experts[]         │  ← MoE routing (optional)
                    └──────────────────────────┘
```

## 📁 Package Structure

```text
dressage/proxy/
├── server.py                     # FastAPI app, CLI entry point
├── session_manager.py            # Per-session step tracking
├── trajectory_store.py           # In-memory segment storage
├── generation_controller.py      # Preemptible generation
├── sglang_client.py              # SGLang router client
├── tool_call_parser.py           # Tool call extraction
├── reasoning_parser.py           # Reasoning content parsing
├── proxy_client.py               # Async client for rollout code
├── tool_call_ids.py              # Deterministic ID generation
├── last_step/                    # Snapshot alignment helpers
│   └── prompt_assistant_mask.py  # Assistant loss mask builder
└── tito/                         # TITO tokenizer
    ├── tito_tokenizer.py         # Qwen35TITOTokenizer
    ├── template_utils.py         # Fixed-template rendering
    └── templates/
        └── qwen3_5_fixed.jinja   # Pinned chat template
```

## 🔗 Integration Points

 | Component | Relationship | 
 | :---------- | :------------ | 
 | [Paddock](./paddock.md) | Paddock coordinates proxy sessions — each rollout creates a session via proxy client | 
 | [Sandbox](./sandbox.md) | BlackboxServer's in-process LLM proxy forwards all agent calls through the Dressage proxy | 
 | [BlackboxServer](./blackbox-server.md) | Injects session/turn headers on every LLM call, routing through proxy | 
 | [Rollout](./rollout.md) | Generate hooks use `ProxyClient` to manage sessions and read trajectories | 
 | [Training](./training.md) | Training layer consumes proxy-produced segments for TITO tokenization and multi-segment expansion | 

---

[← Back to Main README](../README.md) · [Next: Paddock →](./paddock.md)
