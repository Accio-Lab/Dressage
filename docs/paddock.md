# Paddock

**Unified Environment Interaction for Agentic Rollouts**

[← Back to Main README](../README.md) · [Overview](#-overview) · [Design Principles](#-key-design-principles) · [Interface](#-interface-hierarchy) · [Blackbox Mode](#-blackbox-mode) · [Whitebox Mode](#-whitebox-mode) · [Factory](#-factory) · [Comparison](#whitebox-vs-blackbox-mode-comparison)

## 📖 Overview

The **Paddock** is a single-class abstraction that manages all environment interaction during rollouts. It handles sandbox lease creation, agent/tool calls, pause/resume lifecycle, and cleanup — providing a unified interface whether you're running whitebox Python agents or blackbox HTTP agents. The paddock is the coordination layer that ties together the proxy (token recording), sandbox (isolation), and agent (decision-making) into a coherent rollout lifecycle.

> [!TIP]
> Think of the Paddock as the **"what-to-do"** layer, while the [Sandbox](./sandbox.md) layer handles **"where-it-runs"**. The paddock doesn't care whether the agent runs in a local bubblewrap namespace or a remote E2B cloud sandbox — it just needs a `SandboxLease` and, for blackbox rollouts, a `blackbox` service endpoint.

```text
Rollout Generate Function
        │
        ▼
     Paddock
        ├── init(traj_id)
        ├── register_agent / call_agent / execute_cmd   (blackbox)
        │   or tool_call                                (whitebox)
        ├── pause / resume                              (blackbox)
        └── terminate(traj_id)
        │
        ▼
     SandboxProvider
        └── create / terminate sandbox leases
```

The paddock pattern ensures that every rollout follows the same lifecycle contract (`init → interact → terminate`), regardless of the agent type or sandbox backend. This makes it safe to swap agent modes and sandbox providers independently — the two axes are truly orthogonal.

## ✨ Key Design Principles

- **Single Interface** — One paddock class per rollout — blackbox or whitebox, never both. This prevents accidental mode mixing and ensures clean lifecycle management. The generate function creates the paddock once and reuses it across all samples in the batch.
- **Lifecycle Management** — Every trajectory follows `init(traj_id)` → interact → `terminate(traj_id)`. The paddock guarantees cleanup even on failures, preventing sandbox leaks. `init` asks the provider to `create()` a `SandboxLease`, and `terminate` returns that lease through provider `terminate()`.
- **Pluggable Backends** — The same paddock code works with bubblewrap or E2B sandboxes. The paddock receives a `SandboxProvider` at creation time and delegates all placement decisions to it. Swap providers via a single environment variable.
- **State Isolation** — Each trajectory maintains its own `SandboxState` tracking sandbox slot assignment, BlackboxServer endpoint URL, session and instance IDs, current lifecycle phase, and error state. No cross-trajectory contamination.
- **Factory Pattern** — `create_paddock_from_env()` reads `DRESSAGE_PADDOCK_MODE`, `DRESSAGE_SANDBOX_PROVIDER`, and other environment variables to wire up the correct paddock implementation with the correct sandbox provider. Zero configuration in application code.

## 🧱 Interface Hierarchy

The paddock system uses a clean inheritance hierarchy that separates the lifecycle contract from the interaction semantics:

```text
Paddock (abstract base)
├── init(traj_id)          → create sandbox lease, initialize state
└── terminate(traj_id)     → terminate or reclaim sandbox lease

BlackboxPaddock (extends Paddock)
├── register_agent(...)    → start BlackboxServer + backend agent
├── call_agent(...)        → send task prompt, wait for completion
├── execute_cmd(...)       → run shell commands in sandbox
├── pause(...)             → pause generation for weight update
└── resume(...)            → resume generation after update

WhiteboxPaddock (extends Paddock)
└── tool_call(traj_id, tool_id, tool_args) → execute tool in sandbox, return (text, metadata)
```

> [!NOTE]
> Source: `dressage/paddock/interface.py`. The base `Paddock` class defines the lifecycle contract; `BlackboxPaddock` and `WhiteboxPaddock` add the interaction methods specific to each agent paradigm.

## 🤖 Blackbox Mode

For HTTP-based agents running inside sandboxes (e.g., `opencode`, `openclaw`). The blackbox paddock manages the full lifecycle: sandbox acquisition, BlackboxServer startup, agent process management, turn execution, pause/resume for weight updates, and cleanup.

### Complete Lifecycle

```text
BlackboxAgentPaddock
        │
        ├── init(traj_id)         → build SandboxSpec with blackbox service
        │                         → provider.create(...) returns SandboxLease
        │                         → resolve blackbox endpoint + initialize SandboxState
        ├── register_agent(...)   → start BlackboxServer inside sandbox
        │                         → spawn backend agent process (opencode openclaw)
        │                         → configure in-process LLM proxy
        ├── execute_cmd(...)      → run setup commands (clone repos, install deps)
        │                         → optional before/after hooks per sample
        ├── call_agent(...)       → send task prompt to agent
        │                         → wait for agent completion or timeout
        │                         → handle errors, desync detection
        ├── pause / resume        → coordinate with proxy for weight updates
        │                         → GenerationController abort/resume
        └── terminate(traj_id)    → abort active session
                                  → stop backend process
                                  → provider.terminate(lease)
```

### Step Details

 | Method | What It Does |
 | :------- | :------------- |
 | `init(traj_id)` | Builds a `SandboxSpec` containing a `blackbox` `SandboxServiceSpec`, calls provider `create()`, resolves the `blackbox` endpoint from `lease.endpoints` or `get_public_url()`, then initializes per-trajectory `SandboxState`. Unless `DRESSAGE_BLACKBOX_SKIP_HEALTHCHECK` is set, the endpoint is health-checked before use. |
 | `register_agent(...)` | Sends `POST /v1/rollout/register` to the BlackboxServer inside the sandbox. The server spawns the backend agent process, starts the in-process LLM proxy, and configures session routing. Idempotent — re-registration with same params is a no-op. |
 | `execute_cmd(...)` | Runs optional shell commands inside the sandbox. Used for per-sample setup: cloning repositories, installing dependencies, creating workspace directories. Commands are specified via `metadata["blackbox_execute_cmds"]`; parsed `before_agent` / `after_agent` results are accumulated in `metadata["execute_cmds"]`. |
 | `call_agent(...)` | Sends the task prompt to the agent via `POST /v1/sessions/{id}/messages`, reusing one stable `turn_id` across all submit retries (so retries are idempotent). Defaults to `mode="async"` (submit + long-poll `GET /v1/sessions/{id}/turns/{turn_id}?wait=…` until the turn settles); set `DRESSAGE_BLACKBOX_AGENT_CALL_MODE=sync` for the legacy request-bound call. Either way returns a payload matching the legacy synchronous body. The agent makes LLM calls through BlackboxServer's proxy, which routes through the Dressage proxy for recording. |
 | `pause / resume` | Coordinates with the proxy's `GenerationController` for weight updates. `pause` signals all active generation to abort at token boundaries; `resume` re-enables generation after the weight update completes. |
 | `terminate(traj_id)` | Cleanly shuts down the trajectory lease through provider `terminate()`. Guaranteed to run even on errors via try/finally. |

<details>
<summary><b> State Tracking</b></summary>
<br>

Each trajectory maintains a `SandboxState` object that tracks:

- **Sandbox lease** — The provider-created lease (bwrap namespace or E2B sandbox)
- **BlackboxServer URL** — Endpoint for the server running inside the sandbox
- **Session ID** — Active session identifier for turn attribution
- **Instance ID** — Prompt instance identifier for gradient scaling
- **Lifecycle phase** — Current state: `uninitialized → initialized → registered → active → terminated`
- **Error state** — Captures failure details for logging and abort handling

</details>

<details>
<summary><b> Singleton Lifecycle</b></summary>
<br>

The blackbox dispatch hook holds a **process-lifetime paddock singleton** — it is created once at first use and reused across all samples in all batches. Each sample goes through `init → interact → terminate`, but the paddock instance itself persists. This amortizes the cost of sandbox provider initialization and Ray actor discovery.

The singleton is created by `create_paddock_from_env()` on first call and cached. Subsequent calls return the same instance. This pattern is safe because the paddock is stateless between trajectories — all per-trajectory state lives in `SandboxState`.

</details>

### Customization Hooks

 | Feature | Description |
 | :-------- | :------------ |
 | `metadata["blackbox_execute_cmds"]` | Per-sample shell commands to run before/after agent calls. Supports only `before_agent` and `after_agent`; each stage is a list of objects with `name`, `cmd`, boolean `required`, and optional `timeout`. |
 | `DRESSAGE_PADDOCK_CLASS` | Override the default paddock class with `module.Class` for custom lifecycle logic. |
 | `DRESSAGE_BLACKBOX_TYPE` / `metadata["blackbox_type"]` | Select the backend agent type: `opencode` (default, code-editing), `openclaw` (OpenClaw gateway). Per-sample metadata can override the environment default. |
 | `DRESSAGE_BLACKBOX_MAX_STEPS` | Positive int forwarded to `backend_options.proxy.max_steps`; `0` disables the backend proxy step limit. |
 | `DRESSAGE_BLACKBOX_COMPACT_THRESHOLD` | Positive int no greater than the context window; controls backend compaction reserve sizing. |
 | `DRESSAGE_BLACKBOX_AGENT_CALL_MODE` | `async` (default) or `sync`. `async` submits the turn (`202`) and long-polls `GET /turns/{turn_id}` until it settles; `sync` issues a request-bound `/messages` call and returns the result directly. Invalid values fall back to `async`. Both modes reuse one stable `turn_id` across submit retries. |
 | `DRESSAGE_BLACKBOX_AGENT_POLL_WAIT_SEC` | Per-poll long-poll `wait` seconds for async mode (default `30`, server-clamped to `60`). |
 | `DRESSAGE_BLACKBOX_AGENT_POLL_TOTAL_TIMEOUT_SEC` | Total client-side polling budget in seconds for async mode (default `0` = unbounded; the server `backend_timeout` remains the backstop). |

> [!WARNING]
> `blackbox_dispatch` rejects `DRESSAGE_PADDOCK_MODE=whitebox` at startup — use `whitebox_agent` generate function instead. This prevents accidental mode mismatches that would silently produce incorrect trajectories.

## 🔮 Whitebox Mode

For Python agents that call tools directly through the paddock. The whitebox paddock is simpler — it only needs sandbox access for shell/file operations, not a full BlackboxServer.

### Lifecycle

```text
WhiteboxToolPaddock
        │
        ├── init(traj_id)              → provider.create(...) sandbox lease
        │                              → setup working directory
        ├── tool_call(traj_id, ...)    → execute tool in sandbox, return (text, metadata)
        │       ├── shell.exec         → run shell commands
        │       ├── file.read          → read file contents
        │       └── file.write         → write file contents
        │       (repeat as needed for multi-turn agent loop)
        └── terminate(traj_id)         → provider.terminate(lease)
```

### Available Tools

The whitebox paddock exposes three tools that Python agents can call during rollout:

 | Tool | Arguments | Description |
 | :----- | :---------- | :------------ |
 | `shell.exec` | `{"cmd": "..."}` | Execute a shell command inside the sandbox. Returns `(text, metadata)`, where text is stdout or stderr. Supports timeouts. |
 | `file.read` | `{"path": "..."}` | Read file contents from the sandbox filesystem. Returns `(text, metadata)`. |
 | `file.write` | `{"path": "...", "content": "..."}` | Write content to a file in the sandbox. Returns `(text, metadata)`, with empty text and write metadata. |

### Agent Variants

Two whitebox agent base classes are available, depending on whether sandbox access is needed:

 | Class | Sandbox? | Description |
 | :------ | :--------- | :------------ |
 | `WhiteboxAgent` | No sandbox | Pure proxy agent for lightweight Python tools. Agent calls `self.chat(...)` to interact with the LLM via proxy. Tools are implemented as Python functions (API calls, retrieval, computation). No sandbox isolation needed. |
 | `PaddockWhiteboxAgent` | With sandbox | Extends `WhiteboxAgent` with sandbox lifecycle. Agent unpacks `text, metadata = await self.paddock.tool_call(self.session_id, tool_id, args)` to execute shell commands and file operations inside an isolated sandbox. The paddock manages `init/terminate` automatically. |

> [!TIP]
> `PaddockWhiteboxAgent` wraps the `init → interact → terminate` lifecycle inside the agent class itself, so the rollout hook does not need to manage sandbox state manually. The agent's `rollout()` method can freely call `self.paddock.tool_call(...)` without worrying about cleanup.

## 🏭 Factory

The paddock is created from environment variables using `create_paddock_from_env()`. This factory reads the configuration and wires up the correct paddock implementation with the correct sandbox provider — zero boilerplate in application code.

```python
from dressage.paddock.factory import create_paddock_from_env

paddock = create_paddock_from_env()
# Returns BlackboxAgentPaddock or WhiteboxToolPaddock
# with the configured SandboxProvider already attached
```

### Configuration

 | Variable | Values | Description |
 | :--------- | :------- | :------------ |
 | `DRESSAGE_PADDOCK_MODE` | `blackbox` \| `whitebox` | Agent interaction paradigm — determines which paddock subclass is created |
 | `DRESSAGE_PADDOCK_CLASS` | `module.Class` | Custom paddock class override — use for specialized lifecycle logic |
 | `DRESSAGE_SANDBOX_PROVIDER` | `local_bwrap` \| `e2b` | Where sandboxes run — determines which `SandboxProvider` is injected |
 | `DRESSAGE_BLACKBOX_TYPE` | `opencode` \| `openclaw` | Backend agent type (blackbox mode only; defaults to `opencode`) |
 | `DRESSAGE_BLACKBOX_MAX_STEPS` | int | Positive int forwarded to `backend_options.proxy.max_steps`; `0` disables the backend proxy step limit. |
 | `DRESSAGE_BLACKBOX_COMPACT_THRESHOLD` | int | Positive int no greater than the context window; controls backend compaction reserve sizing. |
 | `DRESSAGE_BLACKBOX_PORT` | int | Blackbox service port requested from the provider. |
 | `DRESSAGE_LOCAL_BWRAP_BLACKBOX_PORT` | int | Local bwrap-specific blackbox service port override. |
 | `DRESSAGE_BLACKBOX_SKIP_HEALTHCHECK` | `0` \| `1` | Skip endpoint health checks during blackbox `init`. |
 | `DRESSAGE_PADDOCK_TERMINATE_TIMEOUT_SEC` | float | Best-effort termination timeout. |
 | `DRESSAGE_LOCAL_BWRAP_POOL_MODE` | `blackbox` \| `command_only` | Bwrap pool mode — must match paddock mode |

> [!CAUTION]
> The factory validates that paddock mode and sandbox pool mode are compatible. Using a blackbox paddock with a `command_only` pool (or a whitebox paddock with a `blackbox` pool) raises a clear error before any sandbox lease is created.

## Whitebox vs Blackbox Mode Comparison

 | Dimension | Whitebox | Blackbox |
 | :---------- | :----------- | :----------- |
 | **LLM Caller** | Dressage Python agent via `self.chat(...)` | External agent process (opencode, openclaw) |
 | **Tools** | Python functions or `paddock.tool_call(...)` | Agent's built-in tools inside sandbox |
 | **Paddock Mode** | `whitebox` | `blackbox` |
 | **Generate Hook** | `WhiteboxAgent` or `PaddockWhiteboxAgent` subclass | `blackbox_dispatch.generate` |
 | **Sandbox Needs** | `command_only` pool — shell/file ops only | `blackbox` pool — full BlackboxServer endpoint |
 | **Proxy Integration** | Agent calls proxy directly via `ProxyClient` | BlackboxServer's LLM proxy routes through Dressage proxy |
 | **Complexity** | Simpler — more control over agent logic | More flexible — supports real-world agent frameworks |
 | **Best For** | Custom tools, retrieval, API calls, lightweight envs | Code-editing agents, complex environments, production agents |

```text
Whitebox Flow                           Blackbox Flow
──────────────                          ──────────────
WhiteboxAgent                           BlackboxServer (in sandbox)
  └─ self.chat(...)                       └─ in-process LLM proxy
     └─ proxy.chat_completions               └─ proxy.chat_completions
  └─ self.paddock.tool_call(...)          └─ agent tools (shell, files, etc.)
     └─ sandbox shell/file ops               └─ agent-internal execution
```

## 📁 Package Structure

```text
dressage/paddock/
├── interface.py           # Paddock, BlackboxPaddock, WhiteboxPaddock abstract interfaces
├── blackbox/
│   ├── __init__.py
│   ├── client.py          # BlackboxServer HTTP client
│   ├── paddock.py         # BlackboxAgentPaddock implementation
│   └── common/
│       ├── command.py     # Command execution helpers
│       ├── defaults.py    # Default configuration values
│       ├── http_retry.py  # HTTP retry with backoff
│       ├── state.py       # SandboxState tracking
│       └── utils.py       # Shared utilities
├── whitebox/
│   ├── __init__.py
│   ├── paddock.py         # WhiteboxToolPaddock implementation
│   └── tools.py           # Tool definitions (shell.exec, file.read, file.write)
└── factory.py             # create_paddock_from_env()
```

## 🔗 Integration Points

 | Component | Relationship |
 | :---------- | :------------ |
 | [Proxy](./proxy.md) | Paddock coordinates proxy sessions — creates session via `ProxyClient`, finalizes on trajectory completion |
 | [Sandbox](./sandbox.md) | Paddock uses provider `create()` / `terminate()` for sandbox lease management |
 | [BlackboxServer](./blackbox-server.md) | Blackbox paddock drives the full BlackboxServer lifecycle: register, call, pause, resume, abort |
 | [Rollout](./rollout.md) | Generate hooks hold a paddock singleton and call `init/terminate` per trajectory |
 | [Recipes](./recipes.md) | Recipe agents use `PaddockWhiteboxAgent` for sandbox-backed tool execution |

---

[← Proxy](./proxy.md) · [Back to Main README](../README.md) · [Next: Sandbox →](./sandbox.md)
