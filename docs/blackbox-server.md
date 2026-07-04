# BlackboxServer

**Unified HTTP Adapter for Agentic Blackbox Backends**

[← Back to Main README](../README.md) · [Overview](#-overview) · [Key Features](#-key-features) · [Backends](#-supported-backends) · [Architecture](#️-architecture) · [API Reference](#-api-reference) · [Data Flow](#-data-flow) · [Quick Start](#-quick-start)

## 📖 Overview

BlackboxServer is a bundled HTTP adapter service that **decouples the Dressage rollout manager from concrete agentic backends**. It sits inside sandboxes, manages exactly **one backend agent process** and **one active session at a time**, and transparently proxies all LLM calls back through the Dressage inference proxy.

The key insight: agent frameworks like `opencode`, `openclaw`, Claude Code, and Codex each have their own CLI interfaces, configuration formats, and communication protocols. BlackboxServer provides a **uniform HTTP interface** that the paddock can drive regardless of which backend is behind it. This is what makes it possible to swap agent frameworks with a single environment variable.

```text
blackbox_dispatch (rollout hook)
        │  paddock.register_agent / call_agent / pause / resume
        ▼
BlackboxServer :23456 (inside sandbox)
        │  one backend agent process
        │  one active session at a time
        │  in-process LLM proxy → Dressage Proxy
        ▼
opencode serve / openclaw gateway / claude -p / codex exec / …
        │  agent makes LLM calls
        │  proxy injects session headers
        ▼
Dressage Proxy → ⚡ SGLang Router
```

> [!IMPORTANT]
> BlackboxServer runs **inside** each sandbox slot. In local bubblewrap mode, each bwrap namespace has its own BlackboxServer process. In E2B mode, the server runs as a service in the cloud sandbox. The paddock communicates with it via HTTP from outside the sandbox.

## ✨ Key Features

- **Multi-Backend Support** — Pluggable adapter pattern supports `opencode` (code-editing agent via `opencode serve`), `openclaw` (OpenClaw Gateway via `/v1/chat/completions`), `claude_code` (Claude Code headless CLI via Anthropic Messages), and `codex` (Codex CLI via `codex exec --json`). Adding a new backend means implementing a single `BackendAdapter` class with `initialize`, `send_message`, `abort_session`, `health`, and `capabilities` methods; `pause` / `resume` can be overridden when the backend supports them.
- **In-Process LLM Proxy** — Every BlackboxServer instance runs a lightweight HTTP proxy that intercepts all outgoing LLM calls from the backend agent. This proxy injects session headers (`X-Session-Id`, `X-Instance-Id`, `X-Turn-Id`), routing keys, and partial rollout markers transparently. The agent never knows its calls are being recorded.
- **Turn Idempotency** — Each turn is identified by a `(turn_id, messages_hash)` tuple. Retrying the same turn with the same messages returns cached responses. Retrying with different messages returns `409 Conflict`. This makes the protocol safe for network retries without duplicating agent work.
- **Register & Rebind** — Registration is idempotent: calling `POST /v1/rollout/register` with the same parameters while the server is ready is a no-op. If parameters change, the server returns `409 Conflict` while active or desynced sessions still exist; only after no open sessions remain can it tear down and re-initialize with the new configuration.
- **Health Monitoring** — A background poller periodically checks the backend agent process health. If the agent crashes or becomes unresponsive, the session is marked as `desynced` and the paddock is informed. This prevents silent failures where the agent dies but the rollout keeps waiting.
- **Single-Session Guarantee** — One server instance manages exactly one agent process and one active session. This ensures clean turn-context attribution: every LLM call within a session is guaranteed to carry the correct session/turn headers. For parallel rollout, deploy one BlackboxServer per sandbox slot.

## 🔌 Supported Backends

 | Backend | Status | Description | How It Works |
 | :-------- | :------- | :------------ | :------------- |
 | `opencode` | Implemented | Code-editing agent | Spawns `opencode serve` as a subprocess. Sends tasks via `/api/chat` endpoint. Agent writes code, runs tests, iterates. |
 | `openclaw` | Implemented | OpenClaw Gateway agent | Connects to OpenClaw Gateway's `/v1/chat/completions`. Agent uses OpenClaw's tool ecosystem for complex tasks. |
 | `claude_code` | Implemented | Claude Code agent | Runs `claude -p --output-format stream-json --verbose`; proxy bridges Anthropic Messages to OpenAI-compatible chat completions. |
 | `codex` | Implemented | Codex CLI coding agent | Runs `codex exec --json`; the adapter writes an isolated `CODEX_HOME/config.toml` with a Chat Completions custom provider pointing at the in-process proxy. |

### Adding a New Backend

To add a new backend, create a class that extends `BackendAdapter` in `blackbox_server/adapters/`:

```python
class MyBackendAdapter(BackendAdapter):
    async def initialize(self, binding_context: BindingContext) -> None:
        """Initialize runtime state, proxy config, and backend process."""
        ...

    async def send_message(
        self,
        session_context: SessionContext,
        turn_context: TurnContext,
        new_messages: list[Message],
    ) -> AdapterResponse:
        """Send a user turn and wait for completion."""
        ...

    async def abort_session(self, session_context: SessionContext) -> bool:
        """Abort the active backend session."""
        ...

    async def health(self) -> bool:
        """Check if the agent process is alive."""
        ...

    async def capabilities(self) -> BackendCapabilities:
        """Report supported protocol capabilities."""
        ...
```

Register the adapter in `blackbox_server/adapters/factory.py` and it will be available via `DRESSAGE_BLACKBOX_TYPE`.

## 🏗️ Architecture

The BlackboxServer has a layered internal architecture — FastAPI routes on top, server core in the middle, and the backend adapter + LLM proxy at the bottom:

```text
┌───────────────────────────────────────────────────┐
│              BlackboxServer :23456                │
│                                                   │
│  ┌──────────────────────────────────────────────┐ │
│  │            FastAPI App (api/)                │ │
│  │                                              │ │
│  │     /v1/rollout/register        → register   │ │
│  │     /v1/sessions/{id}/messages  → send turn  │ │
│  │     /v1/sessions/{id}/execute_cmd → shell    │ │
│  │     /v1/sessions/{id}/abort     → abort      │ │
│  │     /health                      → liveness  │ │
│  │     /v1/status                  → full state │ │
│  └──────────────┬───────────────────────────────┘ │
│                 │                                 │
│  ┌──────────────▼───────────────────────────────┐ │
│  │            Server Core (core/)               │ │
│  │                                              │ │
│  │     Register / rebind logic                  │ │
│  │     Session store (in-memory, single session)│ │
│  │     Turn idempotency ledger                  │ │
│  │     Backend health monitor (background task) │ │
│  │     Config change detection (hashing)        │ │
│  └──────────────┬───────────────────────────────┘ │
│                 │                                 │
│  ┌──────────────▼───────────────────────────────┐ │
│  │           Backend Adapter (adapters/)        │ │
│  │                                              │ │
│  │     Spawn agent subprocess                   │ │
│  │     Send messages / receive responses        │ │
│  │     Manage agent lifecycle                   │ │
│  │     Set turn context on LLM proxy            │ │
│  └──────┬───────────────┬───────────────────────┘ │
│         │               │                         │
│  ┌──────▼─────┐         │    control              │
│  │ LLM        │         │ (start/stop/context)    │
│  │ Proxy      │         │                         │
│  │ :AUTO_PORT │         │                         │
│  │            │         │                         │
│  │ → session  │         │                         │
│  │   headers  │         │                         │
│  │ → routing  │         │                         │
│  │   headers  │         │                         │
│  └──────┬─────┘         │                         │
│         │               │                         │
└─────────┼───────────────┼─────────────────────────┘
          │               │ subprocess
          │        ┌──────▼────────────┐
          │        │    opencode serve │
          │        │    or openclaw    │
          │        │    gateway        │
          │        │                   │
          │        │ baseURL → proxy   │
          │        └──────┬────────────┘
          │               │
          └───────────────┘
                  │ HTTP /v1/chat/completions
          ┌──────▼────────────────────────┐
          │       Dressage Proxy          │
          │       SGLang Router           │
          └───────────────────────────────┘
```

## 🌐 API Reference

### Management Endpoints

 | Method | Path | Purpose | Details |
 | :------- | :----- | :-------- | :-------- |
 | `GET` | `/health` | Liveness check | Returns 200 if the server is running. Used by supervisor for health monitoring. |
 | `GET` | `/v1/status` | Full server state | Returns binding info, session state, turn count, backend health, uptime. |
 | `POST` | `/v1/rollout/register` | Register backend | Idempotent registration. Starts agent process and LLM proxy. Returns session binding info. |
 | `POST` | `/v1/rollout/pause` | Pause generation | Forwards pause signal to proxy's `GenerationController`. |
 | `POST` | `/v1/rollout/resume` | ▶ Resume generation | Forwards resume signal after weight update completes. |
 | `GET` | `/v1/rollout/pause_state` | Pause state | Returns the current pause/resume state and in-flight request counters. |

### Session Endpoints

 | Method | Path | Purpose | Details |
 | :------- | :----- | :-------- | :-------- |
 | `POST` | `/v1/sessions/{id}/messages` | Send a user turn | Sends messages to agent, waits for completion. Supports turn idempotency via `turn_id`. |
 | `POST` | `/v1/sessions/{id}/execute_cmd` | Execute command | Run a shell command inside the sandbox. Returns stdout/stderr. |
 | `GET` | `/v1/sessions/{id}` | Get session info | Returns session state, turn history, and metadata. |
 | `POST` | `/v1/sessions/{id}/abort` | Abort session | Cleanly stops the agent session and marks it as aborted. |

## 🔄 Data Flow

### 1 Registration

When the paddock calls `POST /v1/rollout/register`, the BlackboxServer performs a multi-step initialization:

```text
Register Request (blackbox_type, router, bound_session_id, bound_instance_id, ...)
        │
        ├── Hash config → compare with current binding
        │   ├── Same config? → no-op, return existing binding
        │   └── Different config? → full teardown + re-init
        │
        ├── Start in-process LLM proxy on auto-assigned port
        │   └── Configure: upstream_url → Dressage Proxy URL
        │   └── Configure: session headers, routing key
        │
        ├── Start backend agent subprocess
        │   └── Set agent's baseURL → LLM proxy address
        │   └── Wait for agent health check to pass
        │
        └── Create session in session store
            └── Session state = "active"
```

### 2 Turn Execution

For each turn via `POST /v1/sessions/{id}/messages`:

```text
Turn Request (turn_id, messages)
        │
        ├── Check idempotency ledger
        │   ├── Same (turn_id, messages_hash)? → return cached response
        │   └── Same turn_id, different messages? → 409 Conflict
        │
        ├── Set proxy context → (session_id, turn_id)
        │   └── All subsequent LLM calls carry these headers
        │
        ├── Forward messages to backend agent
        │   └── Agent runs its logic (code editing, reasoning, etc.)
        │   └── Agent makes LLM calls → proxy → SGLang
        │   └── Every token is recorded by Dressage proxy
        │
        ├── Store response in idempotency ledger
        │
        └── Return agent response to paddock
```

### 3 Session Termination

```text
Abort Request
        │
        ├── Signal backend agent to stop
        ├── Mark session as "aborted" in store
        └── Clear proxy context
```

## 📡 In-Process LLM Proxy

The LLM proxy is a critical component that runs inside each BlackboxServer instance. It intercepts **every** outgoing LLM call from the backend agent and injects Dressage-specific headers before forwarding to the Dressage proxy. The agent is unaware that its calls are being intercepted.

### Injected Headers

 | Header | Purpose | Example |
 | :------- | :-------- | :-------- |
 | `<sticky_header_name>` | Session routing key (configurable, e.g., `X-SMG-Routing-Key`) | `sess-001` |
 | `X-Session-Id` | Session identifier for trajectory attribution | `sess-001` |
 | `X-Instance-Id` | Instance identifier for prompt-equal scaling | `inst-xyz` |
 | `X-Turn-Id` | Current turn identifier for step ordering | `turn-003` |
 | `X-Dressage-Partial-Rollout` | Injected as `1` on proxied chat calls from BlackboxServer | `1` |

> [!TIP]
> Token-version behavior is controlled by Dressage proxy startup flags such as `--dressage-partial-rollout`, `--record-token-versions`, and `--mask-nonlast-version-tokens`. The `X-Dressage-Partial-Rollout` header is injected by BlackboxServer but is not the feature toggle.

## 📋 Session States

A session progresses through defined states with clear transition rules:

 | State  | Description | Allowed Transitions |
 | :------ |  :------------ | :------------------- |
 | `active` |  Session is healthy and accepting turns. Agent process is alive, proxy is routing correctly. | → `desynced` (on failure) → `aborted` (on explicit abort) |
 | `desynced` |  A turn failed in unknown state — the agent may have partially executed, making turn attribution unreliable. Cannot accept new turns. | → `aborted` (must abort to recover) |
 | `aborted` |  Session has been cleanly or forcibly terminated. Agent process has stopped. | Terminal state — create a new session. |

> [!CAUTION]
> `desynced` is a **terminal** state for the session. The agent may have made partial progress that the proxy can't attribute correctly. The only safe recovery is to abort the session and create a fresh one. The paddock handles this automatically by calling `terminate` and re-initializing.

## 🚀 Quick Start

### Starting the Server

```bash
# Via CLI entry point
blackbox-server

# Via Python module
python -m blackbox_server.main
```

### Register a Backend

Registration requires `blackbox_type`, `router`, `bound_session_id`, and `bound_instance_id`. Optional request fields include `router_api_path`, `system_prompt_file`, `backend_options`, and `server_config`.

```bash
curl -X POST http://127.0.0.1:23456/v1/rollout/register \
    -H 'Content-Type: application/json' \
    -d '{
      "blackbox_type": "opencode",
      "router": "http://<dressage-proxy-host>:<port>",
      "router_api_path": "/v1",
      "bound_session_id": "sess-001",
      "bound_instance_id": "inst-001",
      "backend_options": {
        "provider_id": "sglang",
        "provider_name": "Dressage Proxy",
        "provider_package": "@ai-sdk/openai-compatible",
        "model_id": "proxy-model",
        "model_name": "Dressage Proxy",
        "proxy": {
          "sticky_header_name": "X-SMG-Routing-Key",
          "max_steps": 100,
          "default_temperature": 1.0
        }
      }
    }'
```

### Send a Message

```bash
curl -X POST http://127.0.0.1:23456/v1/sessions/sess-001/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "turn_id": "turn-001",
    "messages": [{"role": "user", "content": "Fix the bug in main.py"}]
  }'
```

### Check Status

```bash
curl http://127.0.0.1:23456/v1/status | python -m json.tool
```

## ⚙️ Environment Variables

<details open>
<summary><b> Server Configuration</b></summary>
<br>

 | Variable | Default | Description |
 | :--------- | :-------- | :------------ |
 | `BBS_HOST` | `0.0.0.0` | Bind host for the FastAPI server |
 | `BBS_PORT` | `23456` | Bind port for the FastAPI server |
 | `BBS_RUNTIME_ROOT` | `/tmp/blackbox_server` | Root directory for runtime files (logs, PID files, etc.) |
 | `BBS_MAX_SESSIONS` | `1` | Maximum tracked sessions. Should always be `1` for single-session guarantee. |
 | `BBS_MAX_TURNS` | `200` | Maximum turns per session before forced termination |
 | `BBS_BACKEND_TIMEOUT` | `960.0` | Timeout for agent calls in seconds (16 minutes default) |
 | `BBS_EXECUTE_CMD_TIMEOUT` | `600.0` | Timeout for `execute_cmd` calls in seconds (10 minutes default) |
 | `BBS_ROUTER_TIMEOUT` | `600000` | Timeout for upstream router requests from the in-process LLM proxy |
 | `BBS_SHUTDOWN_TIMEOUT` | `30.0` | Grace period for shutdown in seconds |
 | `BBS_RUNTIME_HEALTH_CHECK_INTERVAL` | `10.0` | Interval between backend runtime health checks |
 | `BBS_RUNTIME_HEALTH_CHECK_RETRIES` | `3` | Runtime health-check retry count |
 | `BBS_RUNTIME_HEALTH_CHECK_RETRY_DELAY` | `0.5` | Delay between runtime health-check retries |
 | `OPENCODE_BIN` | `opencode` | Path to the `opencode` binary |
 | `OPENCLAW_BIN` | `openclaw` | Path to the `openclaw` binary |
 | `CLAUDE_CODE_BIN` | `claude` | Path to the Claude Code binary |
 | `CODEX_BIN` | `codex` | Path to the Codex CLI binary |

</details>

> [!NOTE]
> `BlackboxServerConfig` has an internal class field default of `8080`, but runtime configuration is loaded through `from_env()`, where `BBS_PORT` defaults to `23456`.

### Backend Proxy Options

`backend_options.proxy` can tune the in-process LLM proxy:

 | Field | Default | Description |
 | :------ | :-------- | :------------ |
 | `sticky_header_name` | `X-SMG-Routing-Key` | Header used for sticky routing/session affinity. |
 | `max_steps` | `100` | Maximum proxied LLM calls before the turn is treated as max-step exceeded. |
 | `default_temperature` | `null` | Default temperature injected when the backend omits one. |

When BlackboxServer registration is built through Dressage paddock defaults, `DRESSAGE_BLACKBOX_TYPE` defaults to `opencode`. `DRESSAGE_BLACKBOX_MAX_STEPS` is forwarded into `backend_options.proxy.max_steps` as a positive integer, and `0` disables the limit. `DRESSAGE_BLACKBOX_COMPACT_THRESHOLD` must be positive and no greater than the context window; it controls backend compaction reserve sizing.

## ⚠️ Important Notes

 | Rule | Description |
 | :----- | :------------ |
 | **One server = one agent** | For parallel rollout, deploy one BlackboxServer per sandbox slot. The bwrap pool does this automatically. |
 | **One bound session** | The LLM proxy holds a single turn context — multiple concurrent sessions would corrupt turn attribution. |
 | **No inline system prompts** | System prompts are configured via `system_prompt_file` at registration time, not in per-turn messages. |
 | **Claude Code uses Anthropic Messages** | The `claude_code` adapter points Claude Code at the in-process proxy, which accepts `/v1/messages`, forwards OpenAI-compatible `/v1/chat/completions` to Dressage, and converts responses back. |
 | **Codex uses isolated local state** | The `codex` adapter sets sandbox-local `CODEX_HOME` / `CODEX_SQLITE_HOME`, removes inherited Codex/OpenAI auth env vars, and does not mount host `~/.codex`. |
 | **Codex defaults to full access inside the outer sandbox** | The `codex` backend defaults to `sandbox_mode=danger-full-access` and `approval_policy=never`; use it only inside Dressage's sandbox boundary or override `backend_options` for a stricter mode. |
 | **Rebinding conflicts while open** | Changing registration parameters returns `409 Conflict` while active or desynced sessions still exist. Rebind only proceeds after no open sessions remain. |
 | **Desynced is terminal** | A desynced session cannot accept new turns. Abort and create a fresh session. |
 | **Timeouts are generous** | Default 16-minute backend timeout accommodates complex coding tasks. Adjust for your workload. |

## 📁 Module Structure

```text
blackbox_server/
├── api/                    # FastAPI route handlers
│   ├── rollout.py             #   Registration, pause, resume endpoints
│   ├── sessions.py            #   Session message, execute_cmd, abort
│   └── health.py              #   Health and status checks
├── adapters/                # Backend implementations
│   ├── base.py                #   Abstract BackendAdapter interface
│   ├── opencode.py            #   opencode adapter (subprocess management)
│   ├── openclaw.py            #   openclaw adapter (gateway client)
│   ├── claude_code.py         #   Claude Code CLI adapter
│   ├── codex.py               #   Codex CLI adapter
│   └── factory.py             #   Adapter factory (type → class mapping)
├── core/                   # Server logic
│   ├── server.py              #   BlackboxServer core (register, rebind, health)
│   ├── models.py              #   Request/response Pydantic models
│   ├── monitoring.py          #   Background health monitor
│   ├── hashing.py             #   Config change detection via SHA hashing
│   ├── command.py             #   Shell command execution utilities
│   └── errors.py              #   Error types and error code mapping
├── proxy/                  # In-process LLM proxy
│   └── rollout_llm_proxy.py   #   HTTP proxy with header injection
├── store/                  # Session store
│   └── session_store.py       #   In-memory session + turn ledger
├── runtime/                # Path and runtime ID resolution
│   └── paths.py               #   Runtime directory layout
├── app.py                  # FastAPI app factory
├── config.py               # Configuration from environment
└── main.py                 # CLI entry point
```

## 🔗 Integration Points

 | Component | Relationship |
 | :---------- | :------------ |
 | [Paddock](./paddock.md) | Blackbox paddock drives the full BlackboxServer lifecycle via HTTP |
 | [Proxy](./proxy.md) | LLM proxy forwards all agent calls through Dressage proxy for token recording |
 | [Sandbox](./sandbox.md) | BlackboxServer runs inside sandbox slots (bwrap or E2B) |
 | [Rollout](./rollout.md) | `blackbox_dispatch` generate function orchestrates the paddock → BlackboxServer flow |

---

[← Sandbox](./sandbox.md) · [Back to Main README](../README.md) · [Next: Rollout →](./rollout.md)
