# Proposal: Pluggable Tool-Call Side-Effect Hooks for Blackbox Rollouts

## Motivation

During blackbox rollouts (opencode / claude_code / codex agents running inside
sandboxes), several workflows need to run side effects at tool-execution
boundaries:

- **Sandbox snapshotting** — capture sandbox state after each tool round for
  post-hoc analysis, reward computation, or reproducible debugging.
- **Network policy toggling** — keep sandboxes network-isolated by default and
  restore egress only while tools that need connectivity are about to run.
- **Per-step observability** — record which tools ran, when, and how long the
  gaps between tool rounds are, correlated with the training trajectory.

The Dressage proxy is the natural place to observe these boundaries: it already
normalizes all agent protocols (Anthropic Messages, OpenAI Responses, OpenAI
chat) into a single OpenAI chat format, so `role=tool` messages and parsed
`tool_calls` are agent-agnostic. However, the proxy never sees *actual* tool
execution — it only sees the request stream. This proposal adds a hook layer
at the two observable step boundaries, without coupling the proxy to any
specific sandbox provider.

## Design

### Hook points (derived from one `/chat/completions` request)

| Hook | Fires when | Semantics |
|---|---|---|
| `after_tool_call` | The normalized request messages end with `role=tool` | The previous step's tools **already executed** in the sandbox (e.g. take a snapshot) |
| `before_tool_call` | The response contains non-empty parsed `tool_calls`, **before** the response is returned | Tools are **about to execute** once the agent receives the response (e.g. restore networking). Runs synchronously; the agent is guaranteed to see a prepared sandbox |

Both are *step-boundary* level side effects, not per-tool interception.

### Interface (`dressage/proxy/tool_call_hooks.py`)

```python
@dataclass(frozen=True)
class ToolCallContext:
    instance_id: str | None        # training instance / rollout worker
    session_id: str                # == trajectory_id
    turn_id: str | None            # one agent call
    step_index: int                # one LLM request
    sandbox_id: str                # provider-native id (e.g. e2b), for
                                   # AsyncSandbox.connect(sandbox_id)
    tool_calls: list[dict] | None  # non-empty only for before-stage
    idempotency_key: str | None    # "session:turn:step:stage"
    stage_metadata: dict[str, Any] # hook results land in trajectory metadata


class ToolCallHook(abc.ABC):
    name: ClassVar[str]            # registry key
    required: ClassVar[bool]       # True: failure aborts the request (502)
    order: ClassVar[int] = 100     # before: ascending; after: descending

    def applies_to(self, ctx) -> bool: ...
    async def before_tool_call(self, ctx) -> None: ...
    async def after_tool_call(self, ctx) -> None: ...
```

### Key decisions

1. **Side-effect only** — hooks return nothing and cannot mutate requests or
   responses. No interceptor semantics.
2. **Pluggable, not built-in** — with no hooks configured the chain is `None`
   and dispatch is a zero-cost no-op.
3. **Run-level configuration** — one chain for the whole proxy lifetime; no
   per-task differentiation (keeps rollout determinism).
4. **Hooks own their provider client** — the proxy only dispatches and hands
   over context. Hooks reconnect to sandboxes themselves via the native
   `sandbox_id`, so the proxy stays provider-agnostic.
5. **Failure semantics mirror `blackbox_execute_cmds`** — `required=True`
   hook failures (or `before_timeout` expiry) abort the request with HTTP 502
   so the agent never executes tools against an unprepared sandbox; optional
   failures are logged and recorded into `stage_metadata`, and the chain
   continues.
6. **Idempotency** — each dispatch claims a `session:turn:step:stage` key;
   duplicate dispatches (HTTP retries, partial-rollout replays) are skipped.
   Keys are purged on session finalize/discard and expire after a TTL
   (default 1h) as a leak backstop.

### `sandbox_id` plumbing (required for e2b)

The provider-native sandbox id must travel from the rollout worker to the
proxy. This PR threads it through the existing register/binding path:

```
paddock.init() → lease.sandbox_id
  → register payload `bound_sandbox_id`          (dressage/paddock/blackbox/client.py)
  → BindingInfo.bound_sandbox_id                 (blackbox_server/core/models.py)
  → RolloutLLMProxy(bound_sandbox_id=...)        (all four adapters' _start_proxy)
  → header X-Dressage-Sandbox-Id per LLM request (blackbox_server/proxy/rollout_llm_proxy.py)
  → proxy parses header → ctx.sandbox_id         (dressage/proxy/server.py)
```

The header is reserved (client-supplied values are stripped), so in-sandbox
agents cannot spoof it. `bound_sandbox_id` also participates in the register
fingerprint so a changed sandbox id correctly triggers a full rebind instead
of an idempotent replay of a stale binding.

### Configuration

```bash
dressage-proxy --tokenizer-path ... \
    --tool-call-hooks e2b_snapshot restore_network \
    --tool-call-hook-before-timeout 5.0
```

Each entry may be a registered hook name, `module:ClassName`,
`/path/file.py:ClassName`, or a bare `/path/file.py` when the file registers
exactly one hook. Loading happens once at startup; bad references fail fast.

Hook results written to `ctx.stage_metadata` are persisted into finalized
trajectory segments under `extra_info["tool_call_hooks"]`, giving full
visibility into what side effects ran for every training sample.

## Files changed

| Layer | File | Change |
|---|---|---|
| new | `dressage/proxy/tool_call_hooks.py` | Context, abstract hook, chain dispatcher, registry, loader, idempotency keys |
| proxy | `dressage/proxy/server.py` | Parse `X-Dressage-Sandbox-Id`; dispatch after (request tail `role=tool`) and before (parsed `tool_calls`, pre-response); `create_app` params; CLI flags; metadata into StepRecord → finalize |
| proxy | `dressage/proxy/session_manager.py` | `StepRecord.tool_call_hook_metadata` field |
| in-sandbox proxy | `blackbox_server/proxy/rollout_llm_proxy.py` | `bound_sandbox_id` ctor param; attach `X-Dressage-Sandbox-Id` header; reserve header |
| binding | `blackbox_server/core/models.py` | `RegisterRequest` / `BindingInfo` gain `bound_sandbox_id` |
| binding | `blackbox_server/core/server.py` | Thread `bound_sandbox_id` into new bindings |
| binding | `blackbox_server/core/hashing.py` | Include `bound_sandbox_id` in the register fingerprint |
| adapters | `opencode.py` / `claude_code.py` / `codex.py` / `openclaw.py` | Pass `bound_sandbox_id` to `RolloutLLMProxy` |
| dressage client | `dressage/paddock/blackbox/client.py`, `paddock.py` | Send `state.sandbox_id` as `bound_sandbox_id` on register |
| example | `examples/tool_call_hooks_example.py` | Four documented example hooks (snapshot, network restore, selective watch, timing) |
| tests | `tests/test_tool_call_hooks.py` | 22 tests: chain semantics, registry/loader, pipeline integration, header forwarding, register plumbing, fingerprint regression |

## Out of scope (explicitly)

- Finalize-time hook fallbacks.
- Per-task hook configuration.
- Response-rewriting interceptors.
- Dynamic tool disabling via config mutation (blocked on opencode runtime
  permission changes; static deny stays with `backend_options` permissions).

## Example

See `examples/tool_call_hooks_example.py` for four ready-to-use hooks:

```bash
dressage-proxy --tokenizer-path ... \
    --tool-call-hooks examples/tool_call_hooks_example.py:e2b_snapshot
```
