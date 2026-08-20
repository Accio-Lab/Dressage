"""Pluggable side-effect hooks around blackbox tool-call boundaries.

The Dressage proxy observes the *request stream* of blackbox CLI agents
(opencode / claude_code / codex ...).  Two step-boundary hook points are
derived from one ``/chat/completions`` request:

- ``after_tool_call``: the normalized request messages end with a
  ``role=tool`` message, which means the previous step's tools already
  executed inside the sandbox (e.g. take a sandbox snapshot).
- ``before_tool_call``: the model response for this request contains
  non-empty ``tool_calls``, which means tools are about to execute once
  the response is returned (e.g. restore sandbox networking).

Hooks are side-effect only: no return values, no request/response
mutation.  They are pluggable and not built in; when nothing is
configured the chain is a no-op.  Configuration is run-level (one chain
for the whole proxy lifetime).

Each hook carries its own provider client and connects to the sandbox
itself using the native ``sandbox_id`` (for e2b:
``AsyncSandbox.connect(sandbox_id)``); the proxy only dispatches and
hands over context.
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import time
from dataclasses import dataclass, field
import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
import sys
from typing import Any, ClassVar, Iterable

logger = logging.getLogger(__name__)

DEFAULT_KEY_TTL_SECONDS = 3600.0
_PRUNE_INTERVAL_SECONDS = 300.0
_PRUNE_MAX_KEYS_PER_ROUND = 10_000

__all__ = [
    "ToolCallContext",
    "ToolCallHook",
    "ToolCallHookChain",
    "ToolCallHookError",
    "register_tool_call_hook",
    "registered_tool_call_hooks",
    "reset_tool_call_hook_registry",
    "load_tool_call_hooks",
    "build_tool_call_hook_chain",
    "build_tool_call_idempotency_key",
]


@dataclass(frozen=True)
class ToolCallContext:
    """Context handed to every hook dispatch.

    Fields follow the rollout hierarchy: ``instance_id`` (training
    instance / rollout worker) > ``session_id`` (== trajectory id) >
    ``turn_id`` (one agent call) > ``step_index`` (one LLM request).
    ``sandbox_id`` is the provider-native sandbox id (e.g. the e2b
    sandbox id) used by hooks to reconnect to the sandbox from the
    proxy side.
    """

    instance_id: str | None
    session_id: str
    turn_id: str | None
    step_index: int
    sandbox_id: str
    tool_calls: list[dict] | None = None
    idempotency_key: str | None = None
    stage_metadata: dict[str, Any] = field(default_factory=dict)


class ToolCallHookError(RuntimeError):
    """Raised when a required hook fails or times out."""


class ToolCallHook(abc.ABC):
    """Base class for pluggable tool-call side-effect hooks."""

    name: ClassVar[str] = ""
    required: ClassVar[bool] = False
    order: ClassVar[int] = 100

    def applies_to(self, ctx: ToolCallContext) -> bool:  # noqa: ARG002
        return True

    async def before_tool_call(self, ctx: ToolCallContext) -> None:  # noqa: ARG002
        return None

    async def after_tool_call(self, ctx: ToolCallContext) -> None:  # noqa: ARG002
        return None


# ---------------------------------------------------------------------------
# Registry (decorator / subclass registration)
# ---------------------------------------------------------------------------

_HOOK_REGISTRY: dict[str, type[ToolCallHook]] = {}


def register_tool_call_hook(hook_cls: type[ToolCallHook]) -> type[ToolCallHook]:
    """Register a hook class; usable directly or as a ``@`` decorator."""

    if not (inspect.isclass(hook_cls) and issubclass(hook_cls, ToolCallHook)):
        raise TypeError("register_tool_call_hook expects a ToolCallHook subclass")
    name = getattr(hook_cls, "name", "")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{hook_cls.__name__} must define a non-empty 'name'")
    key = name.strip()
    existing = _HOOK_REGISTRY.get(key)
    if existing is not None and existing is not hook_cls:
        # The same physical file imported through two references (e.g.
        # "path.py:Hook" and "module.path:Hook") produces two distinct
        # class objects; treat identical qualnames as the same hook.
        if existing.__qualname__ != hook_cls.__qualname__:
            raise ValueError(
                f"tool-call hook name {key!r} is already registered by "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        return hook_cls
    _HOOK_REGISTRY[key] = hook_cls
    return hook_cls


def _register_subclasses(module: Any) -> None:
    for attribute in vars(module).values():
        if (
            inspect.isclass(attribute)
            and issubclass(attribute, ToolCallHook)
            and attribute is not ToolCallHook
            and getattr(attribute, "name", "")
            and attribute.__module__ == module.__name__
        ):
            register_tool_call_hook(attribute)


def registered_tool_call_hooks() -> dict[str, type[ToolCallHook]]:
    return dict(_HOOK_REGISTRY)


def reset_tool_call_hook_registry() -> None:
    _HOOK_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Loading (run-level, one-shot)
# ---------------------------------------------------------------------------


def load_tool_call_hooks(spec: str | Iterable[str] | None) -> list[ToolCallHook]:
    """Load hooks for one run.

    ``spec`` is a list of hook names (or a comma/space separated string).
    Each entry is resolved against the registry.  Entries containing ``:``
    or a path separator are import references processed first so their
    module-level registration side effects run:

    - ``module.path:ClassName``
    - ``/path/to/file.py:ClassName``
    - ``/path/to/file.py`` (allowed when the file registers exactly one hook)
    """

    if spec is None:
        return []
    if isinstance(spec, str):
        names = [item for item in spec.replace(",", " ").split() if item]
    else:
        names: list[str] = []
        for value in spec:
            names.extend(item for item in str(value).split() if item)

    hooks: list[ToolCallHook] = []
    for raw_name in names:
        name = raw_name.strip()
        if not name:
            continue
        if ":" in name or "/" in name or "\\" in name:
            hooks.append(_load_hook_from_reference(name))
            continue
        hook_cls = _HOOK_REGISTRY.get(name)
        if hook_cls is None:
            raise ValueError(
                f"unknown tool-call hook {name!r}; registered hooks: "
                + (", ".join(sorted(_HOOK_REGISTRY)) or "<none>")
            )
        hooks.append(hook_cls())
    return hooks


def _load_hook_from_reference(reference: str) -> ToolCallHook:
    if ":" in reference:
        module_part, _, class_name = reference.rpartition(":")
    else:
        module_part, class_name = reference, None
    module = _import_hook_module(module_part, reference=reference)
    if class_name:
        hook_cls = vars(module).get(class_name)
        if not (inspect.isclass(hook_cls) and issubclass(hook_cls, ToolCallHook)):
            raise ValueError(
                f"{reference!r} does not reference a ToolCallHook subclass"
            )
        return hook_cls()
    registered = [
        hook_cls
        for hook_cls in _HOOK_REGISTRY.values()
        if hook_cls.__module__ == module.__name__
    ]
    if len(registered) == 1:
        return registered[0]()
    raise ValueError(
        f"{reference!r} registers {len(registered)} hooks; "
        "qualify the entry as 'path:ClassName'"
    )


def _import_hook_module(module_part: str, *, reference: str) -> Any:
    looks_like_path = (
        "/" in module_part
        or "\\" in module_part
        or module_part.endswith(".py")
    )
    if not looks_like_path:
        module = importlib.import_module(module_part)
        _register_subclasses(module)
        return module

    path = Path(module_part)
    if not path.is_file():
        raise ValueError(f"tool-call hook file not found: {reference!r}")
    # Distinct files sharing a stem must not collide in sys.modules.
    path_token = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    module_name = f"dressage_tool_call_hooks.{path.stem}_{path_token}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import tool-call hook file: {reference!r}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    _register_subclasses(module)
    return module


def build_tool_call_hook_chain(
    hooks: Iterable[ToolCallHook] | None,
    *,
    before_timeout: float | None = None,
    key_ttl_seconds: float | None = DEFAULT_KEY_TTL_SECONDS,
) -> ToolCallHookChain | None:
    """Build a chain from hooks, or ``None`` when no hooks are configured.

    ``key_ttl_seconds`` bounds how long idempotency keys are remembered
    (pass ``None`` to never expire).  Keys only need to survive
    near-term retries and partial-rollout replays.
    """

    hook_list = list(hooks or [])
    if not hook_list:
        return None
    return ToolCallHookChain(
        hook_list,
        before_timeout=before_timeout,
        key_ttl_seconds=key_ttl_seconds,
    )


def build_tool_call_idempotency_key(
    *,
    session_id: str,
    turn_id: str | None,
    step_index: int,
    stage: str,
) -> str:
    """Stable key (e.g. ``session:turn:step``) for duplicate suppression."""

    return f"{session_id}:{turn_id or '-'}:{step_index}:{stage}"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class ToolCallHookChain:
    """Ordered dispatcher over :class:`ToolCallHook` implementations.

    ``before`` hooks run in ascending ``order``; ``after`` hooks run in
    descending ``order``.  Failures follow the execute-cmd semantics: a
    ``required`` hook failure (or timeout) aborts the request; optional
    failures are logged and the chain continues.
    """

    def __init__(
        self,
        hooks: Iterable[ToolCallHook],
        *,
        before_timeout: float | None = None,
        key_ttl_seconds: float | None = DEFAULT_KEY_TTL_SECONDS,
    ) -> None:
        self._hooks = sorted(hooks, key=lambda hook: hook.order)
        self._before_timeout = before_timeout
        self._key_ttl_seconds = key_ttl_seconds
        # session_id -> {idempotency_key: claimed_at (monotonic)}
        self._seen_keys: dict[str, dict[str, float]] = {}
        self._last_prune = time.monotonic()

    @property
    def hooks(self) -> list[ToolCallHook]:
        return list(self._hooks)

    @property
    def before_timeout(self) -> float | None:
        return self._before_timeout

    @property
    def key_ttl_seconds(self) -> float | None:
        return self._key_ttl_seconds

    def purge_session(self, session_id: str) -> int:
        """Drop all idempotency keys for one session; return the count.

        Called when a session is discarded or finalized so a rebuilt
        session reusing the same id (and step indexes) is not wrongly
        suppressed as a duplicate.
        """

        return len(self._seen_keys.pop(session_id, None) or {})

    def _applicable(self, ctx: ToolCallContext) -> list[ToolCallHook]:
        applicable: list[ToolCallHook] = []
        for hook in self._hooks:
            try:
                if hook.applies_to(ctx):
                    applicable.append(hook)
            except Exception:
                logger.warning(
                    "tool-call hook %s applies_to() failed; skipping hook",
                    type(hook).__name__,
                    exc_info=True,
                )
        return applicable

    def _claim_idempotency(self, ctx: ToolCallContext) -> bool:
        """Return True when this key has not been dispatched yet."""

        key = ctx.idempotency_key
        if not key:
            return True
        now = time.monotonic()
        self._maybe_prune(now)
        session_keys = self._seen_keys.setdefault(ctx.session_id, {})
        claimed_at = session_keys.get(key)
        if claimed_at is not None and not self._expired(claimed_at, now):
            return False
        session_keys[key] = now
        return True

    def _expired(self, claimed_at: float, now: float) -> bool:
        if self._key_ttl_seconds is None:
            return False
        return (now - claimed_at) >= self._key_ttl_seconds

    def _maybe_prune(self, now: float) -> None:
        """Lazily evict expired keys at most once per prune interval.

        Each round scans at most ``_PRUNE_MAX_KEYS_PER_ROUND`` keys so a
        very large key store cannot stall the dispatch hot path; the
        next round resumes from where this one stopped.
        """

        if self._key_ttl_seconds is None:
            return
        if now - self._last_prune < _PRUNE_INTERVAL_SECONDS:
            return
        self._last_prune = now
        scanned = 0
        for session_id, keys in list(self._seen_keys.items()):
            if scanned >= _PRUNE_MAX_KEYS_PER_ROUND:
                break
            for key, claimed_at in list(keys.items()):
                if scanned >= _PRUNE_MAX_KEYS_PER_ROUND:
                    break
                scanned += 1
                if self._expired(claimed_at, now):
                    keys.pop(key, None)
            if not keys:
                self._seen_keys.pop(session_id, None)

    async def run_before(self, ctx: ToolCallContext) -> None:
        if not self._claim_idempotency(ctx):
            logger.debug(
                "skipping duplicate before_tool_call dispatch: key=%s",
                ctx.idempotency_key,
            )
            return
        for hook in self._applicable(ctx):
            await self._run_one(hook, hook.before_tool_call, ctx, stage="before")

    async def run_after(self, ctx: ToolCallContext) -> None:
        if not self._claim_idempotency(ctx):
            logger.debug(
                "skipping duplicate after_tool_call dispatch: key=%s",
                ctx.idempotency_key,
            )
            return
        for hook in reversed(self._applicable(ctx)):
            await self._run_one(hook, hook.after_tool_call, ctx, stage="after")

    async def _run_one(
        self,
        hook: ToolCallHook,
        method: Any,
        ctx: ToolCallContext,
        *,
        stage: str,
    ) -> None:
        label = f"{type(hook).__name__}({getattr(hook, 'name', '')})"
        try:
            coro = method(ctx)
            if self._before_timeout is not None and stage == "before":
                await asyncio.wait_for(coro, timeout=self._before_timeout)
            else:
                await coro
        except asyncio.TimeoutError as exc:
            self._record_failure(
                ctx,
                hook,
                stage,
                exc,
                timed_out=True,
                timeout_seconds=self._before_timeout,
            )
            if hook.required:
                raise ToolCallHookError(
                    f"required tool-call hook {label} timed out after "
                    f"{self._before_timeout}s at stage={stage}"
                ) from exc
            logger.warning(
                "optional tool-call hook %s timed out after %ss at stage=%s; "
                "continuing: session_id=%s",
                label,
                self._before_timeout,
                stage,
                ctx.session_id,
            )
        except Exception as exc:
            self._record_failure(ctx, hook, stage, exc)
            if hook.required:
                raise ToolCallHookError(
                    f"required tool-call hook {label} failed at stage={stage}: {exc}"
                ) from exc
            logger.warning(
                "optional tool-call hook %s failed at stage=%s; continuing: "
                "session_id=%s error=%s",
                label,
                stage,
                ctx.session_id,
                exc,
                exc_info=True,
            )

    @staticmethod
    def _record_failure(
        ctx: ToolCallContext,
        hook: ToolCallHook,
        stage: str,
        exc: BaseException,
        *,
        timed_out: bool = False,
        timeout_seconds: float | None = None,
    ) -> None:
        record = {
            "hook": getattr(hook, "name", "") or type(hook).__name__,
            "stage": stage,
            "required": bool(hook.required),
            "error": {
                "type": f"{type(exc).__module__}.{type(exc).__name__}",
                "message": str(exc) or "timeout",
            },
        }
        if timed_out:
            record["timed_out"] = True
            if timeout_seconds is not None:
                record["timeout_seconds"] = timeout_seconds
        ctx.stage_metadata.setdefault("tool_call_hook_failures", []).append(record)
