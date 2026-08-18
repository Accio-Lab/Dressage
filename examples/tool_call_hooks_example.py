"""Example tool-call hooks for Dressage blackbox rollouts.

This file demonstrates the most common hook patterns.  Enable any of them
with the proxy CLI::

    dressage-proxy --tokenizer-path ... \\
        --tool-call-hooks examples/tool_call_hooks_example.py:E2BSnapshotHook \\
        --tool-call-hook-before-timeout 5.0

Reference forms accepted by ``--tool-call-hooks``:

- ``E2BSnapshotHook``            (a name already registered at import time)
- ``examples.tool_call_hooks_example:E2BSnapshotHook``  (module:Class)
- ``examples/tool_call_hooks_example.py:E2BSnapshotHook`` (path:Class)
- ``examples/tool_call_hooks_example.py`` (bare path; only valid when the
  file registers exactly one hook — this file registers four, so qualify
  the class name)

The e2b examples import the SDK lazily so this file stays importable in
environments without ``e2b`` installed (e.g. CI running the unit tests).
"""

from __future__ import annotations

import logging
import time
from typing import ClassVar

from dressage.proxy.tool_call_hooks import (
    ToolCallContext,
    ToolCallHook,
    register_tool_call_hook,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Example 1: e2b sandbox snapshot after every tool round
# ---------------------------------------------------------------------------


@register_tool_call_hook
class E2BSnapshotHook(ToolCallHook):
    """Take a snapshot point after each tool execution round.

    ``after_tool_call`` fires when a request's normalized message tail is a
    ``role=tool`` message, i.e. the previous step's tools already ran inside
    the sandbox.  The hook reconnects to the sandbox with the provider-native
    id and records a snapshot marker; results written to
    ``ctx.stage_metadata`` land in the finalized trajectory under
    ``extra_info["tool_call_hooks"]``.
    """

    name = "e2b_snapshot"
    required = False  # snapshot failures must not abort the rollout
    order = 100

    def __init__(self) -> None:
        self._snapshots = 0

    async def after_tool_call(self, ctx: ToolCallContext) -> None:
        from e2b import AsyncSandbox  # lazy import; optional dependency

        started = time.monotonic()
        sandbox = await AsyncSandbox.connect(ctx.sandbox_id)
        try:
            # Replace with a real snapshot call for your setup, e.g. an e2b
            # template snapshot or a filesystem checkpoint command:
            await sandbox.commands.run("sync", timeout=10)
        finally:
            await sandbox.disconnect()
        self._snapshots += 1
        ctx.stage_metadata["e2b_snapshot"] = {
            "sandbox_id": ctx.sandbox_id,
            "step_index": ctx.step_index,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            "total_snapshots": self._snapshots,
        }
        logger.info(
            "e2b snapshot taken: session=%s step=%d sandbox=%s",
            ctx.session_id,
            ctx.step_index,
            ctx.sandbox_id,
        )


# ---------------------------------------------------------------------------
# Example 2: restore sandbox networking right before tools execute
# ---------------------------------------------------------------------------


@register_tool_call_hook
class RestoreNetworkHook(ToolCallHook):
    """Re-enable sandbox egress just before the agent executes tools.

    ``before_tool_call`` fires synchronously *before* the tool_calls
    response is returned to the agent, so the sandbox is guaranteed to be
    ready by the time the tools actually run.  ``required=True`` makes a
    failure abort the request (HTTP 502) instead of letting the agent run
    tools against a sandbox that is still network-isolated.
    """

    name = "restore_network"
    required = True
    order = 10  # run before snapshot-style hooks

    async def before_tool_call(self, ctx: ToolCallContext) -> None:
        from e2b import AsyncSandbox

        sandbox = await AsyncSandbox.connect(ctx.sandbox_id)
        try:
            # Example: flip a state file that the sandbox's egress wrapper
            # sources.  Adapt to your own networking setup.
            await sandbox.files.write("/etc/dressage/network_state", "online\n")
        finally:
            await sandbox.disconnect()
        ctx.stage_metadata["network_restored"] = {
            "sandbox_id": ctx.sandbox_id,
            "tools": [tc["function"]["name"] for tc in ctx.tool_calls or []],
        }


# ---------------------------------------------------------------------------
# Example 3: selective observability — only watch one tool
# ---------------------------------------------------------------------------


@register_tool_call_hook
class ToolWatchHook(ToolCallHook):
    """Record every invocation of one specific tool.

    Demonstrates ``applies_to`` filtering and reading parsed tool calls from
    the context.  Steps that do not match skip this hook entirely.
    """

    name = "tool_watch"
    required = False
    order = 200

    watched_tool: ClassVar[str] = "web_search"

    def applies_to(self, ctx: ToolCallContext) -> bool:
        # Only before-stage contexts carry tool_calls; after-stage contexts
        # (tool_calls=None) are filtered out here.
        return any(
            tc.get("function", {}).get("name") == self.watched_tool
            for tc in ctx.tool_calls or []
        )

    async def before_tool_call(self, ctx: ToolCallContext) -> None:
        calls = [
            tc["function"]
            for tc in ctx.tool_calls or []
            if tc.get("function", {}).get("name") == self.watched_tool
        ]
        ctx.stage_metadata.setdefault("watched_calls", []).extend(
            {
                "tool": self.watched_tool,
                "arguments": call.get("arguments"),
                "step_index": ctx.step_index,
            }
            for call in calls
        )


# ---------------------------------------------------------------------------
# Example 4: a no-provider hook that only records timing metrics
# ---------------------------------------------------------------------------


@register_tool_call_hook
class StepTimingHook(ToolCallHook):
    """Pure observability: measure tool-round cadence per session.

    Needs no sandbox access at all — useful for smoke-testing the hook
    pipeline in local runs before wiring real side effects.
    """

    name = "step_timing"
    required = False
    order = 500

    def __init__(self) -> None:
        self._last_seen: dict[str, float] = {}

    async def after_tool_call(self, ctx: ToolCallContext) -> None:
        now = time.monotonic()
        previous = self._last_seen.get(ctx.session_id)
        self._last_seen[ctx.session_id] = now
        if previous is not None:
            ctx.stage_metadata["tool_round_gap_seconds"] = round(now - previous, 3)
