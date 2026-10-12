from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import ClassVar

from harbor.agents.installed.base import (
    ApiConnectionClosedError,
    ContextWindowExceededError,
    ErrorPattern,
    NonZeroAgentExitCodeError,
)
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)


class ClaudeMaxTurnsError(NonZeroAgentExitCodeError):
    """Claude completed structurally but hit the configured turn ceiling."""


class PreinstalledClaudeCode(ClaudeCode):
    """Claude Code agent that never mutates the task image at rollout time."""

    ERROR_PATTERNS: ClassVar[list[ErrorPattern]] = [
        *ClaudeCode.ERROR_PATTERNS,
        ErrorPattern(
            r"GPU Policy transport is temporarily unavailable|API Error:\s*503\b",
            ApiConnectionClosedError,
        ),
    ]

    def __init__(
        self, *args, context_budget_env=None, accept_max_turns=False, **kwargs
    ):
        if type(accept_max_turns) is not bool:
            raise ValueError("accept_max_turns must be a boolean")
        super().__init__(*args, **kwargs)
        self.accept_max_turns = accept_max_turns
        # Numeric budgets are not credentials. Do not put them in extra_env:
        # Harbor's TOKEN-name scrubber would replace numbers throughout trials.
        self.context_budget_env = context_budget_env or {}

    async def run(self, instruction, environment, context):
        with environment.scoped_exec_env(self.context_budget_env):
            try:
                await super().run(instruction, environment, context)
            except ClaudeMaxTurnsError:
                if not self.accept_max_turns:
                    raise
                context.metadata = {
                    **(context.metadata or {}),
                    "agent_terminal_reason": "max_turns",
                }
                logger.info(
                    "agent_budget_ended terminal_reason=max_turns accepted=true"
                )

    def _classify_exec_error(self, command: str, result):
        terminal = _last_claude_result(result.stdout)
        if terminal is None:
            return super()._classify_exec_error(command, result)
        selected = SimpleNamespace(
            return_code=result.return_code,
            stdout=json.dumps(terminal, ensure_ascii=False),
            stderr=result.stderr,
        )
        classified = super()._classify_exec_error(command, selected)
        reason = terminal.get("terminal_reason")
        # A killed process can leave an earlier result in stdout. Never turn
        # SIGKILL/137 or another abnormal exit into a budget-ended episode.
        if reason == "max_turns" and result.return_code == 1:
            return ClaudeMaxTurnsError(str(classified))
        if reason in {"prompt_too_long", "rapid_refill_breaker"}:
            return ContextWindowExceededError(str(classified))
        return classified

    async def install(self, environment: BaseEnvironment) -> None:
        if await self._installed_claude_satisfies_version(environment):
            logger.debug("Preinstalled Claude Code matches the requested version")
            return

        expected = self.version() or "any available version"
        raise RuntimeError(
            "Claude Code must be preinstalled in the task image; "
            f"required={expected!r}. Runtime package installation is disabled."
        )


def _last_claude_result(stdout: str | None) -> dict | None:
    for line in reversed((stdout or "").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("type") == "result":
            return value
    return None
