from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx
from agentic_rollout_contract.client import PolicyServiceClient, RolloutServiceClient
from agentic_rollout_contract.models import (
    ContextBindRequest,
    ExecutionClaim,
    ExecutionState,
    ExecutionTransition,
    RouteControlRequest,
)

logger = logging.getLogger(__name__)


class HarborRemotePolicyHooks:
    def __init__(
        self,
        claim: ExecutionClaim,
        *,
        control_client: RolloutServiceClient,
        policy_client: PolicyServiceClient,
        retry_attempts: int = 8,
        retry_base_sec: float = 0.5,
        retry_max_sec: float = 10.0,
        cleanup_timeout_sec: float = 180.0,
    ) -> None:
        self.claim = claim
        self.control_client = control_client
        self.policy_client = policy_client
        self.retry_attempts = retry_attempts
        self.retry_base_sec = retry_base_sec
        self.retry_max_sec = retry_max_sec
        self.cleanup_timeout_sec = cleanup_timeout_sec
        self.phase = 0
        self.registered = False
        self.trial_id: str | None = None
        self.trial_name: str | None = None

    def bind(self, queue: Any) -> None:
        queue.on_trial_started(self.on_start)
        queue.on_agent_started(self.on_agent_start)
        queue.on_agent_ended(self.on_agent_end)
        queue.on_verification_started(self.on_verification_start)
        queue.on_trial_ended(self.on_end)
        queue.on_trial_cancelled(self.on_cancel)

    async def on_start(self, event: Any) -> None:
        self.trial_id = str(event.trial_id)
        self.trial_name = str(event.trial_name)
        await self._transition(ExecutionState.PREPARING)

    async def on_agent_start(self, event: Any) -> None:
        # Harbor emits AGENT_START after environment/agent setup and before
        # launching the agent. Preparing a sandbox must not materialize a route.
        spec = self.claim.execution
        request = ContextBindRequest(
            worker_id=self.claim.worker_id,
            worker_incarnation=self.claim.worker_incarnation,
            lease_token=self.claim.lease_token,
            execution_id=spec.execution_id,
            reservation_id=self.claim.reservation_id,
            trial_name=self.trial_name,
            trial_id=self.trial_id,
        )
        await self._transition(ExecutionState.WAITING_ROUTE)
        await self._policy_call(
            lambda: self.policy_client.bind(self.claim.context_id, request),
            operation_name=f"bind context {self.claim.context_id}",
        )
        self.registered = True

        self.phase += 1
        request = RouteControlRequest(
            action_id=f"{self.claim.execution.execution_id}:open:{self.phase}",
            execution_id=self.claim.execution.execution_id,
            turn_id=f"{event.trial_id}:agent-phase:{self.phase}",
        )
        await self._policy_call(
            lambda: self.policy_client.control(
                self.claim.context_id, "open-turn", request
            ),
            operation_name=f"open route {self.claim.context_id}",
        )
        await self._transition(ExecutionState.AGENT_RUNNING)

    async def on_agent_end(self, event: Any) -> None:
        request = RouteControlRequest(
            action_id=f"{self.claim.execution.execution_id}:seal:{self.phase}",
            execution_id=self.claim.execution.execution_id,
        )
        await self._policy_call(
            lambda: self.policy_client.control(self.claim.context_id, "seal", request),
            operation_name=f"seal route {self.claim.context_id}",
        )

    async def on_verification_start(self, event: Any) -> None:
        await self._transition(ExecutionState.VERIFYING)

    async def on_end(self, event: Any) -> None:
        if not self.registered:
            return
        if self.phase == 0:
            await self._revoke(f"{self.claim.execution.execution_id}:end-before-agent")
            return
        request = RouteControlRequest(
            action_id=f"{self.claim.execution.execution_id}:seal",
            execution_id=self.claim.execution.execution_id,
        )
        await self._policy_call(
            lambda: self.policy_client.control(self.claim.context_id, "seal", request),
            operation_name=f"seal route {self.claim.context_id}",
        )

    async def on_cancel(self, event: Any) -> None:
        if self.registered:
            await self._revoke(f"{self.claim.execution.execution_id}:cancel")

    async def revoke_after_failure(self) -> None:
        if not self.registered:
            return
        await self._revoke(f"{self.claim.execution.execution_id}:failed")

    async def _revoke(self, action_id: str) -> None:
        await asyncio.wait_for(
            self._policy_call(
                lambda: self.policy_client.revoke(
                    self.claim.context_id,
                    action_id,
                    self.claim.execution.execution_id,
                ),
                operation_name=f"revoke route {self.claim.context_id}",
            ),
            timeout=self.cleanup_timeout_sec,
        )

    async def _policy_call(self, operation: Any, *, operation_name: str) -> Any:
        delay = self.retry_base_sec
        for retry_index in range(self.retry_attempts):
            try:
                return await operation()
            except httpx.HTTPError as exc:
                retryable = isinstance(exc, httpx.TransportError) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and (
                        exc.response.status_code in {408, 425, 429}
                        or exc.response.status_code >= 500
                    )
                )
                if not retryable or retry_index + 1 >= self.retry_attempts:
                    raise
                logger.warning(
                    "%s failed; retrying (%d/%d) error=%s",
                    operation_name,
                    retry_index + 1,
                    self.retry_attempts,
                    type(exc).__name__,
                )
                if delay:
                    await asyncio.sleep(delay * random.uniform(0.8, 1.2))
                delay = min(max(delay * 2, self.retry_base_sec), self.retry_max_sec)

    async def _transition(self, state: ExecutionState) -> None:
        await self.control_client.transition(
            self.claim.execution.execution_id,
            ExecutionTransition(
                worker_id=self.claim.worker_id,
                worker_incarnation=self.claim.worker_incarnation,
                lease_token=self.claim.lease_token,
                state=state,
                trial_id=self.trial_id,
                trial_name=self.trial_name,
            ),
        )
