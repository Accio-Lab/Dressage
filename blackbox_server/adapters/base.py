from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol

from blackbox_server.core.models import (
    AdapterResponse,
    BackendCapabilities,
    BindingContext,
    Message,
    SessionContext,
    TurnContext,
)


LOGGER = logging.getLogger(__name__)


class BackendError(Exception):
    pass


class BackendTransportError(BackendError):
    pass


class BackendProtocolError(BackendError):
    pass


class BackendProcessError(BackendError):
    pass


class ProxyMaxStepsWatcher(Protocol):
    async def wait_for_max_steps_error(
        self,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        ...

    async def consume_max_steps_error(self) -> dict[str, Any] | None:
        ...


class BackendMaxStepsExceededError(BackendError):
    """Raised when a rollout turn exceeds its configured agent LLM step budget."""

    def __init__(
        self,
        message: str,
        *,
        max_steps: int,
        attempted_step: int,
        backend_message: str | None = None,
        raw_error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.max_steps = max_steps
        self.attempted_step = attempted_step
        self.backend_message = backend_message
        self.raw_error_code = raw_error_code

    def details(self) -> dict[str, object]:
        details: dict[str, object] = {
            "max_steps": self.max_steps,
            "attempted_step": self.attempted_step,
        }
        if self.backend_message:
            details["backend_message"] = self.backend_message
        if self.raw_error_code:
            details["raw_error_code"] = self.raw_error_code
        return details


class BackendContextOverflowError(BackendError):
    """Raised when a backend reports that a turn exceeded its context window.

    This is a request/session-level failure rather than a global backend outage.
    The HTTP layer maps it to a non-503 error so callers can distinguish it from
    blackbox server unavailability and retry/label the sample appropriately.
    """

    def __init__(
        self,
        message: str,
        *,
        context_window: int | None = None,
        input_tokens: int | None = None,
        max_tokens: int | None = None,
        backend_message: str | None = None,
        raw_error_code: str | None = None,
        error_details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.context_window = context_window
        self.input_tokens = input_tokens
        self.max_tokens = max_tokens
        self.backend_message = backend_message
        self.raw_error_code = raw_error_code
        self.error_details = dict(error_details or {})

    def details(self) -> dict[str, object]:
        details: dict[str, object] = dict(self.error_details)
        if self.context_window is not None:
            details["context_window"] = self.context_window
        if self.input_tokens is not None:
            details["input_tokens"] = self.input_tokens
        if self.max_tokens is not None:
            details["max_tokens"] = self.max_tokens
        if self.backend_message:
            details["backend_message"] = self.backend_message
        if self.raw_error_code:
            details["raw_error_code"] = self.raw_error_code
        return details


def backend_context_overflow_from_proxy_payload(
    payload: object,
) -> BackendContextOverflowError | None:
    if (
        not isinstance(payload, dict)
        or (payload.get("code") or payload.get("error")) != "context_overflow"
    ):
        return None
    details = payload.get("details")
    if not isinstance(details, dict):
        details = {}
    message = payload.get("message") or "Dressage proxy context window overflow."
    return BackendContextOverflowError(
        str(message),
        context_window=_maybe_int(details.get("context_window")),
        input_tokens=_maybe_int(details.get("input_tokens")),
        max_tokens=_maybe_int(details.get("max_tokens")),
        backend_message=str(message),
        raw_error_code="context_overflow",
        error_details={str(key): value for key, value in details.items()},
    )


def backend_max_steps_from_proxy_payload(
    payload: object,
) -> BackendMaxStepsExceededError | None:
    if not isinstance(payload, dict) or payload.get("error") != "max_steps_exceeded":
        return None
    details = payload.get("details")
    if not isinstance(details, dict):
        details = {}
    message = payload.get("message") or "Turn exceeded max_steps."
    max_steps = _maybe_int(details.get("max_steps")) or 0
    attempted_step = _maybe_int(details.get("attempted_step")) or max_steps
    backend_message = details.get("backend_message") or message
    raw_error_code = details.get("raw_error_code") or "max_steps_exceeded"
    return BackendMaxStepsExceededError(
        str(message),
        max_steps=max_steps,
        attempted_step=attempted_step,
        backend_message=str(backend_message),
        raw_error_code=str(raw_error_code),
    )


def _maybe_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _failed_upstream_backend_error(
    payload: dict[str, Any],
    original_error: BaseException | None,
) -> BackendTransportError:
    parts: list[str] = []
    if original_error is not None:
        parts.append(str(original_error))
    message = payload.get("message")
    if message:
        parts.append(f"upstream error: {message}")
    request_path = payload.get("request_path")
    if request_path:
        parts.append(f"failed upstream payload: {request_path}")
    response_path = payload.get("response_path")
    if response_path:
        parts.append(f"failed upstream response: {response_path}")
    dump_error = payload.get("dump_error")
    if dump_error:
        parts.append(f"failed upstream dump error: {dump_error}")
    return BackendTransportError(
        "; ".join(parts) or "Dressage proxy upstream request failed."
    )


class BackendAdapter(ABC):
    @abstractmethod
    async def initialize(self, binding_context: BindingContext) -> None:
        raise NotImplementedError

    @abstractmethod
    async def send_message(
        self,
        session_context: SessionContext,
        turn_context: TurnContext,
        new_messages: list[Message],
    ) -> AdapterResponse:
        raise NotImplementedError

    @abstractmethod
    async def abort_session(self, session_context: SessionContext) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def capabilities(self) -> BackendCapabilities:
        raise NotImplementedError

    async def pause(
        self,
        *,
        reason: str = "weight_update",
        timeout_seconds: float | None = None,
    ) -> dict:
        del timeout_seconds
        return {
            "status": "noop",
            "reason": reason,
            "quiesced": True,
            "http_inflight_requests": 0,
            "active_sglang_generations": 0,
            "suspended_generations": 0,
        }

    async def resume(
        self,
        *,
        version: str | None = None,
        reason: str = "weight_update",
    ) -> dict:
        return {"status": "noop", "reason": reason, "version": version}

    async def _await_backend_task_or_proxy_max_steps(
        self,
        task: asyncio.Task[Any],
        *,
        session_context: SessionContext,
        proxy: ProxyMaxStepsWatcher | None,
    ) -> Any:
        if proxy is None:
            return await task
        if not hasattr(proxy, "wait_for_max_steps_error"):
            return await task

        max_steps_task = asyncio.create_task(proxy.wait_for_max_steps_error())
        try:
            done, _ = await asyncio.wait(
                {task, max_steps_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if max_steps_task in done:
                payload = max_steps_task.result()
                if payload is not None:
                    await self._abort_backend_after_proxy_max_steps(
                        task,
                        session_context=session_context,
                    )
                    typed_error = backend_max_steps_from_proxy_payload(payload)
                    if typed_error is not None:
                        raise typed_error
                    raise BackendMaxStepsExceededError(
                        "Turn exceeded max_steps.",
                        max_steps=0,
                        attempted_step=0,
                        backend_message="429 Turn exceeded max_steps.",
                        raw_error_code="rate_limit_error",
                    )

            result = await task
            payload = await proxy.consume_max_steps_error()
            typed_error = backend_max_steps_from_proxy_payload(payload)
            if typed_error is not None:
                raise typed_error
            return result
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            raise
        finally:
            if not max_steps_task.done():
                max_steps_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await max_steps_task

    async def _abort_backend_after_proxy_max_steps(
        self,
        task: asyncio.Task[Any],
        *,
        session_context: SessionContext,
    ) -> None:
        with contextlib.suppress(Exception):
            await self.abort_session(session_context)
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, timeout=2.0)

    async def _drain_and_raise_proxy_failure(
        self,
        proxy: Any,
        *,
        original_error: BaseException | None = None,
        drain_timeout: float | None = None,
    ) -> None:
        """Drain the proxy before consuming any failure recorded by its stream."""
        if proxy is None:
            if original_error is not None:
                raise original_error
            return

        drain_error: asyncio.TimeoutError | None = None
        if hasattr(proxy, "drain_turn"):
            try:
                await proxy.drain_turn(timeout=drain_timeout)
            except asyncio.TimeoutError as exc:
                drain_error = exc
                LOGGER.warning(
                    "Timed out draining rollout proxy before consuming failure "
                    "markers; attribution may be incomplete."
                )

        if hasattr(proxy, "consume_context_overflow_error"):
            payload = await proxy.consume_context_overflow_error()
            typed_error = backend_context_overflow_from_proxy_payload(payload)
            if typed_error is not None:
                raise typed_error from original_error

        if hasattr(proxy, "consume_rollout_invalidated_error"):
            payload = await proxy.consume_rollout_invalidated_error()
            if payload is not None:
                error = (
                    payload.get("code")
                    or payload.get("error")
                    or "rollout_invalidated"
                )
                message = (
                    payload.get("message") or "Dressage rollout was invalidated."
                )
                raise BackendTransportError(
                    f"Dressage proxy {error}: {message}"
                ) from original_error

        if hasattr(proxy, "consume_failed_upstream_error"):
            payload = await proxy.consume_failed_upstream_error()
            if payload is not None:
                raise _failed_upstream_backend_error(
                    payload, original_error
                ) from original_error

        if original_error is not None:
            raise original_error
        if drain_error is not None:
            raise drain_error

    @abstractmethod
    async def shutdown(self) -> None:
        raise NotImplementedError
