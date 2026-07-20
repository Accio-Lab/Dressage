"""Preemptible SGLang generation controller for Dressage partial rollout.

This module is intentionally self-contained enough to work with the partial
rollout patch set even if only the new import in server.py was applied first.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from .sglang_client import SGLangResponse, SGLangRouterClient

logger = logging.getLogger(__name__)


class ProxyShuttingDown(RuntimeError):
    """Raised when a generation request arrives while the proxy is stopping."""


class GenerationPreempted(RuntimeError):
    """Raised when non-partial rollout hits an interrupted SGLang request."""


class GenerationStaleEpoch(RuntimeError):
    """Raised when a non-partial rollout request crosses a weight update."""

    def __init__(self, *, expected_epoch: int, current_epoch: int) -> None:
        super().__init__(
            "Dressage rollout epoch changed before generation reached SGLang"
        )
        self.expected_epoch = expected_epoch
        self.current_epoch = current_epoch


class GenerationVersionUnavailable(RuntimeError):
    """SGLang omitted a real weight version for a version-pinned request."""


class GenerationVersionMismatch(RuntimeError):
    """SGLang served a different weight version than the pinned batch."""

    def __init__(self, *, expected: str, observed: str) -> None:
        super().__init__(
            f"SGLang served weight version {observed!r}; expected {expected!r}"
        )
        self.expected = expected
        self.observed = observed


_PREEMPT_FINISH_REASONS = {
    "abort",
    "aborted",
    "preempt",
    "preempted",
    "cancel",
    "cancelled",
    "canceled",
}


@dataclass
class GenerationChunk:
    output_ids: list[int]
    output_logprobs: list[float]
    output_token_texts: list[str]
    text: str
    version: str
    finish_reason: str
    preempted: bool = False


@dataclass
class PreemptibleGenerateResult:
    input_token_ids: list[int]
    input_token_logprobs_raw: list[float]
    input_token_texts: list[str]
    output_ids: list[int]
    output_token_logprobs: list[float]
    output_token_texts: list[str]
    output_versions: list[str]
    all_token_ids: list[int]
    all_logprobs: list[float]
    text: str
    meta_info: dict[str, Any]
    finish_reason: str
    all_logprobs_invalid: bool
    rollout_epoch: int | None = None
    routed_experts: str | None = None
    routed_experts_chunks: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[GenerationChunk] = field(default_factory=list)

    @property
    def weight_version(self) -> str | None:
        if self.output_versions:
            return self.output_versions[-1]
        value = self.meta_info.get("weight_version")
        return None if value is None else str(value)


@dataclass
class _ActiveGeneration:
    request_id: str
    session_id: str | None
    instance_id: str | None
    turn_id: str | None
    routing_key: str | None
    input_ids: list[int]
    generated_ids_at_start: list[int]
    sampling_params: dict[str, Any]
    state: Literal["running", "preempting", "quiesced"] = "running"
    abort_payload: dict[str, Any] | None = None
    abort_succeeded: bool = False
    abort_error: str | None = None
    preempted_chunk_collected: bool = False
    quiesced_event: asyncio.Event = field(default_factory=asyncio.Event)


class GenerationController:
    """Transparent request-level preempt/resume controller.

    On pause, active SGLang requests are aborted by request id. The blackbox
    HTTP call remains suspended. On resume, generation continues from
    original_input_ids + partial_output_ids, and all chunks are stitched into a
    single response for the blackbox backend.
    """

    def __init__(
        self,
        sglang_client: SGLangRouterClient,
        *,
        partial_rollout: bool = False,
    ) -> None:
        self._sglang_client = sglang_client
        self._partial_rollout = partial_rollout
        self._pause_lock = asyncio.Lock()
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._paused = False
        self._pause_reason: str | None = None
        self._current_version: str | None = None
        self._rollout_epoch = 0
        self._shutting_down = False
        self._active: dict[str, _ActiveGeneration] = {}
        self._suspended_generations = 0

    @property
    def current_version(self) -> str | None:
        return self._current_version

    @property
    def current_epoch(self) -> int:
        return self._rollout_epoch

    async def generate_preemptible(
        self,
        input_ids: list[int] | None = None,
        sampling_params: dict[str, Any] | None = None,
        *,
        session_id: str | None,
        instance_id: str | None,
        turn_id: str | None,
        routing_key: str | None = None,
        expected_version: str | None = None,
        expected_epoch: int | None = None,
        logprob_start_len: int = 0,
        context_window: int | None = None,
    ) -> PreemptibleGenerateResult:
        # Keep compatibility with both call styles.  Some deployed server.py
        # versions used generate_preemptible(input_ids, sampling_params, ...),
        # while the newer code uses keyword arguments.
        if input_ids is None:
            raise TypeError("generate_preemptible() missing required argument: input_ids")
        if sampling_params is None:
            raise TypeError("generate_preemptible() missing required argument: sampling_params")
        self._raise_if_shutting_down()

        max_new_tokens = int(sampling_params.get("max_new_tokens") or 0)
        generated_ids: list[int] = []
        chunks: list[GenerationChunk] = []
        input_logprobs: list[float] = []
        input_texts: list[str] = []
        output_ids: list[int] = []
        output_logprobs: list[float] = []
        output_texts: list[str] = []
        output_versions: list[str] = []
        text_parts: list[str] = []
        routed_experts_chunks: list[dict[str, Any]] = []
        finish_reason = "stop"
        meta_info: dict[str, Any] = {}
        all_logprobs_invalid = False
        rollout_epoch: int | None = None
        expect_input_logprobs = logprob_start_len == 0
        input_logprobs_captured = not expect_input_logprobs or len(input_ids) == 0

        while max_new_tokens <= 0 or len(generated_ids) < max_new_tokens:
            await self._wait_until_resumed()
            self._raise_if_shutting_down()
            self._raise_if_stale_epoch(expected_epoch)
            remaining = None if max_new_tokens <= 0 else max_new_tokens - len(generated_ids)
            chunk_sampling_params = dict(sampling_params)
            if remaining is not None:
                chunk_sampling_params["max_new_tokens"] = remaining
            request_token_count = len(input_ids) + len(generated_ids)

            active = _ActiveGeneration(
                request_id=self._new_request_id(session_id=session_id, turn_id=turn_id),
                session_id=session_id,
                instance_id=instance_id,
                turn_id=turn_id,
                routing_key=routing_key,
                input_ids=list(input_ids),
                generated_ids_at_start=list(generated_ids),
                sampling_params=dict(chunk_sampling_params),
            )

            async with self._pause_lock:
                self._raise_if_shutting_down()
                self._raise_if_stale_epoch(expected_epoch)
                if self._paused:
                    continue
                chunk_epoch = self._rollout_epoch
                if rollout_epoch is None:
                    rollout_epoch = chunk_epoch
                self._active[active.request_id] = active

            forced_preempted = False
            chunk_logprob_start_len = (
                0
                if expect_input_logprobs and not input_logprobs_captured
                else -1
            )
            try:
                # Important: abort_request is only a signal.  Partial tokens are
                # returned by this original /generate request after SGLang handles
                # the abort.  Do not read partial output from abort_request's
                # response body.
                response = await self._generate_with_optional_request_id(
                    list(input_ids) + list(generated_ids),
                    chunk_sampling_params,
                    routing_key=routing_key,
                    request_id=active.request_id,
                    logprob_start_len=chunk_logprob_start_len,
                )
                forced_preempted = active.abort_succeeded
            except Exception as exc:
                if self._shutting_down:
                    async with self._pause_lock:
                        self._active.pop(active.request_id, None)
                        active.state = "quiesced"
                        active.quiesced_event.set()
                    raise ProxyShuttingDown("Dressage proxy is shutting down") from exc

                if not active.abort_succeeded:
                    async with self._pause_lock:
                        self._active.pop(active.request_id, None)
                        active.state = "quiesced"
                        active.quiesced_event.set()
                    raise exc

                # Some SGLang versions may terminate the long-poll /generate
                # request with an exception after accepting abort_request.  In
                # that case we have no partial payload to stitch, so we resume
                # from the existing generated_ids prefix.
                forced_preempted = True
                logger.info(
                    "SGLang generation %s aborted but /generate did not return a partial payload; "
                    "resuming from the previous prefix. error=%r",
                    active.request_id,
                    exc,
                )
                response = SGLangResponse(
                    input_token_ids=list(input_ids) + list(generated_ids),
                    output_ids=[],
                    output_token_logprobs=[],
                    output_token_texts=[],
                    all_token_ids=list(input_ids) + list(generated_ids),
                    all_logprobs=[0.0] * (len(input_ids) + len(generated_ids)),
                    text="",
                    meta_info={"finish_reason": {"type": "preempted"}},
                    finish_reason="preempted",
                    input_logprobs_invalid=True,
                    all_logprobs_invalid=True,
                )

            response_input_logprobs = list(
                response.input_token_logprobs_raw[: len(input_ids)]
            )
            if response_input_logprobs and (
                not input_logprobs
                or (
                    expect_input_logprobs
                    and not input_logprobs_captured
                    and not response.input_logprobs_invalid
                )
            ):
                input_logprobs = list(response_input_logprobs)
            if (
                expect_input_logprobs
                and not input_logprobs_captured
                and len(response_input_logprobs) == len(input_ids)
                and not response.input_logprobs_invalid
            ):
                input_logprobs_captured = True
            if not input_texts:
                input_texts = list(response.input_token_texts[: len(input_ids)])

            observed_version = self._response_version(response)
            version_error: RuntimeError | None = None
            if expected_version is not None:
                if observed_version is None:
                    version_error = GenerationVersionUnavailable(
                        "SGLang response omitted weight_version for a version-pinned request"
                    )
                elif str(observed_version) != str(expected_version):
                    version_error = GenerationVersionMismatch(
                        expected=str(expected_version),
                        observed=str(observed_version),
                    )
            if version_error is not None:
                async with self._pause_lock:
                    self._active.pop(active.request_id, None)
                    active.state = "quiesced"
                    active.quiesced_event.set()
                raise version_error

            version = observed_version or "unknown"
            if version != "unknown":
                self._current_version = version
            preempted = forced_preempted or self._is_preempted(response)
            routed_experts = response.routed_experts if hasattr(response, "routed_experts") else None

            chunk = GenerationChunk(
                output_ids=list(response.output_ids),
                output_logprobs=list(response.output_token_logprobs),
                output_token_texts=list(response.output_token_texts),
                text=str(response.text or ""),
                version=version,
                finish_reason=response.finish_reason,
                preempted=preempted,
            )
            chunks.append(chunk)
            output_ids.extend(chunk.output_ids)
            output_logprobs.extend(chunk.output_logprobs)
            output_texts.extend(chunk.output_token_texts)
            output_versions.extend([version] * len(chunk.output_ids))
            generated_ids.extend(chunk.output_ids)
            text_parts.append(chunk.text)
            if routed_experts is not None and (chunk.output_ids or not preempted):
                routed_experts_chunks.append(
                    {
                        "data": routed_experts,
                        "prefix_token_count": request_token_count,
                        "output_token_count": len(chunk.output_ids),
                        "is_first_chunk": request_token_count == len(input_ids),
                    }
                )
            finish_reason = response.finish_reason
            meta_info = dict(response.meta_info or {})
            all_logprobs_invalid = all_logprobs_invalid or response.all_logprobs_invalid

            # Mark model-side quiescence only after the partial chunk from the
            # original /generate response has been appended to generated_ids and
            # chunks.  This guarantees pause() returns only after the prefix that
            # resume will continue from is recoverable.
            async with self._pause_lock:
                self._active.pop(active.request_id, None)
                active.preempted_chunk_collected = preempted
                active.state = "quiesced"
                active.quiesced_event.set()

            if preempted:
                if not self._partial_rollout:
                    raise GenerationPreempted(
                        "SGLang generation was interrupted while partial rollout resume is disabled"
                    )

            if (
                context_window is not None
                and len(input_ids) + len(generated_ids) > context_window
            ):
                meta_info["context_overflow"] = {
                    "phase": "input_output",
                    "context_window": context_window,
                    "input_tokens": len(input_ids),
                    "output_tokens": len(generated_ids),
                    "total_tokens": len(input_ids) + len(generated_ids),
                    "max_tokens": max_new_tokens,
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "last_proxy_step_recorded": False,
                }
                finish_reason = "length"
                break

            if not preempted:
                break

            self._suspended_generations += 1
            try:
                await self._wait_until_resumed()
                self._raise_if_shutting_down()
            finally:
                self._suspended_generations = max(0, self._suspended_generations - 1)

        full_input_logprobs = self._normalize_length(input_logprobs, len(input_ids), 0.0)
        full_input_texts = self._normalize_length(input_texts, len(input_ids), "")
        meta_info = dict(meta_info)
        meta_info["partial_rollout_chunks"] = [
            {
                "num_output_tokens": len(chunk.output_ids),
                "version": chunk.version,
                "finish_reason": chunk.finish_reason,
                "preempted": chunk.preempted,
            }
            for chunk in chunks
        ]
        if output_versions:
            meta_info["weight_version"] = output_versions[-1]

        single_routed_experts = (
            routed_experts_chunks[0]["data"]
            if (
                len(routed_experts_chunks) == 1
                and routed_experts_chunks[0].get("is_first_chunk")
            )
            else None
        )

        return PreemptibleGenerateResult(
            input_token_ids=list(input_ids),
            input_token_logprobs_raw=full_input_logprobs,
            input_token_texts=full_input_texts,
            output_ids=output_ids,
            output_token_logprobs=output_logprobs,
            output_token_texts=output_texts,
            output_versions=output_versions,
            all_token_ids=list(input_ids) + output_ids,
            all_logprobs=full_input_logprobs + output_logprobs,
            text="".join(text_parts),
            meta_info=meta_info,
            finish_reason=finish_reason,
            all_logprobs_invalid=all_logprobs_invalid,
            rollout_epoch=rollout_epoch,
            routed_experts=single_routed_experts,
            routed_experts_chunks=routed_experts_chunks,
            chunks=chunks,
        )

    async def pause(
        self,
        *,
        session_id: str | None = None,
        instance_id: str | None = None,
        reason: str = "weight_update",
        mode: str = "preempt",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if mode != "preempt":
            raise ValueError(f"Unsupported rollout pause mode: {mode}")

        async with self._pause_lock:
            already_paused = self._paused
            self._paused = True
            self._pause_reason = reason
            self._resume_event.clear()
            active = [
                item
                for item in self._active.values()
                if (session_id is None or item.session_id == session_id)
                and (instance_id is None or item.instance_id == instance_id)
            ]
            for item in active:
                item.state = "preempting"

        abort_results = await asyncio.gather(
            *(self._abort_active_with_timeout(item, timeout_seconds=timeout_seconds) for item in active),
            return_exceptions=True,
        )

        abort_attempted_request_ids = [item.request_id for item in active]
        abort_succeeded_request_ids: list[str] = []
        abort_failed_request_ids: list[str] = []
        abort_errors: dict[str, str] = {}
        for item, result in zip(active, abort_results):
            if isinstance(result, Exception):
                item.abort_succeeded = False
                item.abort_error = repr(result)
                abort_failed_request_ids.append(item.request_id)
                abort_errors[item.request_id] = repr(result)
                logger.warning(
                    "failed to abort SGLang request rid=%s session_id=%s instance_id=%s "
                    "turn_id=%s routing_key=%s error=%s",
                    item.request_id,
                    item.session_id,
                    item.instance_id,
                    item.turn_id,
                    item.routing_key,
                    result,
                )
            else:
                item.abort_succeeded = True
                abort_succeeded_request_ids.append(item.request_id)
                target_summary = self._abort_target_summary(result)
                logger.info(
                    "aborted SGLang request rid=%s session_id=%s instance_id=%s "
                    "turn_id=%s routing_key=%s targets=%s errors=%s",
                    item.request_id,
                    item.session_id,
                    item.instance_id,
                    item.turn_id,
                    item.routing_key,
                    target_summary["targets"],
                    target_summary["errors"],
                )

        quiesced = await self._wait_quiesced(active, timeout_seconds=timeout_seconds)
        preempted_request_ids = [
            item.request_id for item in active if item.preempted_chunk_collected
        ]
        fallback = None
        if abort_failed_request_ids and quiesced and not preempted_request_ids:
            fallback = "wait_natural_completion"

        if active:
            logger.info(
                "rollout pause abort summary reason=%s session_id=%s instance_id=%s "
                "attempted_rids=%s succeeded_rids=%s failed_rids=%s preempted_rids=%s "
                "quiesced=%s fallback=%s",
                reason,
                session_id,
                instance_id,
                abort_attempted_request_ids,
                abort_succeeded_request_ids,
                abort_failed_request_ids,
                preempted_request_ids,
                quiesced,
                fallback,
            )

        return {
            "status": "already_paused" if already_paused else "paused",
            "reason": reason,
            "mode": mode,
            "quiesced": quiesced,
            "preempted": bool(preempted_request_ids),
            "fallback": fallback,
            "active_sglang_generations": 0 if quiesced else len([x for x in active if not x.quiesced_event.is_set()]),
            "suspended_generations": self._suspended_generations,
            "version": self._current_version,
            "abort_attempted_request_ids": abort_attempted_request_ids,
            "abort_succeeded_request_ids": abort_succeeded_request_ids,
            "abort_failed_request_ids": abort_failed_request_ids,
            "abort_errors": abort_errors,
            "preempted_request_ids": preempted_request_ids,
        }

    async def resume(self, *, version: str | None = None, reason: str = "weight_update") -> dict[str, Any]:
        readiness: dict[str, Any] | None = None
        async with self._pause_lock:
            if self._shutting_down:
                return {
                    "status": "shutting_down",
                    "reason": reason,
                    "version": self._current_version,
                    "paused": True,
                    "active_sglang_generations": len(self._active),
                    "suspended_generations": self._suspended_generations,
                }
            was_paused = self._paused
            if was_paused:
                wait_until_ready = getattr(self._sglang_client, "wait_until_ready", None)
                if callable(wait_until_ready):
                    readiness = await wait_until_ready()
                    if not readiness.get("ready", False):
                        return {
                            "status": "backend_not_ready",
                            "reason": reason,
                            "version": self._current_version,
                            "paused": True,
                            "readiness": readiness,
                            "active_sglang_generations": len(self._active),
                            "suspended_generations": self._suspended_generations,
                        }
            if version is not None:
                self._current_version = str(version)
            if was_paused:
                self._rollout_epoch += 1
            self._paused = False
            self._pause_reason = None
            self._resume_event.set()
        result = {
            "status": "resumed" if was_paused else "already_running",
            "reason": reason,
            "version": self._current_version,
            "rollout_epoch": self._rollout_epoch,
            "shutting_down": self._shutting_down,
            "active_sglang_generations": len(self._active),
            "suspended_generations": self._suspended_generations,
        }
        if readiness is not None:
            result["readiness"] = readiness
        return result

    async def synchronize_version(
        self, version: str, *, reason: str = "backend_snapshot"
    ) -> dict[str, Any]:
        """Synchronize cached control-plane state at a quiescent batch boundary."""

        normalized = str(version).strip()
        if not normalized:
            raise ValueError("version must be non-empty")
        async with self._pause_lock:
            if self._active or self._paused or self._shutting_down:
                return {
                    "status": "busy",
                    "reason": reason,
                    "version": self._current_version,
                    "rollout_epoch": self._rollout_epoch,
                }
            changed = self._current_version != normalized
            self._current_version = normalized
            if changed:
                self._rollout_epoch += 1
            return {
                "status": "synchronized",
                "reason": reason,
                "changed": changed,
                "version": self._current_version,
                "rollout_epoch": self._rollout_epoch,
            }

    async def shutdown(self, *, timeout_seconds: float = 5.0) -> dict[str, Any]:
        async with self._pause_lock:
            already_shutting_down = self._shutting_down
            self._shutting_down = True
            self._paused = True
            self._pause_reason = "shutdown"
            # Wake waiters so they can observe _shutting_down and exit instead
            # of staying suspended while the HTTP client is closed.
            self._resume_event.set()
            active = list(self._active.values())
            for item in active:
                item.state = "preempting"

        if active:
            await asyncio.gather(
                *(self._abort_active_with_timeout(item, timeout_seconds=timeout_seconds) for item in active),
                return_exceptions=True,
            )
        quiesced = await self._wait_quiesced(active, timeout_seconds=timeout_seconds)
        return {
            "status": "already_shutting_down" if already_shutting_down else "shutting_down",
            "quiesced": quiesced,
            "active_sglang_generations": 0 if quiesced else len([x for x in active if not x.quiesced_event.is_set()]),
            "suspended_generations": self._suspended_generations,
            "version": self._current_version,
        }

    def state(self) -> dict[str, Any]:
        return {
            "paused": self._paused,
            "reason": self._pause_reason,
            "version": self._current_version,
            "rollout_epoch": self._rollout_epoch,
            "shutting_down": self._shutting_down,
            "partial_rollout": self._partial_rollout,
            "active_sglang_generations": len(self._active),
            "suspended_generations": self._suspended_generations,
            "active_request_ids": list(self._active),
        }

    async def _wait_until_resumed(self) -> None:
        while not self._resume_event.is_set():
            await asyncio.sleep(0.01)

    async def _generate_with_optional_request_id(
        self,
        input_ids: list[int],
        sampling_params: dict[str, Any],
        *,
        routing_key: str | None,
        request_id: str,
        logprob_start_len: int,
    ) -> SGLangResponse:
        generate = self._sglang_client.generate
        try:
            signature = inspect.signature(generate)
        except (TypeError, ValueError):
            return await generate(input_ids, sampling_params, routing_key=routing_key)

        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        kwargs: dict[str, Any] = {"routing_key": routing_key}
        if accepts_kwargs or "request_id" in parameters:
            kwargs["request_id"] = request_id
        if accepts_kwargs or "logprob_start_len" in parameters:
            kwargs["logprob_start_len"] = logprob_start_len
        return await generate(input_ids, sampling_params, **kwargs)

    async def _abort_active_with_timeout(
        self,
        active: _ActiveGeneration,
        *,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        if timeout_seconds is None:
            return await self._abort_active(active)
        return await asyncio.wait_for(self._abort_active(active), timeout=timeout_seconds)

    async def _abort_active(self, active: _ActiveGeneration) -> dict[str, Any]:
        if hasattr(self._sglang_client, "abort_request"):
            payload = await self._sglang_client.abort_request(active.request_id, routing_key=active.routing_key)
        else:
            payload = await self._abort_request_fallback(active.request_id, routing_key=active.routing_key)
        active.abort_payload = payload
        active.abort_succeeded = True
        return payload

    async def _abort_request_fallback(self, request_id: str, *, routing_key: str | None) -> dict[str, Any]:
        client = getattr(self._sglang_client, "_client")
        router_url = getattr(self._sglang_client, "_router_url")
        headers = {}
        if routing_key:
            headers["X-SMG-Routing-Key"] = routing_key
        last_exc: Exception | None = None
        for payload in ({"rid": request_id}, {"request_id": request_id}):
            try:
                response = await client.post(f"{router_url}/abort_request", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data if isinstance(data, dict) else {}
            except Exception as exc:  # pragma: no cover - fallback path
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        return {}

    @staticmethod
    def _abort_target_summary(payload: Any) -> dict[str, list[Any]]:
        if not isinstance(payload, dict):
            return {"targets": [], "errors": []}
        targets = payload.get("targets")
        if not isinstance(targets, list):
            targets = []
        errors = payload.get("errors")
        if not isinstance(errors, list):
            errors = []
        return {"targets": targets, "errors": errors}

    async def _wait_quiesced(self, active: list[_ActiveGeneration], *, timeout_seconds: float | None) -> bool:
        if not active:
            return True
        waiters = [item.quiesced_event.wait() for item in active]
        try:
            if timeout_seconds is None:
                await asyncio.gather(*waiters)
            else:
                await asyncio.wait_for(asyncio.gather(*waiters), timeout=timeout_seconds)
            return True
        except asyncio.TimeoutError:
            return False

    def _coerce_response(
        self,
        data: dict[str, Any],
        *,
        input_ids: list[int],
        expect_input_logprobs: bool = True,
    ) -> SGLangResponse:
        coerce = getattr(self._sglang_client, "coerce_response", None)
        if coerce is not None:
            return coerce(
                data,
                input_ids=input_ids,
                expect_input_logprobs=expect_input_logprobs,
            )
        return self._sglang_client._coerce_response(
            data,
            input_ids=input_ids,
            expect_input_logprobs=expect_input_logprobs,
        )

    @staticmethod
    def _extract_partial_payload(payload: dict[str, Any]) -> dict[str, Any]:
        for key in ("data", "result", "partial", "response"):
            value = payload.get(key)
            if isinstance(value, dict) and (value.get("output_ids") or value.get("text") or value.get("meta_info")):
                return value
        return payload

    @staticmethod
    def _has_partial_payload(payload: dict[str, Any]) -> bool:
        return bool(payload.get("output_ids") or payload.get("text") or payload.get("meta_info"))

    @staticmethod
    def _is_preempted(response: SGLangResponse) -> bool:
        reason = str(response.finish_reason or "").lower()
        if reason in _PREEMPT_FINISH_REASONS:
            return True
        finish_reason = response.meta_info.get("finish_reason") if response.meta_info else None
        if isinstance(finish_reason, dict):
            reason = str(finish_reason.get("type", "")).lower()
        else:
            reason = str(finish_reason or "").lower()
        return reason in _PREEMPT_FINISH_REASONS

    @staticmethod
    def _response_version(response: SGLangResponse) -> str | None:
        value = None
        if hasattr(response, "weight_version"):
            value = response.weight_version
        if value is None and response.meta_info:
            value = response.meta_info.get("weight_version")
        if value is not None:
            return str(value)
        return None

    def _raise_if_shutting_down(self) -> None:
        if self._shutting_down:
            raise ProxyShuttingDown("Dressage proxy is shutting down")

    def _raise_if_stale_epoch(self, expected_epoch: int | None) -> None:
        if expected_epoch is not None and self._rollout_epoch != expected_epoch:
            raise GenerationStaleEpoch(
                expected_epoch=expected_epoch,
                current_epoch=self._rollout_epoch,
            )

    @staticmethod
    def _new_request_id(*, session_id: str | None, turn_id: str | None) -> str:
        prefix = "dressage"
        if session_id:
            prefix += f"-{session_id}"
        if turn_id:
            prefix += f"-{turn_id}"
        return f"{prefix}-{uuid.uuid4().hex}"

    @staticmethod
    def _normalize_length(values: list[Any], size: int, default: Any) -> list[Any]:
        result = list(values[:size])
        if len(result) < size:
            result.extend([default] * (size - len(result)))
        return result
