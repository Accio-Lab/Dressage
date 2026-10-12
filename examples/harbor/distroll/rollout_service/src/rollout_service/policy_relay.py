from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import random
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, HTTPException, Request, Response

_ALLOWED_MODEL_PATHS = {
    "v1/messages",
    "v1/chat/completions",
    "v1/responses",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelayRoute:
    attempt_id: str
    route_id: str
    local_token: str


@dataclass(frozen=True)
class BufferedResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes


class RelayLedger:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[str, tuple[str, asyncio.Task[BufferedResponse]]] = {}

    async def execute(
        self,
        request_id: str,
        digest: str,
        operation: Callable[[], Awaitable[BufferedResponse]],
    ) -> BufferedResponse:
        async with self._lock:
            existing = self._tasks.get(request_id)
            if existing is not None:
                existing_digest, task = existing
                if existing_digest != digest:
                    raise ValueError("request id digest conflict")
            else:
                task = asyncio.create_task(operation(), name=f"relay-{request_id[:12]}")
                self._tasks[request_id] = (digest, task)
        try:
            return await asyncio.shield(task)
        except BaseException:
            # A completed exception is not a replayable model response. Keeping it
            # here would make every SDK retry re-await the same failed task even
            # after the cross-cluster path recovered. The GPU Policy ledger is the
            # authoritative idempotency boundary, so a later request with the same
            # id/digest can safely reattach there.
            if task.done():
                async with self._lock:
                    current = self._tasks.get(request_id)
                    if current is not None and current[1] is task:
                        self._tasks.pop(request_id, None)
            raise

    async def clear_attempt(self, attempt_id: str) -> None:
        prefix = f"{attempt_id}:"
        async with self._lock:
            for key in [key for key in self._tasks if key.startswith(prefix)]:
                _, task = self._tasks[key]
                if not task.done():
                    task.cancel()
                self._tasks.pop(key, None)


class PolicyRelayRegistry:
    def __init__(
        self,
        policy_url: str,
        *,
        transport_retry_attempts: int = 20,
        transport_retry_base_sec: float = 0.5,
        transport_retry_max_sec: float = 10.0,
    ) -> None:
        self.policy_url = policy_url.rstrip("/")
        self.transport_retry_attempts = transport_retry_attempts
        self.transport_retry_base_sec = transport_retry_base_sec
        self.transport_retry_max_sec = transport_retry_max_sec
        self.transport_failures_total = 0
        self.transport_retries_total = 0
        self.transport_recovered_total = 0
        self.transport_exhausted_total = 0
        self._routes: dict[str, RelayRoute] = {}
        self._ledger = RelayLedger()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(7200.0, connect=15.0),
            trust_env=False,
            limits=httpx.Limits(
                max_connections=512, max_keepalive_connections=512, keepalive_expiry=5
            ),
        )

    def add(self, route: RelayRoute) -> None:
        self._routes[route.attempt_id] = route

    async def remove(self, attempt_id: str) -> None:
        self._routes.pop(attempt_id, None)
        await self._ledger.clear_attempt(attempt_id)

    async def close(self) -> None:
        await self._client.aclose()

    async def forward(self, attempt_id: str, path: str, request: Request) -> Response:
        route = self._routes.get(attempt_id)
        if route is None:
            raise HTTPException(status_code=404, detail="unknown attempt route")
        normalized_path = path.strip("/")
        if normalized_path not in _ALLOWED_MODEL_PATHS:
            raise HTTPException(status_code=404, detail="unsupported model path")
        if not _authorized(request, route.local_token):
            raise HTTPException(status_code=401, detail="invalid local route token")
        body = await request.body()
        if len(body) > 32 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="model request is too large")
        body_digest = hashlib.sha256(body).hexdigest()
        supplied_id = request.headers.get("x-agentic-rollout-request-id")
        # One HTTP request accepted from the local agent is one logical model
        # call. Cross-cluster retries below this boundary reuse this id; a later
        # agent request gets a new id even when its body is byte-identical.
        logical_id = supplied_id or uuid.uuid4().hex
        request_id = f"{attempt_id}:{logical_id}"
        logger.info(
            "model_request_received execution=%s request_id=%s body_sha256=%s",
            attempt_id,
            logical_id,
            body_digest,
        )

        async def operation() -> BufferedResponse:
            headers = {
                key: value
                for key, value in request.headers.items()
                if key.lower()
                not in {
                    "authorization",
                    "x-api-key",
                    "content-length",
                    "host",
                    "transfer-encoding",
                }
            }
            headers["X-Agentic-Rollout-Request-Id"] = logical_id
            headers["X-Agentic-Rollout-Request-Digest"] = f"sha256:{body_digest}"
            upstream = f"{self.policy_url}/routes/{route.route_id}/{normalized_path}"
            response = await self._post_with_transport_recovery(
                upstream,
                content=body,
                headers=headers,
                attempt_id=attempt_id,
                request_id=request_id,
            )
            return BufferedResponse(
                status_code=response.status_code,
                headers={
                    key: value
                    for key, value in response.headers.items()
                    if key.lower()
                    in {"content-type", "anthropic-ratelimit-tokens-remaining"}
                },
                body=response.content,
            )

        try:
            buffered = await self._ledger.execute(request_id, body_digest, operation)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except httpx.TransportError as exc:
            self.transport_exhausted_total += 1
            request_key = hashlib.sha256(request_id.encode()).hexdigest()[:16]
            logger.error(
                "policy relay transport recovery exhausted attempt_id=%s "
                "request_key=%s error=%s",
                attempt_id,
                request_key,
                type(exc).__name__,
            )
            # A 5xx would invite the black-box agent SDK to create a second
            # logical request after this relay has exhausted its own retry
            # budget. Fail this attempt explicitly; attempt revision is the
            # only recovery boundary above the relay.
            return _non_retry_upstream_error(
                normalized_path,
                "GPU Policy transport is temporarily unavailable",
            )
        if 500 <= buffered.status_code < 600:
            logger.error(
                "GPU Policy returned HTTP %d attempt_id=%s",
                buffered.status_code,
                attempt_id,
            )
            return _non_retry_upstream_error(
                normalized_path,
                "GPU Policy transport is temporarily unavailable "
                f"(upstream HTTP {buffered.status_code})",
            )
        return Response(
            content=buffered.body,
            status_code=buffered.status_code,
            headers=buffered.headers,
        )

    async def _post_with_transport_recovery(
        self,
        upstream: str,
        *,
        content: bytes,
        headers: dict[str, str],
        attempt_id: str,
        request_id: str,
    ) -> httpx.Response:
        delay = self.transport_retry_base_sec
        request_key = hashlib.sha256(request_id.encode()).hexdigest()[:16]
        for retry_index in range(self.transport_retry_attempts):
            try:
                response = await self._client.post(
                    upstream, content=content, headers=headers
                )
            except httpx.TransportError as exc:
                self.transport_failures_total += 1
                if retry_index + 1 >= self.transport_retry_attempts:
                    raise
                self.transport_retries_total += 1
                logger.warning(
                    "policy relay transport failed; retrying attempt_id=%s "
                    "request_key=%s retry=%d/%d error=%s",
                    attempt_id,
                    request_key,
                    retry_index + 1,
                    self.transport_retry_attempts,
                    type(exc).__name__,
                )
                sleep_for = min(delay, self.transport_retry_max_sec)
                if sleep_for:
                    await asyncio.sleep(sleep_for * random.uniform(0.8, 1.2))
                delay = min(
                    max(delay * 2, self.transport_retry_base_sec),
                    self.transport_retry_max_sec,
                )
                continue
            if retry_index:
                self.transport_recovered_total += 1
                logger.info(
                    "policy relay transport recovered attempt_id=%s "
                    "request_key=%s retries=%d",
                    attempt_id,
                    request_key,
                    retry_index,
                )
            return response
        raise AssertionError("transport retry loop exited without a response")

    def transport_metrics(self) -> dict[str, int]:
        return {
            "transport_failures_total": self.transport_failures_total,
            "transport_retries_total": self.transport_retries_total,
            "transport_recovered_total": self.transport_recovered_total,
            "transport_exhausted_total": self.transport_exhausted_total,
        }


def install_relay_routes(app: FastAPI, registry: PolicyRelayRegistry) -> None:
    @app.api_route(
        "/relay/{attempt_id}/{path:path}",
        methods=["POST"],
        response_class=Response,
    )
    async def relay(attempt_id: str, path: str, request: Request) -> Response:
        return await registry.forward(attempt_id, path, request)


def _authorized(request: Request, token: str) -> bool:
    candidates = [request.headers.get("x-api-key", "")]
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        candidates.append(auth[7:].strip())
    return any(
        candidate and hmac.compare_digest(candidate, token) for candidate in candidates
    )


def _non_retry_upstream_error(path: str, message: str) -> Response:
    if path == "v1/messages":
        payload = {
            "type": "error",
            "error": {"type": "upstream_error", "message": message},
        }
    else:
        payload = {"error": {"type": "upstream_error", "message": message}}
    return Response(
        content=json.dumps(payload).encode(),
        status_code=400,
        media_type="application/json",
    )
