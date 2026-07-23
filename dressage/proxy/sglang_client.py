"""Async client for the SGLang Router ``/generate`` endpoint."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx


def _coerce_int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    result: list[int] = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _coerce_logprob_entry(
    entry: Any,
) -> tuple[float | None, int | None, str | None]:
    if isinstance(entry, dict):
        logprob = entry.get("logprob")
        token_id = entry.get("token_id", entry.get("id", entry.get("token")))
        token_text = entry.get(
            "token_text",
            entry.get("text", entry.get("token_str", entry.get("decoded_token"))),
        )
        try:
            logprob_value = None if logprob is None else float(logprob)
        except (TypeError, ValueError):
            logprob_value = None
        try:
            token_id_value = None if token_id is None else int(token_id)
        except (TypeError, ValueError):
            token_id_value = None
        token_text_value = None if token_text is None else str(token_text)
        return logprob_value, token_id_value, token_text_value

    if isinstance(entry, (list, tuple)):
        logprob = entry[0] if len(entry) >= 1 else None
        token_id = entry[1] if len(entry) >= 2 else None
        token_text = entry[2] if len(entry) >= 3 else None
        try:
            logprob_value = None if logprob is None else float(logprob)
        except (TypeError, ValueError):
            logprob_value = None
        try:
            token_id_value = None if token_id is None else int(token_id)
        except (TypeError, ValueError):
            token_id_value = None
        token_text_value = None if token_text is None else str(token_text)
        return logprob_value, token_id_value, token_text_value

    try:
        return float(entry), None, None
    except (TypeError, ValueError):
        return None, None, None


def _parse_logprob_entries(
    entries: Any,
) -> tuple[list[float | None], list[int | None], list[str | None]]:
    logprobs: list[float | None] = []
    token_ids: list[int | None] = []
    token_texts: list[str | None] = []
    for entry in entries or []:
        logprob, token_id, token_text = _coerce_logprob_entry(entry)
        logprobs.append(logprob)
        token_ids.append(token_id)
        token_texts.append(token_text)
    return logprobs, token_ids, token_texts


def _normalize_logprobs(
    raw_values: list[float | None],
    expected_length: int,
    *,
    null_is_valid_zero: bool,
) -> tuple[list[float], bool]:
    invalid = len(raw_values) != expected_length
    normalized: list[float] = []
    for index in range(expected_length):
        value = raw_values[index] if index < len(raw_values) else None
        if value is None:
            normalized.append(0.0)
            if index >= len(raw_values) or not null_is_valid_zero:
                invalid = True
        else:
            normalized.append(float(value))
    return normalized, invalid


def _normalize_texts(
    raw_values: list[str | None],
    expected_length: int,
) -> list[str]:
    return [
        "" if index >= len(raw_values) or raw_values[index] is None else str(raw_values[index])
        for index in range(expected_length)
    ]


@dataclass
class SGLangResponse:
    """Normalized response payload from the router."""

    input_token_ids: list[int] = field(default_factory=list)
    input_token_logprobs_raw: list[float] = field(default_factory=list)
    input_token_texts: list[str] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)
    output_token_logprobs: list[float] = field(default_factory=list)
    output_token_texts: list[str] = field(default_factory=list)
    output_versions: list[str] = field(default_factory=list)
    all_token_ids: list[int] = field(default_factory=list)
    all_logprobs: list[float] = field(default_factory=list)
    text: str = ""
    meta_info: dict = field(default_factory=dict)
    finish_reason: str = "stop"
    input_logprobs_invalid: bool = False
    all_logprobs_invalid: bool = False
    routed_experts: str | None = None

    @property
    def weight_version(self) -> str | None:
        value = self.meta_info.get("weight_version")
        if value is None:
            value = self.meta_info.get("version")
        return None if value is None else str(value)


@dataclass(frozen=True)
class SGLangWorkerInfo:
    """Normalized worker metadata discovered from the router."""

    url: str
    is_healthy: bool = False
    connection_mode: str | None = None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


class SGLangRouterClient:
    """HTTP client that forwards router affinity via ``X-SMG-Routing-Key``."""

    def __init__(
        self,
        router_url: str,
        *,
        timeout: httpx.Timeout | None = None,
        client: httpx.AsyncClient | None = None,
        return_routed_experts: bool = False,
    ):
        self._router_url = router_url.rstrip("/")
        self._owns_client = client is None
        self._return_routed_experts = return_routed_experts
        self._client = client or httpx.AsyncClient(
            timeout=timeout or httpx.Timeout(None), trust_env=False
        )

    async def generate(
        self,
        input_ids: list[int],
        sampling_params: dict[str, Any],
        *,
        return_logprob: bool = True,
        logprob_start_len: int = 0,
        return_routed_experts: bool = False,
        routing_key: str | None = None,
        request_id: str | None = None,
    ) -> SGLangResponse:
        payload = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "return_text_in_logprobs": True,
        }
        if return_routed_experts or self._return_routed_experts:
            payload["return_routed_experts"] = True
        if request_id:
            # SGLang request-level abort is keyed by rid.  Keep the public
            # Python parameter name request_id because the rest of Dressage and
            # blackbox server use that terminology.
            payload["rid"] = request_id
        headers = {}
        if routing_key:
            headers["X-SMG-Routing-Key"] = routing_key

        response = await self._client.post(
            f"{self._router_url}/generate", json=payload, headers=headers
        )
        response.raise_for_status()
        data = response.json()
        return self._coerce_response(
            data,
            input_ids=input_ids,
            expect_input_logprobs=bool(return_logprob and logprob_start_len == 0),
        )

    async def abort_request(
        self,
        request_id: str,
        *,
        routing_key: str | None = None,
        abort_all: bool = False,
    ) -> dict[str, Any]:
        """Signal SGLang to abort an active request by rid.

        sgl-router may not expose /abort_request, while the runtime worker HTTP
        servers do.  Therefore generate still goes through the router, but abort
        is broadcast to healthy HTTP workers discovered from /workers.  The
        partial tokens are not expected in this response; they are returned by
        the original /generate request after the abort is handled.
        """
        headers = {}
        if routing_key:
            headers["X-SMG-Routing-Key"] = routing_key
        payload = {"rid": request_id, "abort_all": abort_all}

        successes: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        try:
            workers = await self.list_workers()
        except Exception as exc:
            workers = []
            errors.append({"target": f"{self._router_url}/workers", "error": repr(exc)})

        for worker in self._candidate_workers(workers):
            target = f"{worker.url}/abort_request"
            try:
                response = await self._client.post(target, json=payload, headers=headers)
                if response.status_code in {200, 204}:
                    successes.append({"target": target, "status_code": response.status_code})
                    continue
                errors.append({
                    "target": target,
                    "status_code": response.status_code,
                    "body": response.text[:500],
                })
            except Exception as exc:
                errors.append({"target": target, "error": repr(exc)})

        if successes:
            return {
                "success": True,
                "request_id": request_id,
                "rid": request_id,
                "abort_all": abort_all,
                "targets": successes,
                "errors": errors,
            }

        # Fallback for future router versions that may add /abort_request.  In
        # the current SGLang router this is commonly a 404.
        router_target = f"{self._router_url}/abort_request"
        try:
            response = await self._client.post(router_target, json=payload, headers=headers)
            if response.status_code in {200, 204}:
                return {
                    "success": True,
                    "request_id": request_id,
                    "rid": request_id,
                    "abort_all": abort_all,
                    "targets": [{"target": router_target, "status_code": response.status_code}],
                    "errors": errors,
                }
            errors.append({
                "target": router_target,
                "status_code": response.status_code,
                "body": response.text[:500],
            })
        except Exception as exc:
            errors.append({"target": router_target, "error": repr(exc)})

        raise RuntimeError(
            f"Failed to abort SGLang request {request_id!r}; errors={errors}"
        )

    def _accessible_router_host(self) -> str:
        parsed = urlsplit(self._router_url)
        host = parsed.hostname or "127.0.0.1"
        if host in {"0.0.0.0", "::"}:
            return "127.0.0.1"
        return host

    @staticmethod
    def _build_netloc(parsed: SplitResult, host: str) -> str:
        auth = ""
        if parsed.username is not None:
            auth = parsed.username
            if parsed.password is not None:
                auth += f":{parsed.password}"
            auth += "@"
        formatted_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{auth}{formatted_host}{port}"

    def _normalize_worker_url(self, worker_url: Any) -> str | None:
        if worker_url is None:
            return None
        url_text = str(worker_url).strip()
        if not url_text:
            return None

        router_parts = urlsplit(self._router_url)
        if "://" not in url_text:
            url_text = f"{router_parts.scheme or 'http'}://{url_text}"
        parsed = urlsplit(url_text)
        host = parsed.hostname
        if host is None:
            return None
        if host in {"0.0.0.0", "::"}:
            host = self._accessible_router_host()
        return urlunsplit(
            parsed._replace(netloc=self._build_netloc(parsed, host))
        ).rstrip("/")

    async def list_workers(self) -> list[SGLangWorkerInfo]:
        response = await self._client.get(f"{self._router_url}/workers")
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid /workers response payload.")

        raw_workers = data.get("workers")
        if not isinstance(raw_workers, list):
            raise ValueError("Invalid /workers response: missing workers list.")

        workers: list[SGLangWorkerInfo] = []
        for raw_worker in raw_workers:
            if not isinstance(raw_worker, dict):
                continue
            normalized_url = self._normalize_worker_url(raw_worker.get("url"))
            if normalized_url is None:
                continue
            connection_mode = raw_worker.get("connection_mode")
            workers.append(
                SGLangWorkerInfo(
                    url=normalized_url,
                    is_healthy=_coerce_bool(raw_worker.get("is_healthy")),
                    connection_mode=(
                        None if connection_mode is None else str(connection_mode)
                    ),
                )
            )
        return workers

    async def get_weight_versions(
        self, *, timeout_seconds: float = 5.0
    ) -> dict[str, str]:
        """Read the authoritative weight version from every healthy HTTP worker.

        The router's ``/generate`` response only tells us which version served a
        completed request.  That observation can be stale at the beginning of a
        new synchronous rollout batch, so integration preflight probes the
        workers' control-plane endpoint instead.  A partial snapshot is unsafe:
        one failed worker probe fails the complete operation.
        """

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        workers = self._candidate_workers(
            await asyncio.wait_for(
                self.list_workers(), timeout=timeout_seconds
            )
        )
        if not workers:
            raise RuntimeError(
                "SGLang router reported no healthy HTTP workers for weight-version probing"
            )

        async def read(worker: SGLangWorkerInfo) -> tuple[str, str]:
            response = await asyncio.wait_for(
                self._client.get(f"{worker.url}/get_weight_version"),
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or data.get("weight_version") is None:
                raise ValueError(
                    f"SGLang worker {worker.url!r} returned no weight_version"
                )
            version = str(data["weight_version"]).strip()
            if not version:
                raise ValueError(
                    f"SGLang worker {worker.url!r} returned an empty weight_version"
                )
            return worker.url, version

        pairs = await asyncio.gather(*(read(worker) for worker in workers))
        final_workers = self._candidate_workers(
            await asyncio.wait_for(
                self.list_workers(), timeout=timeout_seconds
            )
        )
        initial_urls = {worker.url for worker in workers}
        final_urls = {worker.url for worker in final_workers}
        if final_urls != initial_urls:
            raise RuntimeError(
                "SGLang worker topology changed during weight-version probing"
            )
        return dict(pairs)

    @staticmethod
    def _candidate_workers(workers: list[SGLangWorkerInfo]) -> list[SGLangWorkerInfo]:
        return [
            worker
            for worker in workers
            if worker.url
            and worker.is_healthy
            and (worker.connection_mode or "").lower() == "http"
        ]

    async def wait_until_ready(
        self,
        *,
        timeout_seconds: float = 30.0,
        interval_seconds: float = 1.0,
    ) -> dict[str, Any]:
        """Wait until the router reports at least one healthy HTTP worker.

        Keep this intentionally small: /workers is already the control-plane
        source used by parse_function_call/separate_reasoning/abort_request.
        Resume only needs a short gate before releasing suspended generations.
        """
        deadline = time.monotonic() + timeout_seconds
        attempts = 0
        last_error: str | None = None
        last_worker_count = 0
        last_healthy_worker_count = 0

        while True:
            attempts += 1
            try:
                workers = await self.list_workers()
                last_worker_count = len(workers)
                last_healthy_worker_count = len(self._candidate_workers(workers))
                if last_healthy_worker_count > 0:
                    return {
                        "ready": True,
                        "attempts": attempts,
                        "worker_count": last_worker_count,
                        "healthy_worker_count": last_healthy_worker_count,
                    }
                last_error = "router returned no healthy HTTP workers"
            except Exception as exc:
                last_error = repr(exc)

            if time.monotonic() >= deadline:
                return {
                    "ready": False,
                    "attempts": attempts,
                    "worker_count": last_worker_count,
                    "healthy_worker_count": last_healthy_worker_count,
                    "error": last_error,
                }

            await asyncio.sleep(min(interval_seconds, max(0.0, deadline - time.monotonic())))

    async def parse_function_call(
        self,
        text: str,
        *,
        tool_call_parser: str | None = None,
        parser: str | None = None,
        tools: list[dict] | None,
        routing_key: str | None = None,
    ) -> dict[str, Any] | None:
        parser_name = tool_call_parser or parser
        if not parser_name:
            return None

        try:
            workers = await self.list_workers()
        except Exception:
            return None

        headers = {}
        if routing_key:
            headers["X-SMG-Routing-Key"] = routing_key

        for worker in self._candidate_workers(workers):
            payload: dict[str, Any] = {
                "text": text,
                "tool_call_parser": parser_name,
            }
            if tools:
                payload["tools"] = tools

            try:
                response = await self._client.post(
                    f"{worker.url}/parse_function_call",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
            except Exception:
                continue

            if not isinstance(data, dict):
                continue
            if "normal_text" not in data or "calls" not in data:
                continue
            if not isinstance(data.get("calls"), list):
                continue

            normal_text = data.get("normal_text")
            if normal_text is not None and not isinstance(normal_text, str):
                normal_text = str(normal_text)
            return {"normal_text": normal_text, "calls": data.get("calls")}

        return None

    async def separate_reasoning(
        self,
        text: str,
        *,
        reasoning_parser: str | None = None,
        parser: str | None = None,
        routing_key: str | None = None,
    ) -> dict[str, Any] | None:
        parser_name = reasoning_parser or parser
        if not parser_name:
            return None

        try:
            workers = await self.list_workers()
        except Exception:
            return None

        headers = {}
        if routing_key:
            headers["X-SMG-Routing-Key"] = routing_key

        for worker in self._candidate_workers(workers):
            payload: dict[str, Any] = {
                "text": text,
                "reasoning_parser": parser_name,
            }

            try:
                response = await self._client.post(
                    f"{worker.url}/separate_reasoning",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
            except Exception:
                continue

            if not isinstance(data, dict):
                continue
            if "text" not in data:
                continue
            reasoning_text = data.get("reasoning_text", data.get("reasoning_content"))
            if reasoning_text is not None and not isinstance(reasoning_text, str):
                reasoning_text = str(reasoning_text)
            visible_text = data.get("text")
            if visible_text is not None and not isinstance(visible_text, str):
                visible_text = str(visible_text)
            return {"reasoning_text": reasoning_text, "text": visible_text}

        return None

    async def list_models(self) -> dict:
        response = await self._client.get(f"{self._router_url}/v1/models")
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _coerce_response(
        data: dict,
        *,
        input_ids: list[int],
        expect_input_logprobs: bool = True,
    ) -> SGLangResponse:
        meta = data.get("meta_info", {})
        raw_input_logprobs, _, raw_input_texts = _parse_logprob_entries(
            meta.get("input_token_logprobs", [])
        )
        raw_output_logprobs, parsed_output_ids, raw_output_texts = _parse_logprob_entries(
            meta.get("output_token_logprobs", [])
        )

        normalized_input_ids = list(input_ids)
        normalized_output_ids = _coerce_int_list(data.get("output_ids"))
        if not normalized_output_ids:
            normalized_output_ids = [
                token_id for token_id in parsed_output_ids if token_id is not None
            ]

        if expect_input_logprobs:
            normalized_input_logprobs, input_invalid = _normalize_logprobs(
                raw_input_logprobs,
                len(normalized_input_ids),
                null_is_valid_zero=True,
            )
        else:
            normalized_input_logprobs = [0.0] * len(normalized_input_ids)
            input_invalid = False
        normalized_output_logprobs, output_invalid = _normalize_logprobs(
            raw_output_logprobs,
            len(normalized_output_ids),
            null_is_valid_zero=False,
        )

        parsed_output_ids_compact = [
            token_id for token_id in parsed_output_ids if token_id is not None
        ]
        ids_invalid = bool(
            parsed_output_ids_compact
            and parsed_output_ids_compact != normalized_output_ids
        )

        input_token_texts = _normalize_texts(raw_input_texts, len(normalized_input_ids))
        output_token_texts = _normalize_texts(raw_output_texts, len(normalized_output_ids))

        weight_version = meta.get("weight_version", meta.get("version"))
        output_versions = (
            [str(weight_version)] * len(normalized_output_ids)
            if weight_version is not None
            else []
        )

        finish_reason_info = meta.get("finish_reason", {})
        if isinstance(finish_reason_info, dict):
            finish_reason = str(finish_reason_info.get("type", "stop"))
        else:
            finish_reason = str(finish_reason_info or "stop")

        routed_experts_raw = meta.get("routed_experts")

        return SGLangResponse(
            input_token_ids=normalized_input_ids,
            input_token_logprobs_raw=normalized_input_logprobs,
            input_token_texts=input_token_texts,
            output_ids=normalized_output_ids,
            output_token_logprobs=normalized_output_logprobs,
            output_token_texts=output_token_texts,
            output_versions=output_versions,
            all_token_ids=normalized_input_ids + normalized_output_ids,
            all_logprobs=normalized_input_logprobs + normalized_output_logprobs,
            text=data.get("text", ""),
            meta_info=meta,
            finish_reason=finish_reason,
            input_logprobs_invalid=input_invalid,
            all_logprobs_invalid=input_invalid or output_invalid or ids_invalid,
            routed_experts=routed_experts_raw if isinstance(routed_experts_raw, str) else None,
        )
