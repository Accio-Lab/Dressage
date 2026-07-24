"""Shared HTTP retry helpers for blackbox paddocks."""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import logging
from typing import Any

import httpx

from dressage.paddock.blackbox.common.utils import _exception_summary, _jittered_delay

DEFAULT_RETRY_HTTP_STATUS_CODES = frozenset({503})


def response_excerpt(response: httpx.Response, max_chars: int = 1000) -> str:
    """Return a compact response body excerpt for warning logs."""
    text = response.text or ""
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


async def post_json_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    json: dict[str, Any],
    retry_statuses: Collection[int] | None = None,
    operation: str,
    trajectory_id: str,
    max_attempts: int,
    initial_delay: float,
    max_delay: float,
    jitter_fraction: float,
    log_prefix: str,
    logger: logging.Logger,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """POST JSON, retrying connect failures and configured HTTP statuses.

    Non-retryable HTTP status errors are raised directly. The rollout layer records
    the full failure details and keeps the existing ABORTED retry flow.
    """
    retry_status_codes = (
        DEFAULT_RETRY_HTTP_STATUS_CODES if retry_statuses is None else set(retry_statuses)
    )
    delay = min(initial_delay, max_delay) if max_delay > 0 else 0.0
    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.post(url, json=json, headers=headers)
            if response.status_code in retry_status_codes:
                if attempt >= max_attempts:
                    logger.warning(
                        "%s %s for trajectory_id=%s retryable HTTP status "
                        "exhausted on attempt %d/%d: status=%d body=%r",
                        log_prefix,
                        operation,
                        trajectory_id,
                        attempt,
                        max_attempts,
                        response.status_code,
                        response_excerpt(response),
                    )
                    response.raise_for_status()

                sleep_for = _retry_after_or_jittered_delay(
                    response,
                    delay,
                    jitter_fraction,
                    max_delay,
                )
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                delay = _next_delay(delay, initial_delay, max_delay)
                continue

            response.raise_for_status()
            return response
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.PoolTimeout,
        ) as exc:
            if attempt >= max_attempts:
                logger.warning(
                    "%s %s for trajectory_id=%s connect retry exhausted on "
                    "attempt %d/%d: %s",
                    log_prefix,
                    operation,
                    trajectory_id,
                    attempt,
                    max_attempts,
                    _exception_summary(exc),
                )
                raise
            sleep_for = _jittered_delay(delay, jitter_fraction)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            delay = _next_delay(delay, initial_delay, max_delay)

    raise RuntimeError("unreachable blackbox HTTP retry loop exit")


async def get_json_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    retry_statuses: Collection[int] | None = None,
    operation: str,
    trajectory_id: str,
    max_attempts: int,
    initial_delay: float,
    max_delay: float,
    jitter_fraction: float,
    log_prefix: str,
    logger: logging.Logger,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """GET JSON, retrying connect failures and configured HTTP statuses.

    Used for long-poll turn status reads.  Non-retryable HTTP status errors
    (e.g. 404) are raised directly for the caller to treat as fatal.
    """
    retry_status_codes = (
        DEFAULT_RETRY_HTTP_STATUS_CODES if retry_statuses is None else set(retry_statuses)
    )
    delay = min(initial_delay, max_delay) if max_delay > 0 else 0.0
    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.get(url, params=params, headers=headers)
            if response.status_code in retry_status_codes:
                if attempt >= max_attempts:
                    logger.warning(
                        "%s %s for trajectory_id=%s retryable HTTP status "
                        "exhausted on attempt %d/%d: status=%d body=%r",
                        log_prefix,
                        operation,
                        trajectory_id,
                        attempt,
                        max_attempts,
                        response.status_code,
                        response_excerpt(response),
                    )
                    response.raise_for_status()

                sleep_for = _retry_after_or_jittered_delay(
                    response,
                    delay,
                    jitter_fraction,
                    max_delay,
                )
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                delay = _next_delay(delay, initial_delay, max_delay)
                continue

            response.raise_for_status()
            return response
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.PoolTimeout,
            httpx.ReadTimeout,
        ) as exc:
            if attempt >= max_attempts:
                logger.warning(
                    "%s %s for trajectory_id=%s connect retry exhausted on "
                    "attempt %d/%d: %s",
                    log_prefix,
                    operation,
                    trajectory_id,
                    attempt,
                    max_attempts,
                    _exception_summary(exc),
                )
                raise
            sleep_for = _jittered_delay(delay, jitter_fraction)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            delay = _next_delay(delay, initial_delay, max_delay)

    raise RuntimeError("unreachable blackbox HTTP GET retry loop exit")


def _next_delay(delay: float, initial_delay: float, max_delay: float) -> float:
    if max_delay > 0:
        return min(delay * 2 if delay > 0 else initial_delay, max_delay)
    return 0.0


def _retry_after_or_jittered_delay(
    response: httpx.Response,
    delay: float,
    jitter_fraction: float,
    max_delay: float,
) -> float:
    retry_after = _parse_retry_after_seconds(response.headers.get("Retry-After"))
    if retry_after is not None:
        if max_delay > 0:
            return min(retry_after, max_delay)
        return retry_after
    return _jittered_delay(delay, jitter_fraction)


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None

    raw = value.strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)

    return max((retry_at - datetime.now(timezone.utc)).total_seconds(), 0.0)
