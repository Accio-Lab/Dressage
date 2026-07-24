"""Slime custom generation hook for blackbox sandbox rollouts."""

from __future__ import annotations

import logging
import time
from typing import Any

from dressage.config import proxy_url
from dressage.paddock.blackbox.execute_hooks import (
    execute_blackbox_cmds_for_stage,
    parse_blackbox_execute_cmds,
)
from dressage.paddock.blackbox.failures import (
    HARVESTABLE_AGENT_ERRORS,
    agent_response_text,
    expected_abort_from_call_agent_exception,
    failure_from_call_agent_exception,
    failure_from_payload_state,
    record_blackbox_abort_for_retry,
    record_agent_early_stop_metadata,
    record_agent_failure_metadata,
)
from dressage.paddock.blackbox.common.defaults import (
    DEFAULT_BLACKBOX_TYPE,
    merge_backend_options,
    normalize_blackbox_type,
)
from dressage.paddock.lifecycle import (
    exception_summary as _exception_summary,
    schedule_terminate_paddock,
)
from dressage.rollout import multi_segment
from dressage.rollout.artifacts.samples import (
    instance_id as _instance_id,
    set_status as _set_status,
)
from dressage.rollout.artifacts.writer import DEFAULT_WRITER as _ARTIFACT_WRITER
from dressage.rollout.prewarm import (
    claim_prewarm,
    ensure_blackbox_session_id,
)
from dressage.rollout.generate.runtime import (
    get_paddock_from_env,
    get_proxy_client,
    maybe_await,
    paddock_env_args_from_metadata,
)

logger = logging.getLogger(__name__)


def _chat_messages_from_prompt(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return [dict(message) for message in prompt]
    return [{"role": "user", "content": str(prompt)}]


def _backend_options_for_register(
    *,
    args: Any,
    metadata: dict[str, Any],
    blackbox_type: str,
) -> Any:
    backend_options = metadata.get("backend_options")
    return merge_backend_options(blackbox_type, backend_options, args=args)


async def generate(
    args: Any,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Any:
    """Run one blackbox sandbox rollout and write proxy data back to Sample."""
    del sampling_params
    if evaluation:
        raise ValueError("blackbox_dispatch does not support evaluation mode")

    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    metadata.pop("blackbox_error", None)
    metadata.pop("blackbox_error_log_path", None)
    metadata.pop("blackbox_expected_abort", None)
    metadata.pop("blackbox_agent_early_stop", None)
    metadata.pop("blackbox_agent_early_stop_kind", None)
    metadata["execute_cmds"] = []
    session_id = ensure_blackbox_session_id(sample)
    instance_id = _instance_id(sample)
    metadata["session_id"] = session_id
    metadata["instance_id"] = instance_id
    blackbox_type = normalize_blackbox_type(
        metadata.get("blackbox_type") or DEFAULT_BLACKBOX_TYPE
    )

    extra_env_args = None
    if "blackbox_type" in metadata or blackbox_type != DEFAULT_BLACKBOX_TYPE:
        extra_env_args = {"blackbox_type": blackbox_type}
    env_args = paddock_env_args_from_metadata(
        metadata,
        extra_env_args=extra_env_args,
    )
    paddock = None
    state = None
    initialized = False
    agent_response = ""
    try:
        execute_cmd_schedule = parse_blackbox_execute_cmds(
            metadata.get("blackbox_execute_cmds")
        )
        backend_options = _backend_options_for_register(
            args=args,
            metadata=metadata,
            blackbox_type=blackbox_type,
        )
        proxy_client = get_proxy_client()
        prewarm_t0 = time.monotonic()
        handle = await claim_prewarm(session_id)
        prewarm_wait = time.monotonic() - prewarm_t0
        if handle is not None:
            paddock = handle.paddock
            state = handle.state
            env_args = handle.env_args
            initialized = True
            logger.debug(
                "claimed prewarmed sandbox for session_id=%s group_id=%d: "
                "wait=%.2fs",
                session_id,
                handle.group_id,
                prewarm_wait,
            )
        else:
            paddock = get_paddock_from_env(allow_whitebox_mode=False)
            state = await maybe_await(
                paddock.init(
                    session_id,
                    metadata.get("env_type"),
                    env_args,
                )
            )
            initialized = True
        if not hasattr(paddock, "register_agent"):
            raise TypeError(f"{type(paddock).__name__} does not implement register_agent")
        await maybe_await(
            paddock.register_agent(
                state,
                instance_id=instance_id,
                session_id=session_id,
                router_url=proxy_url(),
                blackbox_type=blackbox_type,
                backend_options=backend_options,
            )
        )
        await execute_blackbox_cmds_for_stage(
            paddock,
            state,
            metadata,
            schedule=execute_cmd_schedule,
            session_id=session_id,
            stage="before_agent",
        )
        call_payload: Any = None
        call_succeeded = False
        try:
            call_payload = await maybe_await(
                paddock.call_agent(
                    state,
                    session_id=session_id,
                    messages=_chat_messages_from_prompt(sample.prompt),
                    metadata={"source": "dressage", **metadata},
                )
            )
            call_succeeded = True
        except Exception as exc:
            if agent_failure := failure_from_call_agent_exception(exc):
                record_agent_failure_metadata(metadata, agent_failure)
                if agent_failure.kind in HARVESTABLE_AGENT_ERRORS:
                    record_agent_early_stop_metadata(metadata, agent_failure)
                    logger.info(
                        "harvesting blackbox rollout after agent early stop: "
                        "session_id=%s kind=%s",
                        session_id,
                        agent_failure.kind,
                    )
                else:
                    raise agent_failure from exc
            else:
                raise

        if call_succeeded:
            agent_response = agent_response_text(call_payload)
            if agent_failure := failure_from_payload_state(
                call_payload,
                agent_response=agent_response,
            ):
                record_agent_failure_metadata(metadata, agent_failure)
                raise agent_failure

        await execute_blackbox_cmds_for_stage(
            paddock,
            state,
            metadata,
            schedule=execute_cmd_schedule,
            session_id=session_id,
            stage="after_agent",
        )
        await proxy_client.finalize_session(
            session_id, instance_id=instance_id, label=getattr(sample, "label", None)
        )
        trajectory_payload = await proxy_client.read_trajectory(
            trajectory_id=session_id,
            instance_id=instance_id,
            drain=True,
        )
        try:
            await _ARTIFACT_WRITER.write_session_payload(
                trajectory_payload,
                session_id=session_id,
                instance_id=instance_id,
            )
        except Exception:
            logger.warning(
                "failed to write trajectory payload log for session_id=%s",
                session_id,
                exc_info=True,
            )
        segments = trajectory_payload.get("data") or []
        base_metadata_for_logs = dict(metadata)
        result = multi_segment.expand_segments_to_samples(
            sample,
            segments,
            args=args,
            agent_response=agent_response,
            session_id=session_id,
            instance_id=instance_id,
        )
        log_template = sample
        try:
            await _ARTIFACT_WRITER.write_segment_samples(
                log_template,
                args=args,
                segments=segments,
                base_metadata=base_metadata_for_logs,
                session_id=session_id,
                instance_id=instance_id,
                agent_response=agent_response,
            )
        except Exception:
            logger.warning(
                "failed to write sample logs for session_id=%s",
                session_id,
                exc_info=True,
            )
        return result
    except Exception as exc:
        expected_abort = expected_abort_from_call_agent_exception(exc)
        if expected_abort is None:
            logger.warning(
                "blackbox rollout failed for session_id=%s: %s",
                session_id,
                _exception_summary(exc),
            )
            try:
                error_log_path = await _ARTIFACT_WRITER.write_error(
                    exc,
                    sample=sample,
                    metadata=dict(metadata),
                    session_id=session_id,
                    instance_id=instance_id,
                    blackbox_type=blackbox_type,
                    env_args=dict(env_args),
                    state=state,
                    agent_response=agent_response,
                )
                if error_log_path is not None:
                    metadata["blackbox_error_log_path"] = str(error_log_path)
            except Exception:
                logger.warning(
                    "failed to write trajectory error log for session_id=%s",
                    session_id,
                    exc_info=True,
                )
            record_blackbox_abort_for_retry(metadata, session_id, exc)
        else:
            metadata["blackbox_expected_abort"] = expected_abort
        multi_segment.mark_aborted_no_grad(
            sample, session_id=session_id, instance_id=instance_id
        )
        _set_status(sample, "ABORTED")
        return sample
    finally:
        if initialized and paddock is not None:
            schedule_terminate_paddock(
                paddock,
                session_id=session_id,
                env_args=env_args,
            )
