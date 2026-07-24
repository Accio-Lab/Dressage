"""Convert Dressage trajectory payloads to Harbor ATIF-v1.7 dictionaries."""

from __future__ import annotations

import copy
from importlib import metadata
import json
from typing import Any


ATIF_SCHEMA_VERSION = "ATIF-v1.7"

_SEGMENT_EXTRA_FIELDS = (
    "uid",
    "segment_index",
    "segment_count",
    "turn_id",
    "timestamp",
    "finish_reason",
    "label",
    "aligned_response_length",
    "extra_info",
    "tokens",
    "full_loss_mask",
    "full_logprobs",
    "full_versions",
    "routed_experts",
    "routed_experts_chunks",
)


def trajectory_payload_to_harbor(
    trajectory_payload: dict[str, Any],
    *,
    session_id: str | None = None,
    instance_id: str | None = None,
    agent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a Proxy ``/trajectory/read`` payload to Harbor ATIF-v1.7."""
    if not isinstance(trajectory_payload, dict):
        raise TypeError("trajectory_payload must be a dictionary")
    segments = trajectory_payload.get("data")
    if not isinstance(segments, list) or not segments:
        raise ValueError("trajectory payload contains no segments")
    payload_metadata = {
        key: copy.deepcopy(value)
        for key, value in trajectory_payload.items()
        if key != "data"
    }
    return segments_to_harbor_trajectory(
        segments,
        session_id=session_id,
        instance_id=instance_id,
        agent=agent,
        payload_metadata=payload_metadata,
    )


def segments_to_harbor_trajectory(
    segments: list[dict[str, Any]],
    *,
    session_id: str | None = None,
    instance_id: str | None = None,
    agent: dict[str, Any] | None = None,
    payload_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert ordered Dressage segment snapshots to one ATIF trajectory."""
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments must contain at least one trajectory segment")
    if any(not isinstance(segment, dict) for segment in segments):
        raise TypeError("each trajectory segment must be a dictionary")

    ordered_segments = [
        segment
        for _, segment in sorted(
            enumerate(segments),
            key=lambda item: _segment_sort_key(item[1], item[0]),
        )
    ]
    first_segment = ordered_segments[0]
    effective_session_id = _first_text(
        session_id,
        first_segment.get("session_id"),
        first_segment.get("trajectory_id"),
    )
    trajectory_id = _first_text(
        first_segment.get("trajectory_id"),
        effective_session_id,
    )
    effective_instance_id = _first_text(
        instance_id,
        first_segment.get("instance_id"),
    )

    steps: list[dict[str, Any]] = []
    previous_candidates: list[dict[str, Any]] = []
    segment_extras: list[dict[str, Any]] = []
    for position, segment in enumerate(ordered_segments):
        segment_index = _segment_index(segment, position)
        if position:
            steps.append(_segment_boundary_step(segment, segment_index))

        candidates = _segment_step_candidates(segment, segment_index)
        for candidate_index, candidate in enumerate(candidates):
            step = copy.deepcopy(candidate)
            if (
                candidate_index < len(previous_candidates)
                and candidate == previous_candidates[candidate_index]
            ):
                step["is_copied_context"] = True
            steps.append(step)
        previous_candidates = candidates
        segment_extras.append(_segment_extra(segment))

    if not steps:
        raise ValueError("trajectory segments contain no messages")
    for step_id, step in enumerate(steps, start=1):
        step["step_id"] = step_id

    result: dict[str, Any] = {
        "schema_version": ATIF_SCHEMA_VERSION,
        "agent": _agent_payload(ordered_segments, agent),
        "steps": steps,
        "final_metrics": {"total_steps": len(steps)},
        "extra": {
            "dressage": {
                "payload": copy.deepcopy(payload_metadata or {}),
                "segments": segment_extras,
            }
        },
    }
    if effective_session_id is not None:
        result["session_id"] = effective_session_id
    if trajectory_id is not None:
        result["trajectory_id"] = trajectory_id
    if effective_instance_id is not None:
        result["extra"]["dressage"]["instance_id"] = effective_instance_id
    return result


def _segment_sort_key(segment: dict[str, Any], position: int) -> tuple[int, int, int]:
    try:
        return (0, int(segment.get("segment_index")), position)
    except (TypeError, ValueError):
        return (1, position, position)


def _segment_index(segment: dict[str, Any], position: int) -> int:
    try:
        return int(segment.get("segment_index", position))
    except (TypeError, ValueError):
        return position


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value):
            return str(value)
    return None


def _dressage_version() -> str:
    try:
        return metadata.version("dressage")
    except metadata.PackageNotFoundError:
        return "unknown"


def _agent_payload(
    segments: list[dict[str, Any]],
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    latest_tools = next(
        (
            copy.deepcopy(segment["tools"])
            for segment in reversed(segments)
            if isinstance(segment.get("tools"), list) and segment["tools"]
        ),
        None,
    )
    result: dict[str, Any] = {
        "name": "dressage",
        "version": _dressage_version(),
    }
    if latest_tools is not None:
        result["tool_definitions"] = latest_tools
    if override is not None:
        if not isinstance(override, dict):
            raise TypeError("agent must be a dictionary")
        result.update(copy.deepcopy(override))
    if not isinstance(result.get("name"), str) or not result["name"]:
        raise ValueError("agent.name must be a non-empty string")
    if not isinstance(result.get("version"), str) or not result["version"]:
        raise ValueError("agent.version must be a non-empty string")
    return result


def _segment_boundary_step(
    segment: dict[str, Any],
    segment_index: int,
) -> dict[str, Any]:
    extra_info = segment.get("extra_info")
    extra_info = extra_info if isinstance(extra_info, dict) else {}
    return {
        "source": "system",
        "message": "Dressage trajectory segment boundary",
        "extra": {
            "dressage": {
                "synthetic_segment_boundary": True,
                "segment_index": segment_index,
                "segment_uid": segment.get("uid"),
                "segment_reason": extra_info.get("segment_reason"),
                "segment_reasons": copy.deepcopy(
                    extra_info.get("segment_reasons") or []
                ),
            }
        },
    }


def _segment_step_candidates(
    segment: dict[str, Any],
    segment_index: int,
) -> list[dict[str, Any]]:
    messages = segment.get("messages")
    if not isinstance(messages, list):
        raise ValueError(
            f"segment {segment_index} field 'messages' must be a list"
        )

    candidates: list[dict[str, Any]] = []
    latest_agent_index: int | None = None
    for message_index, raw_message in enumerate(messages):
        if not isinstance(raw_message, dict):
            raw_message = {"role": "unknown", "content": raw_message}
        role = str(raw_message.get("role") or "unknown").lower()
        if role == "tool":
            result = _observation_result(raw_message)
            if latest_agent_index is None:
                raw_call_id = raw_message.get("tool_call_id")
                if raw_call_id is not None:
                    _merge_dressage_extra(
                        result,
                        {"original_tool_call_id": str(raw_call_id)},
                    )
                candidates.append(
                    {
                        "source": "system",
                        "message": "Tool observation",
                        "observation": {"results": [result]},
                        "extra": {"dressage": {"original_role": "tool"}},
                    }
                )
                continue
            agent_step = candidates[latest_agent_index]
            tool_call_ids = {
                item["tool_call_id"] for item in agent_step.get("tool_calls", [])
            }
            raw_call_id = raw_message.get("tool_call_id")
            if raw_call_id is not None and str(raw_call_id) in tool_call_ids:
                result["source_call_id"] = str(raw_call_id)
            elif raw_call_id is not None:
                _merge_dressage_extra(
                    result,
                    {"original_tool_call_id": str(raw_call_id)},
                )
            agent_step.setdefault("observation", {"results": []})[
                "results"
            ].append(result)
            continue

        content, original_content = _text_content(raw_message.get("content"))
        if role == "assistant":
            candidate: dict[str, Any] = {
                "source": "agent",
                "message": content,
            }
            reasoning = raw_message.get("reasoning_content")
            if reasoning is not None:
                candidate["reasoning_content"] = _text_content(reasoning)[0]
            raw_tool_calls = raw_message.get("tool_calls")
            if isinstance(raw_tool_calls, list) and raw_tool_calls:
                candidate["tool_calls"] = [
                    _tool_call(
                        item,
                        segment_index=segment_index,
                        message_index=message_index,
                        tool_index=tool_index,
                    )
                    for tool_index, item in enumerate(raw_tool_calls)
                ]
            if original_content is not None:
                _merge_dressage_extra(
                    candidate,
                    {"original_content": original_content},
                )
            candidates.append(candidate)
            latest_agent_index = len(candidates) - 1
            continue

        source = role if role in {"system", "user"} else "system"
        candidate = {"source": source, "message": content}
        dressage_extra: dict[str, Any] = {}
        if role not in {"system", "user"}:
            dressage_extra["original_role"] = role
        if original_content is not None:
            dressage_extra["original_content"] = original_content
        if dressage_extra:
            _merge_dressage_extra(candidate, dressage_extra)
        candidates.append(candidate)
    return candidates


def _tool_call(
    raw_tool_call: Any,
    *,
    segment_index: int,
    message_index: int,
    tool_index: int,
) -> dict[str, Any]:
    raw = raw_tool_call if isinstance(raw_tool_call, dict) else {}
    function = raw.get("function")
    function = function if isinstance(function, dict) else {}
    call_id = _first_text(raw.get("id"), raw.get("tool_call_id"))
    if call_id is None:
        call_id = f"dressage-{segment_index}-{message_index}-{tool_index}"
    function_name = _first_text(
        function.get("name"),
        raw.get("function_name"),
        raw.get("name"),
    ) or "unknown"
    raw_arguments = function.get("arguments", raw.get("arguments", {}))
    arguments, invalid_arguments = _tool_arguments(raw_arguments)

    result: dict[str, Any] = {
        "tool_call_id": call_id,
        "function_name": function_name,
        "arguments": arguments,
    }
    if isinstance(raw.get("extra"), dict):
        result["extra"] = copy.deepcopy(raw["extra"])
    if invalid_arguments:
        _merge_dressage_extra(
            result,
            {"raw_arguments": copy.deepcopy(raw_arguments)},
        )
    return result


def _tool_arguments(value: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(value, dict):
        return copy.deepcopy(value), False
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}, True
        if isinstance(parsed, dict):
            return parsed, False
    return {}, True


def _observation_result(message: dict[str, Any]) -> dict[str, Any]:
    content, original_content = _text_content(message.get("content"))
    result: dict[str, Any] = {"content": content}
    dressage_extra: dict[str, Any] = {}
    if original_content is not None:
        dressage_extra["original_content"] = original_content
    if message.get("name") is not None:
        dressage_extra["tool_name"] = str(message["name"])
    if dressage_extra:
        _merge_dressage_extra(result, dressage_extra)
    return result


def _text_content(value: Any) -> tuple[str, Any | None]:
    if value is None:
        return "", None
    if isinstance(value, str):
        return value, None
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
        copy.deepcopy(value),
    )


def _merge_dressage_extra(target: dict[str, Any], values: dict[str, Any]) -> None:
    extra = target.setdefault("extra", {})
    if not isinstance(extra, dict):
        extra = {"dressage": {"original_extra": copy.deepcopy(extra)}}
        target["extra"] = extra
    dressage = extra.setdefault("dressage", {})
    if not isinstance(dressage, dict):
        dressage = {"original_extra": copy.deepcopy(dressage)}
        extra["dressage"] = dressage
    dressage.update(values)


def _segment_extra(segment: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: copy.deepcopy(segment[key])
        for key in _SEGMENT_EXTRA_FIELDS
        if key in segment
    }
    result["tool_definitions"] = copy.deepcopy(segment.get("tools") or [])
    return result
