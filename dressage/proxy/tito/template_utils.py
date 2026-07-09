"""Small chat-template helpers used by TITO tokenization."""

from __future__ import annotations

import copy
import json
from typing import Any

from ..reasoning_parser import canonicalize_reasoning_content
from ..tool_call_ids import canonicalize_openclaw_tool_call_id


_TEMPLATE_RELEVANT_KEYS = ("role", "content", "reasoning_content", "tool_calls")


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize OpenAI messages before chat-template rendering and comparison."""

    normalized = copy.deepcopy(messages)
    for message in normalized:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            if message.get("content") is None:
                message["content"] = ""
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            if "id" in tool_call:
                tool_call["id"] = canonicalize_openclaw_tool_call_id(tool_call["id"])
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
    return normalized


def _normalize_value(value: Any) -> Any:
    if value is None or value == "" or value == []:
        return None
    return value


def _canonical_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonical_jsonish(value[key])
            for key in sorted(value)
            if key != "index"
        }
    if isinstance(value, list):
        return [_canonical_jsonish(item) for item in value]
    return value


def message_matches(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Compare fields that affect the chat template output."""

    stored = normalize_messages([stored])[0]
    current = normalize_messages([current])[0]
    for key in _TEMPLATE_RELEVANT_KEYS:
        if key == "reasoning_content":
            stored_value = canonicalize_reasoning_content(stored.get(key))
            current_value = canonicalize_reasoning_content(current.get(key))
        else:
            stored_value = _canonical_jsonish(_normalize_value(stored.get(key)))
            current_value = _canonical_jsonish(_normalize_value(current.get(key)))
        if stored_value != current_value:
            return False
    return True


def assert_messages_append_only_with_allowed_roles(
    stored_messages: list[dict[str, Any]],
    current_messages: list[dict[str, Any]],
    allowed_append_roles: list[str],
) -> None:
    """Validate that current messages extend stored messages append-only."""

    if len(current_messages) < len(stored_messages):
        raise ValueError(
            f"new messages ({len(current_messages)}) are fewer than stored messages "
            f"({len(stored_messages)})"
        )

    for index, stored_message in enumerate(stored_messages):
        if not message_matches(stored_message, current_messages[index]):
            raise ValueError(f"message mismatch at index {index}")

    for index, message in enumerate(current_messages[len(stored_messages) :]):
        role = message.get("role")
        if role not in allowed_append_roles:
            absolute_index = len(stored_messages) + index
            raise ValueError(
                f"appended message at index {absolute_index} has role={role!r}, "
                f"allowed={allowed_append_roles}"
            )


def apply_chat_template(
    messages: list[dict[str, Any]],
    *,
    tokenizer: Any,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool,
    tokenize: bool,
    **kwargs: Any,
) -> str | list[int]:
    """Apply the tokenizer chat template with SGLang-style message normalization."""

    normalized = normalize_messages(messages)
    render_kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
        "return_dict": False,
        **kwargs,
    }
    if tools is not None:
        render_kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(normalized, **render_kwargs)
    except TypeError:
        render_kwargs.pop("tools", None)
        return tokenizer.apply_chat_template(normalized, **render_kwargs)
