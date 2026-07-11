"""Qwen3.5 TITO tokenizer for lineage-local token construction."""

from __future__ import annotations

from typing import Any

from .template_utils import (
    apply_chat_template,
    assert_messages_append_only_with_allowed_roles,
)


_DUMMY_SYSTEM: dict[str, Any] = {"role": "system", "content": "dummy system"}


def _build_dummy_assistant(tool_responses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": " ",
        "tool_calls": [
            {
                "id": response.get("tool_call_id") or f"call0000{index}",
                "type": "function",
                "function": {
                    "name": response.get("name") or "dummy_func",
                    "arguments": {},
                },
            }
            for index, response in enumerate(tool_responses)
        ],
    }


class Qwen35TITOTokenizer:
    """Incrementally tokenize append-only Qwen3.5 chat context."""

    allowed_append_roles = ["tool", "user"]

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        newline_ids = self._encode_text("\n")
        self._newline_id = newline_ids[0] if len(newline_ids) == 1 else None
        self._im_end_id = self._convert_token_to_id("<|im_end|>")

    def _convert_token_to_id(self, token: str) -> int | None:
        converter = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if converter is None:
            return None
        try:
            token_id = converter(token)
        except Exception:
            return None
        if token_id is None:
            return None
        try:
            return int(token_id)
        except (TypeError, ValueError):
            return None

    def _render_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        rendered = apply_chat_template(
            messages,
            tokenizer=self.tokenizer,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            preserve_thinking=True,
        )
        return str(rendered)

    def _encode_text(self, text: str) -> list[int]:
        encoder = getattr(self.tokenizer, "encode", None)
        if encoder is not None:
            encoded = encoder(text, add_special_tokens=False)
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            return [int(token_id) for token_id in encoded]
        return [ord(character) for character in text]

    def _split_appended_segments(
        self, appended_messages: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        segments: list[list[dict[str, Any]]] = []
        index = 0
        while index < len(appended_messages):
            role = appended_messages[index].get("role")
            if role == "tool":
                end = index + 1
                while (
                    end < len(appended_messages)
                    and appended_messages[end].get("role") == "tool"
                ):
                    end += 1
                segments.append(appended_messages[index:end])
                index = end
                continue
            if role == "user":
                segments.append([appended_messages[index]])
                index += 1
                continue
            raise ValueError(f"unsupported appended role for TITO tokenization: {role}")
        return segments

    def _tokenize_rendered_suffix(
        self,
        base_messages: list[dict[str, Any]],
        appended_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        rendered_without = self._render_messages(
            base_messages,
            add_generation_prompt=False,
            tools=tools,
        )
        rendered_with = self._render_messages(
            base_messages + appended_messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        if not rendered_with.startswith(rendered_without):
            roles = [message.get("role") for message in appended_messages]
            if not roles:
                roles = ["generation_prompt"]
            raise ValueError(f"rendered suffix diff failed for {roles}")
        return self._encode_text(rendered_with[len(rendered_without) :])

    def _tokenize_tool_segment(
        self,
        appended_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> list[int]:
        return self._tokenize_rendered_suffix(
            [_DUMMY_SYSTEM, _build_dummy_assistant(appended_messages)],
            appended_messages,
            tools=tools,
        )

    def _tokenize_user_segment(
        self,
        appended_message: dict[str, Any],
        tools: list[dict[str, Any]] | None,
    ) -> list[int]:
        return self._tokenize_rendered_suffix(
            [_DUMMY_SYSTEM],
            [appended_message],
            tools=tools,
        )

    def tokenize_additional_non_assistant(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        assert_messages_append_only_with_allowed_roles(
            old_messages,
            new_messages,
            self.allowed_append_roles,
        )
        appended_messages = new_messages[len(old_messages) :]
        incremental: list[int] = []

        for segment in self._split_appended_segments(appended_messages):
            role = segment[0].get("role")
            if role == "tool":
                incremental.extend(self._tokenize_tool_segment(segment, tools))
            elif role == "user":
                incremental.extend(self._tokenize_user_segment(segment[0], tools))
            else:
                raise ValueError(f"unsupported appended role for TITO tokenization: {role}")

        incremental.extend(
            self._tokenize_rendered_suffix(
                new_messages,
                [],
                tools=tools,
                add_generation_prompt=True,
            )
        )
        return incremental

    def merge_tokens(
        self,
        *,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        incremental = self.tokenize_additional_non_assistant(
            old_messages,
            new_messages,
            tools,
        )
        prefix = list(pretokenized_token_ids)
        if (
            self._im_end_id is not None
            and self._newline_id is not None
            and prefix
            and prefix[-1] == self._im_end_id
        ):
            prefix.append(self._newline_id)
        return prefix + incremental
