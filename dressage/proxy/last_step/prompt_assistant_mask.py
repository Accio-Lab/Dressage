"""Prompt assistant mask helpers for proxy segment alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..reasoning_parser import canonicalize_reasoning_content
from ..tool_call_ids import canonicalize_openclaw_tool_call_id


class ModelMaskTemplateRegistry:
    """Resolve model mask types to mask-only chat template files."""

    def __init__(self):
        self._templates: dict[str, Path] = {}

    def register(self, model_mask_type: str, template_path: Path) -> None:
        self._templates[model_mask_type] = Path(template_path)

    def resolve(self, model_mask_type: str | None) -> Path | None:
        if model_mask_type is None:
            return None
        return self._templates.get(model_mask_type)


def create_default_mask_template_registry() -> ModelMaskTemplateRegistry:
    registry = ModelMaskTemplateRegistry()
    registry.register(
        "qwen3_5",
        Path(__file__).with_name("templates")
        / "qwen3_5_mask_only_chat_template.jinja",
    )
    return registry


class PromptAssistantMaskBuilder:
    """Build dense response masks from the final step prompt and completion."""

    def __init__(
        self,
        tokenizer: Any,
        model_mask_type: str | None,
        registry: ModelMaskTemplateRegistry | None = None,
    ):
        self._tokenizer = tokenizer
        self._model_mask_type = model_mask_type
        self._registry = registry or create_default_mask_template_registry()
        self._mask_template_path = self._registry.resolve(model_mask_type)
        self._mask_chat_template = self._load_mask_chat_template()

    def _load_mask_chat_template(self) -> str | None:
        if self._mask_template_path is None:
            return None
        template = self._mask_template_path.read_text(encoding="utf-8")
        if not template.strip():
            raise ValueError(
                f"Mask-only chat template is empty: {self._mask_template_path}"
            )
        return template

    @staticmethod
    def _deep_copy_jsonish(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: PromptAssistantMaskBuilder._deep_copy_jsonish(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [PromptAssistantMaskBuilder._deep_copy_jsonish(item) for item in value]
        return value

    def normalize_template_messages(self, messages: list[dict]) -> list[dict]:
        normalized_messages: list[dict] = []
        for message in messages:
            normalized_message = {
                key: self._deep_copy_jsonish(value) for key, value in message.items()
            }
            if "reasoning_content" in normalized_message:
                reasoning_content = canonicalize_reasoning_content(
                    normalized_message.get("reasoning_content")
                )
                if reasoning_content is None:
                    normalized_message.pop("reasoning_content", None)
                else:
                    normalized_message["reasoning_content"] = reasoning_content
            if "tool_call_id" in normalized_message:
                normalized_message["tool_call_id"] = canonicalize_openclaw_tool_call_id(
                    normalized_message["tool_call_id"]
                )
            tool_calls = normalized_message.get("tool_calls")
            if not isinstance(tool_calls, list):
                normalized_messages.append(normalized_message)
                continue

            normalized_tool_calls: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                normalized_tool_call = self._deep_copy_jsonish(tool_call)
                if "id" in normalized_tool_call:
                    normalized_tool_call["id"] = canonicalize_openclaw_tool_call_id(
                        normalized_tool_call["id"]
                    )
                function = normalized_tool_call.get("function")
                if not isinstance(function, dict):
                    normalized_tool_calls.append(normalized_tool_call)
                    continue
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                normalized_tool_calls.append(normalized_tool_call)
            normalized_message["tool_calls"] = normalized_tool_calls
            normalized_messages.append(normalized_message)
        return normalized_messages

    def _apply_chat_template(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None,
        add_generation_prompt: bool,
        tokenize: bool,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        chat_template: str | None = None,
        template_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "return_dict": return_dict,
        }
        if return_assistant_tokens_mask:
            kwargs["return_assistant_tokens_mask"] = True
        if chat_template is not None:
            kwargs["chat_template"] = chat_template
        if tools is not None:
            kwargs["tools"] = tools
        if template_kwargs:
            kwargs.update(template_kwargs)
        try:
            return self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _coerce_flat_int_list(value: Any) -> list[int]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, list):
            return []
        if value and isinstance(value[0], list):
            value = value[0]
        result: list[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result

    def tokenize_messages(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        add_generation_prompt: bool,
        chat_template: str | None = None,
        template_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        token_ids = self._apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=False,
            chat_template=chat_template,
            template_kwargs=template_kwargs,
        )
        return self._coerce_flat_int_list(token_ids)

    def render_messages(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        add_generation_prompt: bool,
        chat_template: str | None = None,
        template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        rendered = self._apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            return_dict=False,
            chat_template=chat_template,
            template_kwargs=template_kwargs,
        )
        return str(rendered)

    def get_prompt_ids_and_assistant_masks(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        template_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[int], list[int]]:
        if self._mask_chat_template is None:
            return [], []
        encoded = self._apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            chat_template=self._mask_chat_template,
            template_kwargs=template_kwargs,
        )
        if not hasattr(encoded, "get"):
            return [], []
        return (
            self._coerce_flat_int_list(encoded.get("input_ids", [])),
            self._coerce_flat_int_list(encoded.get("assistant_masks", [])),
        )

    @staticmethod
    def _normalize_mask_length(mask: list[int], token_length: int) -> list[int]:
        normalized = [1 if value else 0 for value in list(mask[:token_length])]
        if len(normalized) < token_length:
            normalized.extend([0] * (token_length - len(normalized)))
        return normalized

    def _output_only_mask(self, step: Any) -> list[int]:
        return self._normalize_mask_length(
            [0] * len(step.prompt_token_ids) + [1] * len(step.response_token_ids),
            len(step.all_token_ids),
        )

    def build_segment_alignment(
        self,
        base_step: Any,
        tools: list[dict] | None,
    ) -> dict[str, Any]:
        train_tokens = list(base_step.all_token_ids)
        dense_logprobs = list(base_step.all_logprobs)
        prompt_assistant_mask: list[int] = []
        mask_template_equivalent = False
        mask_fallback_reason: str | None = None
        normalized_request_messages = list(base_step.normalized_request_messages)

        if self._mask_chat_template is None:
            mask_fallback_reason = "mask_template_not_registered_for_model_mask_type"
        elif base_step.all_logprobs_invalid:
            mask_fallback_reason = "all_logprobs_invalid"
        else:
            formal_rendered = self.render_messages(
                normalized_request_messages,
                tools,
                add_generation_prompt=True,
            )
            mask_rendered = self.render_messages(
                normalized_request_messages,
                tools,
                add_generation_prompt=True,
                chat_template=self._mask_chat_template,
            )
            if formal_rendered != mask_rendered:
                mask_fallback_reason = "mask_template_render_mismatch"
            else:
                prompt_ids, assistant_masks = self.get_prompt_ids_and_assistant_masks(
                    normalized_request_messages,
                    tools,
                )
                if prompt_ids != list(base_step.prompt_token_ids):
                    mask_fallback_reason = "mask_prompt_ids_mismatch"
                elif len(assistant_masks) != len(prompt_ids):
                    mask_fallback_reason = "assistant_mask_length_mismatch"
                else:
                    mask_template_equivalent = True
                    prompt_assistant_mask = list(assistant_masks)

        if mask_template_equivalent:
            response_mask = self._normalize_mask_length(
                prompt_assistant_mask + [1] * len(base_step.response_token_ids),
                len(train_tokens),
            )
        else:
            response_mask = self._output_only_mask(base_step)

        aligned_response_length = sum(response_mask)
        aligned_logprobs = [
            (
                dense_logprobs[index]
                if response_mask[index] and index < len(dense_logprobs)
                else 0.0
            )
            for index in range(len(train_tokens))
        ]
        prompt_assistant_token_count = sum(response_mask[: len(base_step.prompt_token_ids)])
        return {
            "tokens": train_tokens,
            "response_mask": response_mask,
            "response_logprobs": aligned_logprobs,
            "aligned_response_length": aligned_response_length,
            "mask_template_equivalent": mask_template_equivalent,
            "mask_fallback_reason": mask_fallback_reason,
            "prompt_assistant_token_count": prompt_assistant_token_count,
        }
