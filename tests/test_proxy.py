from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import httpx
from jinja2 import Environment, StrictUndefined
import pytest

from dressage.proxy import StepRecord, TrajectoryItem, TrajectorySegment, TurnRecord
from dressage.proxy.last_step import (
    ModelMaskTemplateRegistry,
    PromptAssistantMaskBuilder,
    create_default_mask_template_registry,
)
from dressage.proxy.server import create_app, parse_args
from dressage.proxy.session_manager import IMPLICIT_TURN_ID_PREFIX, SessionManager
from dressage.proxy.sglang_client import SGLangResponse, SGLangRouterClient
from dressage.proxy.reasoning_parser import (
    ProxyReasoningParser,
    canonicalize_reasoning_content,
    parse_qwen3_reasoning,
)
from dressage.proxy.tool_call_parser import (
    ModelToolCallParserRegistry,
    ProxyToolCallParser,
    ToolCallParserSpec,
    create_default_tool_call_parser_registry,
    parse_hermes_tool_calls,
    parse_qwen3_5_tool_calls,
)
from dressage.proxy.tito import load_fixed_template
from dressage.proxy.trajectory_store import TrajectoryStore

_UNSET = object()
_STRICT_TOOL_CALL_ID_RE = re.compile(r"^call[0-9a-f]{8}$")


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict,
        tools=None,
        chat_template=None,
        return_assistant_tokens_mask=False,
        **_,
    ):
        rendered = ""
        assistant_masks: list[int] = []
        for message in messages:
            chunk, chunk_mask = self._render_message(message)
            rendered += chunk
            assistant_masks.extend(chunk_mask)
        tools_text = self._render_tools_marker(tools, chat_template=chat_template)
        rendered += tools_text
        assistant_masks.extend([0] * len(tools_text))
        if add_generation_prompt:
            prompt = self._render_generation_prompt(chat_template=chat_template)
            rendered += prompt
            assistant_masks.extend([0] * len(prompt))
        rendered = self._maybe_adjust_rendered(
            rendered,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
        )
        if not tokenize:
            return rendered
        token_ids = [ord(ch) for ch in rendered]
        if return_dict:
            payload = {"input_ids": token_ids}
            if return_assistant_tokens_mask:
                payload["assistant_masks"] = self._maybe_adjust_assistant_masks(
                    assistant_masks,
                    chat_template=chat_template,
                )
            return payload
        return token_ids

    def _render_tools_marker(self, tools, *, chat_template=None) -> str:
        del tools
        del chat_template
        return ""

    def _render_generation_prompt(self, *, chat_template=None) -> str:
        del chat_template
        return "<assistant>"

    def _maybe_adjust_rendered(
        self,
        rendered: str,
        *,
        chat_template,
        add_generation_prompt: bool,
    ) -> str:
        del chat_template
        del add_generation_prompt
        return rendered

    def _maybe_adjust_assistant_masks(
        self,
        assistant_masks: list[int],
        *,
        chat_template,
    ) -> list[int]:
        del chat_template
        return list(assistant_masks)

    def _render_message(self, message: dict) -> tuple[str, list[int]]:
        role = message.get("role", "unknown")
        chunks = [f"<{role}>"]
        mask = [0] * len(chunks[0])
        reasoning_content = message.get("reasoning_content")
        if role == "assistant" and reasoning_content is not None:
            reasoning_text = f"<think>{reasoning_content}</think>\n\n"
            chunks.append(reasoning_text)
            mask.extend([1] * len(reasoning_text))
        content = message.get("content")
        if content is not None:
            content_text = str(content)
            chunks.append(content_text)
            if role == "assistant":
                mask.extend([1] * len(content_text))
            else:
                mask.extend([0] * len(content_text))
        for tool_call in message.get("tool_calls", []) or []:
            arguments = tool_call["function"]["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            payload = {
                "name": tool_call["function"]["name"],
                "arguments": arguments,
            }
            payload_text = json.dumps(payload)
            prefix = "<tool_call>"
            suffix = "</tool_call>"
            chunks.append(f"{prefix}{payload_text}{suffix}")
            tool_mask_value = 1 if role == "assistant" else 0
            mask.extend([0] * len(prefix))
            mask.extend([tool_mask_value] * len(payload_text))
            mask.extend([0] * len(suffix))
        if "tool_call_id" in message:
            tool_call_id = str(message["tool_call_id"])
            prefix = "<tool_call_id>"
            suffix = "</tool_call_id>"
            chunks.append(f"{prefix}{tool_call_id}{suffix}")
            mask.extend([0] * (len(prefix) + len(tool_call_id) + len(suffix)))
        return "".join(chunks), mask


class EncodeFallbackTokenizer(FakeTokenizer):
    def __init__(self, token_ids: list[int]):
        self.token_ids = list(token_ids)
        self.encode_calls = []

    def encode(self, text, *, add_special_tokens=True):
        self.encode_calls.append(
            {"text": text, "add_special_tokens": add_special_tokens}
        )
        return list(self.token_ids)


class CallableFallbackTokenizer(FakeTokenizer):
    def __init__(self, token_ids: list[int]):
        self.token_ids = list(token_ids)
        self.calls = []

    def __call__(self, text, *, add_special_tokens=True):
        self.calls.append({"text": text, "add_special_tokens": add_special_tokens})
        return {"input_ids": list(self.token_ids)}


class FakeSGLangClient:
    def __init__(
        self,
        responses: list[SGLangResponse],
        *,
        workers: list[dict[str, Any]] | None = None,
        list_workers_error: Exception | None = None,
        parse_function_call_responses: list[dict | Exception | None] | None = None,
        separate_reasoning_responses: list[dict | Exception | None] | None = None,
        weight_versions: dict[str, str] | None = None,
    ):
        self._responses = list(responses)
        self._workers = list(workers or [])
        self._list_workers_error = list_workers_error
        self._parse_function_call_responses = list(parse_function_call_responses or [])
        self._separate_reasoning_responses = list(separate_reasoning_responses or [])
        self._weight_versions = weight_versions
        self._simulate_worker_discovery = workers is not None or list_workers_error is not None
        self.last_routing_key = None
        self.last_parse_routing_key = None
        self.last_reasoning_routing_key = None
        self.calls = []
        self.list_workers_calls = []
        self.parse_function_call_calls = []
        self.separate_reasoning_calls = []
        self.get_weight_versions_calls = 0

    async def get_weight_versions(self):
        self.get_weight_versions_calls += 1
        if self._weight_versions is None:
            raise RuntimeError("weight-version snapshot unavailable in fake client")
        return dict(self._weight_versions)

    async def generate(
        self,
        input_ids,
        sampling_params,
        *,
        routing_key=None,
        return_logprob=True,
        logprob_start_len=0,
    ):
        self.calls.append(
            {
                "input_ids": list(input_ids),
                "sampling_params": dict(sampling_params),
                "routing_key": routing_key,
                "return_logprob": return_logprob,
                "logprob_start_len": logprob_start_len,
            }
        )
        self.last_routing_key = routing_key
        return self._hydrate_response(
            self._responses.pop(0),
            input_ids,
            expect_input_logprobs=bool(return_logprob and logprob_start_len == 0),
        )

    def _hydrate_response(
        self,
        response: SGLangResponse,
        input_ids: list[int],
        *,
        expect_input_logprobs: bool = True,
    ) -> SGLangResponse:
        prompt_ids = list(input_ids)
        raw_prompt_logprobs = response.input_token_logprobs_raw
        if not expect_input_logprobs:
            prompt_logprobs = [0.0] * len(prompt_ids)
            prompt_invalid = False
        elif raw_prompt_logprobs is None:
            prompt_logprobs = [-0.1] * len(prompt_ids)
            prompt_invalid = False
        else:
            prompt_logprobs = list(raw_prompt_logprobs)
            prompt_invalid = len(prompt_logprobs) != len(prompt_ids)
            if len(prompt_logprobs) < len(prompt_ids):
                prompt_logprobs.extend([0.0] * (len(prompt_ids) - len(prompt_logprobs)))
            else:
                prompt_logprobs = prompt_logprobs[: len(prompt_ids)]

        output_ids = list(response.output_ids)
        output_logprobs = list(response.output_token_logprobs)
        output_invalid = len(output_ids) != len(output_logprobs)
        if len(output_logprobs) < len(output_ids):
            output_logprobs.extend([0.0] * (len(output_ids) - len(output_logprobs)))
        else:
            output_logprobs = output_logprobs[: len(output_ids)]

        input_token_texts = list(
            response.input_token_texts
            if response.input_token_texts
            else [chr(token) for token in prompt_ids]
        )
        output_token_texts = list(
            response.output_token_texts
            if response.output_token_texts
            else [chr(token) for token in output_ids]
        )
        return SGLangResponse(
            input_token_ids=prompt_ids,
            input_token_logprobs_raw=prompt_logprobs,
            input_token_texts=input_token_texts,
            output_ids=output_ids,
            output_token_logprobs=output_logprobs,
            output_token_texts=output_token_texts,
            all_token_ids=prompt_ids + output_ids,
            all_logprobs=prompt_logprobs + output_logprobs,
            text=response.text,
            meta_info=response.meta_info,
            finish_reason=response.finish_reason,
            input_logprobs_invalid=(
                response.input_logprobs_invalid or prompt_invalid
            ),
            all_logprobs_invalid=(
                response.all_logprobs_invalid
                or response.input_logprobs_invalid
                or prompt_invalid
                or output_invalid
            ),
        )

    async def list_models(self):
        return {"object": "list", "data": [{"id": "fake-model"}]}

    async def list_workers(self):
        self.list_workers_calls.append({})
        if self._list_workers_error is not None:
            raise self._list_workers_error
        return list(self._workers)

    async def parse_function_call(
        self,
        text,
        *,
        tool_call_parser=None,
        parser=None,
        tools,
        routing_key=None,
    ):
        self.last_parse_routing_key = routing_key
        parser_name = tool_call_parser or parser
        if self._simulate_worker_discovery:
            try:
                workers = await self.list_workers()
            except Exception:
                return None

            eligible_workers = []
            for worker in workers:
                worker_url = worker.get("url") if isinstance(worker, dict) else getattr(worker, "url", None)
                is_healthy = (
                    worker.get("is_healthy")
                    if isinstance(worker, dict)
                    else getattr(worker, "is_healthy", False)
                )
                connection_mode = (
                    worker.get("connection_mode")
                    if isinstance(worker, dict)
                    else getattr(worker, "connection_mode", None)
                )
                if not worker_url or not is_healthy or str(connection_mode).lower() != "http":
                    continue
                eligible_workers.append(worker_url)

            for worker_url in eligible_workers:
                self.parse_function_call_calls.append(
                    {
                        "text": text,
                        "tool_call_parser": parser_name,
                        "tools": tools,
                        "routing_key": routing_key,
                        "worker_url": worker_url,
                    }
                )
                if not self._parse_function_call_responses:
                    return None
                response = self._parse_function_call_responses.pop(0)
                if isinstance(response, Exception):
                    continue
                return response
            return None

        self.parse_function_call_calls.append(
            {
                "text": text,
                "tool_call_parser": parser_name,
                "tools": tools,
                "routing_key": routing_key,
                "worker_url": None,
            }
        )
        if not self._parse_function_call_responses:
            return None
        response = self._parse_function_call_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def separate_reasoning(
        self,
        text,
        *,
        reasoning_parser=None,
        parser=None,
        routing_key=None,
    ):
        self.last_reasoning_routing_key = routing_key
        parser_name = reasoning_parser or parser
        if self._simulate_worker_discovery:
            try:
                workers = await self.list_workers()
            except Exception:
                return None

            eligible_workers = []
            for worker in workers:
                worker_url = worker.get("url") if isinstance(worker, dict) else getattr(worker, "url", None)
                is_healthy = (
                    worker.get("is_healthy")
                    if isinstance(worker, dict)
                    else getattr(worker, "is_healthy", False)
                )
                connection_mode = (
                    worker.get("connection_mode")
                    if isinstance(worker, dict)
                    else getattr(worker, "connection_mode", None)
                )
                if not worker_url or not is_healthy or str(connection_mode).lower() != "http":
                    continue
                eligible_workers.append(worker_url)

            for worker_url in eligible_workers:
                self.separate_reasoning_calls.append(
                    {
                        "text": text,
                        "reasoning_parser": parser_name,
                        "routing_key": routing_key,
                        "worker_url": worker_url,
                    }
                )
                if not self._separate_reasoning_responses:
                    return None
                response = self._separate_reasoning_responses.pop(0)
                if isinstance(response, Exception):
                    continue
                return response
            return None

        self.separate_reasoning_calls.append(
            {
                "text": text,
                "reasoning_parser": parser_name,
                "routing_key": routing_key,
                "worker_url": None,
            }
        )
        if not self._separate_reasoning_responses:
            return None
        response = self._separate_reasoning_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def close(self):
        return None


class BlockingSGLangClient(FakeSGLangClient):
    def __init__(self, responses: list[SGLangResponse]):
        super().__init__(responses)
        self.first_call_started = threading.Event()
        self.release_first_call = threading.Event()
        self.block_first_call = True

    async def generate(
        self,
        input_ids,
        sampling_params,
        *,
        routing_key=None,
        return_logprob=True,
        logprob_start_len=0,
    ):
        self.calls.append(
            {
                "input_ids": list(input_ids),
                "sampling_params": dict(sampling_params),
                "routing_key": routing_key,
                "return_logprob": return_logprob,
                "logprob_start_len": logprob_start_len,
            }
        )
        self.last_routing_key = routing_key
        if self.block_first_call:
            self.block_first_call = False
            self.first_call_started.set()
            await asyncio.to_thread(self.release_first_call.wait)
        return self._hydrate_response(
            self._responses.pop(0),
            input_ids,
            expect_input_logprobs=bool(return_logprob and logprob_start_len == 0),
        )


class TemplateMismatchTokenizer(FakeTokenizer):
    def _maybe_adjust_rendered(
        self,
        rendered: str,
        *,
        chat_template,
        add_generation_prompt: bool,
    ) -> str:
        del add_generation_prompt
        if chat_template is not None:
            return rendered + "<mask-only-mismatch>"
        return rendered


class AssistantMaskLengthMismatchTokenizer(FakeTokenizer):
    def _maybe_adjust_assistant_masks(
        self,
        assistant_masks: list[int],
        *,
        chat_template,
    ) -> list[int]:
        if chat_template is not None and assistant_masks:
            return assistant_masks[:-1]
        return list(assistant_masks)


class NoneVsEmptyToolsTokenizer(FakeTokenizer):
    def _render_tools_marker(self, tools, *, chat_template=None) -> str:
        del chat_template
        if tools is None:
            return "<tools:none>"
        if tools == []:
            return "<tools:empty>"
        return f"<tools:{json.dumps(tools, sort_keys=True)}>"


class FinalizeGuardTokenizer(FakeTokenizer):
    def __init__(self):
        self.reject_tokenization = False

    def apply_chat_template(self, *args, **kwargs):
        if self.reject_tokenization:
            raise AssertionError("finalize must not tokenize concat messages")
        return super().apply_chat_template(*args, **kwargs)


class OnlineTITOGuardTokenizer(FakeTokenizer):
    def __init__(self):
        self.reject_tokenization = False
        self.tokenize_calls = 0

    def apply_chat_template(self, *args, **kwargs):
        if kwargs.get("tokenize"):
            self.tokenize_calls += 1
            if self.reject_tokenization:
                raise AssertionError("append-only concat prompt must use online TITO")
        return super().apply_chat_template(*args, **kwargs)


class OnlineTITOFailureTokenizer(FakeTokenizer):
    def __init__(self):
        self.fail_tito_render = False

    def apply_chat_template(self, *args, **kwargs):
        if self.fail_tito_render and not kwargs.get("tokenize"):
            raise RuntimeError("forced TITO render failure")
        return super().apply_chat_template(*args, **kwargs)


class QwenJinjaTokenizer:
    def __init__(self):
        self.chat_template = load_fixed_template("qwen3_5")

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict=False,
        tools=None,
        chat_template=None,
        return_assistant_tokens_mask=False,
        **template_vars,
    ):
        template = chat_template or self.chat_template
        template = template.replace("{% generation %}", "").replace(
            "{% endgeneration %}", ""
        )

        def raise_exception(message):
            raise RuntimeError(message)

        rendered = Environment().from_string(template).render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            raise_exception=raise_exception,
            **template_vars,
        )
        if not tokenize:
            return rendered
        token_ids = [ord(ch) for ch in rendered]
        if not return_dict:
            return token_ids
        payload = {"input_ids": token_ids}
        if return_assistant_tokens_mask:
            payload["assistant_masks"] = [0] * len(token_ids)
        return payload


def make_response(
    text: str,
    *,
    finish_reason: str = "stop",
    prompt_logprobs=None,
    output_logprobs: list[float] | None = None,
    output_ids: list[int] | None = None,
    all_logprobs_invalid: bool = False,
) -> SGLangResponse:
    if output_logprobs is None:
        output_logprobs = [-0.1 for _ in text]
    return SGLangResponse(
        output_ids=[ord(ch) for ch in text] if output_ids is None else output_ids,
        output_token_logprobs=output_logprobs,
        text=text,
        meta_info={},
        finish_reason=finish_reason,
        input_token_logprobs_raw=prompt_logprobs,
        all_logprobs_invalid=all_logprobs_invalid,
    )


def make_prompt_logprob_missing_response(text: str) -> SGLangResponse:
    return make_response(text, prompt_logprobs=[])


def make_tools(*names: str) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": name, "parameters": {}}}
        for name in names
    ]


def legacy_underscored_tool_call_parser(
    text: str,
) -> tuple[str | None, list[dict] | None]:
    if text != "legacy tool call":
        return text, None
    return None, [
        {
            "id": "call_b1467966",
            "type": "function",
            "index": 0,
            "function": {"name": "search", "arguments": json.dumps({"q": "x"})},
        }
    ]


def sse_data_values(body: str) -> list[str]:
    return [
        line.removeprefix("data:").strip()
        for line in body.splitlines()
        if line.startswith("data:")
    ]


def sse_json_events(body: str) -> list[dict[str, Any]]:
    return [json.loads(data) for data in sse_data_values(body) if data != "[DONE]"]


def assert_stream_usage_chunk(body: str) -> dict[str, int]:
    data_values = sse_data_values(body)
    assert data_values[-1] == "[DONE]"

    events = sse_json_events(body)
    usage_events = [event for event in events if event.get("usage") is not None]
    assert len(usage_events) == 1

    usage_event = usage_events[0]
    usage_index = events.index(usage_event)
    finish_indices = [
        index
        for index, event in enumerate(events)
        if event["choices"] and event["choices"][0]["finish_reason"] is not None
    ]
    assert finish_indices[-1] < usage_index
    assert usage_event["choices"] == []
    assert usage_event["object"] == "chat.completion.chunk"

    usage = usage_event["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert all(
        "usage" in event and event["usage"] is None
        for event in events
        if event is not usage_event
    )
    return usage


def assert_stream_has_no_usage(body: str) -> None:
    data_values = sse_data_values(body)
    assert data_values[-1] == "[DONE]"

    events = sse_json_events(body)
    assert events
    assert all("usage" not in event for event in events)
    assert all(event["choices"] for event in events)


def make_qwen_tool_call_text(
    *,
    prefix: str,
    functions: list[tuple[str, dict[str, Any]]],
) -> str:
    lines = [prefix, "", "<tool_call>"]
    for function_name, parameters in functions:
        lines.append(f"<function={function_name}>")
        for parameter_name, parameter_value in parameters.items():
            lines.append(f"<parameter={parameter_name}>")
            if isinstance(parameter_value, str):
                lines.append(parameter_value)
            else:
                lines.append(json.dumps(parameter_value, ensure_ascii=False))
            lines.append("</parameter>")
        lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def decode_tokens(token_ids: list[int]) -> str:
    return "".join(chr(token) for token in token_ids)


def assert_dense_alignment(item: dict, texts: list[str], *, logprob: float = -0.1) -> None:
    rendered = decode_tokens(item["tokens"])
    expected_mask = [0] * len(item["tokens"])
    cursor = 0
    for text in texts:
        start = rendered.find(text, cursor)
        assert start >= 0, f"missing text {text!r} in {rendered!r}"
        end = start + len(text)
        expected_mask[start:end] = [1] * len(text)
        cursor = end
    assert item["full_loss_mask"] == expected_mask
    assert item["aligned_response_length"] == sum(expected_mask)
    expected_logprobs = [logprob if value else 0.0 for value in expected_mask]
    assert item["full_logprobs"] == expected_logprobs


def assert_alignment_metadata(
    item: dict,
    *,
    mask_template_equivalent: bool,
    mask_fallback_reason: str | None = None,
) -> None:
    extra = item["extra_info"]
    if extra["alignment_method"] == "snapshot_step":
        assert extra["token_build_mode"] == "snapshot"
        assert extra["segment_view"] == "timeline"
        return
    assert extra["alignment_method"] == "snapshot_all_logprobs+assistant_prompt_mask"
    assert extra["token_build_mode"] == "snapshot"
    assert extra["mask_template_equivalent"] is mask_template_equivalent
    if mask_fallback_reason is None:
        assert "mask_fallback_reason" not in extra
    else:
        assert extra["mask_fallback_reason"] == mask_fallback_reason


def assert_concat_mask_for_outputs(item: dict, output_texts: list[str]) -> None:
    rendered = decode_tokens(item["tokens"])
    expected_mask = [0] * len(item["tokens"])
    cursor = 0
    for text in output_texts:
        start = rendered.find(text, cursor)
        assert start >= 0, f"missing output text {text!r} in {rendered!r}"
        end = start + len(text)
        expected_mask[start:end] = [1] * len(text)
        cursor = end
    assert item["full_loss_mask"] == expected_mask
    assert item["aligned_response_length"] == sum(expected_mask)
    assert item["full_logprobs"] == [
        -0.1 if mask_value else 0.0 for mask_value in expected_mask
    ]


def make_client(
    *responses: SGLangResponse,
    tokenizer=None,
    sglang_client=None,
    model_mask_type: str | None = "qwen3_5",
    model_tool_call_type: str | None = "hermes",
    tool_call_parse_backend: str = "sglang_api",
    tool_call_parser=_UNSET,
    tool_call_parser_registry=None,
    model_reasoning_type: str | None = None,
    reasoning_parse_backend: str = "sglang_api",
    token_build_mode: str = "snapshot",
    tito_model: str | None = None,
    record_token_versions: bool = False,
    mask_nonlast_version_tokens: bool = False,
    rollout_temperature: float = 1.0,
    max_output_tokens: int | None = None,
    context_window: int | None = None,
    dynamic_max_tokens: bool = True,
    use_rollout_routing_replay: bool = False,
    partial_rollout: bool = False,
    max_partial_rollout_preempts: int | None = None,
):
    session_manager = SessionManager()
    trajectory_store = TrajectoryStore(min_group_size=1, group_timeout=0.0)
    tokenizer = tokenizer or FakeTokenizer()
    sglang_client = sglang_client or FakeSGLangClient(list(responses))
    create_app_kwargs = dict(
        sglang_router_url="http://router.test",
        tokenizer=tokenizer,
        session_manager=session_manager,
        trajectory_store=trajectory_store,
        sglang_client=sglang_client,
        model_mask_type=model_mask_type,
        model_tool_call_type=model_tool_call_type,
        tool_call_parse_backend=tool_call_parse_backend,
        model_reasoning_type=model_reasoning_type,
        reasoning_parse_backend=reasoning_parse_backend,
        tool_call_parser_registry=tool_call_parser_registry,
        token_build_mode=token_build_mode,
        tito_model=tito_model,
        record_token_versions=record_token_versions,
        mask_nonlast_version_tokens=mask_nonlast_version_tokens,
        rollout_temperature=rollout_temperature,
        max_output_tokens=max_output_tokens,
        context_window=context_window,
        dynamic_max_tokens=dynamic_max_tokens,
        use_rollout_routing_replay=use_rollout_routing_replay,
        partial_rollout=partial_rollout,
        max_partial_rollout_preempts=max_partial_rollout_preempts,
    )
    if tool_call_parser is not _UNSET:
        create_app_kwargs["tool_call_parser"] = tool_call_parser
    app = create_app(**create_app_kwargs)
    return (
        TestClient(app),
        session_manager,
        trajectory_store,
        sglang_client,
    )


def test_concat_mode_infers_tito_model_from_build_model():
    app = create_app(
        sglang_router_url="http://router.test",
        tokenizer=FakeTokenizer(),
        session_manager=SessionManager(),
        trajectory_store=TrajectoryStore(min_group_size=1, group_timeout=0.0),
        sglang_client=FakeSGLangClient([make_response("hello")]),
        token_build_mode="tito",
    )
    client = TestClient(app)

    result = client.get("/health")

    assert result.status_code == 200
    assert result.json()["config"]["token_build_model"] == "qwen3_5"


def test_proxy_defaults_sglang_router_url_from_env(monkeypatch):
    monkeypatch.setenv("SGLANG_ROUTER_URL", "http://router.env")
    app = create_app(
        tokenizer=FakeTokenizer(),
        session_manager=SessionManager(),
        trajectory_store=TrajectoryStore(min_group_size=1, group_timeout=0.0),
        sglang_client=FakeSGLangClient([make_response("hello")]),
    )
    client = TestClient(app)

    result = client.get("/health")

    assert result.status_code == 200
    assert result.json()["config"]["sglang_router_url"] == "http://router.env"
    assert result.json()["config"]["max_output_tokens"] is None


def test_proxy_health_reports_sampling_limits():
    client, _, _, _ = make_client(
        make_response("hello"),
        rollout_temperature=0.7,
        max_output_tokens=2048,
    )

    result = client.get("/health")

    assert result.status_code == 200
    config = result.json()["config"]
    assert config["rollout_temperature"] == 0.7
    assert config["max_output_tokens"] == 2048


def test_integration_capabilities_require_auth_and_report_version():
    tokenizer = FakeTokenizer()
    tokenizer.chat_template = "{{ messages }}"
    app = create_app(
        sglang_router_url="http://router.test",
        tokenizer_path="/models/tokenizer",
        tokenizer=tokenizer,
        session_manager=SessionManager(),
        trajectory_store=TrajectoryStore(min_group_size=1, group_timeout=0.0),
        sglang_client=FakeSGLangClient(
            [make_response("hello")],
            weight_versions={"http://worker.test": "weights-v7"},
        ),
        api_key="proxy-secret",
        token_build_mode="snapshot",
        token_build_model="qwen3_5",
        record_token_versions=True,
        partial_rollout=False,
    )
    client = TestClient(app)

    unauthorized = client.get("/integration/capabilities")
    assert unauthorized.status_code == 401

    headers = {"Authorization": "Bearer proxy-secret"}
    resumed = client.post(
        "/v1/rollout/resume",
        headers=headers,
        json={"version": "weights-v7"},
    )
    assert resumed.status_code == 200

    result = client.get("/integration/capabilities", headers=headers)
    assert result.status_code == 200
    assert result.json() == {
        "schema_version": "dressage.proxy.integration/v1",
        "token_build_mode": "snapshot",
        "token_build_model": "qwen3_5",
        "tokenizer_id": "/models/tokenizer",
        "chat_template_fingerprint": hashlib.sha256(
            tokenizer.chat_template.encode("utf-8")
        ).hexdigest(),
        "record_token_versions": True,
        "mask_nonlast_version_tokens": False,
        "max_output_tokens": None,
        "partial_rollout": False,
        "max_partial_rollout_preempts": None,
        "current_weight_version": "weights-v7",
        "cached_weight_version": "weights-v7",
        "weight_version_authoritative": True,
        "weight_versions_consistent": True,
        "weight_version_source": "worker_snapshot",
        "weight_version_worker_count": 1,
        "supports_expected_version": True,
    }


def test_integration_capabilities_rejects_inconsistent_worker_versions():
    app = create_app(
        tokenizer=FakeTokenizer(),
        session_manager=SessionManager(),
        trajectory_store=TrajectoryStore(min_group_size=1, group_timeout=0.0),
        sglang_client=FakeSGLangClient(
            [make_response("hello")],
            weight_versions={"worker-a": "v7", "worker-b": "v8"},
        ),
        api_key="no-auth",
    )

    payload = TestClient(app).get("/integration/capabilities").json()

    assert payload["weight_version_authoritative"] is True
    assert payload["weight_versions_consistent"] is False
    assert payload["current_weight_version"] is None
    assert payload["weight_version_worker_count"] == 2


@pytest.mark.parametrize(
    ("observed_version", "error_code"),
    [
        (None, "weight_version_unavailable"),
        ("v8", "weight_version_mismatch"),
    ],
)
def test_version_pinned_generation_requires_matching_observed_version(
    observed_version, error_code
):
    response = make_response("hello")
    if observed_version is not None:
        response.meta_info = {"weight_version": observed_version}
    client, _, trajectory_store, _ = make_client(response)

    result = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": f"sess-{error_code}",
            "X-Instance-Id": "instance-a",
            "X-Dressage-Expected-Version": "v7",
        },
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert result.status_code == 502
    assert result.json()["detail"]["error"] == error_code
    assert trajectory_store.stats()["total_items"] == 0


def test_proxy_health_reports_rollout_routing_replay():
    client, _, _, _ = make_client(
        make_response("hello"),
        use_rollout_routing_replay=True,
    )

    result = client.get("/health")

    assert result.status_code == 200
    assert result.json()["config"]["use_rollout_routing_replay"] is True


def test_proxy_rollout_routing_replay_initializes_router_client(monkeypatch):
    captured: dict[str, bool] = {}

    class CapturingSGLangClient(FakeSGLangClient):
        def __init__(
            self,
            router_url,
            *,
            return_routed_experts=False,
        ):
            del router_url
            captured["return_routed_experts"] = return_routed_experts
            super().__init__([make_response("hello")])

    monkeypatch.setattr(
        "dressage.proxy.server.SGLangRouterClient",
        CapturingSGLangClient,
    )

    create_app(
        sglang_router_url="http://router.test",
        tokenizer=FakeTokenizer(),
        session_manager=SessionManager(),
        trajectory_store=TrajectoryStore(min_group_size=1, group_timeout=0.0),
        use_rollout_routing_replay=True,
    )

    assert captured["return_routed_experts"] is True


def test_proxy_sampling_params_use_rollout_temperature_fallback():
    client, _, _, sglang_client = make_client(
        make_response("hello"),
        rollout_temperature=0.7,
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-temp", "X-Instance-Id": "inst"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert sglang_client.calls[0]["sampling_params"]["temperature"] == 0.7


def test_proxy_sampling_params_body_temperature_wins():
    client, _, _, sglang_client = make_client(
        make_response("hello"),
        rollout_temperature=0.7,
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-temp", "X-Instance-Id": "inst"},
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.0,
        },
    )

    assert response.status_code == 200
    assert sglang_client.calls[0]["sampling_params"]["temperature"] == 0.0


def test_token_build_model_infers_qwen_tool_parser():
    raw_tool = make_qwen_tool_call_text(
        prefix="Thinking Process: call tool",
        functions=[("search", {"q": "x"})],
    )
    client, _, _, _ = make_client(
        make_response(raw_tool),
        model_tool_call_type=None,
        tool_call_parse_backend="local",
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-qwen-default", "X-Instance-Id": "inst"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "find"}]},
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["tool_calls"][0]["function"]["name"] == "search"


def test_trajectory_read_with_drain_removes_exact_trajectory():
    client, _, trajectory_store, _ = make_client()
    trajectory_store.write_dict(
        {
            "trajectory_id": "drain-sess",
            "turn_id": "1",
            "instance_id": "drain-inst",
            "messages": [{"role": "assistant", "content": "done"}],
            "tools": None,
            "tokens": [1, 2, 3],
            "full_logprobs": [0.0, -0.1, -0.2],
            "full_loss_mask": [0, 1, 1],
            "full_versions": ["v0", "v1", "v1"],
            "aligned_response_length": 2,
        }
    )

    drained = client.post(
        "/trajectory/read",
        json={
            "trajectory_id": "drain-sess",
            "instance_id": "drain-inst",
            "drain": True,
        },
    )

    assert drained.status_code == 200
    assert drained.json()["success"] is True
    assert drained.json()["drained"] is True
    assert len(drained.json()["data"]) == 1

    second = client.post(
        "/trajectory/read",
        json={"trajectory_id": "drain-sess", "instance_id": "drain-inst"},
    )
    assert second.status_code == 200
    assert second.json()["success"] is False
    assert second.json()["data"] == []


def test_trajectory_store_allows_missing_full_versions():
    trajectory_store = TrajectoryStore(min_group_size=1, group_timeout=0.0)
    trajectory_store.write_dict(
        {
            "trajectory_id": "no-version-sess",
            "turn_id": "1",
            "instance_id": "no-version-inst",
            "messages": [{"role": "assistant", "content": "done"}],
            "tools": None,
            "tokens": [1, 2, 3],
            "full_logprobs": [0.0, -0.1, -0.2],
            "full_loss_mask": [0, 1, 1],
            "aligned_response_length": 2,
        }
    )

    item = trajectory_store.read_trajectory("no-version-sess")[0]

    assert "full_versions" not in item


def test_trajectory_store_write_dict_requires_full_segment_fields():
    trajectory_store = TrajectoryStore(min_group_size=1, group_timeout=0.0)
    try:
        trajectory_store.write_dict(
            {
                "trajectory_id": "legacy-sess",
                "turn_id": "1",
                "instance_id": "legacy-inst",
                "messages": [{"role": "assistant", "content": "done"}],
                "tools": None,
                "tokens": [1, 2, 3],
                "response_logprobs": [0.0, -0.1, -0.2],
                "response_mask": [0, 1, 1],
                "aligned_response_length": 2,
            }
        )
    except ValueError as exc:
        assert "full_logprobs" in str(exc)
    else:
        raise AssertionError("Expected legacy trajectory segment payload to fail")


def test_trajectory_store_write_many_validates_before_publishing_any_record():
    trajectory_store = TrajectoryStore(min_group_size=1, group_timeout=0.0)
    valid = {
        "trajectory_id": "atomic-sess",
        "turn_id": "1",
        "instance_id": "atomic-inst",
        "messages": [{"role": "assistant", "content": "done"}],
        "tools": None,
        "tokens": [1, 2],
        "full_logprobs": [0.0, -0.1],
        "full_loss_mask": [0, 1],
        "aligned_response_length": 1,
    }
    invalid = {**valid, "turn_id": "2"}
    invalid.pop("full_logprobs")

    with pytest.raises(ValueError, match="full_logprobs"):
        trajectory_store.write_many([valid, invalid])

    assert trajectory_store.stats()["total_items"] == 0
    assert trajectory_store.read_trajectory("atomic-sess") == []

    partial = {
        **valid,
        "segment_index": 0,
        "segment_count": 2,
        "extra_info": {
            "segment_view": "timeline",
            "finalization_complete": True,
            "finalization_id": "final-atomic",
        },
    }
    with pytest.raises(ValueError, match="complete 0..N-1"):
        trajectory_store.write_many([partial])
    assert trajectory_store.stats()["total_items"] == 0


def test_finalize_is_idempotent_and_marks_complete_segment_set():
    client, _, trajectory_store, _ = make_client(make_response("done"))
    generated = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "final-once", "X-Instance-Id": "instance-once"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "go"}]},
    )
    assert generated.status_code == 200

    payload = {"session_id": "final-once", "instance_id": "instance-once"}
    first = client.post("/session/finalize", json=payload)
    second = client.post("/session/finalize", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    assert trajectory_store.stats()["total_items"] == 1
    segment = trajectory_store.read_trajectory("final-once")[0]
    assert segment["segment_index"] == 0
    assert segment["segment_count"] == 1
    assert segment["extra_info"]["finalization_complete"] is True
    assert (
        segment["extra_info"]["finalization_id"]
        == first.json()["finalization_id"]
    )


def test_finalize_write_failure_preserves_active_session_and_empty_store():
    class FailingTrajectoryStore(TrajectoryStore):
        def write_many(self, records):
            raise RuntimeError("injected atomic write failure")

    session_manager = SessionManager()
    trajectory_store = FailingTrajectoryStore(min_group_size=1, group_timeout=0.0)
    app = create_app(
        sglang_router_url="http://router.test",
        tokenizer=FakeTokenizer(),
        session_manager=session_manager,
        trajectory_store=trajectory_store,
        sglang_client=FakeSGLangClient([make_response("done")]),
        token_build_mode="snapshot",
    )
    client = TestClient(app, raise_server_exceptions=False)
    generated = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "final-fail", "X-Instance-Id": "instance-fail"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "go"}]},
    )
    assert generated.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "final-fail", "instance_id": "instance-fail"},
    )

    assert finalized.status_code == 500
    assert session_manager.get_session("final-fail") is not None
    assert session_manager.get_finalization_result("final-fail") is None
    assert trajectory_store.stats()["total_items"] == 0


def test_local_reasoning_backend_rejects_unsupported_model_at_startup():
    try:
        create_app(
            sglang_router_url="http://router.test",
            tokenizer=FakeTokenizer(),
            session_manager=SessionManager(),
            trajectory_store=TrajectoryStore(min_group_size=1, group_timeout=0.0),
            sglang_client=FakeSGLangClient([make_response("hello")]),
            model_reasoning_type="deepseek-r1",
            reasoning_parse_backend="local",
        )
    except ValueError as exc:
        assert "local reasoning parser only supports" in str(exc)
    else:
        raise AssertionError("Expected unsupported local reasoning parser to fail")


def test_parse_args_defaults_to_sglang_api_parser_backends(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "dressage.proxy.server",
            "--sglang-router-url",
            "http://router.test",
            "--tokenizer-path",
            "fake-tokenizer",
        ],
    )

    args = parse_args()

    assert args.tool_call_parse_backend == "sglang_api"
    assert args.reasoning_parse_backend == "sglang_api"
    assert args.model_reasoning_type is None
    assert args.context_window is None
    assert args.max_output_tokens is None
    assert args.dynamic_max_tokens is True


def test_parse_args_rejects_non_positive_context_window(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "dressage.proxy.server",
            "--sglang-router-url",
            "http://router.test",
            "--tokenizer-path",
            "fake-tokenizer",
            "--context-window",
            "0",
        ],
    )

    try:
        parse_args()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected --context-window=0 to fail")


@pytest.mark.parametrize("value", ["0", "-1"])
def test_parse_args_rejects_non_positive_max_output_tokens(monkeypatch, value):
    monkeypatch.setattr(
        "sys.argv",
        [
            "dressage.proxy.server",
            "--tokenizer-path",
            "fake-tokenizer",
            "--max-output-tokens",
            value,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


@pytest.mark.parametrize("value", [0, -1])
def test_create_app_rejects_non_positive_max_output_tokens(value):
    with pytest.raises(ValueError, match="max_output_tokens must be greater than 0"):
        create_app(max_output_tokens=value)


def test_parse_args_can_disable_dynamic_max_tokens(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "dressage.proxy.server",
            "--tokenizer-path",
            "fake-tokenizer",
            "--no-dynamic-max-tokens",
        ],
    )

    assert parse_args().dynamic_max_tokens is False


def test_parse_args_accepts_max_partial_rollout_preempts(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "dressage.proxy.server",
            "--tokenizer-path",
            "fake-tokenizer",
            "--max-partial-rollout-preempts",
            "0",
        ],
    )

    assert parse_args().max_partial_rollout_preempts == 0


def test_chat_completion_context_window_input_overflow_skips_sglang():
    sglang_client = FakeSGLangClient([make_response("unused")])
    client, session_manager, _, _ = make_client(
        sglang_client=sglang_client,
        context_window=1,
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-input-overflow", "X-Instance-Id": "inst-1"},
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        },
    )

    body = response.json()
    assert response.status_code == 413
    assert body["error"] == "context_overflow"
    assert body["details"]["phase"] == "input"
    assert body["details"]["last_proxy_step_recorded"] is False
    assert body["details"]["max_tokens"] == 4
    assert sglang_client.calls == []
    assert session_manager.get_session("sess-input-overflow").steps == []


def test_chat_completion_context_window_equal_limit_returns_input_overflow():
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "hi"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    sglang_client = FakeSGLangClient([make_response("unused")])
    client, _, _, _ = make_client(
        tokenizer=tokenizer,
        sglang_client=sglang_client,
        context_window=len(prompt_ids),
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": messages, "max_tokens": 2},
    )

    assert response.status_code == 413
    assert response.json()["details"]["phase"] == "input"
    assert sglang_client.calls == []


def test_chat_completion_clamps_max_tokens_to_remaining_context():
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "hi"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    sglang_client = FakeSGLangClient([make_response("a")])
    client, _, _, _ = make_client(
        tokenizer=tokenizer,
        sglang_client=sglang_client,
        context_window=len(prompt_ids) + 2,
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": messages, "max_tokens": 3},
    )

    assert response.status_code == 200
    assert sglang_client.calls[0]["sampling_params"]["max_new_tokens"] == 1


def test_chat_completion_preserves_smaller_requested_max_tokens():
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "hi"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    sglang_client = FakeSGLangClient([make_response("a")])
    client, _, _, _ = make_client(
        tokenizer=tokenizer,
        sglang_client=sglang_client,
        context_window=len(prompt_ids) + 10,
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": messages, "max_tokens": 2},
    )

    assert response.status_code == 200
    assert sglang_client.calls[0]["sampling_params"]["max_new_tokens"] == 2


def test_chat_completion_uses_requested_max_tokens_without_proxy_cap():
    sglang_client = FakeSGLangClient([make_response("a")])
    client, _, _, _ = make_client(sglang_client=sglang_client)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 512,
        },
    )

    assert response.status_code == 200
    assert sglang_client.calls[0]["sampling_params"]["max_new_tokens"] == 512


def test_chat_completion_caps_explicit_max_tokens():
    tokenizer = FakeTokenizer()
    empty_messages = [{"role": "user", "content": ""}]
    empty_prompt_ids = tokenizer.apply_chat_template(
        empty_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    messages = [
        {"role": "user", "content": "x" * (18534 - len(empty_prompt_ids))}
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    assert len(prompt_ids) == 18534
    sglang_client = FakeSGLangClient([make_response("a")])
    client, _, _, _ = make_client(
        tokenizer=tokenizer,
        sglang_client=sglang_client,
        max_output_tokens=2048,
        context_window=49152,
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": messages,
            "max_tokens": 30618,
        },
    )

    assert response.status_code == 200
    assert sglang_client.calls[0]["sampling_params"]["max_new_tokens"] == 2048


def test_chat_completion_preserves_request_below_output_cap():
    sglang_client = FakeSGLangClient([make_response("a")])
    client, _, _, _ = make_client(
        sglang_client=sglang_client,
        max_output_tokens=2048,
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 512,
        },
    )

    assert response.status_code == 200
    assert sglang_client.calls[0]["sampling_params"]["max_new_tokens"] == 512


def test_chat_completion_combines_output_cap_with_remaining_context():
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "hi"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    sglang_client = FakeSGLangClient([make_response("a")])
    client, _, _, _ = make_client(
        tokenizer=tokenizer,
        sglang_client=sglang_client,
        context_window=len(prompt_ids) + 1025,
        max_output_tokens=2048,
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": messages, "max_tokens": 30618},
    )

    assert response.status_code == 200
    assert sglang_client.calls[0]["sampling_params"]["max_new_tokens"] == 1024


def test_chat_completion_uses_capped_default_when_max_tokens_is_missing():
    sglang_client = FakeSGLangClient([make_response("a")])
    client, _, _, _ = make_client(
        sglang_client=sglang_client,
        max_output_tokens=2048,
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert sglang_client.calls[0]["sampling_params"]["max_new_tokens"] == 2048


def test_chat_completion_omits_max_new_tokens_without_request_cli_or_context_limit():
    sglang_client = FakeSGLangClient([make_response("a")])
    client, _, _, _ = make_client(sglang_client=sglang_client)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert "max_new_tokens" not in sglang_client.calls[0]["sampling_params"]


def test_chat_completion_uses_full_remaining_context_without_request_or_cli_limit():
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "hi"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    sglang_client = FakeSGLangClient([make_response("a")])
    client, _, _, _ = make_client(
        tokenizer=tokenizer,
        sglang_client=sglang_client,
        context_window=len(prompt_ids) + 7,
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": messages},
    )

    assert response.status_code == 200
    assert sglang_client.calls[0]["sampling_params"]["max_new_tokens"] == 6


def test_chat_completion_rejects_when_only_reserved_context_token_remains():
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "hi"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    sglang_client = FakeSGLangClient([make_response("a")])
    client, _, _, _ = make_client(
        tokenizer=tokenizer,
        sglang_client=sglang_client,
        context_window=len(prompt_ids) + 1,
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": messages},
    )

    assert response.status_code == 413
    assert response.json()["error"] == "context_overflow"
    assert sglang_client.calls == []


def test_partial_rollout_uses_remaining_context_as_cumulative_output_limit():
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "hi"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    sglang_client = FakeSGLangClient(
        [
            make_response("a", finish_reason="abort"),
            make_response("b"),
        ]
    )
    client, _, _, _ = make_client(
        tokenizer=tokenizer,
        sglang_client=sglang_client,
        context_window=len(prompt_ids) + 3,
        partial_rollout=True,
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": messages},
    )

    assert response.status_code == 200
    assert [
        call["sampling_params"]["max_new_tokens"]
        for call in sglang_client.calls
    ] == [2, 1]


def test_chat_completion_can_disable_dynamic_max_tokens():
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "hi"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    sglang_client = FakeSGLangClient([make_response("a")])
    client, _, _, _ = make_client(
        tokenizer=tokenizer,
        sglang_client=sglang_client,
        context_window=len(prompt_ids) + 2,
        max_output_tokens=2,
        dynamic_max_tokens=False,
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": messages, "max_tokens": 3},
    )

    assert response.status_code == 200
    assert sglang_client.calls[0]["sampling_params"]["max_new_tokens"] == 2


def test_chat_completion_can_disable_context_derived_max_tokens():
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "hi"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    sglang_client = FakeSGLangClient([make_response("a")])
    client, _, _, _ = make_client(
        tokenizer=tokenizer,
        sglang_client=sglang_client,
        context_window=len(prompt_ids) + 2,
        dynamic_max_tokens=False,
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": messages},
    )

    assert response.status_code == 200
    assert "max_new_tokens" not in sglang_client.calls[0]["sampling_params"]


def test_chat_completion_context_window_input_output_overflow_records_input_only_step():
    messages = [{"role": "user", "content": "hi"}]
    tokenizer = FakeTokenizer()
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    sglang_client = FakeSGLangClient([make_response("abc")])
    client, session_manager, _, _ = make_client(
        tokenizer=tokenizer,
        sglang_client=sglang_client,
        context_window=len(prompt_ids) + 2,
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-output-overflow", "X-Instance-Id": "inst-1"},
        json={"model": "fake-model", "messages": messages, "max_tokens": 3},
    )

    body = response.json()
    session = session_manager.get_session("sess-output-overflow")
    assert response.status_code == 413
    assert body["error"] == "context_overflow"
    assert body["details"]["phase"] == "input_output"
    assert body["details"]["last_proxy_step_recorded"] is True
    assert body["details"]["input_tokens"] == len(prompt_ids)
    assert body["details"]["output_tokens"] == 3
    assert len(session.steps) == 1
    step = session.steps[0]
    assert step.prompt_token_ids == prompt_ids
    assert step.response_token_ids == []
    assert step.response_logprobs == []
    assert step.output_token_texts == []
    assert step.all_token_ids == prompt_ids
    assert step.raw_response_text == ""
    assert step.messages_snapshot[-1] == {"role": "assistant", "content": ""}
    assert step.finish_reason == "length"

    finalize = client.post(
        "/session/finalize",
        json={"session_id": "sess-output-overflow", "instance_id": "inst-1"},
    )
    trajectory = client.post(
        "/trajectory/read",
        json={
            "trajectory_id": "sess-output-overflow",
            "instance_id": "inst-1",
            "drain": True,
        },
    )
    assert finalize.status_code == 200
    assert trajectory.status_code == 200
    item = trajectory.json()["data"][0]
    assert item["finish_reason"] == "length"
    assert item["extra_info"]["output_token_count"] == 0
    assert item["aligned_response_length"] == 0
    assert item["tokens"] == prompt_ids
    assert sglang_client.parse_function_call_calls == []
    assert sglang_client.separate_reasoning_calls == []


def test_concat_context_window_output_overflow_records_only_context_delta():
    tokenizer = FakeTokenizer()
    first_messages = [{"role": "user", "content": "hi"}]
    second_messages = [
        *first_messages,
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "again"},
    ]
    second_prompt_ids = tokenizer.apply_chat_template(
        second_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    client, session_manager, _, _ = make_client(
        tokenizer=tokenizer,
        sglang_client=FakeSGLangClient(
            [make_response("ok"), make_response("abc")]
        ),
        token_build_mode="tito",
        tito_model="qwen3_5",
        context_window=len(second_prompt_ids) + 2,
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-concat-overflow", "X-Instance-Id": "inst"},
        json={"model": "fake-model", "messages": first_messages},
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-concat-overflow", "X-Instance-Id": "inst"},
        json={"model": "fake-model", "messages": second_messages},
    )
    assert second.status_code == 413
    assert second.json()["details"]["output_tokens"] == 3

    overflow_step = session_manager.get_session("sess-concat-overflow").latest_step
    assert overflow_step.response_token_ids == []
    assert overflow_step.concat_output_token_count == 0
    assert all(value == 0 for value in overflow_step.concat_response_mask)
    assert decode_tokens(overflow_step.concat_token_ids) == "<user>again<assistant>"

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-concat-overflow", "instance_id": "inst"},
    )
    assert finalized.status_code == 200
    item = client.post(
        "/trajectory/read",
        json={"trajectory_id": "sess-concat-overflow"},
    ).json()["data"][0]
    assert decode_tokens(item["tokens"]) == (
        "<user>hi<assistant>ok<user>again<assistant>"
    )
    assert_concat_mask_for_outputs(item, ["ok"])
    assert item["finish_reason"] == "length"
    assert item["extra_info"]["output_token_count"] == len("ok")


def test_parse_args_accepts_rollout_routing_replay(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "dressage.proxy.server",
            "--sglang-router-url",
            "http://router.test",
            "--tokenizer-path",
            "fake-tokenizer",
            "--use-rollout-routing-replay",
        ],
    )

    args = parse_args()

    assert args.use_rollout_routing_replay is True


def test_append_only_normalizes_tool_call_index_json_arguments_and_empty_content():
    manager = SessionManager()
    previous_messages = [
        {"role": "user", "content": "find"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_lookup",
                    "type": "function",
                    "index": 0,
                    "function": {
                        "name": "lookup",
                        "arguments": '{ "b": 2, "a": 1 }',
                    },
                },
                {
                    "id": "call_calc",
                    "type": "function",
                    "index": 1,
                    "function": {
                        "name": "calculate",
                        "arguments": {"z": [2, 1], "a": True},
                    },
                },
            ],
        },
    ]
    current_messages = [
        {"role": "user", "content": "find"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_lookup",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"a":1,"b":2}',
                    },
                },
                {
                    "id": "call_calc",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": '{"a":true,"z":[2,1]}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_lookup", "content": "result"},
    ]

    assert manager.is_append_only_continuation(previous_messages, current_messages)


def test_append_only_canonicalizes_openclaw_sanitized_tool_call_ids():
    manager = SessionManager()
    previous_messages = [
        {"role": "user", "content": "find"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_b1467966",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_b1467966", "content": "result"},
    ]
    current_messages = [
        {"role": "user", "content": "find"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "callb1467966",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "callb1467966", "content": "result"},
        {"role": "user", "content": "next"},
    ]

    assert manager.is_append_only_continuation(previous_messages, current_messages)


def test_append_only_normalizes_reasoning_content_boundary_whitespace():
    manager = SessionManager()
    previous_messages = [
        {"role": "user", "content": "find"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": " plan\n",
            "tool_calls": [
                {
                    "id": "call_lookup",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                }
            ],
        },
    ]
    current_messages = [
        {"role": "user", "content": "find"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "plan",
            "tool_calls": [
                {
                    "id": "call_lookup",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_lookup", "content": "result\n"},
    ]

    assert manager.is_append_only_continuation(previous_messages, current_messages)


def test_append_only_still_rejects_missing_nonempty_reasoning_content():
    manager = SessionManager()
    previous_messages = [
        {"role": "user", "content": "find"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "plan",
            "tool_calls": [
                {
                    "id": "call_lookup",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                }
            ],
        },
    ]
    current_messages = [
        {"role": "user", "content": "find"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_lookup",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                }
            ],
        },
    ]

    assert not manager.is_append_only_continuation(previous_messages, current_messages)


def test_append_only_rejects_semantic_tool_call_differences():
    manager = SessionManager()

    def messages(
        *,
        call_id: str = "call_lookup",
        call_type: str = "function",
        name: str = "lookup",
        arguments: Any = '{"a":1}',
    ) -> list[dict]:
        return [
            {"role": "user", "content": "find"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": call_type,
                        "index": 0,
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            },
        ]

    previous_messages = messages()
    for changed_messages in [
        messages(arguments='{"a":2}'),
        messages(call_id="call_other"),
        messages(call_type="custom"),
        messages(name="search"),
    ]:
        assert not manager.is_append_only_continuation(
            previous_messages, changed_messages
        )

    assert not manager.is_append_only_continuation(
        messages(arguments='{"a":1'),
        messages(arguments='{"a":1 '),
    )


def test_backward_compatible_type_aliases_are_exported():
    assert TurnRecord is StepRecord
    assert TrajectoryItem is TrajectorySegment


def test_sglang_router_client_generate_sends_logprob_start_len():
    observed_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/generate"
        payload = json.loads(request.content.decode())
        observed_payloads.append(payload)
        meta_info = {
            "output_token_logprobs": [[-0.3, 99, "c"]],
            "finish_reason": {"type": "stop"},
        }
        if payload["logprob_start_len"] == 0:
            meta_info["input_token_logprobs"] = [
                [-0.1, 97, "a"],
                [-0.2, 98, "b"],
            ]
        return httpx.Response(
            200,
            json={
                "text": "c",
                "output_ids": [99],
                "meta_info": meta_info,
            },
        )

    async def run_test() -> tuple[SGLangResponse, SGLangResponse]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            trust_env=False,
        ) as client:
            router = SGLangRouterClient("http://router.test", client=client)
            full = await router.generate([97, 98], {"max_new_tokens": 1})
            output_only = await router.generate(
                [97, 98],
                {"max_new_tokens": 1},
                logprob_start_len=-1,
            )
            return full, output_only

    full, output_only = asyncio.run(run_test())

    assert observed_payloads[0]["logprob_start_len"] == 0
    assert observed_payloads[1]["logprob_start_len"] == -1
    assert full.input_token_logprobs_raw == [-0.1, -0.2]
    assert full.input_logprobs_invalid is False
    assert output_only.input_token_logprobs_raw == [0.0, 0.0]
    assert output_only.input_logprobs_invalid is False
    assert output_only.all_logprobs_invalid is False


def test_sglang_router_client_get_weight_versions_probes_stable_topology():
    worker_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/workers":
            return httpx.Response(
                200,
                json={
                    "workers": [
                        {
                            "url": "http://worker-a.test:30000",
                            "is_healthy": True,
                            "connection_mode": "Http",
                        },
                        {
                            "url": "http://worker-b.test:30001",
                            "is_healthy": True,
                            "connection_mode": "Http",
                        },
                    ]
                },
            )
        if request.url.path == "/get_weight_version":
            worker_calls.append(str(request.url))
            return httpx.Response(200, json={"weight_version": "v7"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def run_test():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ) as client:
            router = SGLangRouterClient("http://router.test", client=client)
            return await router.get_weight_versions(timeout_seconds=1.0)

    versions = asyncio.run(run_test())

    assert versions == {
        "http://worker-a.test:30000": "v7",
        "http://worker-b.test:30001": "v7",
    }
    assert len(worker_calls) == 2


def test_sglang_response_normalizes_prompt_logprobs_and_all_tokens():
    response = SGLangRouterClient._coerce_response(
        {
            "text": "c",
            "output_ids": [99],
            "meta_info": {
                "input_token_logprobs": [
                    [None, 97, "a"],
                    [-0.2, 98, "b"],
                ],
                "output_token_logprobs": [[-0.3, 99, "c"]],
                "finish_reason": {"type": "stop"},
            },
        },
        input_ids=[97, 98],
    )
    assert response.input_token_ids == [97, 98]
    assert response.input_token_logprobs_raw == [0.0, -0.2]
    assert response.output_ids == [99]
    assert response.output_token_logprobs == [-0.3]
    assert response.input_token_texts == ["a", "b"]
    assert response.output_token_texts == ["c"]
    assert response.all_token_ids == [97, 98, 99]
    assert response.all_logprobs == [0.0, -0.2, -0.3]
    assert response.all_logprobs_invalid is False


def test_sglang_response_output_only_missing_prompt_logprobs_is_valid():
    response = SGLangRouterClient._coerce_response(
        {
            "text": "c",
            "output_ids": [99],
            "meta_info": {
                "output_token_logprobs": [[-0.3, 99, "c"]],
                "finish_reason": {"type": "stop"},
            },
        },
        input_ids=[97, 98],
        expect_input_logprobs=False,
    )

    assert response.input_token_logprobs_raw == [0.0, 0.0]
    assert response.all_logprobs == [0.0, 0.0, -0.3]
    assert response.input_logprobs_invalid is False
    assert response.all_logprobs_invalid is False


def test_sglang_response_full_mode_missing_prompt_logprobs_is_invalid():
    response = SGLangRouterClient._coerce_response(
        {
            "text": "c",
            "output_ids": [99],
            "meta_info": {
                "output_token_logprobs": [[-0.3, 99, "c"]],
                "finish_reason": {"type": "stop"},
            },
        },
        input_ids=[97, 98],
    )

    assert response.input_token_logprobs_raw == [0.0, 0.0]
    assert response.input_logprobs_invalid is True
    assert response.all_logprobs_invalid is True


def test_sglang_response_marks_length_mismatch_invalid():
    response = SGLangRouterClient._coerce_response(
        {
            "text": "bc",
            "output_ids": [98, 99],
            "meta_info": {
                "input_token_logprobs": [[-0.1, 97, "a"]],
                "output_token_logprobs": [[-0.2, 98, "b"]],
                "finish_reason": {"type": "stop"},
            },
        },
        input_ids=[97, 100],
    )
    assert response.input_token_logprobs_raw == [-0.1, 0.0]
    assert response.output_token_logprobs == [-0.2, 0.0]
    assert response.all_token_ids == [97, 100, 98, 99]
    assert response.all_logprobs == [-0.1, 0.0, -0.2, 0.0]
    assert response.all_logprobs_invalid is True


def test_sglang_router_client_parse_function_call_uses_first_healthy_http_worker_and_normalizes_url():
    observed_requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode()) if request.content else None
        observed_requests.append(
            {
                "method": request.method,
                "url": str(request.url),
                "payload": payload,
                "headers": dict(request.headers),
            }
        )
        if request.url.path == "/workers":
            return httpx.Response(
                200,
                json={
                    "workers": [
                        {
                            "url": "http://0.0.0.0:30000",
                            "is_healthy": True,
                            "connection_mode": "Http",
                        },
                        {
                            "url": "http://0.0.0.0:30001",
                            "is_healthy": True,
                            "connection_mode": "Http",
                        },
                    ]
                },
            )
        if request.url.path == "/parse_function_call":
            assert request.url.host == "127.0.0.1"
            assert request.url.port == 30000
            return httpx.Response(
                200,
                json={
                    "normal_text": "Thinking Process: parsed by worker",
                    "calls": [
                        {
                            "name": "get_weather_snapshot",
                            "parameters": {"city": "Shanghai"},
                        }
                    ],
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def run_test():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            trust_env=False,
        ) as client:
            router = SGLangRouterClient("http://0.0.0.0:8000", client=client)
            return await router.parse_function_call(
                "raw",
                tool_call_parser="qwen3_coder",
                tools=make_tools("get_weather_snapshot"),
                routing_key="sess-router",
            )

    parsed = asyncio.run(run_test())

    assert parsed == {
        "normal_text": "Thinking Process: parsed by worker",
        "calls": [{"name": "get_weather_snapshot", "parameters": {"city": "Shanghai"}}],
    }
    parse_requests = [
        request for request in observed_requests if request["url"].endswith("/parse_function_call")
    ]
    assert len(parse_requests) == 1
    assert parse_requests[0]["url"] == "http://127.0.0.1:30000/parse_function_call"
    assert parse_requests[0]["payload"]["tool_call_parser"] == "qwen3_coder"
    assert parse_requests[0]["payload"]["tools"] == make_tools("get_weather_snapshot")
    assert parse_requests[0]["headers"]["x-smg-routing-key"] == "sess-router"


def test_sglang_router_client_parse_function_call_retries_next_worker_after_failures():
    attempted_ports: list[int | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/workers":
            return httpx.Response(
                200,
                json={
                    "workers": [
                        {
                            "url": "http://127.0.0.1:30000",
                            "is_healthy": True,
                            "connection_mode": "Http",
                        },
                        {
                            "url": "http://127.0.0.1:30001",
                            "is_healthy": True,
                            "connection_mode": "Http",
                        },
                    ]
                },
            )
        if request.url.path == "/parse_function_call":
            attempted_ports.append(request.url.port)
            if request.url.port == 30000:
                return httpx.Response(500, json={"error": "first worker failed"})
            return httpx.Response(
                200,
                json={
                    "normal_text": "parsed after retry",
                    "calls": [{"name": "lookup", "parameters": {"q": "ok"}}],
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def run_test():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            trust_env=False,
        ) as client:
            router = SGLangRouterClient("http://router.test", client=client)
            return await router.parse_function_call(
                "raw",
                tool_call_parser="qwen3_coder",
                tools=make_tools("lookup"),
            )

    parsed = asyncio.run(run_test())

    assert parsed == {
        "normal_text": "parsed after retry",
        "calls": [{"name": "lookup", "parameters": {"q": "ok"}}],
    }
    assert attempted_ports == [30000, 30001]


def test_sglang_router_client_separate_reasoning_uses_first_healthy_http_worker():
    observed_requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode()) if request.content else None
        observed_requests.append(
            {
                "method": request.method,
                "url": str(request.url),
                "payload": payload,
                "headers": dict(request.headers),
            }
        )
        if request.url.path == "/workers":
            return httpx.Response(
                200,
                json={
                    "workers": [
                        {
                            "url": "http://0.0.0.0:30000",
                            "is_healthy": True,
                            "connection_mode": "Http",
                        }
                    ]
                },
            )
        if request.url.path == "/separate_reasoning":
            assert request.url.host == "127.0.0.1"
            return httpx.Response(
                200,
                json={"reasoning_text": "plan", "text": "answer"},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def run_test():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            trust_env=False,
        ) as client:
            router = SGLangRouterClient("http://0.0.0.0:8000", client=client)
            return await router.separate_reasoning(
                "raw",
                reasoning_parser="deepseek-r1",
                routing_key="sess-reasoning",
            )

    parsed = asyncio.run(run_test())

    assert parsed == {"reasoning_text": "plan", "text": "answer"}
    reasoning_requests = [
        request
        for request in observed_requests
        if request["url"].endswith("/separate_reasoning")
    ]
    assert len(reasoning_requests) == 1
    assert reasoning_requests[0]["payload"] == {
        "text": "raw",
        "reasoning_parser": "deepseek-r1",
    }
    assert reasoning_requests[0]["headers"]["x-smg-routing-key"] == "sess-reasoning"


def test_mask_template_file_loads_by_default():
    registry = create_default_mask_template_registry()
    assert registry.resolve("qwen3_5") is not None
    builder = PromptAssistantMaskBuilder(FakeTokenizer(), model_mask_type="qwen3_5")
    assert builder._mask_template_path is not None
    assert builder._mask_template_path.name == "qwen3_5_mask_only_chat_template.jinja"
    assert builder._mask_chat_template.strip()


def test_qwen_mask_template_keeps_assistant_tool_call_xml_unescaped():
    registry = create_default_mask_template_registry()
    template_path = registry.resolve("qwen3_5")
    assert template_path is not None
    template = template_path.read_text(encoding="utf-8")
    template = template.replace("{% generation %}", "").replace(
        "{% endgeneration %}", ""
    )

    def raise_exception(message):
        raise RuntimeError(message)

    env = Environment(undefined=StrictUndefined)
    rendered = env.from_string(template).render(
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "check weather"},
            {
                "role": "assistant",
                "content": "Thinking Process: use weather tool",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather_snapshot",
                            "arguments": {
                                "city": "Shanghai",
                                "hours": ["09:00", "21:00"],
                            },
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-weather",
                "content": '{"ok": true}',
            },
            {"role": "user", "content": "summarize"},
        ],
        tools=make_tools("get_weather_snapshot"),
        add_generation_prompt=True,
        add_vision_id=False,
        enable_thinking=True,
        raise_exception=raise_exception,
    )

    assert "<tool_call>\n<function=get_weather_snapshot>" in rendered
    assert "<parameter=city>\nShanghai\n</parameter>" in rendered
    assert "<parameter=hours>\n" in rendered
    assert "</function>\n</tool_call>" in rendered
    assert "&lt;tool_call" not in rendered
    assert "&lt;function=" not in rendered
    assert "&lt;parameter=" not in rendered


def test_qwen_preserve_thinking_keeps_rendered_prefix_after_user_append():
    tokenizer = QwenJinjaTokenizer()
    builder = PromptAssistantMaskBuilder(tokenizer, model_mask_type=None)
    tools = make_tools("send_message")
    old_messages = [
        {"role": "user", "content": "root task"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Need to report the subagent result.",
            "tool_calls": [
                {
                    "id": "callabcdef12",
                    "type": "function",
                    "function": {
                        "name": "send_message",
                        "arguments": {
                            "target": "/root",
                            "message": "Answer: 5",
                        },
                    },
                }
            ],
        },
    ]
    new_messages = old_messages + [
        {"role": "tool", "tool_call_id": "callabcdef12", "content": ""},
        {
            "role": "user",
            "content": '{"type":"agent_message","author":"/root","recipient":"/root"}',
        },
    ]

    default_old = builder.render_messages(
        builder.normalize_template_messages(old_messages),
        tools,
        add_generation_prompt=False,
    )
    default_new = builder.render_messages(
        builder.normalize_template_messages(new_messages),
        tools,
        add_generation_prompt=True,
    )
    assert not default_new.startswith(default_old)

    template_kwargs = {"preserve_thinking": True}
    preserved_old = builder.render_messages(
        builder.normalize_template_messages(old_messages),
        tools,
        add_generation_prompt=False,
        template_kwargs=template_kwargs,
    )
    preserved_new = builder.render_messages(
        builder.normalize_template_messages(new_messages),
        tools,
        add_generation_prompt=True,
        template_kwargs=template_kwargs,
    )
    assert preserved_new.startswith(preserved_old)


def test_mask_template_file_missing_fails_fast():
    missing_path = Path("/tmp/does-not-exist-qwen3_5-mask-only-template.jinja")
    registry = ModelMaskTemplateRegistry()
    registry.register("qwen3_5", missing_path)
    try:
        PromptAssistantMaskBuilder(
            FakeTokenizer(),
            model_mask_type="qwen3_5",
            registry=registry,
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected missing mask template file to fail fast")


def test_unregistered_model_mask_type_disables_prompt_mask():
    builder = PromptAssistantMaskBuilder(FakeTokenizer(), model_mask_type="unknown")
    assert builder._mask_template_path is None
    assert builder._mask_chat_template is None


def test_unregistered_model_mask_type_falls_back_to_output_only():
    client, session_manager, _, _ = make_client(
        make_response("hello"),
        make_response("follow"),
        model_mask_type="unknown",
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-SMG-Routing-Key": "sess-unregistered", "X-Instance-Id": "inst-1"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-unregistered", "X-Instance-Id": "inst-1"},
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-unregistered").full_messages
            + [{"role": "user", "content": "again"}],
        },
    )
    assert second.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-unregistered", "instance_id": "inst-1"},
    )
    assert finalized.status_code == 200

    item = client.post("/trajectory/read", json={"trajectory_id": "sess-unregistered"}).json()[
        "data"
    ][0]
    assert_alignment_metadata(
        item,
        mask_template_equivalent=False,
        mask_fallback_reason="mask_template_not_registered_for_model_mask_type",
    )
    assert item["extra_info"]["output_token_count"] == len("hello")
    assert_dense_alignment(item, ["hello"])


def test_default_tool_call_parser_registry_resolves_hermes_and_qwen():
    registry = create_default_tool_call_parser_registry()
    hermes = registry.resolve("hermes")
    qwen = registry.resolve("qwen3_5")

    assert hermes is not None
    assert hermes.local_parser is parse_hermes_tool_calls
    assert hermes.sglang_tool_call_parser_name is None

    assert qwen is not None
    assert qwen.local_parser is parse_qwen3_5_tool_calls
    assert qwen.sglang_tool_call_parser_name == "qwen3_coder"


def test_custom_tool_call_parser_registry_binds_one_spec_per_type():
    registry = ModelToolCallParserRegistry()
    registry.register(
        "custom",
        ToolCallParserSpec(
            local_parser=parse_qwen3_5_tool_calls,
            sglang_tool_call_parser_name="custom_parser",
        ),
    )

    resolved = registry.resolve("custom")

    assert resolved is not None
    assert resolved.local_parser is parse_qwen3_5_tool_calls
    assert resolved.sglang_tool_call_parser_name == "custom_parser"


def test_parse_hermes_tool_calls_generates_strict_safe_ids():
    content, tool_calls = parse_hermes_tool_calls(
        'before <tool_call>{"name":"lookup","arguments":{"q":"x"}}</tool_call>'
    )

    assert content == "before"
    assert tool_calls is not None
    assert _STRICT_TOOL_CALL_ID_RE.fullmatch(tool_calls[0]["id"])


def test_parse_qwen3_reasoning_extracts_thinking_with_and_without_open_tag():
    parsed = parse_qwen3_reasoning("<think>plan</think>\n\nanswer")
    assert parsed.reasoning_content == "plan"
    assert parsed.text == "answer"

    parsed_with_boundary_whitespace = parse_qwen3_reasoning(
        "<think>\n plan\n</think>\n\nanswer"
    )
    assert parsed_with_boundary_whitespace.reasoning_content == "plan"
    assert parsed_with_boundary_whitespace.text == "answer"

    parsed_without_open = parse_qwen3_reasoning("plan</think>\n\nanswer")
    assert parsed_without_open.reasoning_content == "plan"
    assert parsed_without_open.text == "answer"

    blank_reasoning = parse_qwen3_reasoning("<think>\n\t\n</think>\n\nanswer")
    assert blank_reasoning.reasoning_content is None
    assert blank_reasoning.text == "answer"

    no_reasoning = parse_qwen3_reasoning("plain answer")
    assert no_reasoning.reasoning_content is None
    assert no_reasoning.text == "plain answer"


def test_canonicalize_reasoning_content_strips_boundary_whitespace():
    assert canonicalize_reasoning_content(" plan\n") == "plan"
    assert canonicalize_reasoning_content("\n\t\n") is None
    assert canonicalize_reasoning_content(None) is None


def test_proxy_reasoning_parser_sglang_api_allows_arbitrary_parser_name():
    sglang_client = FakeSGLangClient(
        [],
        separate_reasoning_responses=[
            {"reasoning_text": " api plan\n", "text": "api answer"}
        ],
    )
    parser = ProxyReasoningParser(
        sglang_client,
        model_reasoning_type="gpt-oss",
        backend="sglang_api",
    )

    parsed = asyncio.run(parser.parse("raw text", routing_key="sess-reasoning-api"))

    assert parsed.reasoning_content == "api plan"
    assert parsed.text == "api answer"
    assert sglang_client.separate_reasoning_calls[0]["reasoning_parser"] == "gpt-oss"
    assert sglang_client.separate_reasoning_calls[0]["routing_key"] == "sess-reasoning-api"


def test_proxy_reasoning_parser_sglang_api_blank_reasoning_is_none():
    sglang_client = FakeSGLangClient(
        [],
        separate_reasoning_responses=[{"reasoning_text": "\n\t\n", "text": "answer"}],
    )
    parser = ProxyReasoningParser(
        sglang_client,
        model_reasoning_type="qwen3",
        backend="sglang_api",
    )

    parsed = asyncio.run(parser.parse("raw", routing_key="sess-blank-reasoning"))

    assert parsed.reasoning_content is None
    assert parsed.text == "answer"


def test_proxy_reasoning_parser_repairs_qwen_completion_only_thinking():
    raw_text = "api plan</think>\n\napi answer"
    sglang_client = FakeSGLangClient(
        [],
        separate_reasoning_responses=[
            {"reasoning_text": "api plan", "text": "api answer"}
        ],
    )
    parser = ProxyReasoningParser(
        sglang_client,
        model_reasoning_type="qwen3",
        backend="sglang_api",
    )

    parsed = asyncio.run(parser.parse(raw_text, routing_key="sess-qwen-repair"))

    assert parsed.reasoning_content == "api plan"
    assert parsed.text == "api answer"
    assert sglang_client.separate_reasoning_calls[0]["text"] == "<think>" + raw_text


def test_proxy_reasoning_parser_does_not_double_repair_full_qwen_thinking():
    raw_text = "<think>api plan</think>\n\napi answer"
    sglang_client = FakeSGLangClient(
        [],
        separate_reasoning_responses=[
            {"reasoning_text": "api plan", "text": "api answer"}
        ],
    )
    parser = ProxyReasoningParser(
        sglang_client,
        model_reasoning_type="qwen3_5",
        backend="sglang_api",
    )

    parsed = asyncio.run(parser.parse(raw_text, routing_key="sess-qwen-full"))

    assert parsed.reasoning_content == "api plan"
    assert parsed.text == "api answer"
    assert sglang_client.separate_reasoning_calls[0]["text"] == raw_text


def test_proxy_reasoning_parser_hybrid_falls_back_to_local_qwen3():
    sglang_client = FakeSGLangClient(
        [],
        separate_reasoning_responses=[httpx.HTTPError("boom")],
    )
    parser = ProxyReasoningParser(
        sglang_client,
        model_reasoning_type="qwen3",
        backend="hybrid",
    )

    parsed = asyncio.run(
        parser.parse("<think>local plan</think>\n\nlocal answer", routing_key="sess")
    )

    assert parsed.reasoning_content == "local plan"
    assert parsed.text == "local answer"


def test_proxy_reasoning_parser_local_rejects_unsupported_parser():
    parser = ProxyReasoningParser(
        FakeSGLangClient([]),
        model_reasoning_type="deepseek-r1",
        backend="local",
    )

    try:
        asyncio.run(parser.parse("raw", routing_key="sess"))
    except ValueError as exc:
        assert "local reasoning parser only supports" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_parse_qwen3_5_tool_calls_extracts_multiple_functions_and_mixed_parameters():
    raw_text = make_qwen_tool_call_text(
        prefix="<think>plan</think>\n\nThinking Process: inspect tools",
        functions=[
            (
                "get_weather_snapshot",
                {
                    "city": "Shanghai",
                    "hours": ["09:00", "15:00"],
                },
            ),
            (
                "annotate_trip",
                {
                    "payload": {"umbrella": True},
                    "note": "bring light coat",
                },
            ),
        ],
    )

    content, tool_calls = parse_qwen3_5_tool_calls(raw_text)

    assert content is not None
    assert "<tool_call>" not in content
    assert "Thinking Process: inspect tools" in content
    assert tool_calls is not None
    assert all(
        _STRICT_TOOL_CALL_ID_RE.fullmatch(tool_call["id"])
        for tool_call in tool_calls
    )
    assert [tool_call["function"]["name"] for tool_call in tool_calls] == [
        "get_weather_snapshot",
        "annotate_trip",
    ]
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "city": "Shanghai",
        "hours": ["09:00", "15:00"],
    }
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {
        "payload": {"umbrella": True},
        "note": "bring light coat",
    }


def test_parse_qwen3_5_tool_calls_keeps_malformed_blocks_in_content():
    raw_text = (
        "Thinking Process: malformed\n"
        "<tool_call>\n"
        "<function=broken>\n"
        "<parameter=city>\nShanghai\n</parameter>\n"
        "</tool_call>"
    )

    content, tool_calls = parse_qwen3_5_tool_calls(raw_text)

    assert content == raw_text
    assert tool_calls is None


def test_proxy_tool_call_parser_local_backend_only_uses_local_parser():
    raw_text = make_qwen_tool_call_text(
        prefix="Thinking Process: local parser",
        functions=[("get_weather_snapshot", {"city": "Shanghai"})],
    )
    sglang_client = FakeSGLangClient(
        [],
        parse_function_call_responses=[
            {"normal_text": "api text", "calls": [{"name": "ignored", "parameters": {}}]}
        ],
    )
    parser = ProxyToolCallParser(
        sglang_client,
        model_tool_call_type="qwen3_5",
        backend="local",
    )

    content, tool_calls = asyncio.run(
        parser.parse(
            raw_text,
            make_tools("get_weather_snapshot"),
            routing_key="sess-local",
        )
    )

    assert content == "Thinking Process: local parser"
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "get_weather_snapshot"
    assert sglang_client.parse_function_call_calls == []


def test_proxy_tool_call_parser_sglang_api_backend_returns_raw_text_when_api_has_no_calls():
    raw_text = make_qwen_tool_call_text(
        prefix="Thinking Process: api only",
        functions=[("get_weather_snapshot", {"city": "Shanghai"})],
    )
    sglang_client = FakeSGLangClient(
        [],
        parse_function_call_responses=[{"normal_text": "cleaned", "calls": []}],
    )
    parser = ProxyToolCallParser(
        sglang_client,
        model_tool_call_type="qwen3_5",
        backend="sglang_api",
    )

    content, tool_calls = asyncio.run(
        parser.parse(
            raw_text,
            make_tools("get_weather_snapshot"),
            routing_key="sess-api",
        )
    )

    assert content == raw_text
    assert tool_calls is None
    assert sglang_client.parse_function_call_calls[0]["tool_call_parser"] == "qwen3_coder"
    assert sglang_client.parse_function_call_calls[0]["routing_key"] == "sess-api"


def test_proxy_tool_call_parser_hybrid_falls_back_to_local_on_api_failures():
    raw_text = make_qwen_tool_call_text(
        prefix="Thinking Process: hybrid fallback",
        functions=[("get_weather_snapshot", {"city": "Shanghai"})],
    )
    tools = make_tools("get_weather_snapshot")

    for api_response in (
        httpx.HTTPError("boom"),
        {"malformed": True},
        {"normal_text": "cleaned", "calls": []},
    ):
        sglang_client = FakeSGLangClient(
            [],
            parse_function_call_responses=[api_response],
        )
        parser = ProxyToolCallParser(
            sglang_client,
            model_tool_call_type="qwen3_5",
            backend="hybrid",
        )

        content, tool_calls = asyncio.run(
            parser.parse(raw_text, tools, routing_key="sess-hybrid")
        )

        assert content == "Thinking Process: hybrid fallback"
        assert tool_calls is not None
        assert tool_calls[0]["function"]["name"] == "get_weather_snapshot"


def test_proxy_tool_call_parser_unregistered_type_falls_back_to_raw_text():
    parser = ProxyToolCallParser(
        FakeSGLangClient([]),
        model_tool_call_type="unknown",
        backend="hybrid",
    )

    content, tool_calls = asyncio.run(
        parser.parse(
            "plain raw text",
            make_tools("lookup"),
            routing_key="sess-unknown",
        )
    )

    assert content == "plain raw text"
    assert tool_calls is None


def test_proxy_tool_call_parser_sglang_api_normalizes_parameter_shapes():
    sglang_client = FakeSGLangClient(
        [],
        parse_function_call_responses=[
            {
                "normal_text": "parsed by api",
                "calls": [
                    {"name": "first", "parameters": {"city": "Shanghai"}},
                    {"name": "second", "parameters": ["09:00", "15:00"]},
                    {"name": "third", "parameters": "raw=1"},
                ],
            }
        ],
    )
    parser = ProxyToolCallParser(
        sglang_client,
        model_tool_call_type="qwen3_5",
        backend="sglang_api",
    )

    content, tool_calls = asyncio.run(
        parser.parse(
            "raw",
            make_tools("first", "second", "third"),
            routing_key="sess-shapes",
        )
    )

    assert content == "parsed by api"
    assert tool_calls is not None
    assert all(
        _STRICT_TOOL_CALL_ID_RE.fullmatch(tool_call["id"])
        for tool_call in tool_calls
    )
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Shanghai"}
    assert json.loads(tool_calls[1]["function"]["arguments"]) == ["09:00", "15:00"]
    assert tool_calls[2]["function"]["arguments"] == "raw=1"


def test_proxy_tool_call_parser_sglang_api_skips_parse_without_tools():
    sglang_client = FakeSGLangClient(
        [],
        parse_function_call_responses=[
            {"normal_text": "unused", "calls": [{"name": "ignored", "parameters": {}}]}
        ],
    )
    parser = ProxyToolCallParser(
        sglang_client,
        model_tool_call_type="qwen3_5",
        backend="sglang_api",
    )

    no_tools_content, no_tools_calls = asyncio.run(
        parser.parse("raw-none", None, routing_key="sess-no-tools")
    )
    empty_tools_content, empty_tools_calls = asyncio.run(
        parser.parse("raw-empty", [], routing_key="sess-no-tools")
    )

    assert no_tools_content == "raw-none"
    assert no_tools_calls is None
    assert empty_tools_content == "raw-empty"
    assert empty_tools_calls is None
    assert sglang_client.parse_function_call_calls == []


def test_proxy_tool_call_parser_sglang_api_returns_raw_text_when_worker_discovery_unavailable():
    raw_text = make_qwen_tool_call_text(
        prefix="Thinking Process: api fallback raw",
        functions=[("get_weather_snapshot", {"city": "Shanghai"})],
    )
    tools = make_tools("get_weather_snapshot")

    scenarios = [
        httpx.HTTPStatusError(
            "workers 404",
            request=httpx.Request("GET", "http://router.test/workers"),
            response=httpx.Response(404),
        ),
        ValueError("malformed workers payload"),
        None,
    ]

    for list_workers_error in scenarios:
        workers = None
        if list_workers_error is None:
            workers = [
                {
                    "url": "http://127.0.0.1:30000",
                    "is_healthy": False,
                    "connection_mode": "Http",
                }
            ]
        sglang_client = FakeSGLangClient(
            [],
            workers=workers,
            list_workers_error=list_workers_error,
        )
        parser = ProxyToolCallParser(
            sglang_client,
            model_tool_call_type="qwen3_5",
            backend="sglang_api",
        )

        content, tool_calls = asyncio.run(
            parser.parse(raw_text, tools, routing_key="sess-workers")
        )

        assert content == raw_text
        assert tool_calls is None


def test_proxy_tool_call_parser_hybrid_falls_back_to_local_when_worker_discovery_unavailable():
    raw_text = make_qwen_tool_call_text(
        prefix="Thinking Process: hybrid worker fallback",
        functions=[("get_weather_snapshot", {"city": "Shanghai"})],
    )
    tools = make_tools("get_weather_snapshot")

    scenarios = [
        httpx.HTTPStatusError(
            "workers 404",
            request=httpx.Request("GET", "http://router.test/workers"),
            response=httpx.Response(404),
        ),
        ValueError("malformed workers payload"),
        None,
    ]

    for list_workers_error in scenarios:
        workers = None
        if list_workers_error is None:
            workers = [
                {
                    "url": "http://127.0.0.1:30000",
                    "is_healthy": False,
                    "connection_mode": "Http",
                }
            ]
        sglang_client = FakeSGLangClient(
            [],
            workers=workers,
            list_workers_error=list_workers_error,
        )
        parser = ProxyToolCallParser(
            sglang_client,
            model_tool_call_type="qwen3_5",
            backend="hybrid",
        )

        content, tool_calls = asyncio.run(
            parser.parse(raw_text, tools, routing_key="sess-workers")
        )

        assert content == "Thinking Process: hybrid worker fallback"
        assert tool_calls is not None
        assert tool_calls[0]["function"]["name"] == "get_weather_snapshot"


def test_qwen_hybrid_chat_completion_preserves_content_and_tool_calls_without_false_rewrite():
    tool_text = make_qwen_tool_call_text(
        prefix="Thinking Process: use weather tool",
        functions=[
            (
                "get_weather_snapshot",
                {"city": "Shanghai", "hours": ["09:00", "21:00"]},
            )
        ],
    )
    tools = make_tools("get_weather_snapshot")
    client, session_manager, _, sglang_client = make_client(
        make_response(tool_text),
        make_response("Bring a light waterproof jacket."),
        model_tool_call_type="qwen3_5",
        tool_call_parse_backend="hybrid",
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-qwen", "X-Instance-Id": "inst-qwen"},
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "check weather"}],
            "tools": tools,
        },
    )
    assert first.status_code == 200
    completion = first.json()
    choice = completion["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "Thinking Process: use weather tool"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather_snapshot"
    assert sglang_client.parse_function_call_calls[0]["tool_call_parser"] == "qwen3_coder"

    assistant_message = session_manager.get_session("sess-qwen").full_messages[-1]
    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-qwen", "X-Instance-Id": "inst-qwen"},
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-qwen").full_messages
            + [
                {
                    "role": "tool",
                    "tool_call_id": assistant_message["tool_calls"][0]["id"],
                    "content": '{"ok": true}',
                }
            ],
            "tools": tools,
        },
    )
    assert second.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-qwen", "instance_id": "inst-qwen"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["history_rewritten"] is False
    assert finalized.json()["num_turns"] == 1
    assert finalized.json()["num_segments"] == 2


def test_qwen_streaming_emits_content_before_tool_calls():
    raw_text = make_qwen_tool_call_text(
        prefix="Thinking Process: stream tool",
        functions=[("get_weather_snapshot", {"city": "Shanghai"})],
    )
    client, _, _, _ = make_client(
        make_response(raw_text),
        model_tool_call_type="qwen3_5",
        tool_call_parse_backend="hybrid",
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-qwen-stream", "X-Instance-Id": "inst-qwen-stream"},
        json={
            "model": "fake-model",
            "stream": True,
            "messages": [{"role": "user", "content": "check weather"}],
            "tools": make_tools("get_weather_snapshot"),
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"content": "Thinking' in body or '"content":"Thinking' in body
    assert '"tool_calls"' in body
    assert body.index("content") < body.index("tool_calls")
    assert_stream_usage_chunk(body)


def test_streaming_emits_usage_chunk_before_done():
    raw_text = "hello usage"
    client, _, _, sglang_client = make_client(make_response(raw_text))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-stream-usage",
            "X-Instance-Id": "inst-stream-usage",
        },
        json={
            "model": "fake-model",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    events = sse_json_events(body)
    usage = assert_stream_usage_chunk(body)
    usage_event = [event for event in events if event.get("usage") is not None][0]
    assert usage_event["model"] == "fake-model"
    assert usage["prompt_tokens"] == len(sglang_client.calls[0]["input_ids"])
    assert usage["completion_tokens"] == len(raw_text)


def test_streaming_include_usage_true_and_bad_options_default_to_usage():
    cases = [
        ("explicit-true", {"include_usage": True}),
        ("null-options", None),
        ("bad-options", "bad"),
    ]
    client, _, _, _ = make_client(
        make_response("explicit usage"),
        make_response("null usage"),
        make_response("bad usage"),
    )

    for index, (case_name, stream_options) in enumerate(cases):
        body_json = {
            "model": "fake-model",
            "stream": True,
            "messages": [{"role": "user", "content": case_name}],
        }
        body_json["stream_options"] = stream_options
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={
                "X-Session-Id": f"sess-stream-usage-{index}",
                "X-Instance-Id": "inst-stream-usage-options",
            },
            json=body_json,
        ) as response:
            body = "".join(response.iter_text())

        assert response.status_code == 200
        assert_stream_usage_chunk(body)


def test_streaming_include_usage_false_omits_usage_fields():
    client, _, _, _ = make_client(make_response("no usage"))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-stream-no-usage",
            "X-Instance-Id": "inst-stream-no-usage",
        },
        json={
            "model": "fake-model",
            "stream": True,
            "stream_options": {"include_usage": False},
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert_stream_has_no_usage(body)


def test_completion_usage_falls_back_to_tokenizer_encode_without_changing_recorded_ids(
    caplog,
):
    raw_text = "fallback text"
    tokenizer = EncodeFallbackTokenizer([101, 102, 103])
    client, session_manager, _, _ = make_client(
        make_response(raw_text, output_ids=[]),
        make_response(raw_text, output_ids=[]),
        tokenizer=tokenizer,
    )
    caplog.set_level(logging.WARNING, logger="dressage.proxy.server")

    non_stream = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-usage-fallback-nonstream",
            "X-Instance-Id": "inst-usage-fallback",
        },
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert non_stream.status_code == 200
    assert non_stream.json()["usage"]["completion_tokens"] == 3

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-usage-fallback-stream",
            "X-Instance-Id": "inst-usage-fallback",
        },
        json={
            "model": "fake-model",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert assert_stream_usage_chunk(body)["completion_tokens"] == 3
    assert tokenizer.encode_calls == [
        {"text": raw_text, "add_special_tokens": False},
        {"text": raw_text, "add_special_tokens": False},
    ]
    assert (
        session_manager.get_session("sess-usage-fallback-nonstream")
        .latest_step
        .response_token_ids
        == []
    )
    assert (
        session_manager.get_session("sess-usage-fallback-stream")
        .latest_step
        .response_token_ids
        == []
    )
    assert "estimated completion_tokens for public usage" in caplog.text


def test_completion_usage_falls_back_to_callable_tokenizer():
    raw_text = "callable fallback"
    tokenizer = CallableFallbackTokenizer([201, 202, 203, 204])
    client, session_manager, _, _ = make_client(
        make_response(raw_text, output_ids=[]),
        tokenizer=tokenizer,
    )

    response = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-usage-fallback-callable",
            "X-Instance-Id": "inst-usage-fallback",
        },
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] == 4
    assert tokenizer.calls == [
        {"text": raw_text, "add_special_tokens": False},
    ]
    assert (
        session_manager.get_session("sess-usage-fallback-callable")
        .latest_step
        .response_token_ids
        == []
    )


def test_qwen_sglang_api_chat_completion_uses_parsed_calls_and_normal_text():
    sglang_client = FakeSGLangClient(
        [make_response("raw qwen text")],
        parse_function_call_responses=[
            {
                "normal_text": "Thinking Process: parsed by api",
                "calls": [
                    {
                        "name": "get_weather_snapshot",
                        "parameters": {"city": "Shanghai", "hours": ["09:00"]},
                    }
                ],
            }
        ],
    )
    client, _, _, _ = make_client(
        sglang_client=sglang_client,
        model_tool_call_type="qwen3_5",
        tool_call_parse_backend="sglang_api",
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-qwen-api", "X-Instance-Id": "inst-qwen-api"},
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "check weather"}],
            "tools": make_tools("get_weather_snapshot"),
        },
    )

    assert response.status_code == 200
    completion = response.json()
    choice = completion["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "Thinking Process: parsed by api"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "city": "Shanghai",
        "hours": ["09:00"],
    }
    assert sglang_client.parse_function_call_calls[0]["tool_call_parser"] == "qwen3_coder"


def test_chat_completion_reasoning_parser_cleans_public_content_and_preserves_raw_training_text():
    raw_text = "<think>\nplan carefully\n</think>\n\nfinal answer<|im_end|>"
    sglang_client = FakeSGLangClient(
        [make_response(raw_text)],
        separate_reasoning_responses=[
            {"reasoning_text": "plan carefully\n", "text": "final answer<|im_end|>"}
        ],
    )
    client, session_manager, _, _ = make_client(
        sglang_client=sglang_client,
        model_reasoning_type="qwen3",
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-reasoning", "X-Instance-Id": "inst"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == "final answer"
    assert message["reasoning_content"] == "plan carefully"
    assert "</think>" not in message["content"]
    assert "<|im_end|>" not in message["content"]
    assert sglang_client.separate_reasoning_calls[0]["reasoning_parser"] == "qwen3"

    session = session_manager.get_session("sess-reasoning")
    assert session is not None
    step = session.latest_step
    assert step is not None
    assert step.raw_response_text == raw_text
    assert step.response_token_ids == [ord(ch) for ch in raw_text]
    assert step.response_logprobs == [-0.1 for _ in raw_text]
    assert session.full_messages[-1]["content"] == "final answer"
    assert session.full_messages[-1]["reasoning_content"] == "plan carefully"


def test_chat_completion_repairs_completion_only_reasoning_for_public_response():
    raw_text = "plan carefully\n</think>\n\nfinal answer<|im_end|>"
    sglang_client = FakeSGLangClient(
        [make_response(raw_text)],
        separate_reasoning_responses=[
            {"reasoning_text": "plan carefully\n", "text": "final answer<|im_end|>"}
        ],
    )
    client, session_manager, _, _ = make_client(
        sglang_client=sglang_client,
        model_reasoning_type="qwen3",
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-reasoning-repair", "X-Instance-Id": "inst"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == "final answer"
    assert message["reasoning_content"] == "plan carefully"
    assert "</think>" not in message["content"]
    assert sglang_client.separate_reasoning_calls[0]["text"] == "<think>" + raw_text

    session = session_manager.get_session("sess-reasoning-repair")
    assert session is not None
    step = session.latest_step
    assert step is not None
    assert step.raw_response_text == raw_text
    assert step.response_token_ids == [ord(ch) for ch in raw_text]
    assert session.full_messages[-1]["content"] == "final answer"
    assert session.full_messages[-1]["reasoning_content"] == "plan carefully"


def test_chat_completion_reasoning_then_tool_call_parse():
    visible_tool_text = make_qwen_tool_call_text(
        prefix="",
        functions=[("get_weather_snapshot", {"city": "Shanghai"})],
    )
    raw_text = f"<think>need weather</think>\n\n{visible_tool_text}<|im_end|>"
    sglang_client = FakeSGLangClient(
        [make_response(raw_text)],
        separate_reasoning_responses=[
            {"reasoning_text": "need weather", "text": visible_tool_text}
        ],
        parse_function_call_responses=[
            {
                "normal_text": None,
                "calls": [
                    {
                        "name": "get_weather_snapshot",
                        "parameters": {"city": "Shanghai"},
                    }
                ],
            }
        ],
    )
    client, session_manager, _, _ = make_client(
        sglang_client=sglang_client,
        model_tool_call_type="qwen3_5",
        model_reasoning_type="qwen3",
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-reasoning-tool", "X-Instance-Id": "inst"},
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "check weather"}],
            "tools": make_tools("get_weather_snapshot"),
        },
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] is None
    assert message["reasoning_content"] == "need weather"
    assert message["tool_calls"][0]["function"]["name"] == "get_weather_snapshot"
    assert sglang_client.parse_function_call_calls[0]["text"] == visible_tool_text

    session = session_manager.get_session("sess-reasoning-tool")
    assert session is not None
    assert session.latest_step.raw_response_text == raw_text


def test_streaming_emits_reasoning_content_before_clean_content():
    raw_text = "<think>stream plan</think>\n\nstream answer"
    sglang_client = FakeSGLangClient(
        [make_response(raw_text)],
        separate_reasoning_responses=[
            {"reasoning_text": "stream plan", "text": "stream answer"}
        ],
    )
    client, _, _, _ = make_client(
        sglang_client=sglang_client,
        model_reasoning_type="qwen3",
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-reasoning-stream", "X-Instance-Id": "inst"},
        json={
            "model": "fake-model",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"reasoning_content"' in body
    assert '"content"' in body
    assert body.index("reasoning_content") < body.index("content")
    assert "</think>" not in body
    assert_stream_usage_chunk(body)


def test_chat_completion_tool_call_backend_defaults_to_sglang_api():
    local_parseable_text = '<tool_call>{"name": "local", "arguments": {}}</tool_call>'
    sglang_client = FakeSGLangClient(
        [make_response(local_parseable_text)],
        parse_function_call_responses=[
            {
                "normal_text": "api normal",
                "calls": [{"name": "api_tool", "parameters": {"ok": True}}],
            }
        ],
    )
    client, _, _, _ = make_client(
        sglang_client=sglang_client,
        model_tool_call_type="qwen3_5",
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-default-tool-api", "X-Instance-Id": "inst"},
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "use tool"}],
            "tools": make_tools("api_tool"),
        },
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == "api normal"
    assert message["tool_calls"][0]["function"]["name"] == "api_tool"
    assert sglang_client.parse_function_call_calls[0]["tool_call_parser"] == "qwen3_coder"


def test_chat_completion_uses_implicit_single_turn_and_finalize_as_trajectory():
    client, session_manager, _, sglang_client = make_client(
        make_response("hello"),
        make_response("follow"),
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-SMG-Routing-Key": "sess-1", "X-Instance-Id": "inst-1"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200
    assert sglang_client.last_routing_key == "sess-1"

    session = session_manager.get_session("sess-1")
    assert session is not None
    assert session.instance_id == "inst-1"
    assert session.turn_mode == "implicit"
    implicit_turn_id = session.active_turn_id
    assert implicit_turn_id is not None
    assert implicit_turn_id.startswith(IMPLICIT_TURN_ID_PREFIX)

    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-1", "X-Instance-Id": "inst-1"},
        json={
            "model": "fake-model",
            "messages": session.full_messages + [{"role": "user", "content": "again"}],
        },
    )
    assert second.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-1", "instance_id": "inst-1"},
    )
    assert finalized.status_code == 200
    payload = finalized.json()
    assert payload["trajectory_id"] == "sess-1"
    assert payload["session_id"] == "sess-1"
    assert payload["num_steps"] == 2
    assert payload["num_turns"] == 1
    assert payload["num_segments"] == 2
    assert payload["history_rewritten"] is False

    exact = client.post("/trajectory/read", json={"trajectory_id": "sess-1"})
    assert exact.status_code == 200
    assert exact.json()["mode"] == "trajectory"
    exact_data = exact.json()["data"]
    assert len(exact_data) == 2
    assert exact_data[0]["trajectory_id"] == "sess-1"
    assert exact_data[0]["session_id"] == "sess-1"
    assert exact_data[0]["turn_id"] == implicit_turn_id
    assert len(exact_data[0]["tokens"]) == len(exact_data[0]["full_logprobs"])
    assert len(exact_data[0]["tokens"]) == len(exact_data[0]["full_loss_mask"])
    assert "full_versions" not in exact_data[0]
    assert_dense_alignment(exact_data[0], ["hello"])
    assert_dense_alignment(exact_data[1], ["follow"])
    assert exact_data[0]["extra_info"]["num_steps"] == 1
    assert exact_data[0]["extra_info"]["num_turns"] == 1
    assert exact_data[0]["extra_info"]["turn_ids"] == [implicit_turn_id]
    assert exact_data[0]["extra_info"]["segment_reason"] == "initial"
    assert exact_data[0]["extra_info"]["segment_reasons"] == ["initial"]
    assert_alignment_metadata(exact_data[0], mask_template_equivalent=True)
    assert exact_data[0]["extra_info"]["output_token_count"] == len("hello")
    assert exact_data[1]["extra_info"]["output_token_count"] == len("follow")
    assert payload["token_build_mode"] == "snapshot"


def test_finalize_records_token_versions_only_when_enabled():
    client, _, _, _ = make_client(
        make_response("versioned"),
        record_token_versions=True,
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-versioned", "X-Instance-Id": "inst-versioned"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-versioned", "instance_id": "inst-versioned"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["record_token_versions"] is True

    exact = client.post("/trajectory/read", json={"trajectory_id": "sess-versioned"})
    item = exact.json()["data"][0]
    assert "full_versions" in item
    assert len(item["full_versions"]) == len(item["tokens"])


def test_finalize_marks_nonlast_version_mask_proxy_config():
    client, _, _, _ = make_client(
        make_response("old"),
        record_token_versions=True,
        mask_nonlast_version_tokens=True,
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-mask-config", "X-Instance-Id": "inst"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-mask-config", "instance_id": "inst"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["mask_nonlast_version_tokens"] is True

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["config"]["mask_nonlast_version_tokens"] is True

    item = client.post(
        "/trajectory/read",
        json={"trajectory_id": "sess-mask-config"},
    ).json()["data"][0]
    assert item["extra_info"]["mask_nonlast_version_tokens"] is True
    assert "full_loss_mask_before_nonlast_version_mask" not in item["extra_info"]
    assert_dense_alignment(item, ["old"])


def test_non_partial_session_rejects_stale_epoch_before_sglang():
    first = make_response("one")
    first.meta_info = {"finish_reason": {"type": "stop"}, "weight_version": "v1"}
    second = make_response("two")
    second.meta_info = {"finish_reason": {"type": "stop"}, "weight_version": "v2"}
    client, session_manager, _, sglang_client = make_client(
        first,
        second,
        token_build_mode="tito",
        tito_model="qwen3_5",
        record_token_versions=True,
    )

    first_response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-cross-version", "X-Instance-Id": "inst-cross-version"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first_response.status_code == 200

    paused = client.post("/v1/rollout/pause", json={"reason": "weight_update"})
    assert paused.status_code == 200
    resumed = client.post("/v1/rollout/resume", json={"reason": "weight_update"})
    assert resumed.status_code == 200

    second_response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-cross-version", "X-Instance-Id": "inst-cross-version"},
        json={
            "model": "fake-model",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "one"},
                {"role": "user", "content": "next"},
            ],
        },
    )

    assert second_response.status_code == 502
    assert second_response.json()["detail"]["error"] == "trajectory_version_changed"
    assert len(sglang_client.calls) == 1
    session = session_manager.get_session("sess-cross-version")
    assert session is not None
    assert len(session.steps) == 1


def test_partial_rollout_rejects_staleness_exceeded():
    first = make_response("a", finish_reason="abort", output_logprobs=[-0.11])
    first.meta_info = {"finish_reason": {"type": "abort"}, "weight_version": "v1"}
    second = make_response("b", finish_reason="abort", output_logprobs=[-0.22])
    second.meta_info = {"finish_reason": {"type": "abort"}, "weight_version": "v2"}
    third = make_response("c", output_logprobs=[-0.33])
    third.meta_info = {"finish_reason": {"type": "stop"}, "weight_version": "v3"}
    client, session_manager, _, sglang_client = make_client(
        first,
        second,
        third,
        partial_rollout=True,
        max_partial_rollout_preempts=1,
    )

    response = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-staleness",
            "X-Instance-Id": "inst-staleness",
        },
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["error"] == "partial_rollout_staleness_exceeded"
    assert detail["versions"] == ["v1", "v2", "v3"]
    assert detail["version_span"] == 3
    assert detail["version_switches"] == 2
    assert detail["max_preempts"] == 1
    assert detail["max_version_span"] == 2
    assert len(sglang_client.calls) == 3
    session = session_manager.get_session("sess-staleness")
    assert session is not None
    assert len(session.steps) == 0


def test_non_partial_first_step_waiting_for_resume_binds_generated_epoch():
    first = make_response("one")
    first.meta_info = {"finish_reason": {"type": "stop"}, "weight_version": "v2"}
    second = make_response("two")
    second.meta_info = {"finish_reason": {"type": "stop"}, "weight_version": "v2"}
    client, session_manager, _, sglang_client = make_client(
        first,
        second,
        token_build_mode="tito",
        tito_model="qwen3_5",
        record_token_versions=True,
    )

    paused = client.post("/v1/rollout/pause", json={"reason": "weight_update"})
    assert paused.status_code == 200

    result: dict[str, Any] = {}

    def post_first_step() -> None:
        try:
            result["response"] = client.post(
                "/v1/chat/completions",
                headers={
                    "X-Session-Id": "sess-first-step-pause",
                    "X-Instance-Id": "inst-first-step-pause",
                },
                json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
            )
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=post_first_step)
    thread.start()
    threading.Event().wait(0.05)

    resumed = client.post("/v1/rollout/resume", json={"reason": "weight_update"})
    assert resumed.status_code == 200
    thread.join(2)

    assert thread.is_alive() is False
    assert "error" not in result
    assert result["response"].status_code == 200
    session = session_manager.get_session("sess-first-step-pause")
    assert session is not None
    assert session.rollout_epoch == resumed.json()["rollout_epoch"]

    second_response = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-first-step-pause",
            "X-Instance-Id": "inst-first-step-pause",
        },
        json={
            "model": "fake-model",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "one"},
                {"role": "user", "content": "next"},
            ],
        },
    )

    assert second_response.status_code == 200
    assert len(sglang_client.calls) == 2


def test_finalize_concat_mode_uses_tito_recorded_step_fragments():
    client, session_manager, _, sglang_client = make_client(
        make_response("hello"),
        make_response("follow"),
        token_build_mode="tito",
        tito_model="qwen3_5",
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-concat", "X-Instance-Id": "inst-concat"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200
    assert sglang_client.calls[0]["logprob_start_len"] == -1

    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-concat", "X-Instance-Id": "inst-concat"},
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-concat").full_messages
            + [{"role": "user", "content": "again"}],
        },
    )
    assert second.status_code == 200
    assert sglang_client.calls[1]["logprob_start_len"] == -1

    finalized = client.post(
        "/session/finalize",
        json={
            "session_id": "sess-concat",
            "instance_id": "inst-concat",
        },
    )
    assert finalized.status_code == 200
    payload = finalized.json()
    assert payload["trajectory_id"] == "sess-concat"
    assert payload["token_build_mode"] == "tito"

    exact = client.post("/trajectory/read", json={"trajectory_id": "sess-concat"})
    assert exact.status_code == 200
    item = exact.json()["data"][0]
    expected_rendered = (
        "<user>hi<assistant>"
        "hello"
        "<user>again<assistant>"
        "follow"
    )
    assert decode_tokens(item["tokens"]) == expected_rendered
    assert decode_tokens(item["tokens"]).count("<user>hi") == 1
    assert_concat_mask_for_outputs(item, ["hello", "follow"])
    extra = item["extra_info"]
    assert extra["alignment_method"] == "tito"
    assert extra["token_build_mode"] == "tito"
    assert extra["context_token_count"] == len(
        "<user>hi<assistant><user>again<assistant>"
    )
    assert extra["context_delta_token_count"] == extra["context_token_count"]
    assert extra["output_token_count"] == len("hello") + len("follow")
    assert extra["num_steps"] == 2
    assert "mask_template_equivalent" not in extra
    assert "mask_fallback_reason" not in extra


def test_concat_append_only_prompt_uses_online_tito_without_full_tokenize():
    tokenizer = OnlineTITOGuardTokenizer()
    client, session_manager, _, sglang_client = make_client(
        make_response("hello"),
        make_response("follow"),
        tokenizer=tokenizer,
        token_build_mode="tito",
        tito_model="qwen3_5",
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-concat-online", "X-Instance-Id": "inst"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200

    tokenizer.reject_tokenization = True
    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-concat-online", "X-Instance-Id": "inst"},
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-concat-online").full_messages
            + [{"role": "user", "content": "again"}],
        },
    )
    assert second.status_code == 200

    assert decode_tokens(sglang_client.calls[1]["input_ids"]) == (
        "<user>hi<assistant>hello<user>again<assistant>"
    )
    assert sglang_client.calls[0]["logprob_start_len"] == -1
    assert sglang_client.calls[1]["logprob_start_len"] == -1
    step = session_manager.get_session("sess-concat-online").latest_step
    assert decode_tokens(step.concat_token_ids) == "<user>again<assistant>follow"
    assert step.segment_boundary_before is False
    assert step.concat_incremental_tokenization_failed is False


def test_concat_online_tito_failure_falls_back_and_starts_new_segment():
    tokenizer = OnlineTITOFailureTokenizer()
    client, session_manager, _, sglang_client = make_client(
        make_response("hello"),
        make_response("follow"),
        tokenizer=tokenizer,
        token_build_mode="tito",
        tito_model="qwen3_5",
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-concat-tito-fail", "X-Instance-Id": "inst"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200

    tokenizer.fail_tito_render = True
    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-concat-tito-fail", "X-Instance-Id": "inst"},
        json={
            "model": "fake-model",
            "messages": session_manager.get_session(
                "sess-concat-tito-fail"
            ).full_messages
            + [{"role": "user", "content": "again"}],
        },
    )
    assert second.status_code == 200

    step = session_manager.get_session("sess-concat-tito-fail").latest_step
    assert sglang_client.calls[0]["logprob_start_len"] == -1
    assert sglang_client.calls[1]["logprob_start_len"] == -1
    assert step.segment_boundary_before is True
    assert step.segment_reasons_before == ["tito_routing_render_failed"]

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-concat-tito-fail", "instance_id": "inst"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["num_segments"] == 2

    items = client.post(
        "/trajectory/read", json={"trajectory_id": "sess-concat-tito-fail"}
    ).json()["data"]
    assert items[1]["extra_info"]["segment_reason"] == "tito_routing_render_failed"
    assert items[1]["extra_info"]["segment_reasons"] == [
        "tito_routing_render_failed"
    ]
    assert decode_tokens(items[1]["tokens"]) == (
        "<user>hi<assistant>hello<user>again<assistant>follow"
    )


def test_concat_reasoning_round_trip_stays_in_one_segment():
    first_raw = "<think>plan</think>\n\nhello"
    client, session_manager, _, _ = make_client(
        make_response(first_raw),
        make_response("follow"),
        token_build_mode="tito",
        tito_model="qwen3_5",
        model_reasoning_type="qwen3",
        reasoning_parse_backend="local",
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-concat-reasoning", "X-Instance-Id": "inst"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-concat-reasoning", "X-Instance-Id": "inst"},
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-concat-reasoning").full_messages
            + [{"role": "user", "content": "again"}],
        },
    )
    assert second.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-concat-reasoning", "instance_id": "inst"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["num_segments"] == 1

    item = client.post(
        "/trajectory/read", json={"trajectory_id": "sess-concat-reasoning"}
    ).json()["data"][0]
    rendered = decode_tokens(item["tokens"])
    assert first_raw in rendered
    assert "follow" in rendered
    assert_concat_mask_for_outputs(item, [first_raw, "follow"])


def test_concat_reasoning_boundary_whitespace_round_trip_stays_in_one_segment():
    first_raw = "<think>\nplan\n</think>\n\nhello"
    sglang_client = FakeSGLangClient(
        [make_response(first_raw), make_response("follow")],
        separate_reasoning_responses=[
            {"reasoning_text": "plan\n", "text": "hello"},
        ],
    )
    client, session_manager, _, _ = make_client(
        sglang_client=sglang_client,
        token_build_mode="tito",
        tito_model="qwen3_5",
        model_reasoning_type="qwen3",
    )

    first = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-concat-reasoning-trim",
            "X-Instance-Id": "inst",
        },
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200
    assert first.json()["choices"][0]["message"]["reasoning_content"] == "plan"

    replayed_messages = [
        dict(message)
        for message in session_manager.get_session(
            "sess-concat-reasoning-trim"
        ).full_messages
    ]
    replayed_messages[-1]["reasoning_content"] = " plan\n"
    second = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-concat-reasoning-trim",
            "X-Instance-Id": "inst",
        },
        json={
            "model": "fake-model",
            "messages": replayed_messages + [{"role": "user", "content": "again"}],
        },
    )
    assert second.status_code == 200
    assert session_manager.get_session(
        "sess-concat-reasoning-trim"
    ).latest_step.segment_boundary_before is False

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-concat-reasoning-trim", "instance_id": "inst"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["num_segments"] == 1

    item = client.post(
        "/trajectory/read", json={"trajectory_id": "sess-concat-reasoning-trim"}
    ).json()["data"][0]
    assert "concat_incremental_tokenization_failed" not in item["extra_info"]
    assert_concat_mask_for_outputs(item, [first_raw, "follow"])


def test_concat_missing_reasoning_content_starts_new_segment():
    first_raw = "<think>plan</think>\n\nhello"
    client, session_manager, _, _ = make_client(
        make_response(first_raw),
        make_response("follow"),
        token_build_mode="tito",
        tito_model="qwen3_5",
        model_reasoning_type="qwen3",
        reasoning_parse_backend="local",
    )

    first = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-concat-lost-reasoning",
            "X-Instance-Id": "inst",
            "X-Turn-Id": "turn-1",
        },
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200

    replayed_messages = [
        {key: value for key, value in message.items() if key != "reasoning_content"}
        for message in session_manager.get_session("sess-concat-lost-reasoning").full_messages
    ]
    second = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-concat-lost-reasoning",
            "X-Instance-Id": "inst",
            "X-Turn-Id": "turn-2",
        },
        json={
            "model": "fake-model",
            "messages": replayed_messages + [{"role": "user", "content": "again"}],
        },
    )
    assert second.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-concat-lost-reasoning", "instance_id": "inst"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["num_segments"] == 2

    items = client.post(
        "/trajectory/read", json={"trajectory_id": "sess-concat-lost-reasoning"}
    ).json()["data"]
    assert items[1]["extra_info"]["segment_reason"] == "initial"
    assert items[1]["extra_info"]["segment_reasons"] == ["initial"]


def test_finalize_concat_mode_does_not_tokenize_messages():
    tokenizer = FinalizeGuardTokenizer()
    client, session_manager, _, _ = make_client(
        make_response("hello"),
        make_response("follow"),
        tokenizer=tokenizer,
        token_build_mode="tito",
        tito_model="qwen3_5",
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-concat-finalize", "X-Instance-Id": "inst"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-concat-finalize", "X-Instance-Id": "inst"},
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-concat-finalize").full_messages
            + [{"role": "user", "content": "again"}],
        },
    )
    assert second.status_code == 200

    tokenizer.reject_tokenization = True
    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-concat-finalize", "instance_id": "inst"},
    )

    assert finalized.status_code == 200
    item = client.post(
        "/trajectory/read", json={"trajectory_id": "sess-concat-finalize"}
    ).json()["data"][0]
    assert_concat_mask_for_outputs(item, ["hello", "follow"])


def test_finalize_concat_mode_keeps_tool_call_output_and_masks_tool_response_context():
    raw_tool = '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>'
    client, session_manager, _, _ = make_client(
        make_response(raw_tool),
        make_response("done"),
        token_build_mode="tito",
        tito_model="qwen3_5",
        tool_call_parse_backend="local",
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-concat-tool", "X-Instance-Id": "inst-tool"},
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "find x"}],
        },
    )
    assert first.status_code == 200
    session = session_manager.get_session("sess-concat-tool")
    tool_call_id = session.full_messages[-1]["tool_calls"][0]["id"]

    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-concat-tool", "X-Instance-Id": "inst-tool"},
        json={
            "model": "fake-model",
            "messages": session.full_messages
            + [
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "result",
                }
            ],
        },
    )
    assert second.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={
            "session_id": "sess-concat-tool",
            "instance_id": "inst-tool",
        },
    )
    assert finalized.status_code == 200

    exact = client.post("/trajectory/read", json={"trajectory_id": "sess-concat-tool"})
    item = exact.json()["data"][0]
    rendered = decode_tokens(item["tokens"])
    assert raw_tool in rendered
    assert "<tool>result" in rendered
    assert "<tool_call_id>" in rendered
    assert_concat_mask_for_outputs(item, [raw_tool, "done"])
    result_start = rendered.index("result")
    result_end = result_start + len("result")
    assert item["full_loss_mask"][result_start:result_end] == [0] * len("result")


def test_concat_openclaw_sanitized_legacy_tool_call_replay_stays_in_one_segment():
    client, session_manager, _, _ = make_client(
        make_response("legacy tool call"),
        make_response("done"),
        tool_call_parser=legacy_underscored_tool_call_parser,
        token_build_mode="tito",
        tito_model="qwen3_5",
    )

    first = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-concat-openclaw",
            "X-Instance-Id": "inst-concat-openclaw",
        },
        json={"model": "fake-model", "messages": [{"role": "user", "content": "find"}]},
    )
    assert first.status_code == 200
    session = session_manager.get_session("sess-concat-openclaw")
    assert session is not None
    assert session.full_messages[-1]["tool_calls"][0]["id"] == "call_b1467966"

    replayed_messages = json.loads(json.dumps(session.full_messages))
    replayed_messages[-1]["tool_calls"][0]["id"] = "callb1467966"
    second = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-concat-openclaw",
            "X-Instance-Id": "inst-concat-openclaw",
        },
        json={
            "model": "fake-model",
            "messages": replayed_messages
            + [
                {
                    "role": "tool",
                    "tool_call_id": "callb1467966",
                    "content": "result",
                }
            ],
        },
    )
    assert second.status_code == 200
    latest_step = session_manager.get_session("sess-concat-openclaw").latest_step
    assert latest_step.segment_boundary_before is False
    assert latest_step.concat_incremental_tokenization_failed is False

    finalized = client.post(
        "/session/finalize",
        json={
            "session_id": "sess-concat-openclaw",
            "instance_id": "inst-concat-openclaw",
        },
    )
    assert finalized.status_code == 200
    assert finalized.json()["history_rewritten"] is False
    assert finalized.json()["num_segments"] == 1

    item = client.post(
        "/trajectory/read", json={"trajectory_id": "sess-concat-openclaw"}
    ).json()["data"][0]
    assert "concat_incremental_tokenization_failed" not in item["extra_info"]


def test_concat_message_prefix_mismatch_starts_new_segment():
    client, _, _, _ = make_client(
        make_response("first"),
        make_response("second"),
        token_build_mode="tito",
        tito_model="qwen3_5",
    )

    first = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-concat-prefix",
            "X-Instance-Id": "inst-prefix",
            "X-Turn-Id": "turn-1",
        },
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-concat-prefix",
            "X-Instance-Id": "inst-prefix",
            "X-Turn-Id": "turn-2",
        },
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "fresh"}],
        },
    )
    assert second.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-concat-prefix", "instance_id": "inst-prefix"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["num_segments"] == 2

    items = client.post(
        "/trajectory/read", json={"trajectory_id": "sess-concat-prefix"}
    ).json()["data"]
    assert items[1]["extra_info"]["segment_reason"] == "initial"
    assert items[1]["extra_info"]["segment_reasons"] == ["initial"]
    assert decode_tokens(items[1]["tokens"]) == "<user>fresh<assistant>second"


def test_tito_lineage_routing_keeps_interleaved_subagents_out_of_main_base():
    client, session_manager, _, _ = make_client(
        make_response("m2"),
        make_response("s2"),
        make_response("m3"),
        make_response("s4"),
        make_response("m5"),
        token_build_mode="tito",
        tito_model="qwen3_5",
    )

    def post(messages: list[dict]) -> None:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Session-Id": "sess-lineage", "X-Instance-Id": "inst-lineage"},
            json={"model": "fake-model", "messages": messages},
        )
        assert response.status_code == 200

    post([{"role": "user", "content": "m1"}])
    main_after_m2 = list(session_manager.get_session("sess-lineage").full_messages)
    post([{"role": "user", "content": "s1"}])
    post(main_after_m2 + [{"role": "user", "content": "s2"}])
    main_after_m3 = list(session_manager.get_session("sess-lineage").full_messages)
    post([{"role": "user", "content": "s3"}])
    post(main_after_m3 + [{"role": "user", "content": "s4"}])

    session = session_manager.get_session("sess-lineage")
    steps = session.steps
    assert steps[2].route_base_step_id == steps[0].step_id
    assert steps[4].route_base_step_id == steps[2].step_id
    assert [step.route_type for step in steps] == [
        "create",
        "create",
        "append",
        "create",
        "append",
    ]

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-lineage", "instance_id": "inst-lineage"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["num_segments"] == 3
    assert finalized.json()["num_timeline_segments"] == 5

    lineage = client.post("/trajectory/read", json={"trajectory_id": "sess-lineage"})
    assert lineage.status_code == 200
    lineage_items = lineage.json()["data"]
    assert len(lineage_items) == 3
    assert {item["extra_info"]["segment_view"] for item in lineage_items} == {"lineage"}
    assert [item["extra_info"]["lineage_index"] for item in lineage_items] == [0, 1, 2]
    assert [item["extra_info"]["num_steps"] for item in lineage_items] == [3, 1, 1]
    assert_concat_mask_for_outputs(lineage_items[0], ["m2", "m3", "m5"])

    timeline = client.post(
        "/trajectory/read",
        json={"trajectory_id": "sess-lineage", "segment_view": "timeline"},
    )
    assert timeline.status_code == 200
    timeline_items = timeline.json()["data"]
    assert len(timeline_items) == 5
    assert {item["extra_info"]["segment_view"] for item in timeline_items} == {"timeline"}
    assert [item["extra_info"]["route_base_step_id"] for item in timeline_items] == [
        None,
        None,
        steps[0].step_id,
        None,
        steps[2].step_id,
    ]

    drained = client.post(
        "/trajectory/read",
        json={
            "trajectory_id": "sess-lineage",
            "segment_view": "timeline",
            "drain": True,
        },
    )
    assert len(drained.json()["data"]) == 5
    empty = client.post("/trajectory/read", json={"trajectory_id": "sess-lineage"})
    assert empty.json()["data"] == []


def test_tito_lineage_routing_preserves_thinking_when_agent_message_appends():
    def send_message_parser(text: str):
        if text.strip() != "call-send":
            return text, None
        return None, [
            {
                "id": "callabcdef12",
                "type": "function",
                "index": 0,
                "function": {
                    "name": "send_message",
                    "arguments": json.dumps(
                        {"target": "/root", "message": "Answer: 5"}
                    ),
                },
            }
        ]

    client, session_manager, _, _ = make_client(
        make_response("<think>\nNeed to report.\n</think>\n\ncall-send"),
        make_response("Answer: 5"),
        tokenizer=QwenJinjaTokenizer(),
        token_build_mode="tito",
        tito_model="qwen3_5",
        model_reasoning_type="qwen3_5",
        reasoning_parse_backend="local",
        tool_call_parser=send_message_parser,
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-preserve-thinking", "X-Instance-Id": "inst"},
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "root task"}],
            "tools": make_tools("send_message"),
        },
    )
    assert first.status_code == 200
    session = session_manager.get_session("sess-preserve-thinking")
    assert session is not None
    assert session.latest_step.route_type == "create"

    second_messages = session.full_messages + [
        {"role": "tool", "tool_call_id": "callabcdef12", "content": ""},
        {
            "role": "user",
            "content": '{"type":"agent_message","author":"/root","recipient":"/root"}',
        },
    ]
    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-preserve-thinking", "X-Instance-Id": "inst"},
        json={
            "model": "fake-model",
            "messages": second_messages,
            "tools": make_tools("send_message"),
        },
    )
    assert second.status_code == 200

    steps = session_manager.get_session("sess-preserve-thinking").steps
    assert [step.route_type for step in steps] == ["create", "append"]
    assert steps[1].route_base_step_id == steps[0].step_id
    assert steps[1].lineage_id == steps[0].lineage_id
    assert steps[1].segment_boundary_before is False
    assert steps[1].concat_incremental_tokenization_failed is False

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-preserve-thinking", "instance_id": "inst"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["num_segments"] == 1

    lineage = client.post(
        "/trajectory/read", json={"trajectory_id": "sess-preserve-thinking"}
    )
    assert lineage.status_code == 200
    items = lineage.json()["data"]
    assert len(items) == 1
    assert items[0]["extra_info"]["num_steps"] == 2


def test_finalize_rejects_request_token_build_mode_without_finalizing_session():
    client, _, _, _ = make_client(make_response("hello"))
    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-bad-mode", "X-Instance-Id": "inst-bad-mode"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200

    rejected_batch = client.post(
        "/session/finalize",
        json={
            "session_id": "sess-bad-mode",
            "instance_id": "inst-bad-mode",
            "token_build_modes": ["snapshot", "tito"],
        },
    )
    assert rejected_batch.status_code == 400

    rejected = client.post(
        "/session/finalize",
        json={
            "session_id": "sess-bad-mode",
            "instance_id": "inst-bad-mode",
            "token_build_mode": "both",
        },
    )
    assert rejected.status_code == 400

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-bad-mode", "instance_id": "inst-bad-mode"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["token_build_mode"] == "snapshot"


def test_streaming_tool_calls_stay_in_one_implicit_turn_without_false_rewrite():
    raw_tool = '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>'
    client, session_manager, _, _ = make_client(
        make_response(raw_tool),
        make_response("done"),
        tool_call_parse_backend="local",
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-tools", "X-Instance-Id": "inst-tools"},
        json={
            "model": "fake-model",
            "stream": True,
            "messages": [{"role": "user", "content": "find x"}],
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"tool_calls"' in body
    assert "[DONE]" in body
    assert_stream_usage_chunk(body)

    session = session_manager.get_session("sess-tools")
    assert session is not None
    assistant_message = session.full_messages[-1]
    assert assistant_message["content"] is None
    assert assistant_message["tool_calls"][0]["function"]["name"] == "search"

    followup_messages = session.full_messages + [
        {
            "role": "tool",
            "tool_call_id": assistant_message["tool_calls"][0]["id"],
            "content": "result",
        }
    ]
    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-tools", "X-Instance-Id": "inst-tools"},
        json={"model": "fake-model", "messages": followup_messages},
    )
    assert second.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-tools", "instance_id": "inst-tools"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["history_rewritten"] is False
    assert finalized.json()["num_turns"] == 1
    assert finalized.json()["num_segments"] == 2


def test_tool_call_history_replay_without_index_and_minified_arguments_is_append_only():
    raw_tool = (
        '<tool_call>{"name": "search", "arguments": {"b": 2, "a": 1}}</tool_call>'
    )
    client, session_manager, _, _ = make_client(
        make_response(raw_tool),
        make_response("done"),
        tool_call_parse_backend="local",
    )

    first = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-sdk-tool-history",
            "X-Instance-Id": "inst-sdk-tool-history",
        },
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "find"}],
        },
    )
    assert first.status_code == 200
    session = session_manager.get_session("sess-sdk-tool-history")
    assert session is not None
    assistant_message = session.full_messages[-1]
    tool_call = assistant_message["tool_calls"][0]
    assert "index" in tool_call

    sdk_replayed_assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tool_call["id"],
                "type": tool_call["type"],
                "function": {
                    "name": tool_call["function"]["name"],
                    "arguments": '{"a":1,"b":2}',
                },
            }
        ],
    }
    second = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-sdk-tool-history",
            "X-Instance-Id": "inst-sdk-tool-history",
        },
        json={
            "model": "fake-model",
            "messages": [
                {"role": "user", "content": "find"},
                sdk_replayed_assistant,
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": "result",
                },
            ],
        },
    )
    assert second.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={
            "session_id": "sess-sdk-tool-history",
            "instance_id": "inst-sdk-tool-history",
        },
    )
    assert finalized.status_code == 200
    assert finalized.json()["history_rewritten"] is False
    assert finalized.json()["num_segments"] == 2


def test_openclaw_sanitized_legacy_tool_call_replay_is_append_only():
    client, session_manager, _, _ = make_client(
        make_response("legacy tool call"),
        make_response("done"),
        tool_call_parser=legacy_underscored_tool_call_parser,
    )

    first = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-openclaw-sanitize",
            "X-Instance-Id": "inst-openclaw-sanitize",
        },
        json={"model": "fake-model", "messages": [{"role": "user", "content": "find"}]},
    )
    assert first.status_code == 200
    session = session_manager.get_session("sess-openclaw-sanitize")
    assert session is not None
    assert session.full_messages[-1]["tool_calls"][0]["id"] == "call_b1467966"

    replayed_messages = json.loads(json.dumps(session.full_messages))
    replayed_messages[-1]["tool_calls"][0]["id"] = "callb1467966"
    second = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-openclaw-sanitize",
            "X-Instance-Id": "inst-openclaw-sanitize",
        },
        json={
            "model": "fake-model",
            "messages": replayed_messages
            + [
                {
                    "role": "tool",
                    "tool_call_id": "callb1467966",
                    "content": "result",
                }
            ],
        },
    )
    assert second.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={
            "session_id": "sess-openclaw-sanitize",
            "instance_id": "inst-openclaw-sanitize",
        },
    )
    assert finalized.status_code == 200
    assert finalized.json()["history_rewritten"] is False
    assert finalized.json()["num_segments"] == 2


def test_finalize_exact_read_and_batch_drain_for_explicit_multi_turn_sessions():
    client, session_manager, _, _ = make_client(
        make_response("first"),
        make_response("second"),
        make_response("third"),
    )

    first = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-multi",
            "X-Instance-Id": "inst-multi",
            "X-Turn-Id": "turn-1",
        },
        json={"model": "fake-model", "messages": [{"role": "user", "content": "step one"}]},
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-multi",
            "X-Instance-Id": "inst-multi",
            "X-Turn-Id": "turn-1",
        },
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-multi").full_messages
            + [{"role": "user", "content": "step two"}],
        },
    )
    assert second.status_code == 200

    third = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-multi",
            "X-Instance-Id": "inst-multi",
            "X-Turn-Id": "turn-2",
        },
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-multi").full_messages
            + [{"role": "user", "content": "new turn"}],
        },
    )
    assert third.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-multi", "instance_id": "inst-multi"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["trajectory_id"] == "sess-multi"
    assert finalized.json()["num_steps"] == 3
    assert finalized.json()["num_turns"] == 2
    assert finalized.json()["num_segments"] == 3

    exact = client.post("/trajectory/read", json={"trajectory_id": "sess-multi"})
    assert exact.status_code == 200
    exact_data = exact.json()["data"]
    assert len(exact_data) == 3
    assert exact_data[2]["turn_id"] == "turn-2"
    assert exact_data[0]["segment_index"] == 0
    assert exact_data[0]["segment_count"] == 3
    assert_dense_alignment(exact_data[0], ["first"])
    assert_dense_alignment(exact_data[1], ["second"])
    assert_dense_alignment(exact_data[2], ["third"])
    assert exact_data[0]["extra_info"]["num_steps"] == 1
    assert exact_data[2]["extra_info"]["num_turns"] == 1
    assert exact_data[2]["extra_info"]["turn_ids"] == ["turn-2"]
    assert exact_data[0]["extra_info"]["trajectory_num_segments"] == 3
    assert exact_data[0]["extra_info"]["segment_reason"] == "initial"
    assert exact_data[0]["extra_info"]["segment_reasons"] == ["initial"]
    assert_alignment_metadata(exact_data[0], mask_template_equivalent=True)
    assert exact_data[2]["extra_info"]["output_token_count"] == len("third")

    exact_legacy = client.post("/trajectory/read", json={"session_id": "sess-multi"})
    assert exact_legacy.status_code == 200
    assert len(exact_legacy.json()["data"]) == 3

    drained = client.post(
        "/trajectory/read",
        json={"max_groups": 1},
    )
    assert drained.status_code == 200
    drained_groups = drained.json()["data"]
    assert len(drained_groups) == 1
    assert len(drained_groups[0]) == 3
    assert drained_groups[0][2]["turn_id"] == "turn-2"
    assert drained_groups[0][2]["extra_info"]["turn_ids"] == ["turn-2"]

    empty = client.post(
        "/trajectory/read",
        json={"max_groups": 1},
    )
    assert empty.status_code == 200
    assert empty.json()["data"] == []


def test_same_turn_rewrite_splits_segments_without_creating_new_turn():
    client, _, _, _ = make_client(
        make_response("first"),
        make_response("second"),
    )

    first = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-rewrite",
            "X-Instance-Id": "inst-rewrite",
            "X-Turn-Id": "turn-1",
        },
        json={"model": "fake-model", "messages": [{"role": "user", "content": "step one"}]},
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-rewrite",
            "X-Instance-Id": "inst-rewrite",
            "X-Turn-Id": "turn-1",
        },
        json={"model": "fake-model", "messages": [{"role": "user", "content": "rewritten history"}]},
    )
    assert second.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-rewrite", "instance_id": "inst-rewrite"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["history_rewritten"] is True
    assert finalized.json()["num_turns"] == 1
    assert finalized.json()["num_segments"] == 2

    exact = client.post("/trajectory/read", json={"trajectory_id": "sess-rewrite"})
    assert exact.status_code == 200
    exact_data = exact.json()["data"]
    assert len(exact_data) == 2
    assert {item["turn_id"] for item in exact_data} == {"turn-1"}
    assert exact_data[0]["extra_info"]["segment_reason"] == "initial"
    assert exact_data[0]["extra_info"]["segment_reasons"] == ["initial"]
    assert exact_data[1]["extra_info"]["segment_reason"] == "history_rewrite"
    assert exact_data[1]["extra_info"]["segment_reasons"] == ["history_rewrite"]
    assert_dense_alignment(exact_data[0], ["first"])
    assert_dense_alignment(exact_data[1], ["second"])
    assert_alignment_metadata(exact_data[0], mask_template_equivalent=True)
    assert_alignment_metadata(exact_data[1], mask_template_equivalent=True)
    assert exact_data[0]["extra_info"]["num_turns"] == 1
    assert exact_data[1]["extra_info"]["num_turns"] == 1
    assert exact_data[1]["extra_info"]["turn_ids"] == ["turn-1"]


def test_invalid_turn_mixing_returns_bad_request():
    client, session_manager, _, _ = make_client(
        make_response("implicit"),
        make_response("explicit"),
        make_response("explicit again"),
        make_response("turn one"),
        make_response("turn two"),
    )

    implicit = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-implicit-mix", "X-Instance-Id": "inst-mix"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert implicit.status_code == 200

    implicit_then_explicit = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-implicit-mix",
            "X-Instance-Id": "inst-mix",
            "X-Turn-Id": "turn-1",
        },
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-implicit-mix").full_messages,
        },
    )
    assert implicit_then_explicit.status_code == 400

    explicit = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-explicit-mix",
            "X-Instance-Id": "inst-mix",
            "X-Turn-Id": "turn-1",
        },
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert explicit.status_code == 200

    explicit_then_implicit = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-explicit-mix", "X-Instance-Id": "inst-mix"},
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-explicit-mix").full_messages,
        },
    )
    assert explicit_then_implicit.status_code == 400

    turn_one = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-return-old-turn",
            "X-Instance-Id": "inst-mix",
            "X-Turn-Id": "turn-1",
        },
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert turn_one.status_code == 200

    turn_two = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-return-old-turn",
            "X-Instance-Id": "inst-mix",
            "X-Turn-Id": "turn-2",
        },
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-return-old-turn").full_messages
            + [{"role": "user", "content": "new turn"}],
        },
    )
    assert turn_two.status_code == 200

    return_old_turn = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-return-old-turn",
            "X-Instance-Id": "inst-mix",
            "X-Turn-Id": "turn-1",
        },
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-return-old-turn").full_messages,
        },
    )
    assert return_old_turn.status_code == 400


def test_empty_response_step_keeps_mask_on_last_step_output_only():
    client, session_manager, _, _ = make_client(
        make_response(""),
        make_response("done"),
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-empty", "X-Instance-Id": "inst-empty"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-empty", "X-Instance-Id": "inst-empty"},
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-empty").full_messages
            + [{"role": "user", "content": "again"}],
        },
    )
    assert second.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-empty", "instance_id": "inst-empty"},
    )
    assert finalized.status_code == 200

    items = client.post("/trajectory/read", json={"trajectory_id": "sess-empty"}).json()["data"]
    item = items[0]
    assert_alignment_metadata(item, mask_template_equivalent=True)
    assert item["extra_info"]["output_token_count"] == 0
    assert_dense_alignment(items[1], ["done"])


def test_missing_prompt_logprobs_falls_back_to_last_step_output_only():
    client, session_manager, _, sglang_client = make_client(
        make_response("bad"),
        make_prompt_logprob_missing_response("good"),
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-mismatch", "X-Instance-Id": "inst-mismatch"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-mismatch", "X-Instance-Id": "inst-mismatch"},
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-mismatch").full_messages
            + [{"role": "user", "content": "again"}],
        },
    )
    assert second.status_code == 200
    assert [call["logprob_start_len"] for call in sglang_client.calls] == [0, 0]

    client.post(
        "/session/finalize",
        json={"session_id": "sess-mismatch", "instance_id": "inst-mismatch"},
    )
    item = client.post("/trajectory/read", json={"trajectory_id": "sess-mismatch"}).json()[
        "data"
    ][1]
    assert_alignment_metadata(
        item,
        mask_template_equivalent=False,
        mask_fallback_reason="all_logprobs_invalid",
    )
    assert item["extra_info"]["output_token_count"] == len("good")
    assert_dense_alignment(item, ["good"])


def test_template_mismatch_falls_back_to_last_step_output_only():
    client, session_manager, _, _ = make_client(
        make_response("bad"),
        make_response("next"),
        tokenizer=TemplateMismatchTokenizer(),
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-prefix-empty", "X-Instance-Id": "inst-prefix"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-prefix-empty", "X-Instance-Id": "inst-prefix"},
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-prefix-empty").full_messages
            + [{"role": "user", "content": "again"}],
        },
    )
    assert second.status_code == 200

    client.post(
        "/session/finalize",
        json={"session_id": "sess-prefix-empty", "instance_id": "inst-prefix"},
    )
    item = client.post(
        "/trajectory/read", json={"trajectory_id": "sess-prefix-empty"}
    ).json()["data"][1]
    assert_alignment_metadata(
        item,
        mask_template_equivalent=False,
        mask_fallback_reason="mask_template_render_mismatch",
    )
    assert_dense_alignment(item, ["next"])


def test_assistant_mask_length_mismatch_falls_back_to_last_step_output_only():
    client, session_manager, _, _ = make_client(
        make_response("bad"),
        make_response("next"),
        tokenizer=AssistantMaskLengthMismatchTokenizer(),
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-prefix", "X-Instance-Id": "inst-prefix"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-prefix", "X-Instance-Id": "inst-prefix"},
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-prefix").full_messages
            + [{"role": "user", "content": "again"}],
        },
    )
    assert second.status_code == 200

    client.post(
        "/session/finalize",
        json={"session_id": "sess-prefix", "instance_id": "inst-prefix"},
    )
    item = client.post("/trajectory/read", json={"trajectory_id": "sess-prefix"}).json()[
        "data"
    ][1]
    assert_alignment_metadata(
        item,
        mask_template_equivalent=False,
        mask_fallback_reason="assistant_mask_length_mismatch",
    )
    assert_dense_alignment(item, ["next"])


def test_tools_change_splits_segment_without_marking_history_rewrite():
    tool_a = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    tool_b = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    client, session_manager, _, _ = make_client(
        make_response("alpha"),
        make_response("beta"),
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-tools-change", "X-Instance-Id": "inst-tools"},
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tool_a,
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-tools-change", "X-Instance-Id": "inst-tools"},
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-tools-change").full_messages
            + [{"role": "user", "content": "again"}],
            "tools": tool_b,
        },
    )
    assert second.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-tools-change", "instance_id": "inst-tools"},
    )
    assert finalized.status_code == 200
    assert finalized.json()["history_rewritten"] is False
    assert finalized.json()["num_segments"] == 2

    items = client.post("/trajectory/read", json={"trajectory_id": "sess-tools-change"}).json()[
        "data"
    ]
    assert items[0]["extra_info"]["segment_reason"] == "initial"
    assert items[1]["extra_info"]["segment_reason"] == "tools_changed"
    assert items[1]["extra_info"]["segment_reasons"] == ["tools_changed"]
    assert_dense_alignment(items[0], ["alpha"])
    assert_dense_alignment(items[1], ["beta"])
    assert_alignment_metadata(items[0], mask_template_equivalent=True)
    assert_alignment_metadata(items[1], mask_template_equivalent=True)


def test_rewrite_and_tools_change_prioritize_history_rewrite():
    tool_a = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    tool_b = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    client, _, _, _ = make_client(
        make_response("first"),
        make_response("second"),
    )

    first = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-rewrite-tools",
            "X-Instance-Id": "inst-rewrite-tools",
            "X-Turn-Id": "turn-1",
        },
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "step one"}],
            "tools": tool_a,
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-rewrite-tools",
            "X-Instance-Id": "inst-rewrite-tools",
            "X-Turn-Id": "turn-1",
        },
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "rewritten"}],
            "tools": tool_b,
        },
    )
    assert second.status_code == 200

    client.post(
        "/session/finalize",
        json={"session_id": "sess-rewrite-tools", "instance_id": "inst-rewrite-tools"},
    )
    items = client.post("/trajectory/read", json={"trajectory_id": "sess-rewrite-tools"}).json()[
        "data"
    ]
    assert items[1]["extra_info"]["segment_reason"] == "history_rewrite"
    assert items[1]["extra_info"]["segment_reasons"] == ["history_rewrite"]


def test_none_and_empty_tools_split_when_tokenizer_distinguishes_them():
    client, session_manager, _, _ = make_client(
        make_response("one"),
        make_response("two"),
        tokenizer=NoneVsEmptyToolsTokenizer(),
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-none-empty", "X-Instance-Id": "inst-tools"},
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-none-empty", "X-Instance-Id": "inst-tools"},
        json={
            "model": "fake-model",
            "messages": session_manager.get_session("sess-none-empty").full_messages
            + [{"role": "user", "content": "again"}],
            "tools": [],
        },
    )
    assert second.status_code == 200

    client.post(
        "/session/finalize",
        json={"session_id": "sess-none-empty", "instance_id": "inst-tools"},
    )
    items = client.post("/trajectory/read", json={"trajectory_id": "sess-none-empty"}).json()[
        "data"
    ]
    assert len(items) == 2
    assert items[1]["extra_info"]["segment_reason"] == "tools_changed"


def test_same_session_requests_are_serialized():
    blocking_client = BlockingSGLangClient(
        [make_response("first"), make_response("second")]
    )
    app = create_app(
        sglang_router_url="http://router.test",
        tokenizer=FakeTokenizer(),
        session_manager=SessionManager(),
        trajectory_store=TrajectoryStore(min_group_size=1, group_timeout=0.0),
        sglang_client=blocking_client,
    )

    async def run_test() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    headers={
                        "X-Session-Id": "sess-serial",
                        "X-Instance-Id": "inst-serial",
                    },
                    json={
                        "model": "fake-model",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            )
            assert await asyncio.to_thread(blocking_client.first_call_started.wait, 2.0)

            second = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    headers={
                        "X-Session-Id": "sess-serial",
                        "X-Instance-Id": "inst-serial",
                    },
                    json={
                        "model": "fake-model",
                        "messages": [{"role": "user", "content": "rewrite"}],
                    },
                )
            )
            await asyncio.sleep(0.2)
            assert len(blocking_client.calls) == 1

            blocking_client.release_first_call.set()
            first_response = await first
            second_response = await second
            assert first_response.status_code == 200
            assert second_response.status_code == 200
            assert len(blocking_client.calls) == 2

    asyncio.run(run_test())


def test_finalize_waits_for_inflight_request_and_reuse_returns_conflict():
    blocking_client = BlockingSGLangClient([make_response("hello")])
    app = create_app(
        sglang_router_url="http://router.test",
        tokenizer=FakeTokenizer(),
        session_manager=SessionManager(),
        trajectory_store=TrajectoryStore(min_group_size=1, group_timeout=0.0),
        sglang_client=blocking_client,
    )

    async def run_test() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            request_task = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    headers={
                        "X-Session-Id": "sess-finalized",
                        "X-Instance-Id": "inst-finalized",
                    },
                    json={
                        "model": "fake-model",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            )
            assert await asyncio.to_thread(blocking_client.first_call_started.wait, 2.0)

            finalize_task = asyncio.create_task(
                client.post(
                    "/session/finalize",
                    json={
                        "session_id": "sess-finalized",
                        "instance_id": "inst-finalized",
                    },
                )
            )
            await asyncio.sleep(0.2)
            assert not finalize_task.done()

            blocking_client.release_first_call.set()
            request_response = await request_task
            finalize_response = await finalize_task
            assert request_response.status_code == 200
            assert finalize_response.status_code == 200

            reused = await client.post(
                "/v1/chat/completions",
                headers={
                    "X-Session-Id": "sess-finalized",
                    "X-Instance-Id": "inst-finalized",
                },
                json={
                    "model": "fake-model",
                    "messages": [{"role": "user", "content": "again"}],
                },
            )
            assert reused.status_code == 409

    asyncio.run(run_test())


def test_chat_completion_returns_503_for_sglang_upstream_request_error():
    class FailingSGLangClient(FakeSGLangClient):
        async def generate(
            self,
            input_ids,
            sampling_params,
            *,
            routing_key=None,
            return_logprob=True,
        ):
            del input_ids, sampling_params, routing_key, return_logprob
            raise httpx.ConnectError("router unavailable")

    client, _, _, _ = make_client(sglang_client=FailingSGLangClient([]))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "sglang_upstream_unavailable"


def test_generation_controller_resume_uses_output_only_after_prompt_capture():
    from dressage.proxy.generation_controller import GenerationController

    first = make_response("a", finish_reason="abort", output_logprobs=[-0.11])
    first.meta_info = {"finish_reason": {"type": "abort"}, "weight_version": "v0"}
    second = make_response("b", output_logprobs=[-0.22])
    second.meta_info = {"finish_reason": {"type": "stop"}, "weight_version": "v1"}
    sglang_client = FakeSGLangClient([first, second])
    controller = GenerationController(sglang_client, partial_rollout=True)

    async def run_test():
        return await controller.generate_preemptible(
            [1, 2],
            {"max_new_tokens": 2},
            session_id="sess-preempt-logprob",
            instance_id="inst-preempt-logprob",
            turn_id="turn-preempt-logprob",
        )

    result = asyncio.run(run_test())

    assert [call["logprob_start_len"] for call in sglang_client.calls] == [0, -1]
    assert sglang_client.calls[1]["input_ids"] == [1, 2, ord("a")]
    assert result.output_ids == [ord("a"), ord("b")]
    assert result.output_token_logprobs == [-0.11, -0.22]
    assert result.output_versions == ["v0", "v1"]


def test_generation_controller_rejects_preempt_without_partial_resume():
    from dressage.proxy.generation_controller import (
        GenerationController,
        GenerationPreempted,
    )

    first = make_response("a", finish_reason="abort", output_logprobs=[-0.11])
    first.meta_info = {"finish_reason": {"type": "abort"}, "weight_version": "v0"}
    second = make_response("b", output_logprobs=[-0.22])
    second.meta_info = {"finish_reason": {"type": "stop"}, "weight_version": "v1"}
    sglang_client = FakeSGLangClient([first, second])
    controller = GenerationController(sglang_client)

    async def run_test():
        try:
            await controller.generate_preemptible(
                [1, 2],
                {"max_new_tokens": 2},
                session_id="sess-preempt-disabled",
                instance_id="inst-preempt-disabled",
                turn_id="turn-preempt-disabled",
            )
        except GenerationPreempted:
            return
        raise AssertionError("Expected preempted generation to be rejected")

    asyncio.run(run_test())

    assert len(sglang_client.calls) == 1


def test_generation_controller_rejects_stale_epoch_before_sglang():
    from dressage.proxy.generation_controller import (
        GenerationController,
        GenerationStaleEpoch,
    )

    sglang_client = FakeSGLangClient([make_response("late")])
    controller = GenerationController(sglang_client)

    async def run_test():
        await controller.pause(reason="weight_update")
        task = asyncio.create_task(
            controller.generate_preemptible(
                [1],
                {"max_new_tokens": 1},
                session_id="sess-stale-epoch",
                instance_id="inst-stale-epoch",
                turn_id="turn-stale-epoch",
                expected_epoch=0,
            )
        )
        await asyncio.sleep(0)
        assert task.done() is False
        await controller.resume(reason="weight_update")
        try:
            await task
        except GenerationStaleEpoch as exc:
            assert exc.expected_epoch == 0
            assert exc.current_epoch == 1
            return
        raise AssertionError("Expected stale epoch generation to be rejected")

    asyncio.run(run_test())

    assert len(sglang_client.calls) == 0


def test_generation_controller_resume_keeps_full_until_prompt_capture():
    from dressage.proxy.generation_controller import GenerationController

    no_partial = make_response("", finish_reason="abort", prompt_logprobs=[])
    no_partial.meta_info = {"finish_reason": {"type": "abort"}, "weight_version": "v0"}
    recovered = make_response("z", output_logprobs=[-0.33])
    recovered.meta_info = {"finish_reason": {"type": "stop"}, "weight_version": "v1"}
    sglang_client = FakeSGLangClient([no_partial, recovered])
    controller = GenerationController(sglang_client, partial_rollout=True)

    async def run_test():
        return await controller.generate_preemptible(
            [1, 2],
            {"max_new_tokens": 1},
            session_id="sess-preempt-no-payload",
            instance_id="inst-preempt-no-payload",
            turn_id="turn-preempt-no-payload",
        )

    result = asyncio.run(run_test())

    assert [call["logprob_start_len"] for call in sglang_client.calls] == [0, 0]
    assert result.input_token_logprobs_raw == [-0.1, -0.1]
    assert result.output_ids == [ord("z")]
    assert result.output_token_logprobs == [-0.33]


def test_generation_controller_output_only_resume_stays_output_only():
    from dressage.proxy.generation_controller import GenerationController

    first = make_response("a", finish_reason="abort", output_logprobs=[-0.11])
    first.meta_info = {"finish_reason": {"type": "abort"}, "weight_version": "v0"}
    second = make_response("b", output_logprobs=[-0.22])
    second.meta_info = {"finish_reason": {"type": "stop"}, "weight_version": "v1"}
    sglang_client = FakeSGLangClient([first, second])
    controller = GenerationController(sglang_client, partial_rollout=True)

    async def run_test():
        return await controller.generate_preemptible(
            [1, 2],
            {"max_new_tokens": 2},
            session_id="sess-output-only-preempt",
            instance_id="inst-output-only-preempt",
            turn_id="turn-output-only-preempt",
            logprob_start_len=-1,
        )

    result = asyncio.run(run_test())

    assert [call["logprob_start_len"] for call in sglang_client.calls] == [-1, -1]
    assert result.input_token_logprobs_raw == [0.0, 0.0]
    assert result.output_token_logprobs == [-0.11, -0.22]


def test_generation_controller_context_window_overflow_stops_partial_resume():
    from dressage.proxy.generation_controller import GenerationController

    first = make_response("abc", finish_reason="abort")
    first.meta_info = {"finish_reason": {"type": "abort"}, "weight_version": "v0"}
    second = make_response("z")
    sglang_client = FakeSGLangClient([first, second])
    controller = GenerationController(sglang_client, partial_rollout=True)

    async def run_test():
        return await controller.generate_preemptible(
            [1, 2],
            {"max_new_tokens": 10},
            session_id="sess-partial-overflow",
            instance_id="inst-partial-overflow",
            turn_id="turn-partial-overflow",
            context_window=4,
        )

    result = asyncio.run(run_test())

    assert len(sglang_client.calls) == 1
    assert result.output_ids == [ord("a"), ord("b"), ord("c")]
    assert result.finish_reason == "length"
    assert result.meta_info["context_overflow"]["phase"] == "input_output"
    assert result.meta_info["context_overflow"]["total_tokens"] == 5


def test_generation_controller_shutdown_rejects_new_generations():
    from dressage.proxy.generation_controller import GenerationController, ProxyShuttingDown

    controller = GenerationController(FakeSGLangClient([make_response("ok")]))

    async def run_test() -> None:
        result = await controller.shutdown(timeout_seconds=0.1)
        assert result["status"] == "shutting_down"
        assert controller.state()["paused"] is True
        assert controller.state()["shutting_down"] is True
        try:
            await controller.generate_preemptible(
                [1],
                {"max_new_tokens": 1},
                session_id="sess-shutdown",
                instance_id="inst-shutdown",
                turn_id="turn-shutdown",
            )
        except ProxyShuttingDown:
            return
        raise AssertionError("Expected shutdown controller to reject generation")

    asyncio.run(run_test())
