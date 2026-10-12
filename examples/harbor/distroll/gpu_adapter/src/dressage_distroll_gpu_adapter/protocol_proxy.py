"""Pinned Dressage protocol adapter; does not change tokens or TQ ownership.

The upstream converter stops before the standalone usage frame and reuses a
message ID across calls. Keep these fixes at the supported proxy_factory seam.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from uuid import uuid4

from blackbox_server.proxy.rollout_llm_proxy import (
    RolloutLLMProxy,
    _anthropic_sse,
    _AnthropicStreamState,
    _first_openai_choice,
    _iter_openai_sse_payloads,
    _openai_finish_reason_to_anthropic,
    _openai_usage_to_anthropic_usage,
)

from .runtime_debug import event


class _StreamState(_AnthropicStreamState):
    def __init__(self):
        super().__init__()
        self.message_id = "msg_" + uuid4().hex

    def message_start(self, *, model, usage=None):
        original = super().message_start(model=model)
        payload = json.loads(original.split(b"data: ", 1)[1])
        payload["message"]["id"] = self.message_id
        payload["message"]["usage"] = {
            "input_tokens": (usage or {}).get("input_tokens", 0),
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        return _anthropic_sse("message_start", payload)

    async def finish(self, *, stop_reason, usage):
        async for frame in super().finish(stop_reason=stop_reason, usage=usage):
            if frame.startswith(b"event: message_delta\n"):
                payload = json.loads(frame.split(b"data: ", 1)[1])
                payload["usage"] = usage
                frame = _anthropic_sse("message_delta", payload)
            yield frame


class AgentProtocolProxy(RolloutLLMProxy):
    def _anthropic_messages_to_openai_chat_completion(self, payload):
        messages = payload.get("messages") or []
        roles, blocks = Counter(), Counter()
        uses, results = [], []
        # Counts only: no prompt, tool input/output, API key, or raw tool IDs.
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            roles[role if role in {"user", "assistant", "system"} else "other"] += 1
            content = message.get("content")
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                blocks[
                    kind
                    if kind in {"text", "thinking", "tool_use", "tool_result"}
                    else "other"
                ] += 1
                if kind == "tool_use" and isinstance(block.get("id"), str):
                    uses.append(block["id"])
                if kind == "tool_result" and isinstance(block.get("tool_use_id"), str):
                    results.append(block["tool_use_id"])
        event(
            "agent_request_structure",
            session_id=self.bound_session_id,
            message_count=len(messages),
            roles=dict(roles),
            blocks=dict(blocks),
            duplicate_tool_use_ids=len(uses) - len(set(uses)),
            unmatched_tool_results=len(set(results) - set(uses)),
        )
        return super()._anthropic_messages_to_openai_chat_completion(payload)

    def _openai_chat_completion_to_anthropic_message(self, payload):
        result = super()._openai_chat_completion_to_anthropic_message(payload)
        result["id"] = "msg_" + uuid4().hex
        event(
            "agent_nonstream_response",
            session_id=self.bound_session_id,
            message_id=result["id"],
            **result["usage"],
        )
        return result

    async def _iter_anthropic_events_from_openai_stream(self, first_bytes, byte_iter):
        state = _StreamState()
        started = time.monotonic()
        usage = None
        finish = None
        done = False
        chunks = 0
        frames = []
        buffered_bytes = 0
        limit = int(os.environ.get("POLICY_MAX_MODEL_RESPONSE_BYTES", "8388608"))

        def buffer(frame):
            nonlocal buffered_bytes
            buffered_bytes += len(frame)
            if buffered_bytes > limit:
                raise ValueError("Anthropic response exceeds bounded buffer")
            frames.append(frame)

        outcome = "aborted"
        identity = {"session_id": self.bound_session_id, "message_id": state.message_id}
        event("agent_stream_start", **identity)
        try:
            # The backend is a completed-turn pseudo-stream, and Policy already
            # buffers complete replies for replay. Read usage before emitting
            # message_start so Claude's retained assistant messages see it too.
            async for payload in _iter_openai_sse_payloads(first_bytes, byte_iter):
                if payload == "[DONE]":
                    done = True
                    break
                chunk = json.loads(payload)
                if not isinstance(chunk, dict) or "error" in chunk:
                    raise ValueError("invalid or failed upstream SSE response")
                chunks += 1
                if chunk.get("model"):
                    state.model = str(chunk["model"])
                if isinstance(chunk.get("usage"), dict):
                    raw_usage = chunk["usage"]
                    if not (
                        "prompt_tokens" in raw_usage or "input_tokens" in raw_usage
                    ):
                        raise ValueError("upstream usage has no input token count")
                    if not (
                        "completion_tokens" in raw_usage or "output_tokens" in raw_usage
                    ):
                        raise ValueError("upstream usage has no output token count")
                    usage = _openai_usage_to_anthropic_usage(raw_usage)
                    if any(value < 0 for value in usage.values()):
                        raise ValueError("negative upstream token count")
                choice = _first_openai_choice(chunk)
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    async for frame in state.emit_text_like_block(
                        "thinking",
                        str(reasoning),
                        delta_type="thinking_delta",
                        delta_key="thinking",
                    ):
                        buffer(frame)
                if delta.get("content"):
                    async for frame in state.emit_text_like_block(
                        "text",
                        str(delta["content"]),
                        delta_type="text_delta",
                        delta_key="text",
                    ):
                        buffer(frame)
                for tool in delta.get("tool_calls") or []:
                    if isinstance(tool, dict):
                        async for frame in state.emit_tool_call_delta(tool):
                            buffer(frame)
                if choice.get("finish_reason") is not None:
                    finish = _openai_finish_reason_to_anthropic(choice["finish_reason"])
            # This adapter targets Dressage's pinned pseudo-stream, which always
            # emits all three. Never turn a truncated response into success.
            if not done or finish is None or usage is None:
                raise ValueError(
                    "incomplete upstream stream: finish/usage/DONE required"
                )
            yield state.message_start(model="proxy-model", usage=usage)
            for frame in frames:
                yield frame
            async for frame in state.finish(stop_reason=finish, usage=usage):
                yield frame
            outcome = "complete"
        except BaseException as exc:
            outcome = type(exc).__name__
            raise
        finally:
            event(
                "agent_stream_end",
                **identity,
                outcome=outcome,
                elapsed_sec=round(time.monotonic() - started, 3),
                chunks=chunks,
                finish_seen=finish is not None,
                usage_seen=usage is not None,
                done_seen=done,
                input_tokens=(usage or {}).get("input_tokens"),
                output_tokens=(usage or {}).get("output_tokens"),
            )
