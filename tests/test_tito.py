from __future__ import annotations

import json

from jinja2 import Environment

from dressage.proxy.tito import Qwen35TITOTokenizer, load_fixed_template


class TinyTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict=False,
        tools=None,
        **_,
    ):
        del return_dict
        del tools
        rendered = ""
        for message in messages:
            role = message.get("role", "unknown")
            rendered += f"<{role}>"
            content = message.get("content")
            if content is not None:
                rendered += str(content)
            for tool_call in message.get("tool_calls", []) or []:
                arguments = tool_call["function"].get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                rendered += "<tool_call>"
                rendered += json.dumps(
                    {
                        "name": tool_call["function"]["name"],
                        "arguments": arguments,
                    }
                )
                rendered += "</tool_call>"
            if "tool_call_id" in message:
                rendered += (
                    "<tool_call_id>"
                    + str(message["tool_call_id"])
                    + "</tool_call_id>"
                )
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return [ord(character) for character in rendered]
        return rendered

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def convert_tokens_to_ids(self, token):
        if token == "<|im_end|>":
            return 1
        return None


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
        **template_vars,
    ):
        del return_dict

        def raise_exception(message):
            raise RuntimeError(message)

        rendered = Environment().from_string(self.chat_template).render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            raise_exception=raise_exception,
            **template_vars,
        )
        if tokenize:
            return [ord(character) for character in rendered]
        return rendered

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def convert_tokens_to_ids(self, token):
        del token
        return None


def decode(token_ids: list[int]) -> str:
    return "".join(chr(token_id) for token_id in token_ids)


def test_fixed_template_loads_without_user_query_guard():
    template = load_fixed_template("qwen3_5")
    assert "No user query found in messages" not in template
    assert "<tool_response>" in template


def test_user_segment_incremental():
    tokenizer = Qwen35TITOTokenizer(TinyTokenizer())
    old_messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    new_messages = old_messages + [{"role": "user", "content": "again"}]

    token_ids = tokenizer.tokenize_additional_non_assistant(
        old_messages,
        new_messages,
    )

    assert decode(token_ids) == "<user>again<assistant>"


def test_tool_segment_incremental():
    tokenizer = Qwen35TITOTokenizer(TinyTokenizer())
    old_messages = [
        {"role": "user", "content": "find"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-search",
                    "type": "function",
                    "function": {"name": "search", "arguments": {"q": "x"}},
                }
            ],
        },
    ]
    new_messages = old_messages + [
        {
            "role": "tool",
            "tool_call_id": "call-search",
            "content": "result",
        }
    ]

    token_ids = tokenizer.tokenize_additional_non_assistant(
        old_messages,
        new_messages,
    )
    rendered = decode(token_ids)

    assert "<tool>result" in rendered
    assert "<tool_call_id>call-search</tool_call_id>" in rendered
    assert rendered.endswith("<assistant>")


def test_system_append_rejected():
    tokenizer = Qwen35TITOTokenizer(TinyTokenizer())

    try:
        tokenizer.tokenize_additional_non_assistant(
            [{"role": "user", "content": "hi"}],
            [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "late system"},
            ],
        )
    except ValueError as exc:
        assert "role='system'" in str(exc)
    else:
        raise AssertionError("Expected system append to be rejected")


def test_merge_inserts_newline_after_im_end():
    tokenizer = Qwen35TITOTokenizer(TinyTokenizer())
    merged = tokenizer.merge_tokens(
        old_messages=[{"role": "user", "content": "hi"}],
        new_messages=[
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "again"},
        ],
        pretokenized_token_ids=[1],
    )

    assert merged[:2] == [1, ord("\n")]


def test_merge_empty_prefix():
    tokenizer = Qwen35TITOTokenizer(TinyTokenizer())
    merged = tokenizer.merge_tokens(
        old_messages=[],
        new_messages=[{"role": "user", "content": "hi"}],
        pretokenized_token_ids=[],
    )

    assert decode(merged) == "<user>hi<assistant>"


def test_merge_tokens_preserves_thinking_for_appended_tool_and_user():
    tokenizer = Qwen35TITOTokenizer(QwenJinjaTokenizer())
    old_messages = [
        {"role": "user", "content": "root task"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Need to report.",
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
    old_rendered = tokenizer._render_messages(
        old_messages,
        add_generation_prompt=False,
    )

    merged = tokenizer.merge_tokens(
        old_messages=old_messages,
        new_messages=new_messages,
        pretokenized_token_ids=[ord(character) for character in old_rendered],
    )
    full_rendered = tokenizer._render_messages(
        new_messages,
        add_generation_prompt=True,
    )

    assert decode(merged) == full_rendered
