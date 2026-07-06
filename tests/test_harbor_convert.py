from __future__ import annotations

import pytest


def _segment(
    *,
    index: int,
    messages: list[dict],
    tools: list[dict] | None = None,
    reason: str = "initial",
    count: int = 2,
) -> dict:
    return {
        "uid": f"segment-{index}",
        "session_id": "bbs-session",
        "trajectory_id": "bbs-trajectory",
        "instance_id": "instance-1",
        "turn_id": f"turn-{index}",
        "segment_index": index,
        "segment_count": count,
        "messages": messages,
        "tools": tools or [],
        "tokens": [index, index + 1],
        "full_loss_mask": [0, 1],
        "full_logprobs": [0.0, -0.1],
        "finish_reason": "stop",
        "extra_info": {
            "segment_reason": reason,
            "segment_reasons": [reason],
        },
    }


def test_inline_trajectory_payload_matches_exact_atif() -> None:
    from dressage.rollout.artifacts.harbor_convert import trajectory_payload_to_harbor

    segment = _segment(
        index=0,
        count=1,
        messages=[
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    )

    actual = trajectory_payload_to_harbor(
        {
            "success": True,
            "mode": "trajectory",
            "drained": True,
            "data": [segment],
        },
        agent={"name": "dressage", "version": "test"},
    )

    assert actual == {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "dressage", "version": "test"},
        "steps": [
            {"source": "system", "message": "rules", "step_id": 1},
            {"source": "user", "message": "question", "step_id": 2},
            {"source": "agent", "message": "answer", "step_id": 3},
        ],
        "final_metrics": {"total_steps": 3},
        "extra": {
            "dressage": {
                "payload": {
                    "success": True,
                    "mode": "trajectory",
                    "drained": True,
                },
                "segments": [
                    {
                        "uid": "segment-0",
                        "segment_index": 0,
                        "segment_count": 1,
                        "turn_id": "turn-0",
                        "finish_reason": "stop",
                        "extra_info": {
                            "segment_reason": "initial",
                            "segment_reasons": ["initial"],
                        },
                        "tokens": [0, 1],
                        "full_loss_mask": [0, 1],
                        "full_logprobs": [0.0, -0.1],
                        "tool_definitions": [],
                    }
                ],
                "instance_id": "instance-1",
            }
        },
        "session_id": "bbs-session",
        "trajectory_id": "bbs-trajectory",
    }


def test_tool_calls_observations_and_invalid_arguments_are_schema_safe() -> None:
    from dressage.rollout.artifacts.harbor_convert import segments_to_harbor_trajectory

    messages = [
        {"role": "developer", "content": {"policy": "safe"}},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "inspect first",
            "tool_calls": [
                {
                    "id": "call-ok",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"path":"README.md"}',
                    },
                },
                {
                    "type": "function",
                    "function": {"name": "broken", "arguments": "not-json"},
                },
                {
                    "id": "call-list",
                    "function": {"name": "list_arg", "arguments": [1, 2]},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call-ok", "content": "contents"},
        {"role": "tool", "tool_call_id": "missing", "content": {"ok": False}},
    ]

    result = segments_to_harbor_trajectory(
        [_segment(index=0, messages=messages)],
        agent={"name": "dressage", "version": "test"},
    )

    developer, agent_step = result["steps"]
    assert developer == {
        "step_id": 1,
        "source": "system",
        "message": '{"policy":"safe"}',
        "extra": {
            "dressage": {
                "original_role": "developer",
                "original_content": {"policy": "safe"},
            }
        },
    }
    assert agent_step["message"] == ""
    assert agent_step["reasoning_content"] == "inspect first"
    assert agent_step["tool_calls"][0] == {
        "tool_call_id": "call-ok",
        "function_name": "read",
        "arguments": {"path": "README.md"},
    }
    assert agent_step["tool_calls"][1]["tool_call_id"] == "dressage-0-1-1"
    assert agent_step["tool_calls"][1]["arguments"] == {}
    assert agent_step["tool_calls"][1]["extra"]["dressage"]["raw_arguments"] == "not-json"
    assert agent_step["tool_calls"][2]["arguments"] == {}
    assert agent_step["tool_calls"][2]["extra"]["dressage"]["raw_arguments"] == [1, 2]
    assert agent_step["observation"]["results"] == [
        {"source_call_id": "call-ok", "content": "contents"},
        {
            "content": '{"ok":false}',
            "extra": {
                "dressage": {
                    "original_tool_call_id": "missing",
                    "original_content": {"ok": False},
                }
            },
        },
    ]


def test_orphan_tool_message_becomes_system_observation() -> None:
    from dressage.rollout.artifacts.harbor_convert import segments_to_harbor_trajectory

    result = segments_to_harbor_trajectory(
        [
            _segment(
                index=0,
                messages=[
                    {"role": "tool", "tool_call_id": "orphan", "content": "result"}
                ],
            )
        ],
        agent={"name": "dressage", "version": "test"},
    )

    assert result["steps"] == [
        {
            "step_id": 1,
            "source": "system",
            "message": "Tool observation",
            "observation": {
                "results": [
                    {
                        "content": "result",
                        "extra": {
                            "dressage": {"original_tool_call_id": "orphan"}
                        },
                    }
                ]
            },
            "extra": {"dressage": {"original_role": "tool"}},
        }
    ]


def test_multi_segment_expands_boundaries_and_marks_only_identical_context() -> None:
    from dressage.rollout.artifacts.harbor_convert import segments_to_harbor_trajectory

    first_messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "draft"},
    ]
    second_messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "rewritten question"},
        {"role": "assistant", "content": "draft"},
    ]

    result = segments_to_harbor_trajectory(
        [
            _segment(index=1, messages=second_messages, reason="history_rewrite"),
            _segment(index=0, messages=first_messages),
        ],
        agent={"name": "dressage", "version": "test"},
    )

    assert [step["step_id"] for step in result["steps"]] == list(range(1, 8))
    boundary = result["steps"][3]
    assert boundary["source"] == "system"
    assert boundary["extra"]["dressage"] == {
        "synthetic_segment_boundary": True,
        "segment_index": 1,
        "segment_uid": "segment-1",
        "segment_reason": "history_rewrite",
        "segment_reasons": ["history_rewrite"],
    }
    assert "context_management" not in boundary["extra"]
    assert result["steps"][4]["is_copied_context"] is True
    assert "is_copied_context" not in result["steps"][5]
    assert result["steps"][6]["is_copied_context"] is True
    assert result["final_metrics"] == {"total_steps": 7}

    dressage_extra = result["extra"]["dressage"]
    assert dressage_extra["instance_id"] == "instance-1"
    assert dressage_extra["segments"][0]["tokens"] == [0, 1]
    assert dressage_extra["segments"][1]["extra_info"]["segment_reason"] == "history_rewrite"


def test_payload_metadata_and_explicit_identity_take_precedence() -> None:
    from dressage.rollout.artifacts.harbor_convert import trajectory_payload_to_harbor

    payload = {
        "success": True,
        "mode": "trajectory",
        "drained": True,
        "meta_info": {"source": "proxy"},
        "data": [_segment(index=0, messages=[{"role": "user", "content": "hi"}])],
    }
    result = trajectory_payload_to_harbor(
        payload,
        session_id="explicit-session",
        instance_id="explicit-instance",
        agent={"name": "custom", "version": "1.2.3", "model_name": "model"},
    )

    assert result["session_id"] == "explicit-session"
    assert result["trajectory_id"] == "bbs-trajectory"
    assert result["agent"] == {
        "name": "custom",
        "version": "1.2.3",
        "model_name": "model",
    }
    assert result["extra"]["dressage"]["instance_id"] == "explicit-instance"
    assert result["extra"]["dressage"]["payload"] == {
        "success": True,
        "mode": "trajectory",
        "drained": True,
        "meta_info": {"source": "proxy"},
    }


def test_default_agent_uses_package_version_and_latest_tools(monkeypatch) -> None:
    from dressage.rollout.artifacts import harbor_convert

    monkeypatch.setattr(harbor_convert.metadata, "version", lambda _name: "9.8.7")
    first_tool = {
        "type": "function",
        "function": {"name": "first", "parameters": {}},
    }
    latest_tool = {
        "type": "function",
        "function": {"name": "latest", "parameters": {}},
    }

    result = harbor_convert.segments_to_harbor_trajectory(
        [
            _segment(
                index=0,
                messages=[{"role": "user", "content": "one"}],
                tools=[first_tool],
            ),
            _segment(
                index=1,
                messages=[{"role": "user", "content": "two"}],
                tools=[latest_tool],
            ),
        ]
    )

    assert result["agent"] == {
        "name": "dressage",
        "version": "9.8.7",
        "tool_definitions": [latest_tool],
    }
    assert result["extra"]["dressage"]["segments"][0]["tool_definitions"] == [first_tool]


def test_default_agent_falls_back_when_package_metadata_is_unavailable(monkeypatch) -> None:
    from dressage.rollout.artifacts import harbor_convert

    def missing(_name: str) -> str:
        raise harbor_convert.metadata.PackageNotFoundError

    monkeypatch.setattr(harbor_convert.metadata, "version", missing)

    result = harbor_convert.segments_to_harbor_trajectory(
        [_segment(index=0, messages=[{"role": "user", "content": "hi"}])]
    )

    assert result["agent"]["version"] == "unknown"


@pytest.mark.parametrize(
    "value",
    [
        {"data": []},
        {"data": [_segment(index=0, messages=[])]},
    ],
)
def test_empty_trajectory_is_rejected(value: dict) -> None:
    from dressage.rollout.artifacts.harbor_convert import trajectory_payload_to_harbor

    with pytest.raises(ValueError):
        trajectory_payload_to_harbor(value)
