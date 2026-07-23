"""Session tracking for proxy-mediated agent conversations."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from .reasoning_parser import canonicalize_reasoning_content
from .tool_call_ids import canonicalize_openclaw_tool_call_id

IMPLICIT_TURN_ID_PREFIX = "__implicit_turn__:"


logger = logging.getLogger(__name__)


class SessionFinalizedError(RuntimeError):
    """Raised when a caller attempts to reuse a finalized session."""


RouteType = Literal["append", "branch", "create"]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


@dataclass
class Lineage:
    id: str
    index: int
    latest_step_id: str | None
    branch_from_step_id: str | None = None


@dataclass(frozen=True)
class Route:
    lineage_id: str
    type: RouteType
    base_step_id: str | None


class SessionPrefixTree:
    """Message-prefix candidate index for lineage routing.

    The tree only narrows the candidate set.  Route correctness is decided by
    rendered chat-template prefix equality in ``server.py``.
    """

    def __init__(self) -> None:
        self._by_fingerprint: dict[str, set[str]] = {}
        self._fingerprint_by_step_id: dict[str, str] = {}
        self._all_step_ids: list[str] = []

    @staticmethod
    def fingerprint(messages: list[dict[str, Any]]) -> str:
        payload = _canonical_json(messages)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def insert(self, step: "StepRecord") -> None:
        fingerprint = self.fingerprint(step.normalized_messages_snapshot)
        self._fingerprint_by_step_id[step.step_id] = fingerprint
        self._by_fingerprint.setdefault(fingerprint, set()).add(step.step_id)
        self._all_step_ids.append(step.step_id)

    def candidates(self, current_messages: list[dict[str, Any]]) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for prefix_len in range(len(current_messages) + 1):
            fingerprint = self.fingerprint(current_messages[:prefix_len])
            for step_id in self._by_fingerprint.get(fingerprint, set()):
                if step_id in seen:
                    continue
                seen.add(step_id)
                candidates.append(step_id)
        for step_id in self._all_step_ids:
            if step_id in seen:
                continue
            seen.add(step_id)
            candidates.append(step_id)
        return candidates


@dataclass
class StepRecord:
    """One assistant generation step captured by the proxy."""

    turn_id: str
    request_messages: list[dict]
    normalized_request_messages: list[dict]
    prompt_token_ids: list[int]
    prompt_token_logprobs: list[float]
    snapshot_token_ids: list[int]
    response_token_ids: list[int]
    response_logprobs: list[float]
    all_token_ids: list[int]
    all_logprobs: list[float]
    input_token_texts: list[str]
    output_token_texts: list[str]
    messages_snapshot: list[dict]
    raw_response_text: str
    step_id: str = ""
    lineage_id: str = ""
    lineage_index: int = 0
    route_type: RouteType = "create"
    route_base_step_id: str | None = None
    normalized_messages_snapshot: list[dict] = field(default_factory=list)
    snapshot_rendered: str = ""
    snapshot_rendered_len: int = 0
    snapshot_tools_hash: str | None = None
    lineage_segment_boundary_before: bool = False
    lineage_segment_reasons_before: list[str] = field(default_factory=list)
    prompt_versions: list[str] = field(default_factory=list)
    response_versions: list[str] = field(default_factory=list)
    all_versions: list[str] = field(default_factory=list)
    all_logprobs_invalid: bool = False
    concat_token_ids: list[int] = field(default_factory=list)
    concat_response_logprobs: list[float] = field(default_factory=list)
    concat_response_mask: list[int] = field(default_factory=list)
    concat_versions: list[str] = field(default_factory=list)
    concat_context_token_count: int = 0
    concat_output_token_count: int = 0
    concat_logprobs_invalid: bool = False
    concat_incremental_tokenization_failed: bool = False
    response_routed_experts: str | None = None
    response_routed_experts_chunks: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    segment_boundary_before: bool = False
    rewrite_reason: str | None = None
    segment_reason_before: str | None = None
    segment_reasons_before: list[str] = field(default_factory=list)
    finish_reason: str = "stop"
    request_version: str | None = None
    response_version: str | None = None
    timestamp: float = field(default_factory=time.time)


TurnRecord = StepRecord


@dataclass
class Session:
    """Conversation session keyed by canonical ``session_id``."""

    session_id: str
    instance_id: str
    steps: list[StepRecord] = field(default_factory=list)
    history_rewritten: bool = False
    rewrite_reason: str | None = None
    rewrite_detected_at: float | None = None
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    turn_mode: Literal["implicit", "explicit"] | None = None
    implicit_turn_id: str | None = None
    turn_ids: list[str] = field(default_factory=list)
    active_turn_id: str | None = None
    rollout_epoch: int | None = None
    lineages: dict[str, Lineage] = field(default_factory=dict)
    steps_by_id: dict[str, StepRecord] = field(default_factory=dict)
    prefix_tree: SessionPrefixTree = field(default_factory=SessionPrefixTree)
    request_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _turn_counter: int = field(default=0, repr=False)
    _step_counter: int = field(default=0, repr=False)
    _lineage_counter: int = field(default=0, repr=False)

    def next_turn_id(self) -> str:
        self._turn_counter += 1
        return str(self._turn_counter)

    def next_step_id(self) -> str:
        self._step_counter += 1
        return f"step-{self._step_counter:06d}"

    def create_lineage(self, *, branch_from_step_id: str | None = None) -> Lineage:
        self._lineage_counter += 1
        lineage = Lineage(
            id=f"lineage-{self._lineage_counter:06d}",
            index=self._lineage_counter - 1,
            latest_step_id=None,
            branch_from_step_id=branch_from_step_id,
        )
        self.lineages[lineage.id] = lineage
        return lineage

    @property
    def turns(self) -> list[StepRecord]:
        return self.steps

    @property
    def full_messages(self) -> list[dict]:
        if not self.steps:
            return []
        return self.steps[-1].messages_snapshot

    @property
    def latest_tools(self) -> list[dict[str, Any]] | None:
        for step in reversed(self.steps):
            if step.tools is not None:
                return step.tools
        return None

    @property
    def latest_step(self) -> StepRecord | None:
        if not self.steps:
            return None
        return self.steps[-1]


class SessionManager:
    """Thread-safe manager for multi-turn conversations."""

    def __init__(self, session_timeout: float = 3200.0):
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._finalized_session_ids: dict[str, float] = {}
        self._finalization_results: dict[str, dict[str, Any]] = {}
        self._session_timeout = session_timeout

    @staticmethod
    def _normalize_tool_call_arguments(arguments: Any) -> Any:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return arguments
        elif (
            not isinstance(arguments, (dict, list, int, float, bool))
            and arguments is not None
        ):
            return arguments

        try:
            return json.dumps(
                arguments,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except (TypeError, ValueError):
            return arguments

    @classmethod
    def _normalize_tool_call(cls, tool_call: Any) -> Any:
        if not isinstance(tool_call, dict):
            return tool_call

        normalized: dict[str, Any] = {
            "id": canonicalize_openclaw_tool_call_id(tool_call.get("id")),
            "type": tool_call.get("type"),
        }
        function = tool_call.get("function")
        if isinstance(function, dict):
            normalized["function"] = {
                "name": function.get("name"),
                "arguments": cls._normalize_tool_call_arguments(
                    function.get("arguments")
                ),
            }
        else:
            normalized["function"] = function
        return normalized

    @classmethod
    def _normalize_tool_calls(cls, tool_calls: Any) -> Any:
        if not isinstance(tool_calls, list):
            return tool_calls
        return [cls._normalize_tool_call(tool_call) for tool_call in tool_calls]

    @staticmethod
    def _has_tool_calls(message: dict) -> bool:
        tool_calls = message.get("tool_calls")
        return isinstance(tool_calls, list) and bool(tool_calls)

    @classmethod
    def _normalize_message(cls, message: dict) -> dict:
        content = message.get("content")
        if cls._has_tool_calls(message) and content == "":
            content = None
        normalized = {
            "role": message.get("role"),
            "content": content,
        }
        if "name" in message:
            normalized["name"] = message["name"]
        if "reasoning_content" in message:
            reasoning_content = canonicalize_reasoning_content(
                message["reasoning_content"]
            )
            if reasoning_content is not None:
                normalized["reasoning_content"] = reasoning_content
        if "tool_call_id" in message:
            normalized["tool_call_id"] = canonicalize_openclaw_tool_call_id(
                message["tool_call_id"]
            )
        if "tool_calls" in message:
            normalized["tool_calls"] = cls._normalize_tool_calls(
                message["tool_calls"]
            )
        return normalized

    def is_append_only_continuation(
        self,
        previous_messages: list[dict],
        current_messages: list[dict],
    ) -> bool:
        if len(current_messages) < len(previous_messages):
            return False
        prev_norm = [self._normalize_message(msg) for msg in previous_messages]
        curr_norm = [
            self._normalize_message(msg) for msg in current_messages[: len(previous_messages)]
        ]
        return prev_norm == curr_norm

    def get_or_create_session(
        self,
        session_id: str | None,
        messages: list[dict],
        instance_id: str | None = None,
    ) -> tuple[Session, bool]:
        del messages  # Reserved for future session bootstrapping heuristics.

        with self._lock:
            self._cleanup_expired_locked()
            if session_id and session_id in self._finalized_session_ids:
                raise SessionFinalizedError(f"Session {session_id} has already been finalized.")
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_active = time.time()
                if instance_id and instance_id != session.instance_id:
                    session.instance_id = instance_id
                return session, False

            sid = session_id or str(uuid.uuid4())
            iid = instance_id or str(uuid.uuid4())
            session = Session(session_id=sid, instance_id=iid)
            self._sessions[sid] = session
            return session, True

    def ensure_session_active(self, session_id: str, session: Session) -> None:
        with self._lock:
            self._cleanup_expired_locked()
            if self._finalized_session_ids.get(session_id) is not None:
                raise SessionFinalizedError(f"Session {session_id} has already been finalized.")
            current = self._sessions.get(session_id)
            if current is not session:
                raise SessionFinalizedError(f"Session {session_id} is no longer active.")

    @staticmethod
    def _is_reserved_turn_id(turn_id: str) -> bool:
        return turn_id.startswith(IMPLICIT_TURN_ID_PREFIX)

    def resolve_turn_id(self, *, session_id: str, requested_turn_id: str | None) -> str:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")

            now = time.time()
            session.last_active = now

            if requested_turn_id is None:
                if session.turn_mode == "explicit":
                    raise ValueError(
                        "turn_id is required once the session starts using explicit turns."
                    )
                if session.turn_mode is None:
                    session.turn_mode = "implicit"
                    session.implicit_turn_id = (
                        f"{IMPLICIT_TURN_ID_PREFIX}{uuid.uuid4().hex}"
                    )
                    session.turn_ids.append(session.implicit_turn_id)
                    session.active_turn_id = session.implicit_turn_id
                return session.implicit_turn_id or ""

            if self._is_reserved_turn_id(requested_turn_id):
                raise ValueError("turn_id uses a reserved implicit-turn prefix.")
            if session.turn_mode == "implicit":
                raise ValueError(
                    "Cannot provide explicit turn_id after the session started in implicit single-turn mode."
                )

            session.turn_mode = "explicit"
            if session.active_turn_id is None:
                session.turn_ids.append(requested_turn_id)
                session.active_turn_id = requested_turn_id
                return requested_turn_id

            if requested_turn_id == session.active_turn_id:
                return requested_turn_id

            if requested_turn_id in session.turn_ids:
                raise ValueError(
                    "Cannot return to a previous turn_id after the session has advanced to a newer turn."
                )

            session.turn_ids.append(requested_turn_id)
            session.active_turn_id = requested_turn_id
            return requested_turn_id

    def record_step(
        self,
        *,
        session_id: str,
        turn_id: str,
        request_messages: list[dict],
        normalized_request_messages: list[dict],
        prompt_token_ids: list[int],
        prompt_token_logprobs: list[float],
        snapshot_token_ids: list[int],
        response_token_ids: list[int],
        response_logprobs: list[float],
        response_versions: list[str] | None = None,
        all_token_ids: list[int],
        all_logprobs: list[float],
        all_versions: list[str] | None = None,
        prompt_versions: list[str] | None = None,
        input_token_texts: list[str],
        output_token_texts: list[str],
        messages: list[dict],
        raw_response_text: str,
        all_logprobs_invalid: bool = False,
        concat_token_ids: list[int] | None = None,
        concat_response_logprobs: list[float] | None = None,
        concat_response_mask: list[int] | None = None,
        concat_versions: list[str] | None = None,
        concat_context_token_count: int = 0,
        concat_output_token_count: int = 0,
        concat_logprobs_invalid: bool = False,
        concat_incremental_tokenization_failed: bool = False,
        response_routed_experts: str | None = None,
        response_routed_experts_chunks: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        segment_boundary_before: bool = False,
        rewrite_reason: str | None = None,
        segment_reason_before: str | None = None,
        segment_reasons_before: list[str] | None = None,
        step_id: str | None = None,
        lineage_id: str | None = None,
        lineage_index: int | None = None,
        route_type: RouteType = "create",
        route_base_step_id: str | None = None,
        normalized_messages_snapshot: list[dict] | None = None,
        snapshot_rendered: str = "",
        snapshot_rendered_len: int = 0,
        snapshot_tools_hash: str | None = None,
        lineage_segment_boundary_before: bool = False,
        lineage_segment_reasons_before: list[str] | None = None,
        finish_reason: str = "stop",
        request_version: str | None = None,
        response_version: str | None = None,
    ) -> StepRecord | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning(
                    "dropping trajectory step for missing session %s after "
                    "possible stale-cleanup race",
                    session_id,
                )
                return None
            if step_id is None:
                step_id = session.next_step_id()
            if lineage_id is None:
                lineage = session.create_lineage(branch_from_step_id=route_base_step_id)
                lineage_id = lineage.id
                lineage_index = lineage.index
            elif lineage_index is None:
                lineage = session.lineages.get(lineage_id)
                lineage_index = lineage.index if lineage is not None else 0

            step = StepRecord(
                turn_id=turn_id,
                request_messages=request_messages,
                normalized_request_messages=normalized_request_messages,
                prompt_token_ids=prompt_token_ids,
                prompt_token_logprobs=prompt_token_logprobs,
                snapshot_token_ids=snapshot_token_ids,
                response_token_ids=response_token_ids,
                response_logprobs=response_logprobs,
                response_versions=list(response_versions or []),
                all_token_ids=all_token_ids,
                all_logprobs=all_logprobs,
                all_versions=list(all_versions or []),
                prompt_versions=list(prompt_versions or []),
                input_token_texts=input_token_texts,
                output_token_texts=output_token_texts,
                messages_snapshot=messages,
                raw_response_text=raw_response_text,
                step_id=step_id,
                lineage_id=lineage_id,
                lineage_index=int(lineage_index or 0),
                route_type=route_type,
                route_base_step_id=route_base_step_id,
                normalized_messages_snapshot=list(normalized_messages_snapshot or messages),
                snapshot_rendered=snapshot_rendered,
                snapshot_rendered_len=snapshot_rendered_len,
                snapshot_tools_hash=snapshot_tools_hash,
                lineage_segment_boundary_before=lineage_segment_boundary_before,
                lineage_segment_reasons_before=list(
                    lineage_segment_reasons_before or []
                ),
                all_logprobs_invalid=all_logprobs_invalid,
                concat_token_ids=list(concat_token_ids or []),
                concat_response_logprobs=list(concat_response_logprobs or []),
                concat_response_mask=list(concat_response_mask or []),
                concat_versions=list(concat_versions or []),
                concat_context_token_count=concat_context_token_count,
                concat_output_token_count=concat_output_token_count,
                concat_logprobs_invalid=concat_logprobs_invalid,
                concat_incremental_tokenization_failed=(
                    concat_incremental_tokenization_failed
                ),
                response_routed_experts=response_routed_experts,
                response_routed_experts_chunks=[
                    dict(item) for item in (response_routed_experts_chunks or [])
                ],
                tools=tools,
                segment_boundary_before=segment_boundary_before,
                rewrite_reason=rewrite_reason,
                segment_reason_before=segment_reason_before,
                segment_reasons_before=list(segment_reasons_before or []),
                finish_reason=finish_reason,
                request_version=request_version,
                response_version=response_version,
            )
            session.steps.append(step)
            session.steps_by_id[step.step_id] = step
            if step.normalized_messages_snapshot:
                session.prefix_tree.insert(step)
            lineage = session.lineages.get(step.lineage_id)
            if lineage is not None:
                lineage.latest_step_id = step.step_id
            session.last_active = time.time()
            return step

    def record_turn(self, **kwargs: Any) -> None:
        self.record_step(**kwargs)

    def mark_history_rewritten(self, session_id: str, reason: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.history_rewritten:
                return
            session.history_rewritten = True
            session.rewrite_reason = reason
            session.rewrite_detected_at = time.time()

    def get_session(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def get_finalization_result(self, session_id: str) -> dict[str, Any] | None:
        """Return the cached response for an already finalized session."""

        with self._lock:
            self._cleanup_expired_locked()
            result = self._finalization_results.get(session_id)
            return None if result is None else copy.deepcopy(result)

    def finalize_session(
        self,
        session_id: str,
        *,
        result: dict[str, Any] | None = None,
    ) -> Session | None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                self._finalized_session_ids[session_id] = time.time()
                if result is not None:
                    self._finalization_results[session_id] = copy.deepcopy(result)
            return session

    def active_count(self) -> int:
        with self._lock:
            self._cleanup_expired_locked()
            return len(self._sessions)

    def _cleanup_expired_locked(self) -> None:
        now = time.time()
        expired: list[str] = []
        for session_id, session in self._sessions.items():
            if now - session.last_active <= self._session_timeout:
                continue
            # A request can legitimately outlive the idle timeout.  Evicting
            # while it owns request_lock would make its eventual record_step
            # disappear and can turn a complete finalize into a partial one.
            if session.request_lock.locked():
                session.last_active = now
                continue
            expired.append(session_id)
        for session_id in expired:
            del self._sessions[session_id]
        expired_finalized = [
            session_id
            for session_id, finalized_at in self._finalized_session_ids.items()
            if now - finalized_at > self._session_timeout
        ]
        for session_id in expired_finalized:
            del self._finalized_session_ids[session_id]
            self._finalization_results.pop(session_id, None)
