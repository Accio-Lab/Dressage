"""Base class for Dressage whitebox rollout agents.

A whitebox agent drives the Dressage proxy via ``chat_completions`` calls.
The proxy records segments (tokens / loss masks / logprobs); the framework
drains them back into training samples at the end of the trajectory.

Two layers:

  * **WhiteboxAgent** — pure proxy, no sandbox. For agents whose tools are
    simple Python functions (API calls, search, etc.) that don't need a
    sandbox environment.

  * **PaddockWhiteboxAgent** — adds paddock sandbox lifecycle (init /
    terminate) and trajectory + sample logging via shared rollout helpers.
    For agents that execute code in a sandbox.

Subclass contract::

    class MyAgent(WhiteboxAgent):
        name = "my_agent"

        async def rollout(self, sample, sampling_params) -> str:
            response = await self.chat({"messages": [...], "model": "..."})
            return extract_assistant_content(response)

    generate = make_generate(MyAgent)
    # --custom-generate-function-path mypkg.my_agent:generate
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from dressage.rollout import multi_segment
from dressage.rollout.artifacts.samples import (
    instance_id as _resolve_instance_id,
    set_status as _set_status,
)
from dressage.rollout.generate.runtime import (
    get_paddock_from_env,
    get_proxy_client,
    maybe_await,
    paddock_env_args_from_metadata,
)

if TYPE_CHECKING:
    from dressage.proxy.proxy_client import ProxyClient

logger = logging.getLogger(__name__)


def _stamp_runtime_metadata(sample: Any, session_id: str, instance_id: str) -> None:
    sample.session_id = session_id
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    metadata["session_id"] = session_id
    metadata["instance_id"] = instance_id


def _agent_session_id(sample: Any, prefix: str) -> str:
    session_id = getattr(sample, "session_id", None)
    if session_id is None:
        session_id = uuid.uuid4().hex
    session_id = str(session_id)
    if prefix and not session_id.startswith(f"{prefix}-"):
        session_id = f"{prefix}-{session_id}"
    return session_id


def extract_assistant_content(response: dict[str, Any]) -> str:
    """Pull ``choices[0].message.content`` from an OpenAI-shaped response."""
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def extract_finish_reason(response: dict[str, Any]) -> str:
    """Pull ``choices[0].finish_reason`` with a ``"stop"`` fallback."""
    choices = response.get("choices") or []
    if not choices:
        return "stop"
    return choices[0].get("finish_reason") or "stop"


# ---------------------------------------------------------------------------
# WhiteboxAgent — proxy-only base class
# ---------------------------------------------------------------------------


class WhiteboxAgent(ABC):
    """Pure-proxy whitebox agent. Subclass and implement :meth:`rollout`.

    The framework (via :func:`make_generate`) manages session ids, drains
    the proxy after rollout, and converts segments to training samples.
    Inside ``rollout`` use ``self.chat(body)`` to drive the conversation.
    Raise to abort.
    """

    name: ClassVar[str]
    session_prefix: ClassVar[str] = "wb"

    args: Any
    session_id: str
    instance_id: str

    @property
    def proxy(self) -> ProxyClient:
        return get_proxy_client()

    @abstractmethod
    async def rollout(
        self, sample: Any, sampling_params: dict[str, Any],
    ) -> str:
        """Drive one trajectory. Return accumulated assistant text."""
        raise NotImplementedError

    async def chat(
        self,
        body: dict[str, Any],
        *,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        return await self.proxy.chat_completions(
            body,
            session_id=self.session_id,
            instance_id=self.instance_id,
            turn_id=turn_id,
        )

    async def setup(self, sample: Any) -> None:
        """Called before ``rollout()``. Override for initialization."""

    async def teardown(self) -> None:
        """Called in ``finally`` after rollout + finalize. Override for cleanup."""

    def _abort(self, sample: Any) -> Any:
        multi_segment.mark_aborted_no_grad(
            sample, session_id=self.session_id, instance_id=self.instance_id,
        )
        _set_status(sample, "ABORTED")
        return sample

    async def _finalize_trajectory(
        self, sample: Any, agent_response: str,
    ) -> Any:
        """Drain proxy, convert segments to training samples."""
        try:
            await self.proxy.finalize_session(
                self.session_id,
                instance_id=self.instance_id,
                label=getattr(sample, "label", None),
            )
            payload = await self.proxy.read_trajectory(
                trajectory_id=self.session_id,
                instance_id=self.instance_id,
                drain=True,
            )
            segments = payload.get("data") or []
        except Exception:
            logger.exception(
                "drain failed session=%s instance=%s",
                self.session_id, self.instance_id,
            )
            return self._abort(sample)

        if not segments:
            logger.warning(
                "no segments returned session=%s instance=%s",
                self.session_id, self.instance_id,
            )
            return self._abort(sample)

        return multi_segment.expand_segments_to_samples(
            sample,
            segments,
            args=self.args,
            agent_response=agent_response,
            session_id=self.session_id,
            instance_id=self.instance_id,
        )


# ---------------------------------------------------------------------------
# PaddockWhiteboxAgent — adds sandbox lifecycle and logging
# ---------------------------------------------------------------------------


class PaddockWhiteboxAgent(WhiteboxAgent):
    """WhiteboxAgent with paddock sandbox lifecycle and logging.

    Subclasses get ``self.paddock`` for tool execution in the sandbox.
    The framework handles ``paddock.init`` / ``paddock.terminate``
    automatically via ``setup`` / ``teardown``.
    """

    _paddock: Any = None
    _paddock_initialized: bool = False
    _env_args: dict[str, Any] = {}

    @property
    def paddock(self) -> Any:
        if self._paddock is None:
            self._paddock = get_paddock_from_env(
                allow_whitebox_mode=True, mode="whitebox"
            )
        return self._paddock

    async def setup(self, sample: Any) -> None:
        metadata = getattr(sample, "metadata", None) or {}
        self._env_args = paddock_env_args_from_metadata(metadata)
        await maybe_await(
            self.paddock.init(self.session_id, metadata.get("env_type"), self._env_args)
        )
        self._paddock_initialized = True

    async def teardown(self) -> None:
        if not self._paddock_initialized or self._paddock is None:
            return
        from dressage.paddock.lifecycle import terminate_paddock_best_effort

        try:
            await terminate_paddock_best_effort(
                self._paddock,
                session_id=self.session_id,
                env_args=self._env_args,
            )
        except Exception:
            logger.warning(
                "failed to terminate paddock session=%s",
                self.session_id,
                exc_info=True,
            )

    async def _finalize_trajectory(
        self, sample: Any, agent_response: str,
    ) -> Any:
        from dressage.rollout.artifacts.writer import DEFAULT_WRITER

        try:
            await self.proxy.finalize_session(
                self.session_id,
                instance_id=self.instance_id,
                label=getattr(sample, "label", None),
            )
            payload = await self.proxy.read_trajectory(
                trajectory_id=self.session_id,
                instance_id=self.instance_id,
                drain=True,
            )
            segments = payload.get("data") or []
        except Exception:
            logger.exception(
                "drain failed session=%s instance=%s",
                self.session_id, self.instance_id,
            )
            return self._abort(sample)

        if not segments:
            logger.warning(
                "no segments returned session=%s instance=%s",
                self.session_id, self.instance_id,
            )
            return self._abort(sample)

        try:
            await DEFAULT_WRITER.write_session_payload(
                payload,
                session_id=self.session_id,
                instance_id=self.instance_id,
            )
        except Exception:
            logger.warning(
                "failed to write trajectory payload log session=%s",
                self.session_id,
                exc_info=True,
            )

        base_metadata = dict(getattr(sample, "metadata", None) or {})

        result = multi_segment.expand_segments_to_samples(
            sample,
            segments,
            args=self.args,
            agent_response=agent_response,
            session_id=self.session_id,
            instance_id=self.instance_id,
        )
        log_sample = sample

        try:
            await DEFAULT_WRITER.write_segment_samples(
                log_sample,
                args=self.args,
                segments=segments,
                base_metadata=base_metadata,
                session_id=self.session_id,
                instance_id=self.instance_id,
                agent_response=agent_response,
            )
        except Exception:
            logger.warning(
                "failed to write sample logs session=%s",
                self.session_id,
                exc_info=True,
            )

        return result


# ---------------------------------------------------------------------------
# slime adapter
# ---------------------------------------------------------------------------


def make_generate(agent_cls: type[WhiteboxAgent]):
    """Build the ``async def generate`` that slime's
    ``--custom-generate-function-path`` loads.

    A fresh agent instance is created per call. The proxy client is cached
    at module scope.
    """

    async def generate(
        args: Any,
        sample: Any,
        sampling_params: dict[str, Any],
        evaluation: bool = False,
    ) -> Any:
        if evaluation:
            raise ValueError(
                f"{agent_cls.__name__} does not support evaluation mode"
            )

        agent = agent_cls()
        agent.args = args
        agent.session_id = _agent_session_id(sample, agent.session_prefix)
        agent.instance_id = _resolve_instance_id(sample)
        _stamp_runtime_metadata(sample, agent.session_id, agent.instance_id)

        try:
            await agent.setup(sample)
            try:
                agent_response = await agent.rollout(sample, sampling_params)
            except Exception:
                logger.exception(
                    "rollout raised session=%s instance=%s",
                    agent.session_id, agent.instance_id,
                )
                return agent._abort(sample)
            return await agent._finalize_trajectory(
                sample, agent_response or "",
            )
        finally:
            await agent.teardown()

    generate.__name__ = f"{agent_cls.__name__}_generate"
    generate.__qualname__ = generate.__name__
    generate.__module__ = agent_cls.__module__
    return generate
