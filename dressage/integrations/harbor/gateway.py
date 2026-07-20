"""Process-shared Harbor model gateway and route lifecycle primitives.

All proxy instances and their asyncio primitives live on a single dedicated
event loop.  Harbor hooks use :class:`AttemptHandle`; they never touch proxy
coroutines from Harbor's own loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import hmac
import inspect
import json
import os
import re
import secrets
import socket
import threading
import time
from collections import OrderedDict
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping, MutableMapping

import uvicorn

from dressage.integrations.harbor.config import HarborIntegrationConfig


ASGIScope = MutableMapping[str, Any]
ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]
ASGISend = Callable[[dict[str, Any]], Awaitable[None]]
ProxyFactory = Callable[..., Any]

_MODEL_PATHS: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/responses"),
        ("POST", "/v1/messages"),
    }
)
_AGENT_CREDENTIAL_HEADERS = frozenset(
    {
        b"authorization",
        b"proxy-authorization",
        b"x-api-key",
        b"api-key",
        b"anthropic-api-key",
        b"cookie",
    }
)
_TOKEN_HEADERS = frozenset(
    {b"authorization", b"x-api-key", b"api-key", b"anthropic-api-key"}
)
_ENV_SAFE = re.compile(r"[^A-Za-z0-9_]")


class GatewayError(RuntimeError):
    pass


class GatewayConfigurationError(GatewayError):
    pass


class RouteConflictError(GatewayError):
    pass


class RouteStateError(GatewayError):
    pass


class AttemptState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    ACTIVE = "active"
    DRAINING = "draining"
    BROKEN = "broken"
    CLOSED = "closed"


@dataclass(frozen=True)
class RouteSpec:
    trial_name: str
    trial_id: str
    instance_id: str
    token: str = field(repr=False)
    upstream_origin: str = "http://127.0.0.1:8800"
    router_api_path: str = "/v1"
    sticky_header_name: str = "X-SMG-Routing-Key"
    model_override: str | None = None
    expected_version: str | None = None
    upstream_headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    sampling_mode: str | None = None
    sampling_temperature: float | None = None
    sampling_top_p: float | None = None
    sampling_seed_base: int | None = None
    verify_tls: bool = True
    max_steps: int | None = 100
    default_temperature: float | None = None
    debug_log_dir: str | Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "trial_name",
            "trial_id",
            "instance_id",
            "token",
            "upstream_origin",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"RouteSpec.{name} must be a non-empty string")
        if not self.router_api_path.startswith("/"):
            raise ValueError("RouteSpec.router_api_path must start with '/'")
        if self.expected_version is not None and not str(self.expected_version).strip():
            raise ValueError("RouteSpec.expected_version must not be empty")
        if self.sampling_mode not in {None, "fill_missing", "force"}:
            raise ValueError(
                "RouteSpec.sampling_mode must be 'fill_missing', 'force', or None"
            )
        if self.sampling_temperature is not None and self.sampling_temperature < 0:
            raise ValueError("RouteSpec.sampling_temperature must be non-negative")
        if self.sampling_top_p is not None and not 0 < self.sampling_top_p <= 1:
            raise ValueError("RouteSpec.sampling_top_p must be in (0, 1]")
        object.__setattr__(
            self, "upstream_headers", MappingProxyType(dict(self.upstream_headers))
        )


@dataclass
class _SecretRecord:
    job_id: str
    slot_id: str
    env_name: str
    token: str = field(repr=False)


class SecretSlotRegistry:
    """Thread-safe, in-memory tokens mirrored to ephemeral host env slots."""

    def __init__(self, *, token_bytes: int = 32) -> None:
        if token_bytes < 16:
            raise ValueError("route tokens must contain at least 16 random bytes")
        self._token_bytes = token_bytes
        self._records: dict[str, _SecretRecord] = {}
        self._lock = threading.RLock()

    def create(self, job_id: str, slot_id: str, env_name: str | None = None) -> str:
        if not job_id or not slot_id:
            raise ValueError("job_id and slot_id must be non-empty")
        with self._lock:
            if slot_id in self._records:
                raise KeyError(f"secret slot {slot_id!r} already exists")
            selected_env = env_name or self._default_env_name(slot_id)
            if any(
                record.env_name == selected_env for record in self._records.values()
            ):
                raise KeyError(
                    f"secret environment slot {selected_env!r} already exists"
                )
            if selected_env in os.environ:
                raise KeyError(
                    f"secret environment slot {selected_env!r} already exists outside the registry"
                )
            token = self._new_token()
            self._records[slot_id] = _SecretRecord(job_id, slot_id, selected_env, token)
            os.environ[selected_env] = token
            return token

    def current(self, slot_id: str) -> str:
        with self._lock:
            return self._get(slot_id).token

    def fingerprint(self, slot_id: str) -> str:
        with self._lock:
            token = self._get(slot_id).token
            return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def env_name(self, slot_id: str) -> str:
        with self._lock:
            return self._get(slot_id).env_name

    def rotate(self, slot_id: str) -> str:
        with self._lock:
            record = self._get(slot_id)
            previous = record.token
            token = self._new_token()
            while hmac.compare_digest(
                previous, token
            ):  # pragma: no cover - cryptographically negligible
                token = self._new_token()
            record.token = token
            os.environ[record.env_name] = token
            return token

    def delete(self, slot_id: str) -> None:
        with self._lock:
            record = self._records.pop(slot_id, None)
            if record is not None and os.environ.get(record.env_name) == record.token:
                os.environ.pop(record.env_name, None)

    def close_job(self, job_id: str) -> None:
        with self._lock:
            slot_ids = [
                slot_id
                for slot_id, record in self._records.items()
                if record.job_id == job_id
            ]
            for slot_id in slot_ids:
                self.delete(slot_id)

    def close(self) -> None:
        with self._lock:
            for slot_id in tuple(self._records):
                self.delete(slot_id)

    def _get(self, slot_id: str) -> _SecretRecord:
        try:
            return self._records[slot_id]
        except KeyError as exc:
            raise KeyError(f"unknown secret slot {slot_id!r}") from exc

    def _new_token(self) -> str:
        return secrets.token_urlsafe(self._token_bytes)

    @staticmethod
    def _default_env_name(slot_id: str) -> str:
        normalized = _ENV_SAFE.sub("_", slot_id).strip("_").upper()
        digest = hashlib.sha256(slot_id.encode("utf-8")).hexdigest()[:12].upper()
        prefix = normalized[:32] or "SLOT"
        return f"DRESSAGE_HARBOR_ROUTE_{prefix}_{digest}"


@dataclass
class _RouteBinding:
    spec: RouteSpec
    token_digest: bytes
    proxy: Any
    lifespan: AbstractAsyncContextManager[Any]
    state: AttemptState = AttemptState.STARTING
    broken_reason: str | None = None
    inflight: int = 0
    inflight_zero: asyncio.Event = field(default_factory=asyncio.Event)
    semaphore: asyncio.Semaphore | None = None
    created_at: float = field(default_factory=time.monotonic)
    request_sequence: int = 0

    def __post_init__(self) -> None:
        self.inflight_zero.set()


class AttemptHandle:
    """Cross-loop-safe facade over one live Trial route."""

    def __init__(self, runtime: "GatewayRuntime", binding: _RouteBinding) -> None:
        self._runtime = runtime
        self._binding = binding

    @property
    def token(self) -> str:
        return self._binding.spec.token

    @property
    def trial_id(self) -> str:
        return self._binding.spec.trial_id

    @property
    def trial_name(self) -> str:
        return self._binding.spec.trial_name

    @property
    def state(self) -> AttemptState:
        return self._binding.state

    @property
    def proxy(self) -> Any:
        """Read-only identity access; proxy coroutines must not be called directly."""

        return self._binding.proxy

    @property
    def broken_reason(self) -> str | None:
        return self._binding.broken_reason

    async def open_turn(
        self, turn_id: str, backend_session_id: str | None = None
    ) -> None:
        async def operation() -> None:
            binding = self._binding
            if binding.state != AttemptState.READY:
                raise RouteStateError(
                    f"cannot open turn for {self.trial_id}: route is {binding.state.value}"
                )
            await binding.proxy.open_turn(
                turn_id, backend_session_id=backend_session_id
            )
            binding.state = AttemptState.ACTIVE

        await self._runtime.call(operation)

    async def quiesce(self, timeout: float | None = None) -> dict[str, dict[str, Any]]:
        async def operation() -> dict[str, dict[str, Any]]:
            binding = self._binding
            if binding.state == AttemptState.READY:
                return {}
            if binding.state == AttemptState.BROKEN:
                raise RouteStateError(binding.broken_reason or "route is broken")
            if binding.state != AttemptState.ACTIVE:
                raise RouteStateError(
                    f"cannot quiesce {self.trial_id}: route is {binding.state.value}"
                )
            binding.state = AttemptState.DRAINING
            selected_timeout = timeout
            if selected_timeout is None:
                selected_timeout = self._runtime.config.gateway.limits.drain_timeout_sec
            try:
                await asyncio.wait_for(
                    binding.inflight_zero.wait(), timeout=selected_timeout
                )
                await binding.proxy.drain_turn(timeout=selected_timeout)
                errors: dict[str, dict[str, Any]] = {}
                consumers = (
                    ("context_overflow", binding.proxy.consume_context_overflow_error),
                    (
                        "rollout_invalidated",
                        binding.proxy.consume_rollout_invalidated_error,
                    ),
                    ("failed_upstream", binding.proxy.consume_failed_upstream_error),
                    ("max_steps", binding.proxy.consume_max_steps_error),
                )
                for name, consumer in consumers:
                    payload = await consumer()
                    if payload is not None:
                        errors[name] = payload
                await binding.proxy.clear_turn()
                binding.state = AttemptState.READY
                return errors
            except BaseException as exc:
                binding.state = AttemptState.BROKEN
                binding.broken_reason = f"quiesce failed: {type(exc).__name__}: {exc}"
                raise

        return await self._runtime.call(operation)

    async def mark_broken(self, reason: str) -> None:
        async def operation() -> None:
            if self._binding.state == AttemptState.CLOSED:
                return
            self._binding.broken_reason = reason
            self._binding.state = AttemptState.BROKEN

        await self._runtime.call(operation)

    async def close(self, *, tombstone: bool = True) -> None:
        await self._runtime.call(
            lambda: self._runtime._close_binding(self._binding, tombstone=tombstone)
        )


class GatewayLease:
    def __init__(self, runtime: "GatewayRuntime") -> None:
        self._runtime = runtime
        self._released = False
        self._release_task: asyncio.Task[None] | None = None

    @property
    def runtime(self) -> "GatewayRuntime":
        return self._runtime

    @property
    def secret_slots(self) -> SecretSlotRegistry:
        return self._runtime.secret_slots

    @property
    def advertise_url(self) -> str:
        return self._runtime.advertise_url

    @property
    def public_url(self) -> str:
        return self.advertise_url

    async def register(self, spec: RouteSpec) -> AttemptHandle:
        if self._released:
            raise GatewayError("gateway lease is already released")
        return await self._runtime.call(lambda: self._runtime._register(spec))

    async def start_attempt(self, spec: RouteSpec) -> AttemptHandle:
        return await self.register(spec)

    async def release(self) -> None:
        if self._released:
            return
        if self._release_task is None:
            self._release_task = asyncio.create_task(
                self._runtime._release_lease(),
                name="harbor-gateway-lease-release",
            )
        await asyncio.shield(self._release_task)
        self._released = True

    async def aclose(self) -> None:
        await self.release()

    async def __aenter__(self) -> "GatewayLease":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.release()


class GatewayRuntime:
    """Process singleton owner for the Uvicorn thread and child proxy apps."""

    _singleton: "GatewayRuntime | None" = None
    _singleton_lock = threading.Lock()

    def __init__(self, *, proxy_factory: ProxyFactory | None = None) -> None:
        self._validate_harbor_runtime = proxy_factory is None
        self._proxy_factory = proxy_factory or self._default_proxy_factory
        self.secret_slots = SecretSlotRegistry()
        self._config: HarborIntegrationConfig | None = None
        self._config_fingerprint: str | None = None
        self._refcount = 0
        self._lease_lock = threading.RLock()
        self._start_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: uvicorn.Server | None = None
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._startup_error: BaseException | None = None
        self._bound_port: int | None = None
        self._unix_socket_path: Path | None = None
        self._routes: dict[bytes, _RouteBinding] = {}
        self._tombstones: OrderedDict[bytes, float] = OrderedDict()
        self._global_semaphore: asyncio.Semaphore | None = None
        self._asgi_app = _GatewayASGI(self)

    @classmethod
    def get(cls) -> "GatewayRuntime":
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    @property
    def config(self) -> HarborIntegrationConfig:
        if self._config is None:
            raise GatewayError("gateway runtime has not been acquired")
        return self._config

    @property
    def advertise_url(self) -> str:
        configured = self.config.gateway.advertise_url
        if configured is not None:
            return str(configured).rstrip("/")
        if self._bound_port is None:
            raise GatewayError("gateway is not listening yet")
        host = self.config.gateway.listen_host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self._bound_port}"

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive() and self._ready.is_set()

    @property
    def unix_socket_path(self) -> Path | None:
        return self._unix_socket_path

    async def acquire(self, config: HarborIntegrationConfig) -> GatewayLease:
        if self._validate_harbor_runtime:
            from dressage.integrations.harbor.compat import require_harbor_runtime

            require_harbor_runtime()
        fingerprint = config.gateway_fingerprint()
        with self._lease_lock:
            if self._refcount and self._config_fingerprint != fingerprint:
                raise GatewayConfigurationError(
                    "the process GatewayRuntime is already leased with incompatible gateway/backend/security configuration"
                )
            if self._refcount == 0:
                self._config = config
                self._config_fingerprint = fingerprint
            self._refcount += 1
        try:
            await asyncio.to_thread(self._start_blocking)
        except BaseException:
            with self._lease_lock:
                self._refcount -= 1
                if self._refcount == 0:
                    self._config = None
                    self._config_fingerprint = None
            raise
        return GatewayLease(self)

    async def call(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        loop = self._loop
        if loop is None or not loop.is_running():
            raise GatewayError("gateway owner loop is not running")
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - this is an async API
            current_loop = None
        if current_loop is loop:
            return await operation()

        result: concurrent.futures.Future[Any] = concurrent.futures.Future()
        cancel_requested = threading.Event()
        owner_task: list[asyncio.Task[Any] | None] = [None]

        def submit() -> None:
            if result.done() or cancel_requested.is_set():
                result.cancel()
                return
            try:
                task = loop.create_task(operation())
            except BaseException as exc:
                if not result.done():
                    result.set_exception(exc)
                return
            owner_task[0] = task
            if cancel_requested.is_set():
                task.cancel()

            def completed(done: asyncio.Task[Any]) -> None:
                if result.done():
                    # Retrieve the exception to avoid an owner-loop
                    # "Task exception was never retrieved" warning.
                    if not done.cancelled():
                        done.exception()
                    return
                try:
                    if done.cancelled():
                        result.cancel()
                    else:
                        error = done.exception()
                        if error is not None:
                            result.set_exception(error)
                        else:
                            result.set_result(done.result())
                except concurrent.futures.InvalidStateError:
                    # The caller can be cancelled between the done() check and
                    # this cross-thread state transition.
                    pass

            task.add_done_callback(completed)

        try:
            loop.call_soon_threadsafe(submit)
        except RuntimeError as exc:
            raise GatewayError("gateway owner loop stopped during dispatch") from exc
        try:
            return await asyncio.wrap_future(result)
        except asyncio.CancelledError:
            cancel_requested.set()
            result.cancel()

            def cancel_owner_task() -> None:
                task = owner_task[0]
                if task is not None and not task.done():
                    task.cancel()

            try:
                loop.call_soon_threadsafe(cancel_owner_task)
            except RuntimeError:
                pass
            raise

    async def _release_lease(self) -> None:
        should_stop = False
        with self._lease_lock:
            if self._refcount <= 0:
                return
            self._refcount -= 1
            should_stop = self._refcount == 0
        if not should_stop:
            return
        await self.call(self._shutdown_on_owner)
        await self._wait_until_stopped()
        with self._lease_lock:
            self._config = None
            self._config_fingerprint = None

    async def _register(self, spec: RouteSpec) -> AttemptHandle:
        self._cleanup_tombstones()
        if len(self._routes) >= self.config.gateway.limits.max_active_routes:
            raise GatewayError("gateway max_active_routes limit reached")
        digest = _token_digest(spec.token)
        if digest in self._routes:
            raise RouteConflictError("route token is already active")
        if digest in self._tombstones:
            raise RouteConflictError("route token is tombstoned and cannot be reused")

        proxy = self._create_proxy(spec)
        lifespan_factory = getattr(
            getattr(proxy.app, "router", None), "lifespan_context", None
        )
        if not callable(lifespan_factory):
            raise GatewayError("RolloutLLMProxy.app does not expose a lifespan context")
        lifespan = lifespan_factory(proxy.app)
        binding = _RouteBinding(
            spec=spec,
            token_digest=digest,
            proxy=proxy,
            lifespan=lifespan,
            semaphore=asyncio.Semaphore(
                self.config.gateway.limits.max_inflight_per_route
            ),
        )
        self._routes[digest] = binding
        try:
            await lifespan.__aenter__()
        except BaseException as exc:
            self._routes.pop(digest, None)
            binding.state = AttemptState.BROKEN
            binding.broken_reason = (
                f"proxy lifespan startup failed: {type(exc).__name__}: {exc}"
            )
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise GatewayError(binding.broken_reason) from exc
        binding.state = AttemptState.READY
        return AttemptHandle(self, binding)

    async def _close_binding(self, binding: _RouteBinding, *, tombstone: bool) -> None:
        if binding.state == AttemptState.CLOSED:
            return
        binding.state = AttemptState.CLOSED
        self._routes.pop(binding.token_digest, None)
        timeout = self.config.gateway.limits.drain_timeout_sec
        try:
            await asyncio.wait_for(binding.inflight_zero.wait(), timeout=timeout)
        finally:
            try:
                await binding.lifespan.__aexit__(None, None, None)
            finally:
                if tombstone:
                    expires_at = (
                        time.monotonic() + self.config.gateway.limits.tombstone_ttl_sec
                    )
                    self._tombstones[binding.token_digest] = expires_at
                    self._tombstones.move_to_end(binding.token_digest)
                    self._trim_tombstones()

    async def _shutdown_on_owner(self) -> None:
        bindings = tuple(self._routes.values())
        results = await asyncio.gather(
            *(self._close_binding(binding, tombstone=False) for binding in bindings),
            return_exceptions=True,
        )
        for binding, result in zip(bindings, results):
            if isinstance(result, BaseException):
                binding.state = AttemptState.CLOSED
        self._routes.clear()
        self._tombstones.clear()
        self.secret_slots.close()
        if self._server is not None:
            self._server.should_exit = True

    def _create_proxy(self, spec: RouteSpec) -> Any:
        kwargs = {
            "upstream_origin": spec.upstream_origin,
            "router_api_path": spec.router_api_path,
            "bound_session_id": spec.trial_id,
            "bound_instance_id": spec.instance_id,
            "sticky_header_name": spec.sticky_header_name,
            "max_steps": spec.max_steps,
            "default_temperature": spec.default_temperature,
            "debug_log_dir": spec.debug_log_dir,
            "upstream_headers": dict(spec.upstream_headers),
            "expected_version": spec.expected_version,
            "model_override": spec.model_override,
            "sampling_mode": spec.sampling_mode,
            "sampling_temperature": spec.sampling_temperature,
            "sampling_top_p": spec.sampling_top_p,
            "sampling_seed_base": spec.sampling_seed_base,
            "verify_tls": spec.verify_tls,
        }
        try:
            signature = inspect.signature(self._proxy_factory)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and not any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
        return self._proxy_factory(**kwargs)

    @staticmethod
    def _default_proxy_factory(**kwargs: Any) -> Any:
        from blackbox_server.proxy.rollout_llm_proxy import RolloutLLMProxy

        return RolloutLLMProxy(**kwargs)

    def _start_blocking(self) -> None:
        with self._start_lock:
            if self.is_running:
                return
            self._ready.clear()
            self._stopped.clear()
            self._startup_error = None
            self._thread = threading.Thread(
                target=self._thread_main,
                name="dressage-harbor-gateway",
                daemon=True,
            )
            self._thread.start()
            if not self._ready.wait(timeout=15.0):
                raise GatewayError("timed out starting the Harbor gateway owner loop")
            if self._startup_error is not None:
                raise GatewayError(
                    "failed to start the Harbor gateway"
                ) from self._startup_error

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._loop = None
            self._stopped.set()

    async def _serve(self) -> None:
        gateway = self.config.gateway
        family = socket.AF_INET6 if ":" in gateway.listen_host else socket.AF_INET
        tcp_socket = socket.socket(family, socket.SOCK_STREAM)
        tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        unix_socket: socket.socket | None = None
        try:
            tcp_socket.bind((gateway.listen_host, gateway.listen_port))
            tcp_socket.listen(2048)
            tcp_socket.setblocking(False)
            self._bound_port = int(tcp_socket.getsockname()[1])
            sockets = [tcp_socket]
            if self.config.environment.mode == "bwrap":
                runtime_root = self.config.environment.runtime_root
                runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
                runtime_root.chmod(0o700)
                socket_dir = runtime_root / "gateway"
                socket_dir.mkdir(mode=0o700, exist_ok=True)
                socket_dir.chmod(0o700)
                self._unix_socket_path = socket_dir / "gateway.sock"
                self._unix_socket_path.unlink(missing_ok=True)
                unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                unix_socket.bind(str(self._unix_socket_path))
                self._unix_socket_path.chmod(0o600)
                unix_socket.listen(2048)
                unix_socket.setblocking(False)
                sockets.append(unix_socket)
            self._global_semaphore = asyncio.Semaphore(
                gateway.limits.max_inflight_global
            )
            server_config = uvicorn.Config(
                self._asgi_app,
                host=gateway.listen_host,
                port=self._bound_port,
                workers=1,
                log_level=gateway.log_level,
                lifespan="off",
                access_log=False,
            )
            self._server = uvicorn.Server(server_config)
            self._ready.set()
            await self._server.serve(sockets=sockets)
        finally:
            tcp_socket.close()
            if unix_socket is not None:
                unix_socket.close()
            if self._unix_socket_path is not None:
                self._unix_socket_path.unlink(missing_ok=True)
                self._unix_socket_path = None
            self._server = None
            self._global_semaphore = None

    async def _wait_until_stopped(self, *, timeout: float = 15.0) -> None:
        """Wait for the owner thread without scheduling work on an executor.

        Python shuts down ``concurrent.futures`` workers before running normal
        ``atexit`` callbacks.  The Harbor rollout fallback closes its process
        lease from such a callback, so using ``asyncio.to_thread`` here races
        interpreter shutdown.  Polling the thread-owned event keeps the close
        path usable after the default executor is no longer available.
        """

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not self._stopped.is_set():
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise GatewayError("timed out stopping the Harbor gateway thread")
            await asyncio.sleep(min(0.05, remaining))
        # _thread_main sets _stopped immediately before returning.  Joining at
        # this point only reaps an already-finished (or finishing) thread and
        # does not depend on the interpreter's default executor.
        self._join_thread()

    def _join_thread(self) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=15.0)
            if thread.is_alive():
                raise GatewayError("timed out stopping the Harbor gateway thread")
        self._thread = None
        self._bound_port = None

    def _cleanup_tombstones(self) -> None:
        now = time.monotonic()
        expired = [
            digest
            for digest, expires_at in self._tombstones.items()
            if expires_at <= now
        ]
        for digest in expired:
            self._tombstones.pop(digest, None)

    def _trim_tombstones(self) -> None:
        self._cleanup_tombstones()
        maximum = self.config.gateway.limits.max_tombstones
        while len(self._tombstones) > maximum:
            self._tombstones.popitem(last=False)


class _GatewayASGI:
    def __init__(self, runtime: GatewayRuntime) -> None:
        self._runtime = runtime

    async def __call__(
        self, scope: ASGIScope, receive: ASGIReceive, send: ASGISend
    ) -> None:
        if scope.get("type") != "http":
            await _send_json(send, 404, {"error": "not_found"})
            return
        await self._handle_http(scope, receive, send)

    async def _handle_http(
        self, scope: ASGIScope, receive: ASGIReceive, send: ASGISend
    ) -> None:
        runtime = self._runtime
        runtime._cleanup_tombstones()
        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", "/")).rstrip("/") or "/"
        allowed = (method, path) in _MODEL_PATHS
        if method == "GET" and path == "/v1/models":
            allowed = runtime.config.security.allow_model_listing
        if not allowed:
            await _send_json(send, 404, {"error": "not_found"})
            return

        headers = list(scope.get("headers", []))
        token, conflict = _extract_route_token(headers)
        if conflict:
            await _send_json(send, 400, {"error": "conflicting_route_tokens"})
            return
        if token is None:
            await _send_json(send, 401, {"error": "missing_route_token"})
            return
        digest = _token_digest(token)
        binding = runtime._routes.get(digest)
        if binding is None:
            if digest in runtime._tombstones:
                await _send_json(send, 410, {"error": "route_closed"})
            else:
                await _send_json(send, 401, {"error": "unknown_route_token"})
            return
        if (
            time.monotonic() - binding.created_at
            >= runtime.config.gateway.limits.route_ttl_sec
        ):
            await runtime._close_binding(binding, tombstone=True)
            await _send_json(send, 410, {"error": "route_expired"})
            return
        if binding.state == AttemptState.BROKEN:
            await _send_json(send, 503, {"error": "route_broken"})
            return
        if binding.state == AttemptState.CLOSED:
            await _send_json(send, 410, {"error": "route_closed"})
            return
        if binding.state != AttemptState.ACTIVE:
            await _send_json(send, 409, {"error": "route_not_active"})
            return

        header_size = sum(len(name) + len(value) for name, value in headers)
        if header_size > runtime.config.gateway.limits.request_header_max_bytes:
            await _send_json(send, 431, {"error": "request_headers_too_large"})
            return

        acquired = await self._acquire_capacity(binding)
        if not acquired:
            await _send_json(send, 503, {"error": "gateway_busy"})
            return
        try:
            if binding.state != AttemptState.ACTIVE:
                await _send_json(send, 409, {"error": "route_not_active"})
                return
            body = await _read_body(
                receive, runtime.config.gateway.limits.request_body_max_bytes
            )
            if body is None:
                await _send_json(send, 413, {"error": "request_body_too_large"})
                return
            request_sequence = binding.request_sequence
            binding.request_sequence += 1
            rewritten_body = _rewrite_payload(body, binding.spec, request_sequence)
            rewritten_scope = dict(scope)
            rewritten_scope["headers"] = _rewrite_agent_headers(
                headers, len(rewritten_body)
            )
            delegated_receive = _BufferedReceive(rewritten_body, receive)
            tracked_send = _ActivitySend(send)
            try:
                await _dispatch_with_idle_timeout(
                    binding.proxy.app(rewritten_scope, delegated_receive, tracked_send),
                    tracked_send,
                    timeout=runtime.config.gateway.limits.sse_idle_timeout_sec,
                )
            except asyncio.TimeoutError:
                if not tracked_send.response_started:
                    await _send_json(send, 504, {"error": "upstream_idle_timeout"})
        finally:
            self._release_capacity(binding)

    async def _acquire_capacity(self, binding: _RouteBinding) -> bool:
        global_semaphore = self._runtime._global_semaphore
        route_semaphore = binding.semaphore
        if global_semaphore is None or route_semaphore is None:
            return False
        deadline = (
            asyncio.get_running_loop().time()
            + self._runtime.config.gateway.limits.queue_timeout_sec
        )
        acquired_global = False
        try:
            await asyncio.wait_for(
                global_semaphore.acquire(),
                timeout=max(deadline - asyncio.get_running_loop().time(), 0),
            )
            acquired_global = True
            await asyncio.wait_for(
                route_semaphore.acquire(),
                timeout=max(deadline - asyncio.get_running_loop().time(), 0),
            )
        except asyncio.TimeoutError:
            if acquired_global:
                global_semaphore.release()
            return False
        except BaseException:
            if acquired_global:
                global_semaphore.release()
            raise
        binding.inflight += 1
        binding.inflight_zero.clear()
        return True

    def _release_capacity(self, binding: _RouteBinding) -> None:
        global_semaphore = self._runtime._global_semaphore
        route_semaphore = binding.semaphore
        if route_semaphore is not None:
            route_semaphore.release()
        if global_semaphore is not None:
            global_semaphore.release()
        binding.inflight = max(binding.inflight - 1, 0)
        if binding.inflight == 0:
            binding.inflight_zero.set()


class _BufferedReceive:
    def __init__(self, body: bytes, original_receive: ASGIReceive) -> None:
        self._body = body
        self._original_receive = original_receive
        self._sent = False

    async def __call__(self) -> dict[str, Any]:
        if not self._sent:
            self._sent = True
            return {"type": "http.request", "body": self._body, "more_body": False}
        return await self._original_receive()


class _ActivitySend:
    def __init__(self, send: ASGISend) -> None:
        self._send = send
        self.activity = asyncio.Event()
        self.response_started = False

    async def __call__(self, message: dict[str, Any]) -> None:
        if message.get("type") == "http.response.start":
            self.response_started = True
        await self._send(message)
        self.activity.set()


async def _dispatch_with_idle_timeout(
    awaitable: Awaitable[Any],
    tracked_send: _ActivitySend,
    *,
    timeout: float,
) -> Any:
    task = asyncio.create_task(awaitable)
    try:
        while True:
            if task.done():
                return await task
            tracked_send.activity.clear()
            activity_task = asyncio.create_task(tracked_send.activity.wait())
            try:
                done, _ = await asyncio.wait(
                    {task, activity_task},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if task in done:
                    return await task
                if activity_task not in done:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise asyncio.TimeoutError
            finally:
                if not activity_task.done():
                    activity_task.cancel()
                await asyncio.gather(activity_task, return_exceptions=True)
    except BaseException:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        raise


async def _read_body(receive: ASGIReceive, maximum: int) -> bytes | None:
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        if message.get("type") == "http.disconnect":
            return b""
        body = message.get("body", b"")
        size += len(body)
        if size > maximum:
            return None
        chunks.append(body)
        if not message.get("more_body", False):
            return b"".join(chunks)


def _rewrite_payload(body: bytes, spec: RouteSpec, request_sequence: int) -> bytes:
    if not body:
        return body
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    changed = False
    if spec.model_override is not None and "model" in payload:
        payload["model"] = spec.model_override
        changed = True
    if spec.sampling_mode is not None:
        for key, value in (
            ("temperature", spec.sampling_temperature),
            ("top_p", spec.sampling_top_p),
        ):
            if value is None:
                continue
            if spec.sampling_mode == "force" or payload.get(key) is None:
                payload[key] = value
                changed = True
    if spec.sampling_mode is not None and spec.sampling_seed_base is not None:
        if spec.sampling_mode == "force" or payload.get("seed") is None:
            payload["seed"] = spec.sampling_seed_base + request_sequence
            changed = True
    if not changed:
        return body
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _extract_route_token(headers: list[tuple[bytes, bytes]]) -> tuple[str | None, bool]:
    values: list[str] = []
    for raw_name, raw_value in headers:
        name = raw_name.lower()
        if name not in _TOKEN_HEADERS:
            continue
        value = raw_value.decode("latin-1").strip()
        if name == b"authorization":
            scheme, separator, credential = value.partition(" ")
            if not separator or scheme.lower() != "bearer":
                continue
            value = credential.strip()
        if value:
            values.append(value)
    if not values:
        return None, False
    first = values[0]
    return first, any(not hmac.compare_digest(first, value) for value in values[1:])


def _rewrite_agent_headers(
    headers: list[tuple[bytes, bytes]], body_length: int
) -> list[tuple[bytes, bytes]]:
    rewritten = [
        (name, value)
        for name, value in headers
        if name.lower() not in _AGENT_CREDENTIAL_HEADERS
        and name.lower() not in {b"content-length", b"transfer-encoding"}
    ]
    rewritten.append((b"content-length", str(body_length).encode("ascii")))
    return rewritten


def _token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


async def _send_json(send: ASGISend, status: int, payload: Mapping[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


__all__ = [
    "AttemptHandle",
    "AttemptState",
    "GatewayConfigurationError",
    "GatewayError",
    "GatewayLease",
    "GatewayRuntime",
    "RouteConflictError",
    "RouteSpec",
    "RouteStateError",
    "SecretSlotRegistry",
]
