from __future__ import annotations

import asyncio
import ipaddress
import logging
import signal
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class GatewayModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TcpRelaySpec(GatewayModel):
    name: str = Field(min_length=1, max_length=100)
    listen_host: str = Field(min_length=1)
    listen_port: int = Field(ge=0, le=65535)
    upstream_host: str = Field(min_length=1)
    upstream_port: int = Field(ge=1, le=65535)
    source_host: str | None = None
    max_connections: int = Field(default=4096, ge=1)
    connect_timeout_sec: float = Field(default=15.0, gt=0)
    buffer_bytes: int = Field(default=256 * 1024, ge=4096, le=4 * 1024 * 1024)
    tcp_keepalive_idle_sec: int = Field(default=60, ge=1)
    tcp_keepalive_interval_sec: int = Field(default=15, ge=1)
    tcp_keepalive_count: int = Field(default=4, ge=1)

    @field_validator("source_host")
    @classmethod
    def validate_source_host(cls, value: str | None) -> str | None:
        if value is not None:
            ipaddress.ip_address(value)
        return value


class NetworkGatewayConfig(GatewayModel):
    schema_version: Literal["rollout-network-gateway/v0"] = "rollout-network-gateway/v0"
    relays: list[TcpRelaySpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_relays(self) -> NetworkGatewayConfig:
        names = [relay.name for relay in self.relays]
        listeners = [(relay.listen_host, relay.listen_port) for relay in self.relays]
        if len(names) != len(set(names)):
            raise ValueError("network gateway relay names must be unique")
        if len(listeners) != len(set(listeners)):
            raise ValueError("network gateway listen endpoints must be unique")
        return self


def load_network_gateway_config(path: str | Path) -> NetworkGatewayConfig:
    selected = Path(path).expanduser().resolve()
    payload = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"network gateway config must be a YAML object: {selected}")
    return NetworkGatewayConfig.model_validate(payload)


class TcpRelay:
    def __init__(self, spec: TcpRelaySpec) -> None:
        self.spec = spec
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._semaphore = asyncio.Semaphore(spec.max_connections)

    @property
    def bound_port(self) -> int | None:
        if self._server is None or not self._server.sockets:
            return None
        return int(self._server.sockets[0].getsockname()[1])

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._accept,
            host=self.spec.listen_host,
            port=self.spec.listen_port,
            backlog=min(self.spec.max_connections, 4096),
            start_serving=True,
        )
        logger.info(
            "TCP relay %s listening on %s:%s -> %s:%s source=%s via=%s",
            self.spec.name,
            self.spec.listen_host,
            self.bound_port,
            self.spec.upstream_host,
            self.spec.upstream_port,
            self.spec.source_host or "automatic",
            "direct",
        )

    async def close(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if server is not None:
            await server.wait_closed()

    def _accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.create_task(
            self._handle(reader, writer),
            name=f"tcp-relay-{self.spec.name}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("TCP relay %s handler escaped: %r", self.spec.name, error)

    async def _handle(
        self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter
    ) -> None:
        peer = client_writer.get_extra_info("peername")
        peer_host = str(peer[0]) if isinstance(peer, tuple) and peer else ""

        if self._semaphore.locked():
            logger.warning(
                "TCP relay %s rejected peer %s at connection limit",
                self.spec.name,
                peer_host,
            )
            await _close_writer(client_writer)
            return

        async with self._semaphore:
            upstream_writer: asyncio.StreamWriter | None = None
            try:
                _configure_keepalive(client_writer, self.spec)
                async with asyncio.timeout(self.spec.connect_timeout_sec):
                    upstream_reader, upstream_writer = await _open_upstream(self.spec)
                _configure_keepalive(upstream_writer, self.spec)
                try:
                    async with asyncio.TaskGroup() as group:
                        group.create_task(
                            _pump(
                                client_reader,
                                upstream_writer,
                                buffer_bytes=self.spec.buffer_bytes,
                            )
                        )
                        group.create_task(
                            _pump(
                                upstream_reader,
                                client_writer,
                                buffer_bytes=self.spec.buffer_bytes,
                            )
                        )
                except* (OSError, TimeoutError, RuntimeError) as errors:
                    logger.warning(
                        "TCP relay %s stream failed peer=%s errors=%r",
                        self.spec.name,
                        peer_host,
                        errors.exceptions,
                    )
            except asyncio.CancelledError:
                raise
            except (OSError, TimeoutError, RuntimeError) as error:
                logger.warning(
                    "TCP relay %s connection failed peer=%s error=%r",
                    self.spec.name,
                    peer_host,
                    error,
                )
            finally:
                await _close_writer(client_writer)
                if upstream_writer is not None:
                    await _close_writer(upstream_writer)


class NetworkGateway:
    def __init__(self, config: NetworkGatewayConfig) -> None:
        self.config = config
        self.relays = [TcpRelay(spec) for spec in config.relays]

    async def start(self) -> None:
        started: list[TcpRelay] = []
        try:
            for relay in self.relays:
                await relay.start()
                started.append(relay)
        except BaseException:
            await asyncio.gather(
                *(relay.close() for relay in reversed(started)),
                return_exceptions=True,
            )
            raise

    async def close(self) -> None:
        await asyncio.gather(
            *(relay.close() for relay in reversed(self.relays)),
            return_exceptions=False,
        )

    async def run_until_stopped(self) -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        installed_signals: list[signal.Signals] = []
        for selected in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(selected, stop.set)
                installed_signals.append(selected)
            except NotImplementedError:  # pragma: no cover - non-POSIX fallback
                pass
        await self.start()
        try:
            await stop.wait()
        finally:
            await self.close()
            for selected in installed_signals:
                loop.remove_signal_handler(selected)


@asynccontextmanager
async def running_gateway(
    config: NetworkGatewayConfig,
) -> AsyncIterator[NetworkGateway]:
    gateway = NetworkGateway(config)
    await gateway.start()
    try:
        yield gateway
    finally:
        await gateway.close()


async def _pump(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    buffer_bytes: int,
) -> None:
    while chunk := await reader.read(buffer_bytes):
        writer.write(chunk)
        await writer.drain()
    if writer.can_write_eof():
        writer.write_eof()
        await writer.drain()


async def _open_upstream(
    spec: TcpRelaySpec,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    local_addr = (spec.source_host, 0) if spec.source_host else None
    return await asyncio.open_connection(
        host=spec.upstream_host,
        port=spec.upstream_port,
        local_addr=local_addr,
    )


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except (ConnectionError, OSError):
        pass


def _configure_keepalive(writer: asyncio.StreamWriter, spec: TcpRelaySpec) -> None:
    selected = writer.get_extra_info("socket")
    if selected is None:
        return
    selected.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for option_name, value in (
        ("TCP_KEEPIDLE", spec.tcp_keepalive_idle_sec),
        ("TCP_KEEPINTVL", spec.tcp_keepalive_interval_sec),
        ("TCP_KEEPCNT", spec.tcp_keepalive_count),
    ):
        option = getattr(socket, option_name, None)
        if option is not None:
            selected.setsockopt(socket.IPPROTO_TCP, option, value)
