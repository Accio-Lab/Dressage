from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import uvicorn

from .api import create_control_app
from .config import (
    control_settings_from_env,
    load_inventory,
    worker_settings_from_env,
)
from .store import PostgresStore
from .task_resource_index import load_inventory_task_resource_index
from .worker import create_worker_app


def control_main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = control_settings_from_env()
    inventory = load_inventory(settings.inventory_path)
    uvicorn.run(
        create_control_app(settings, inventory),
        host=settings.listen_host,
        port=settings.listen_port,
        log_level="info",
    )


def worker_main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = worker_settings_from_env()
    inventory = load_inventory(settings.inventory_path)
    app = create_worker_app(
        settings,
        inventory,
        resource_index=load_inventory_task_resource_index(inventory),
    )

    async def run() -> None:
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=settings.listen_host,
                port=settings.listen_port,
                log_level="info",
            )
        )
        server_task = asyncio.create_task(server.serve(), name="worker-uvicorn")
        while not server.started:
            if server_task.done():
                await server_task
                return
            await asyncio.sleep(0.05)
        daemon_task = app.state.worker_task
        done, _ = await asyncio.wait(
            {server_task, daemon_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if daemon_task in done and not server_task.done():
            daemon_error = None if daemon_task.cancelled() else daemon_task.exception()
            server.should_exit = True
            await server_task
            if daemon_error is not None:
                raise RuntimeError("worker daemon failed") from daemon_error
            raise RuntimeError("worker daemon stopped while HTTP service was active")
        await server_task

    asyncio.run(run())


def migrate_main() -> None:
    async def run() -> None:
        settings = control_settings_from_env()
        inventory = load_inventory(settings.inventory_path)
        store = PostgresStore(
            inventory,
            settings.database_url,
            pool_min_size=settings.database_pool_min_size,
            pool_max_size=settings.database_pool_max_size,
            resource_index=load_inventory_task_resource_index(inventory),
            admission=settings.admission,
        )
        await store.initialize()
        await store.close()

    asyncio.run(run())


def network_gateway_main() -> None:
    from .network_gateway import NetworkGateway, load_network_gateway_config

    logging.basicConfig(level=logging.INFO)
    config_path = Path(
        os.environ.get(
            "ROLLOUT_NETWORK_GATEWAY_CONFIG",
            "/etc/dressage-distroll/network-gateway.yaml",
        )
    )
    gateway = NetworkGateway(load_network_gateway_config(config_path))
    asyncio.run(gateway.run_until_stopped())
