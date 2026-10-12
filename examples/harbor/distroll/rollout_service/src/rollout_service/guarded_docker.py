from __future__ import annotations

import asyncio
import functools
import json
import subprocess
from pathlib import Path
from typing import Any

from harbor.environments.docker.docker import DockerEnvironment

from .sandbox_guard import current_guard


class GuardedDockerEnvironment(DockerEnvironment):
    """Harbor Docker environment with PID/cgroup/storage safety limits."""

    _DOCKER_COMPOSE_PREBUILT_PATH = Path(__file__).with_name(
        "docker-compose-prebuilt-never.yaml"
    )

    @staticmethod
    async def _collect_buffered_output(process, *, timeout_sec, stdin_data=None):
        try:
            return await DockerEnvironment._collect_buffered_output(
                process, timeout_sec=timeout_sec, stdin_data=stdin_data
            )
        except asyncio.CancelledError:
            # Upstream's buffered path only reaps its CLI on timeout. A
            # cancelled compose up must not outlive the creation counter.
            await DockerEnvironment._terminate_process(process)
            raise

    def __init__(
        self,
        *args: Any,
        guard_pids_limit: int,
        guard_cgroup_parent: str,
        guard_storage_mb: int | None = None,
        guard_storage_size_enforced: bool = False,
        guard_execution_image: str,
        guard_worker_id: str,
        guard_attempt_id: str,
        guard_client_scope: str,
        **kwargs: Any,
    ) -> None:
        self._guard_pids_limit = guard_pids_limit
        self._guard_cgroup_parent = guard_cgroup_parent
        self._guard_storage_mb = guard_storage_mb
        self._guard_storage_size_enforced = guard_storage_size_enforced
        self._guard_execution_image = guard_execution_image
        self._guard_labels = {
            "io.dressage.rollout.managed": "true",
            "io.dressage.rollout.worker_id": guard_worker_id,
            "io.dressage.rollout.attempt_id": guard_attempt_id,
            "io.dressage.rollout.client_scope": guard_client_scope,
        }
        super().__init__(*args, **kwargs)
        self._apply_execution_image(guard_execution_image)

    def _apply_execution_image(self, guard_execution_image: str) -> None:
        # task.toml keeps the human-readable admitted tag. Harbor receives the
        # immutable execution ref only inside this per-trial environment object,
        # so concurrent trials cannot overwrite process-global state.
        self.task_env_config = self.task_env_config.model_copy(
            update={"docker_image": guard_execution_image}
        )
        # DockerEnvironment snapshots the task image into Compose infra vars
        # during super().__init__. Keep that per-trial snapshot aligned too;
        # otherwise compose still asks Docker for the human-readable alias.
        self._env_vars.prebuilt_image_name = guard_execution_image

    async def start(self, force_build: bool) -> None:
        image = self.task_env_config.docker_image
        if force_build or not image or image != self._guard_execution_image:
            raise RuntimeError(
                "rollout requires the frozen execution image; runtime build or "
                "task-image tag execution is disabled"
            )
        watch = current_guard.get()
        if watch is None:
            await super().start(force_build=False)
            return
        watch.begin_start()
        try:
            await super().start(force_build=False)
            if watch.owner.settings.mode == "off":
                if watch.sealed or watch.triggered.is_set():
                    raise RuntimeError("execution interrupted sandbox creation")
                return
            containers = await self._run_docker_compose_command(
                ["ps", "--quiet", "main"], timeout_sec=5
            )
            identifiers = containers.stdout.strip().splitlines()
            if len(identifiers) != 1:
                raise RuntimeError("resource guard could not identify main sandbox")
            await watch.bind(identifiers[0])
        finally:
            watch.creating -= 1

    async def stop(self, delete: bool) -> None:
        """Remove sandbox resources without evicting managed prebuilt images."""

        try:
            watch = current_guard.get()
            if watch is not None:
                # Read the last counters before deleting the cgroup. Sampling
                # never waits for Control and must not delay an emergency stop.
                try:
                    async with asyncio.timeout(3):
                        await watch.owner.sample(watch)
                except TimeoutError:
                    self.logger.warning("Final resource guard sample timed out")
            if watch is None or not watch.triggered.is_set():
                await self.prepare_logs_for_host()
            if self._keep_containers:
                try:
                    await self._run_docker_compose_command(["stop"])
                except (OSError, RuntimeError, TimeoutError) as exc:
                    self.logger.warning("Docker compose stop failed: %s", exc)
            elif delete:
                try:
                    await self._run_docker_compose_command(
                        ["down", "--volumes", "--remove-orphans"]
                    )
                except (OSError, RuntimeError, TimeoutError) as exc:
                    self.logger.warning("Docker compose down failed: %s", exc)
            else:
                try:
                    await self._run_docker_compose_command(["down"])
                except (OSError, RuntimeError, TimeoutError) as exc:
                    self.logger.warning("Docker compose down failed: %s", exc)
        finally:
            self._cleanup_mounts_compose_file()
            self._cleanup_resources_compose_file()
            self._cleanup_env_compose_file()
            self._cleanup_egress_control_services_compose_file()

    async def _ensure_egress_control_sidecar_image_built(self) -> None:
        from harbor.environments.docker.utils import (
            _compute_image_name,
            default_docker_platform,
            docker_image_exists,
        )
        from harbor.utils.container_cache import docker_build_context_hash

        platform = await default_docker_platform()
        hash_key = docker_build_context_hash(
            context=self._EGRESS_CONTROL_SIDECAR_CONTEXT_PATH,
            dockerfile_path=self._egress_control_sidecar_dockerfile_path(),
            build_args={},
            platform=platform,
        )
        image_name = _compute_image_name(
            self._EGRESS_CONTROL_SIDECAR_DOCKER_NAME, hash_key
        )
        if not await docker_image_exists(image_name):
            raise RuntimeError(
                "Harbor egress-control sidecar is not prewarmed; run the CPU "
                "runtime-image prewarm step before rollout"
            )
        self._env_vars.egress_control_sidecar_image_name = image_name

    @staticmethod
    @functools.cache
    def _egress_control_kernel_support() -> bool:
        image = DockerEnvironment._EGRESS_CONTROL_KERNEL_PROBE_IMAGE
        inspected = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        if inspected.returncode != 0:
            raise RuntimeError(
                "Harbor kernel-probe image is not prewarmed; rollout will not pull it"
            )
        result = subprocess.run(
            [
                "docker",
                "container",
                "run",
                "--rm",
                "--pull",
                "never",
                image,
                "sh",
                "-c",
                DockerEnvironment._EGRESS_CONTROL_KERNEL_PROBE_SCRIPT,
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        return result.returncode == 0

    def _write_resources_compose_file(self) -> Path | None:
        path = super()._write_resources_compose_file()
        if path is None:
            raise RuntimeError("Harbor did not create a resource compose override")
        payload = json.loads(path.read_text(encoding="utf-8"))
        main = payload.setdefault("services", {}).setdefault("main", {})
        labels = main.setdefault("labels", {})
        if not isinstance(labels, dict):
            raise TypeError("Harbor resource compose labels must be an object")
        labels.update(self._guard_labels)
        main["pids_limit"] = self._guard_pids_limit
        main["cgroup_parent"] = self._guard_cgroup_parent
        if self._guard_storage_size_enforced:
            if self._guard_storage_mb is None:
                raise ValueError("guard_storage_mb is required for storage enforcement")
            main["storage_opt"] = {"size": f"{self._guard_storage_mb}M"}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
