from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dressage.sandbox.factory import create_sandbox_provider_from_env
from dressage.sandbox.local.bwrap.provider import LocalBwrapSandboxProvider
from dressage.sandbox.remote.e2b.provider import E2BSandboxProvider
from dressage.sandbox.types import SandboxServiceSpec, SandboxSpec


class FakeLocalManager:
    def __init__(self, pool_mode: str = "blackbox") -> None:
        self.pool_mode = pool_mode
        self.calls = []

    async def acquire(self, trajectory_id, env_type=None, env_args=None):
        self.calls.append(("acquire", trajectory_id, env_type, env_args))
        payload = {
            "lease_id": f"lease-{trajectory_id}",
            "trajectory_id": trajectory_id,
            "node_id": "node-a",
            "node_ip": "10.0.0.12",
            "slot_id": 3,
            "port": 31003,
            "generation": 5,
            "pool_mode": self.pool_mode,
        }
        if self.pool_mode == "blackbox":
            payload["sandbox_url"] = "http://10.0.0.12:31003"
        return payload

    async def release(self, trajectory_id=None, lease_id=None, reason=None):
        self.calls.append(("release", trajectory_id, lease_id, reason))
        return {"released": True, "trajectory_id": trajectory_id, "lease_id": lease_id}

    async def run_command(self, **kwargs):
        self.calls.append(("run_command", kwargs))
        return {"cmd": kwargs["command"], "stdout": "ok\n", "stderr": "", "returncode": 0, "timed_out": False}

    async def read_file(self, **kwargs):
        self.calls.append(("read_file", kwargs))
        return "hello"

    async def write_file(self, **kwargs):
        self.calls.append(("write_file", kwargs))
        return {"path": kwargs["path"], "bytes": len(kwargs["content"])}


class FakeE2BCommandResult:
    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        returncode: int | None = 0,
        timed_out: bool = False,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.timed_out = timed_out


class FakeE2BCommands:
    def __init__(
        self,
        events: list[tuple[Any, ...]],
        results: dict[str, FakeE2BCommandResult] | None = None,
    ) -> None:
        self.events = events
        self.results = results or {}

    async def run(self, cmd: str, **kwargs):
        self.events.append(("run", cmd, kwargs))
        return self.results.get(cmd, FakeE2BCommandResult(stdout=f"{cmd}\n"))


class FakeE2BSandbox:
    sandbox_id = "sandbox-1"

    def __init__(
        self,
        events: list[tuple[Any, ...]],
        results: dict[str, FakeE2BCommandResult] | None = None,
    ) -> None:
        self.events = events
        self.commands = FakeE2BCommands(events, results)

    async def get_host(self, port):
        self.events.append(("get_host", port))
        return "sandbox.e2b.test"

    async def kill(self):
        self.events.append(("kill",))
        return True


def test_sandbox_provider_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("DRESSAGE_SANDBOX_PROVIDER", "ray_pool")
    with pytest.raises(ValueError, match="e2b|local_bwrap"):
        create_sandbox_provider_from_env()


def test_e2b_provider_requires_sample_or_default_template(monkeypatch):
    asyncio.run(_run_e2b_provider_requires_sample_or_default_template(monkeypatch))


async def _run_e2b_provider_requires_sample_or_default_template(monkeypatch):
    monkeypatch.delenv("DRESSAGE_SANDBOX_DEFAULT_IMAGE", raising=False)
    provider = E2BSandboxProvider(sandbox_factory=lambda **kwargs: object())
    with pytest.raises(ValueError, match="DRESSAGE_SANDBOX_DEFAULT_IMAGE"):
        await provider.create(SandboxSpec(trajectory_id="traj-1"))


def test_e2b_provider_uses_default_template_from_env(monkeypatch):
    asyncio.run(_run_e2b_provider_uses_default_template_from_env(monkeypatch))


async def _run_e2b_provider_uses_default_template_from_env(monkeypatch):
    class FakeSandbox:
        sandbox_id = "sandbox-1"

    calls = []

    async def sandbox_factory(**kwargs):
        calls.append(kwargs)
        return FakeSandbox()

    monkeypatch.setenv("DRESSAGE_SANDBOX_DEFAULT_IMAGE", "default-template")
    provider = E2BSandboxProvider(sandbox_factory=sandbox_factory)
    lease = await provider.create(SandboxSpec(trajectory_id="traj-1"))

    assert calls[0]["template"] == "default-template"
    assert lease.metadata["template"] == "default-template"


def test_e2b_provider_sample_template_overrides_default(monkeypatch):
    asyncio.run(_run_e2b_provider_sample_template_overrides_default(monkeypatch))


async def _run_e2b_provider_sample_template_overrides_default(monkeypatch):
    class FakeSandbox:
        sandbox_id = "sandbox-1"

    calls = []

    async def sandbox_factory(**kwargs):
        calls.append(kwargs)
        return FakeSandbox()

    monkeypatch.setenv("DRESSAGE_SANDBOX_DEFAULT_IMAGE", "default-template")
    provider = E2BSandboxProvider(sandbox_factory=sandbox_factory)
    lease = await provider.create(
        SandboxSpec(
            trajectory_id="traj-1",
            env_args={"sandbox_image": "sample-template"},
        )
    )

    assert calls[0]["template"] == "sample-template"
    assert lease.metadata["template"] == "sample-template"


def test_e2b_provider_reads_envs_and_metadata_from_sandbox_extra_params(monkeypatch):
    asyncio.run(_run_e2b_provider_reads_envs_and_metadata_from_sandbox_extra_params(monkeypatch))


async def _run_e2b_provider_reads_envs_and_metadata_from_sandbox_extra_params(monkeypatch):
    class FakeSandbox:
        sandbox_id = "sandbox-1"

    calls = []

    async def sandbox_factory(**kwargs):
        calls.append(kwargs)
        return FakeSandbox()

    monkeypatch.setenv("DRESSAGE_SANDBOX_DEFAULT_IMAGE", "default-template")
    provider = E2BSandboxProvider(sandbox_factory=sandbox_factory)
    await provider.create(
        SandboxSpec(
            trajectory_id="traj-1",
            env={"BASE": "1"},
            env_args={
                "sandbox_extra_params": {
                    "e2b_envs": {"A": "2"},
                    "e2b_metadata": {"trace": "abc"},
                }
            },
            metadata={"paddock_mode": "blackbox"},
        )
    )

    assert calls[0]["envs"] == {"BASE": "1", "A": "2"}
    assert calls[0]["metadata"] == {
        "trajectory_id": "traj-1",
        "paddock_mode": "blackbox",
        "trace": "abc",
    }


def test_e2b_provider_ignores_top_level_e2b_envs_and_metadata(monkeypatch):
    asyncio.run(_run_e2b_provider_ignores_top_level_e2b_envs_and_metadata(monkeypatch))


async def _run_e2b_provider_ignores_top_level_e2b_envs_and_metadata(monkeypatch):
    class FakeSandbox:
        sandbox_id = "sandbox-1"

    calls = []

    async def sandbox_factory(**kwargs):
        calls.append(kwargs)
        return FakeSandbox()

    monkeypatch.setenv("DRESSAGE_SANDBOX_DEFAULT_IMAGE", "default-template")
    provider = E2BSandboxProvider(sandbox_factory=sandbox_factory)
    await provider.create(
        SandboxSpec(
            trajectory_id="traj-1",
            env_args={
                "e2b_envs": {"A": "ignored"},
                "e2b_metadata": {"trace": "ignored"},
            },
        )
    )

    assert calls[0]["envs"] == {}
    assert calls[0]["metadata"] == {"trajectory_id": "traj-1"}


@pytest.mark.parametrize("key", ["e2b_envs", "e2b_metadata"])
def test_e2b_provider_rejects_non_dict_extra_param(monkeypatch, key):
    asyncio.run(_run_e2b_provider_rejects_non_dict_extra_param(monkeypatch, key))


async def _run_e2b_provider_rejects_non_dict_extra_param(monkeypatch, key):
    monkeypatch.setenv("DRESSAGE_SANDBOX_DEFAULT_IMAGE", "default-template")
    provider = E2BSandboxProvider(sandbox_factory=lambda **kwargs: object())

    with pytest.raises(ValueError, match=key):
        await provider.create(
            SandboxSpec(
                trajectory_id="traj-1",
                env_args={"sandbox_extra_params": {key: "not-a-dict"}},
            )
        )


def test_e2b_provider_rejects_unknown_sandbox_extra_params_key(monkeypatch):
    asyncio.run(_run_e2b_provider_rejects_unknown_sandbox_extra_params_key(monkeypatch))


async def _run_e2b_provider_rejects_unknown_sandbox_extra_params_key(monkeypatch):
    monkeypatch.setenv("DRESSAGE_SANDBOX_DEFAULT_IMAGE", "default-template")
    provider = E2BSandboxProvider(sandbox_factory=lambda **kwargs: object())

    with pytest.raises(ValueError, match="env_key"):
        await provider.create(
            SandboxSpec(
                trajectory_id="traj-1",
                env_args={"sandbox_extra_params": {"env_key": "unsupported"}},
            )
        )


def test_e2b_provider_runs_string_sandbox_cmd_before_public_url():
    asyncio.run(_run_e2b_provider_runs_string_sandbox_cmd_before_public_url())


async def _run_e2b_provider_runs_string_sandbox_cmd_before_public_url():
    events: list[tuple[Any, ...]] = []

    async def sandbox_factory(**kwargs):
        return FakeE2BSandbox(
            events,
            results={
                "python -V": FakeE2BCommandResult(stdout="Python 3.11\n"),
            },
        )

    provider = E2BSandboxProvider(template="blackbox-template", sandbox_factory=sandbox_factory)
    lease = await provider.create(
        SandboxSpec(
            trajectory_id="traj-1",
            env_args={"sandbox_cmd": "python -V"},
            services=(SandboxServiceSpec(name="blackbox", port=31000),),
            timeout_sec=42,
        )
    )

    assert events == [
        ("run", "python -V", {"timeout": 42}),
        ("get_host", 31000),
    ]
    assert lease.endpoint("blackbox").url == "https://sandbox.e2b.test"
    assert lease.metadata["sandbox_cmd_results"] == [
        {
            "cmd": "python -V",
            "stdout": "Python 3.11\n",
            "stderr": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "returncode": 0,
            "timed_out": False,
        }
    ]
    assert lease.metadata["sandbox_cmd_result"] == lease.metadata["sandbox_cmd_results"][-1]


def test_e2b_provider_runs_list_sandbox_cmd_as_sequential_commands():
    asyncio.run(_run_e2b_provider_runs_list_sandbox_cmd_as_sequential_commands())


async def _run_e2b_provider_runs_list_sandbox_cmd_as_sequential_commands():
    events: list[tuple[Any, ...]] = []

    async def sandbox_factory(**kwargs):
        return FakeE2BSandbox(
            events,
            results={
                "python -V": FakeE2BCommandResult(stdout="Python\n"),
                "pwd": FakeE2BCommandResult(stdout="/workspace\n"),
                "ls -la": FakeE2BCommandResult(stdout="total 0\n"),
            },
        )

    provider = E2BSandboxProvider(template="blackbox-template", sandbox_factory=sandbox_factory)
    lease = await provider.create(
        SandboxSpec(
            trajectory_id="traj-1",
            env_args={"sandbox_cmd": ["python -V", "pwd", "ls -la"]},
            services=(SandboxServiceSpec(name="blackbox", port=31000),),
            timeout_sec=42,
        )
    )

    assert events == [
        ("run", "python -V", {"timeout": 42}),
        ("run", "pwd", {"timeout": 42}),
        ("run", "ls -la", {"timeout": 42}),
        ("get_host", 31000),
    ]
    assert [item["cmd"] for item in lease.metadata["sandbox_cmd_results"]] == [
        "python -V",
        "pwd",
        "ls -la",
    ]


def test_e2b_provider_skips_missing_or_empty_sandbox_cmd():
    asyncio.run(_run_e2b_provider_skips_missing_or_empty_sandbox_cmd())


async def _run_e2b_provider_skips_missing_or_empty_sandbox_cmd():
    events: list[tuple[Any, ...]] = []

    async def sandbox_factory(**kwargs):
        return FakeE2BSandbox(events)

    provider = E2BSandboxProvider(template="whitebox-template", sandbox_factory=sandbox_factory)
    lease = await provider.create(
        SandboxSpec(
            trajectory_id="traj-1",
            env_args={"sandbox_cmd": ["", "   "]},
            timeout_sec=42,
        )
    )

    assert events == []
    assert "sandbox_cmd_results" not in lease.metadata
    assert "sandbox_cmd_result" not in lease.metadata


def test_e2b_provider_sandbox_cmd_nonzero_exit_aborts_and_kills():
    asyncio.run(_run_e2b_provider_sandbox_cmd_nonzero_exit_aborts_and_kills())


async def _run_e2b_provider_sandbox_cmd_nonzero_exit_aborts_and_kills():
    events: list[tuple[Any, ...]] = []

    async def sandbox_factory(**kwargs):
        return FakeE2BSandbox(
            events,
            results={
                "python -V": FakeE2BCommandResult(stdout="Python\n"),
                "pwd": FakeE2BCommandResult(stderr="bad\n", returncode=2),
            },
        )

    provider = E2BSandboxProvider(template="blackbox-template", sandbox_factory=sandbox_factory)
    with pytest.raises(RuntimeError, match="sandbox_cmd failed"):
        await provider.create(
            SandboxSpec(
                trajectory_id="traj-1",
                env_args={"sandbox_cmd": ["python -V", "pwd", "ls -la"]},
                services=(SandboxServiceSpec(name="blackbox", port=31000),),
                timeout_sec=42,
            )
        )

    assert events == [
        ("run", "python -V", {"timeout": 42}),
        ("run", "pwd", {"timeout": 42}),
        ("kill",),
    ]


def test_e2b_provider_sandbox_cmd_timeout_aborts_and_kills():
    asyncio.run(_run_e2b_provider_sandbox_cmd_timeout_aborts_and_kills())


async def _run_e2b_provider_sandbox_cmd_timeout_aborts_and_kills():
    events: list[tuple[Any, ...]] = []

    async def sandbox_factory(**kwargs):
        return FakeE2BSandbox(
            events,
            results={
                "sleep 10": FakeE2BCommandResult(timed_out=True, returncode=None),
            },
        )

    provider = E2BSandboxProvider(template="blackbox-template", sandbox_factory=sandbox_factory)
    with pytest.raises(RuntimeError, match="sandbox_cmd failed"):
        await provider.create(
            SandboxSpec(
                trajectory_id="traj-1",
                env_args={"sandbox_cmd": ["sleep 10", "echo done"]},
                timeout_sec=42,
            )
        )

    assert events == [
        ("run", "sleep 10", {"timeout": 42}),
        ("kill",),
    ]


def test_e2b_provider_public_url_failure_kills_created_sandbox():
    asyncio.run(_run_e2b_provider_public_url_failure_kills_created_sandbox())


async def _run_e2b_provider_public_url_failure_kills_created_sandbox():
    events: list[tuple[Any, ...]] = []

    class PublicUrlFailureSandbox(FakeE2BSandbox):
        async def get_host(self, port):
            self.events.append(("get_host", port))
            raise RuntimeError("public URL unavailable")

    async def sandbox_factory(**kwargs):
        return PublicUrlFailureSandbox(events)

    provider = E2BSandboxProvider(
        template="blackbox-template",
        sandbox_factory=sandbox_factory,
    )
    with pytest.raises(RuntimeError, match="public URL unavailable"):
        await provider.create(
            SandboxSpec(
                trajectory_id="traj-public-url-failure",
                services=(SandboxServiceSpec(name="blackbox", port=31000),),
            )
        )

    assert events == [("get_host", 31000), ("kill",)]


def test_e2b_provider_rejects_invalid_sandbox_cmd_before_create():
    asyncio.run(_run_e2b_provider_rejects_invalid_sandbox_cmd_before_create())


async def _run_e2b_provider_rejects_invalid_sandbox_cmd_before_create():
    calls = []

    async def sandbox_factory(**kwargs):
        calls.append(kwargs)
        return FakeE2BSandbox([])

    provider = E2BSandboxProvider(template="blackbox-template", sandbox_factory=sandbox_factory)
    with pytest.raises(ValueError, match="sandbox_cmd must be a string or list of strings"):
        await provider.create(
            SandboxSpec(
                trajectory_id="traj-1",
                env_args={"sandbox_cmd": ["python -V", 123]},
                timeout_sec=42,
            )
        )

    assert calls == []


def test_local_bwrap_provider_wraps_ray_pool_manager():
    asyncio.run(_run_local_bwrap_provider_wraps_ray_pool_manager())


async def _run_local_bwrap_provider_wraps_ray_pool_manager():
    manager = FakeLocalManager()
    provider = LocalBwrapSandboxProvider(manager=manager)
    lease = await provider.create(
        SandboxSpec(
            trajectory_id="traj-1",
            env_type="env",
            env_args={"x": 1},
            metadata={"paddock_mode": "blackbox"},
        )
    )

    assert lease.provider == "local_bwrap"
    assert lease.sandbox_id == "lease-traj-1"
    assert lease.capabilities == {"command", "file", "public_url"}
    assert lease.endpoint("blackbox").url == "http://10.0.0.12:31003"

    cmd = await provider.run_command(lease, "echo ok")
    assert cmd.stdout == "ok\n"
    assert await provider.read_file(lease, "/workspace/a.txt") == "hello"
    written = await provider.write_file(lease, "/workspace/a.txt", "hello")
    assert written["bytes"] == 5
    released = await provider.terminate(lease)
    assert released["released"] is True


def test_local_bwrap_provider_releases_slot_when_lease_is_invalid():
    asyncio.run(_run_local_bwrap_provider_releases_slot_when_lease_is_invalid())


async def _run_local_bwrap_provider_releases_slot_when_lease_is_invalid():
    class MissingUrlManager(FakeLocalManager):
        async def acquire(self, trajectory_id, env_type=None, env_args=None):
            payload = await super().acquire(trajectory_id, env_type, env_args)
            payload.pop("sandbox_url", None)
            return payload

    manager = MissingUrlManager()
    provider = LocalBwrapSandboxProvider(manager=manager)

    with pytest.raises(RuntimeError, match="did not return sandbox_url"):
        await provider.create(
            SandboxSpec(
                trajectory_id="traj-invalid",
                metadata={"paddock_mode": "blackbox"},
            )
        )

    assert manager.calls[-1] == (
        "release",
        "traj-invalid",
        "lease-traj-invalid",
        "paddock_create_failed",
    )


def test_local_bwrap_provider_command_only_lease_has_no_blackbox_endpoint():
    asyncio.run(_run_local_bwrap_provider_command_only_lease_has_no_blackbox_endpoint())


async def _run_local_bwrap_provider_command_only_lease_has_no_blackbox_endpoint():
    manager = FakeLocalManager(pool_mode="command_only")
    provider = LocalBwrapSandboxProvider(manager=manager)
    lease = await provider.create(
        SandboxSpec(
            trajectory_id="traj-1",
            metadata={"paddock_mode": "whitebox"},
        )
    )

    assert lease.capabilities == {"command", "file"}
    assert "blackbox" not in lease.endpoints
    with pytest.raises(ValueError, match="does not expose public URLs"):
        await provider.get_public_url(lease, port=31000, service_name="blackbox")


def test_local_bwrap_provider_rejects_paddock_pool_mismatch():
    asyncio.run(_run_local_bwrap_provider_rejects_paddock_pool_mismatch())


async def _run_local_bwrap_provider_rejects_paddock_pool_mismatch():
    manager = FakeLocalManager(pool_mode="blackbox")
    provider = LocalBwrapSandboxProvider(manager=manager)

    with pytest.raises(RuntimeError, match="pool mode does not match paddock mode"):
        await provider.create(
            SandboxSpec(
                trajectory_id="traj-1",
                metadata={"paddock_mode": "whitebox"},
            )
        )


def test_e2b_whitebox_create_does_not_request_public_url():
    asyncio.run(_run_e2b_whitebox_create_does_not_request_public_url())


async def _run_e2b_whitebox_create_does_not_request_public_url():
    class FakeSandbox:
        sandbox_id = "sandbox-1"

        async def get_host(self, port):
            raise AssertionError(f"unexpected public URL request for port {port}")

    calls = []

    async def sandbox_factory(**kwargs):
        calls.append(kwargs)
        return FakeSandbox()

    provider = E2BSandboxProvider(template="whitebox-template", sandbox_factory=sandbox_factory)
    lease = await provider.create(
        SandboxSpec(
            trajectory_id="traj-1",
            metadata={"paddock_mode": "whitebox"},
        )
    )

    assert lease.provider == "e2b"
    assert lease.sandbox_id == "sandbox-1"
    assert lease.capabilities == {"command", "file"}
    assert lease.endpoints == {}
    assert calls[0]["template"] == "whitebox-template"
