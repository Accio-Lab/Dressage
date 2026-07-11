from __future__ import annotations

from pathlib import Path

from dressage import config


def test_log_dir_defaults_to_repo_log(monkeypatch):
    monkeypatch.delenv("LOG_DIR", raising=False)
    monkeypatch.delenv("REPO_ROOT", raising=False)

    assert config.log_dir() == Path(__file__).resolve().parents[1] / "log"


def test_log_helpers_use_log_dir_run_name_and_legacy_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DRESSAGE_RUN_NAME", "run-a")
    monkeypatch.delenv("DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR", raising=False)
    monkeypatch.delenv("DRESSAGE_TRAJECTORY_ERROR_LOG_DIR", raising=False)
    monkeypatch.delenv("PROXY_LOG_FILE", raising=False)
    monkeypatch.delenv("PROXY_PID_FILE", raising=False)

    assert config.trajectory_payload_log_dir() == tmp_path / "logs" / "traj_payload" / "run-a"
    assert config.trajectory_error_log_dir() == tmp_path / "logs" / "traj_err" / "run-a"
    assert config.proxy_log_file() == tmp_path / "logs" / "proxy" / "run-a.log"
    assert config.proxy_pid_file() == tmp_path / "logs" / "proxy" / "run-a.pid"

    monkeypatch.setenv("DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR", str(tmp_path / "legacy"))
    assert config.trajectory_payload_log_dir() == tmp_path / "legacy"


def test_open_source_sandbox_defaults(monkeypatch):
    monkeypatch.delenv("DRESSAGE_SANDBOX_PROVIDER", raising=False)
    monkeypatch.delenv("DRESSAGE_PADDOCK_MODE", raising=False)
    monkeypatch.delenv("DRESSAGE_LOCAL_BWRAP_POOL_MODE", raising=False)

    assert config.sandbox_provider() == "local_bwrap"
    assert config.paddock_mode() == "blackbox"
    assert config.local_bwrap_pool_mode(mode="blackbox") == "blackbox"
    assert config.local_bwrap_pool_mode(mode="whitebox") == "command_only"


def test_token_build_model_qwen35_defaults():
    defaults = config.token_build_defaults(
        token_build_mode="tito",
        token_build_model="qwen3_5",
    )

    assert defaults.model_mask_type == "qwen3_5"
    assert defaults.model_tool_call_type == "qwen3_5"
    assert defaults.model_reasoning_type == "qwen3"
    assert defaults.tito_model == "qwen3_5"
