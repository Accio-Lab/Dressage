from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dressage.paddock.blackbox.common.defaults import merge_backend_options


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BLACKBOX_GENERATE_FLAG = (
    "--custom-generate-function-path "
    "dressage.rollout.generate.blackbox_dispatch.generate"
)


def _args(rollout_temperature: float = 0.7) -> SimpleNamespace:
    return SimpleNamespace(
        max_tokens_per_gpu=8,
        context_parallel_size=2,
        rollout_max_response_len=4,
        rollout_temperature=rollout_temperature,
    )


def test_opencode_dynamic_defaults_include_proxy_temperature() -> None:
    options = merge_backend_options("opencode", {}, args=_args(0.7))

    assert options["proxy"]["default_temperature"] == 0.7


def test_openclaw_dynamic_defaults_include_proxy_temperature() -> None:
    options = merge_backend_options("openclaw", {}, args=_args(0.7))

    assert options["proxy"]["default_temperature"] == 0.7


def test_claude_code_dynamic_defaults_include_proxy_temperature() -> None:
    options = merge_backend_options("claude_code", {}, args=_args(0.7))

    assert options["proxy"]["default_temperature"] == 0.7


def test_codex_dynamic_defaults_include_proxy_temperature() -> None:
    options = merge_backend_options("codex", {}, args=_args(0.7))

    assert options["proxy"]["default_temperature"] == 0.7


def test_claude_code_static_defaults_include_proxy_model() -> None:
    options = merge_backend_options("claude_code", None)

    assert options["model"]["id"] == "proxy-model"
    assert options["model"]["name"] == "Dressage Proxy"
    assert options["system_prompt_mode"] == "append"
    assert options["gateway"]["auth_token"] == "blackbox-local"


def test_codex_static_defaults_include_proxy_model_and_permissions() -> None:
    options = merge_backend_options("codex", None)

    assert options["model"]["id"] == "proxy-model"
    assert options["model"]["name"] == "Dressage Proxy"
    assert options["model_provider_id"] == "dressage_proxy"
    assert options["sandbox_mode"] == "danger-full-access"
    assert options["approval_policy"] == "never"
    assert options["skip_git_repo_check"] is True
    assert options["ignore_rules"] is True
    assert options["web_search"] == "disabled"


def test_explicit_proxy_temperature_override_wins() -> None:
    options = merge_backend_options(
        "opencode",
        {"proxy": {"default_temperature": 0.2}},
        args=_args(0.7),
    )

    assert options["proxy"]["default_temperature"] == 0.2


def test_opencode_explicit_compact_threshold_is_independent_of_output(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DRESSAGE_BLACKBOX_COMPACT_THRESHOLD", "80000")
    args = SimpleNamespace(
        max_tokens_per_gpu=12000,
        context_parallel_size=8,
        rollout_max_response_len=32768,
        rollout_temperature=1.0,
    )

    options = merge_backend_options("opencode", {}, args=args)

    assert options["model_limit"] == {
        "context": 96000,
        "output": 32768,
        "input": 96000,
    }
    assert options["compaction"]["reserved"] == 16000


def test_openclaw_explicit_compact_threshold_sets_reserve_tokens(monkeypatch) -> None:
    monkeypatch.setenv("DRESSAGE_BLACKBOX_COMPACT_THRESHOLD", "10")

    options = merge_backend_options("openclaw", {}, args=_args())

    assert options["context_window"] == 16
    assert options["max_tokens"] == 4
    assert options["request"]["max_tokens"] == 4
    assert options["compaction"]["reserve_tokens"] == 6
    assert options["compaction"]["reserve_tokens_floor"] == 6


def test_claude_code_explicit_compact_threshold_sets_auto_compact_pct(monkeypatch) -> None:
    monkeypatch.setenv("DRESSAGE_BLACKBOX_COMPACT_THRESHOLD", "10")

    options = merge_backend_options("claude_code", {}, args=_args())

    assert options["compaction"]["auto"] is True
    assert options["compaction"]["auto_compact_pct_override"] == 62


def test_opencode_compact_threshold_rejects_value_over_context(monkeypatch) -> None:
    monkeypatch.setenv("DRESSAGE_BLACKBOX_COMPACT_THRESHOLD", "17")

    try:
        merge_backend_options("opencode", {}, args=_args())
    except ValueError as exc:
        assert "DRESSAGE_BLACKBOX_COMPACT_THRESHOLD" in str(exc)
        assert "context window (16)" in str(exc)
    else:
        raise AssertionError("compact threshold above context should be rejected")


def test_openclaw_compact_threshold_rejects_value_over_context(monkeypatch) -> None:
    monkeypatch.setenv("DRESSAGE_BLACKBOX_COMPACT_THRESHOLD", "17")

    try:
        merge_backend_options("openclaw", {}, args=_args())
    except ValueError as exc:
        assert "DRESSAGE_BLACKBOX_COMPACT_THRESHOLD" in str(exc)
        assert "context window (16)" in str(exc)
    else:
        raise AssertionError("compact threshold above context should be rejected")


def test_claude_code_compact_threshold_rejects_value_over_context(monkeypatch) -> None:
    monkeypatch.setenv("DRESSAGE_BLACKBOX_COMPACT_THRESHOLD", "17")

    try:
        merge_backend_options("claude_code", {}, args=_args())
    except ValueError as exc:
        assert "DRESSAGE_BLACKBOX_COMPACT_THRESHOLD" in str(exc)
        assert "context window (16)" in str(exc)
    else:
        raise AssertionError("compact threshold above context should be rejected")


def test_explicit_reserved_override_wins_over_compact_threshold(monkeypatch) -> None:
    monkeypatch.setenv("DRESSAGE_BLACKBOX_COMPACT_THRESHOLD", "12")

    options = merge_backend_options(
        "opencode",
        {"compaction": {"reserved": 3}},
        args=_args(),
    )

    assert options["model_limit"]["input"] == 16
    assert options["compaction"]["reserved"] == 3


def test_blackbox_max_steps_env_is_added_to_proxy_options(monkeypatch) -> None:
    monkeypatch.setenv("DRESSAGE_BLACKBOX_MAX_STEPS", "37")

    opencode = merge_backend_options("opencode", {}, args=_args())
    openclaw = merge_backend_options("openclaw", {}, args=_args())
    claude_code = merge_backend_options("claude_code", {}, args=_args())
    codex = merge_backend_options("codex", {}, args=_args())

    assert opencode["proxy"]["max_steps"] == 37
    assert openclaw["proxy"]["max_steps"] == 37
    assert claude_code["proxy"]["max_steps"] == 37
    assert codex["proxy"]["max_steps"] == 37


def test_explicit_proxy_max_steps_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("DRESSAGE_BLACKBOX_MAX_STEPS", "37")

    options = merge_backend_options(
        "opencode",
        {"proxy": {"max_steps": 12}},
        args=_args(),
    )

    assert options["proxy"]["max_steps"] == 12


def test_blackbox_max_steps_env_zero_disables_limit(monkeypatch) -> None:
    monkeypatch.setenv("DRESSAGE_BLACKBOX_MAX_STEPS", "0")

    options = merge_backend_options("opencode", {}, args=_args())

    assert options["proxy"]["max_steps"] is None


def test_blackbox_max_steps_env_rejects_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("DRESSAGE_BLACKBOX_MAX_STEPS", "-1")

    try:
        merge_backend_options("opencode", {}, args=_args())
    except ValueError as exc:
        assert "DRESSAGE_BLACKBOX_MAX_STEPS" in str(exc)
    else:
        raise AssertionError("negative max steps should be rejected")


def test_blackbox_ray_scripts_forward_blackbox_env_in_runtime_env() -> None:
    scripts = sorted((_REPO_ROOT / "examples" / "scripts").rglob("*.sh"))
    blackbox_scripts = [
        path
        for path in scripts
        if _BLACKBOX_GENERATE_FLAG in path.read_text(encoding="utf-8")
    ]

    assert blackbox_scripts
    for path in blackbox_scripts:
        text = path.read_text(encoding="utf-8")
        assert (
            '"DRESSAGE_BLACKBOX_MAX_STEPS": "${DRESSAGE_BLACKBOX_MAX_STEPS}"'
            in text
        ), f"{path} does not forward max steps through Ray runtime_env"
        assert (
            '"DRESSAGE_BLACKBOX_COMPACT_THRESHOLD": '
            '"${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD}"'
            in text
        ), f"{path} does not forward compact threshold through Ray runtime_env"


def test_docker_scripts_include_codex_binary_env() -> None:
    dockerfile = (_REPO_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    run_script = (_REPO_ROOT / "docker" / "run.sh").read_text(encoding="utf-8")

    assert "CODEX_BIN=/usr/local/bin/codex" in dockerfile
    assert "ARG CODEX_VERSION=0.142.5" in dockerfile
    assert "chatgpt.com/codex/install.sh" in dockerfile
    assert '--release "${CODEX_VERSION}"' in dockerfile
    assert "/usr/local/bin/codex --version | grep -F \"${CODEX_VERSION}\"" in dockerfile
    assert "npm install --global" not in dockerfile
    assert "@openai/codex" not in dockerfile
    assert '-e "CODEX_BIN=/usr/local/bin/codex"' in run_script
