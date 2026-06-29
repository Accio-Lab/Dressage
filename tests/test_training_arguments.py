from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from dressage.rollout.staleness import config_from_args


def test_staleness_config_uses_args_attribute_set_by_custom_config():
    args = SimpleNamespace(dressage_staleness_keep_versions=3)

    assert config_from_args(args).keep_versions == 3


def test_generated_custom_config_file_carries_staleness():
    config_path = Path("examples/scripts/default/dressage_staleness.yaml")
    data = yaml.safe_load(config_path.read_text())

    assert set(data) == {"dressage_staleness_keep_versions"}
    config = config_from_args(SimpleNamespace(**data))
    assert config.enabled is (data["dressage_staleness_keep_versions"] > 0)


def test_slime_common_parser_loads_custom_config_attributes():
    source = Path("slime/slime/utils/arguments.py").read_text()

    assert '"--custom-config-path"' in source
    assert "if args.custom_config_path:" in source
    assert "setattr(args, k, v)" in source


def test_whitebox_async_scripts_use_staleness_yaml_and_record_versions():
    script_paths = {
        Path("examples/scripts/run_hotpotqa_whitebox_agent_qwen3.5_4b_async.sh"): "hotpotqa",
        Path("examples/scripts/run_alfworld_whitebox_agent_qwen3.5_4b_async.sh"): "alfworld",
    }

    for path, recipe_name in script_paths.items():
        source = path.read_text()
        assert '--custom-config-path "${SCRIPT_DIR}/default/dressage_staleness.yaml"' in source
        assert "--record-token-versions" in source
        assert f"# --wandb-group {recipe_name}-qwen3.5-4B-whitebox-async" in source
        assert "DRESSAGE_STALENESS_KEEP_VERSIONS" not in source
        assert "DRESSAGE_CUSTOM_CONFIG_PATH" not in source
        assert "wandb_v1_" not in source
        assert "WANDB_KEY:-" not in source


def test_whitebox_partial_async_scripts_enable_staleness_controls():
    script_paths = sorted(
        Path("examples/scripts").glob(
            "run_*_whitebox_agent_qwen3.5_4b_partial_rollout_async.sh"
        )
    )
    assert script_paths

    for path in script_paths:
        source = path.read_text()
        assert "--rollout-function-path dressage.rollout.partial_async_rollout.generate_rollout_partial_async" in source
        assert "ROLLOUT_FUNCTION_PATH" not in source
        assert '--custom-config-path "${SCRIPT_DIR}/default/dressage_staleness.yaml"' in source
        assert "DRESSAGE_CUSTOM_CONFIG_PATH" not in source
        assert "DRESSAGE_WHITEBOX_PARTIAL_ROLLOUT" in source
        assert "--dressage-partial-rollout" in source
        assert "--mask-nonlast-version-tokens" in source
        assert "wandb_v1_" not in source
        assert "WANDB_KEY:-" not in source


def test_blackbox_sync_scripts_do_not_use_staleness_custom_config():
    script_paths = [
        Path("examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local.sh"),
        Path("examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_remote.sh"),
    ]

    for path in script_paths:
        source = path.read_text()
        assert "dressage.rollout.sync_rollout.generate_rollout_sync" in source
        assert '--custom-config-path "${SCRIPT_DIR}/default/dressage_staleness.yaml"' not in source
        assert "DRESSAGE_CUSTOM_CONFIG_PATH" not in source


def test_blackbox_async_scripts_use_staleness_custom_config():
    defaults_source = Path("examples/scripts/default/dressage_env_defaults.sh").read_text()
    assert "DRESSAGE_CUSTOM_CONFIG_PATH" not in defaults_source

    script_paths = [
        Path("examples/scripts/run_blackbox_qwen3.5_4b_async_local.sh"),
        Path("examples/scripts/run_blackbox_qwen3.5_4b_async_remote.sh"),
        Path("examples/scripts/run_blackbox_qwen3.5_4b_partial_rollout_async_local.sh"),
        Path("examples/scripts/run_blackbox_qwen3.5_4b_partial_rollout_async_remote.sh"),
    ]

    for path in script_paths:
        source = path.read_text()
        assert '--custom-config-path "${SCRIPT_DIR}/default/dressage_staleness.yaml"' in source
        assert "DRESSAGE_CUSTOM_CONFIG_PATH" not in source


def test_blackbox_async_scripts_record_token_versions_for_staleness():
    script_paths = [
        Path("examples/scripts/run_blackbox_qwen3.5_4b_async_local.sh"),
        Path("examples/scripts/run_blackbox_qwen3.5_4b_async_remote.sh"),
        Path("examples/scripts/run_blackbox_qwen3.5_4b_partial_rollout_async_local.sh"),
        Path("examples/scripts/run_blackbox_qwen3.5_4b_partial_rollout_async_remote.sh"),
    ]

    for path in script_paths:
        source = path.read_text()
        assert "--record-token-versions" in source


def test_train_async_entrypoint_uses_slime_common_parser():
    source = Path("dressage/training/train_async_with_rollout_pause.py").read_text()

    assert "from slime.utils.arguments import parse_args" in source
    assert "add_dressage_arguments" not in source
    assert "train(parse_args())" in source
