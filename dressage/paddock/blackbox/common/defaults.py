"""Default blackbox sandbox and backend configuration."""

from __future__ import annotations

import copy
import os
from typing import Any

DEFAULT_BLACKBOX_TYPE = "opencode"

BLACKBOX_MAX_STEPS_ENV = "DRESSAGE_BLACKBOX_MAX_STEPS"
BLACKBOX_COMPACT_THRESHOLD_ENV = "DRESSAGE_BLACKBOX_COMPACT_THRESHOLD"

DEFAULT_SERVER_CONFIG = {
    "runtime_root": "/workspace_sandbox/blackbox_server_runtime",
    "backend_timeout": 1800,
}

DEFAULT_OPENCODE_COMPACTION: dict[str, Any] = {
    "auto": True,
    "prune": True,
}

_OPENCODE_STATIC_BACKEND_DEFAULTS: dict[str, Any] = {
    "provider_id": "sglang",
    "provider_name": "Dressage Proxy",
    "provider_package": "@ai-sdk/openai-compatible",
    "model_id": "proxy-model",
    "model_name": "Dressage Proxy",
}

_OPENCLAW_STATIC_BACKEND_DEFAULTS: dict[str, Any] = {
    "agent_id": "default",
    "provider_id": "sglang",
    "model_id": "proxy-model",
    "model_name": "Dressage Proxy",
    "api_key": "sglang-local",
}

_CLAUDE_CODE_STATIC_BACKEND_DEFAULTS: dict[str, Any] = {
    "model": {
        "id": "proxy-model",
        "name": "Dressage Proxy",
        "supported_capabilities": [
            "thinking",
            "adaptive_thinking",
            "interleaved_thinking",
        ],
    },
    "max_turns": 20,
    "permission_mode": "default",
    "setting_sources": "user",
    "system_prompt_mode": "append",
    "gateway": {
        "auth_token": "blackbox-local",
    },
    "thinking": {
        "enabled": True,
        "interleaved": True,
        "budget_tokens": None,
    },
    "compaction": {
        "auto": True,
    },
    "compat": {
        "disable_prompt_caching": True,
        "disable_nonessential_traffic": True,
    },
    "subagents": {
        "enabled": False,
    },
}

_CODEX_STATIC_BACKEND_DEFAULTS: dict[str, Any] = {
    "model": {
        "id": "proxy-model",
        "name": "Dressage Proxy",
    },
    "model_provider_id": "dressage_proxy",
    "sandbox_mode": "danger-full-access",
    "approval_policy": "never",
    "skip_git_repo_check": True,
    "ignore_rules": True,
    "web_search": "disabled",
}

_KNOWN_BACKEND_DEFAULTS: dict[str, dict[str, Any]] = {
    "opencode": _OPENCODE_STATIC_BACKEND_DEFAULTS,
    "openclaw": _OPENCLAW_STATIC_BACKEND_DEFAULTS,
    "claude_code": _CLAUDE_CODE_STATIC_BACKEND_DEFAULTS,
    "codex": _CODEX_STATIC_BACKEND_DEFAULTS,
}

_DYNAMIC_BACKEND_KEYS: dict[str, tuple[str, ...]] = {
    "opencode": ("model_limit", "compaction", "proxy"),
    "openclaw": ("context_window", "max_tokens", "request", "compaction", "proxy"),
    "claude_code": ("compaction", "proxy"),
    "codex": ("proxy",),
}


def normalize_blackbox_type(blackbox_type: Any) -> str:
    """Return the canonical registry key for a blackbox backend."""
    value = str(blackbox_type or DEFAULT_BLACKBOX_TYPE).strip().lower()
    return value or DEFAULT_BLACKBOX_TYPE


def server_config_for(blackbox_type: Any) -> dict[str, Any]:
    """Return blackbox server defaults for a blackbox backend."""
    del blackbox_type
    return copy.deepcopy(DEFAULT_SERVER_CONFIG)


def backend_defaults_for(blackbox_type: Any, args: Any | None = None) -> dict[str, Any]:
    """Return backend registration defaults for a blackbox backend."""
    backend = normalize_blackbox_type(blackbox_type)
    defaults = copy.deepcopy(_KNOWN_BACKEND_DEFAULTS.get(backend, {}))
    defaults = _deep_merge(defaults, _environment_backend_defaults(backend))
    if args is not None and backend in _DYNAMIC_BACKEND_KEYS:
        defaults = _deep_merge(defaults, dynamic_backend_defaults_for(backend, args))
    return defaults


def dynamic_backend_defaults_for(blackbox_type: Any, args: Any) -> dict[str, Any]:
    """Return rollout-argument-derived backend defaults."""
    backend = normalize_blackbox_type(blackbox_type)
    if backend == "codex":
        return {
            "proxy": {
                "default_temperature": _rollout_temperature_arg(args),
            },
        }

    limits = _dynamic_token_defaults(args, backend)

    if backend == "opencode":
        return {
            "model_limit": {
                "context": limits["context"],
                "output": limits["output"],
                "input": limits["input"],
            },
            "proxy": {
                "default_temperature": _rollout_temperature_arg(args),
            },
            "compaction": {
                **DEFAULT_OPENCODE_COMPACTION,
                "reserved": limits["reserved"],
            },
        }

    if backend == "openclaw":
        return {
            "context_window": limits["context"],
            "max_tokens": limits["output"],
            "request": {
                "max_tokens": limits["output"],
            },
            "proxy": {
                "default_temperature": _rollout_temperature_arg(args),
            },
            "compaction": {
                "reserve_tokens": limits["reserved"],
                "reserve_tokens_floor": limits["reserved"],
            },
        }

    if backend == "claude_code":
        compaction: dict[str, Any] = {"auto": True}
        compact_threshold = _compact_threshold(limits["context"], backend)
        if compact_threshold is not None:
            compaction["auto_compact_pct_override"] = max(
                1,
                min(100, int(compact_threshold * 100 / limits["context"])),
            )
        return {
            "proxy": {
                "default_temperature": _rollout_temperature_arg(args),
            },
            "compaction": compaction,
        }

    return {}


def merge_backend_options(
    blackbox_type: Any,
    backend_options: Any,
    *,
    args: Any | None = None,
) -> Any:
    """Merge user backend options over defaults without mutating the input."""
    backend = normalize_blackbox_type(blackbox_type)
    if backend_options is not None and not isinstance(backend_options, dict):
        return backend_options

    defaults = copy.deepcopy(_KNOWN_BACKEND_DEFAULTS.get(backend, {}))
    defaults = _deep_merge(defaults, _environment_backend_defaults(backend))
    if _should_add_dynamic_defaults(backend, backend_options, args):
        defaults = _deep_merge(defaults, dynamic_backend_defaults_for(backend, args))

    if backend_options is None:
        return defaults
    if not defaults:
        return copy.deepcopy(backend_options)
    return _deep_merge(defaults, backend_options)


def _environment_backend_defaults(backend: str) -> dict[str, Any]:
    if backend not in _DYNAMIC_BACKEND_KEYS:
        return {}

    raw_max_steps = os.environ.get(BLACKBOX_MAX_STEPS_ENV)
    if raw_max_steps is None or not raw_max_steps.strip():
        return {}

    normalized = raw_max_steps.strip().lower()
    if normalized == "0":
        max_steps: int | None = None
    else:
        try:
            max_steps = int(normalized)
        except ValueError as exc:
            raise ValueError(
                f"{BLACKBOX_MAX_STEPS_ENV} must be a positive integer or 0 "
                f"to disable the limit, got {raw_max_steps!r}"
            ) from exc
        if max_steps <= 0:
            raise ValueError(
                f"{BLACKBOX_MAX_STEPS_ENV} must be a positive integer or 0 "
                f"to disable the limit, got {raw_max_steps!r}"
            )

    return {"proxy": {"max_steps": max_steps}}


def _should_add_dynamic_defaults(
    backend: str,
    backend_options: dict[str, Any] | None,
    args: Any | None,
) -> bool:
    if args is None or backend not in _DYNAMIC_BACKEND_KEYS:
        return False
    if backend_options is None:
        return True
    dynamic_keys = _DYNAMIC_BACKEND_KEYS[backend]
    return any(
        key not in backend_options or isinstance(backend_options[key], dict)
        for key in dynamic_keys
    )


def _dynamic_token_defaults(args: Any, backend: str) -> dict[str, int]:
    max_tokens_per_gpu = _positive_int_arg(
        args, "max_tokens_per_gpu", "--max-tokens-per-gpu", backend
    )
    cp_size = _context_parallel_size(args)
    default_output = _positive_int_arg(
        args, "rollout_max_response_len", "--rollout-max-response-len", backend
    )
    default_context = max_tokens_per_gpu * cp_size
    legacy_input_limit = default_context - default_output
    if legacy_input_limit <= 0:
        raise ValueError(
            f"{backend} backend_options require "
            "--max-tokens-per-gpu * --context-parallel-size to be greater than "
            "--rollout-max-response-len; got "
            f"{max_tokens_per_gpu} * {cp_size} = {default_context} and "
            f"--rollout-max-response-len={default_output}"
        )
    compact_threshold = _compact_threshold(default_context, backend)
    if compact_threshold is None:
        default_input = legacy_input_limit
        proportional_reserved = max(1, legacy_input_limit // 4)
        default_reserved = min(
            proportional_reserved,
            8192,
            legacy_input_limit - 1,
        )
    else:
        # OpenCode/OpenClaw compact at context/input limit minus reserved tokens.
        default_input = default_context
        default_reserved = default_context - compact_threshold
    return {
        "context": default_context,
        "output": default_output,
        "input": default_input,
        "reserved": default_reserved,
    }


def _compact_threshold(context: int, backend: str) -> int | None:
    if backend not in {"opencode", "openclaw", "claude_code"}:
        return None

    raw_value = os.environ.get(BLACKBOX_COMPACT_THRESHOLD_ENV)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        threshold = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{BLACKBOX_COMPACT_THRESHOLD_ENV} must be a positive integer no "
            f"greater than the context window, got {raw_value!r}"
        ) from exc
    if threshold <= 0 or threshold > context:
        raise ValueError(
            f"{BLACKBOX_COMPACT_THRESHOLD_ENV} must be a positive integer no "
            f"greater than the context window ({context}), got {raw_value!r}"
        )
    return threshold


def _positive_int_arg(args: Any, attr: str, flag: str, backend: str) -> int:
    value = getattr(args, attr, None)
    if value is None:
        raise ValueError(
            f"{backend} backend_options require {flag}; adjust the training script "
            "to provide --max-tokens-per-gpu, --context-parallel-size, and "
            "--rollout-max-response-len"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{flag} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{flag} must be a positive integer, got {value!r}")
    return parsed


def _rollout_temperature_arg(args: Any) -> float:
    value = getattr(args, "rollout_temperature", 1.0)
    if value is None:
        return 1.0
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--rollout-temperature must be a non-negative float, got {value!r}"
        ) from exc
    if parsed < 0:
        raise ValueError(
            f"--rollout-temperature must be a non-negative float, got {value!r}"
        )
    return parsed


def _context_parallel_size(args: Any) -> int:
    value = getattr(args, "context_parallel_size", None)
    if value is None:
        value = getattr(args, "cp_size", 1)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--context-parallel-size must be a positive integer, got {value!r}"
        ) from exc
    if parsed <= 0:
        raise ValueError(
            f"--context-parallel-size must be a positive integer, got {value!r}"
        )
    return parsed


def _deep_merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
