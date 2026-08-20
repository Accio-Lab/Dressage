from __future__ import annotations

import hashlib
import json
import os
from typing import Any
from urllib.parse import urlparse, urlunparse

from blackbox_server.core.models import Message, RegisterRequest


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: canonicalize(value[key]) for key in sorted(value) if value[key] is not None}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    return value


def stable_json(value: Any) -> str:
    return json.dumps(canonicalize(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def normalize_router(router: str, router_api_path: str = "/v1") -> tuple[str, str]:
    raw = router.strip()
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    if not parsed.netloc:
        raise ValueError(f"Invalid router value: {router}")
    effective_path = parsed.path or router_api_path or "/v1"
    if not effective_path.startswith("/"):
        effective_path = f"/{effective_path}"
    normalized_base = urlunparse((parsed.scheme or "http", parsed.netloc, effective_path, "", "", ""))
    return normalized_base.rstrip("/"), effective_path


def normalize_system_prompt_file(path: str | None) -> str | None:
    if path is None:
        return None
    return os.path.abspath(path)


def binding_request_fingerprint(request: RegisterRequest, router_base_url: str) -> str:
    system_prompt_file = normalize_system_prompt_file(request.system_prompt_file)
    server_config = {}
    if request.server_config is not None:
        server_config = request.server_config.explicit_values()
    payload = {
        "blackbox_type": request.blackbox_type,
        "router_base_url": router_base_url,
        "bound_session_id": request.bound_session_id,
        "bound_instance_id": request.bound_instance_id,
        "bound_sandbox_id": request.bound_sandbox_id,
        "system_prompt_file": system_prompt_file,
        "backend_options": request.backend_options,
        "server_config": server_config,
    }
    return sha256_hex(payload)


def message_request_fingerprint(messages: list[Message]) -> str:
    payload = [message.model_dump(mode="json", exclude_none=True) for message in messages]
    return sha256_hex(payload)
