#!/usr/bin/env python3
"""Fail-closed checks for image dependencies plus host-mounted GPU source."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Any


def _load_lock(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "dressage-distroll-gpu-runtime-lock/v1":
        raise RuntimeError(f"unsupported GPU runtime lock: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str | None:
    if (root / ".git").exists():
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    marker = root / ".dressage-source-revision"
    if not marker.is_file():
        return None
    revision = marker.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError(f"invalid source revision marker: {marker}")
    return revision


def _module_root(module_name: str) -> tuple[Path, Path]:
    module = importlib.import_module(module_name)
    origin = Path(module.__file__).resolve()
    return origin, origin.parent.parent


def _check_symbol(value: str) -> str:
    module_name, symbol_name = value.split(":", 1)
    module = importlib.import_module(module_name)
    symbol = getattr(module, symbol_name)
    if not callable(symbol):
        raise TypeError(f"configured symbol is not callable: {value}")
    return f"{module.__file__}:{symbol_name}"


def _verify_source(
    name: str,
    root: Path,
    lock: dict[str, Any],
    *,
    expected_root: Path,
) -> tuple[str | None, dict[str, str]]:
    source_lock = lock["sources"][name]
    if root != expected_root:
        raise RuntimeError(
            f"{name} source shadowing detected: expected {expected_root}, got {root}"
        )
    revision = _git_revision(root)
    expected_revision = source_lock["revision"]
    if revision is None:
        raise RuntimeError(f"{name} source revision evidence is missing: {root}")
    if name == "dressage" and (root / ".git").exists():
        # The example's feature commits change HEAD, not the reviewed core.
        for arguments in (
            ["merge-base", "--is-ancestor", expected_revision, "HEAD"],
            [
                "diff",
                "--quiet",
                expected_revision,
                "--",
                "dressage",
                "blackbox_server",
                "pyproject.toml",
            ],
        ):
            subprocess.run(
                ["git", "-c", f"safe.directory={root}", "-C", str(root), *arguments],
                check=True,
            )
    elif revision != expected_revision:
        raise RuntimeError(
            f"{name} revision mismatch: expected {expected_revision}, got {revision}"
        )

    actual_hashes = {}
    for relative, expected_hash in lock["source_hashes"][name].items():
        source_path = root / relative
        if not source_path.is_file():
            raise RuntimeError(f"missing {name} source file: {source_path}")
        actual_hash = _sha256(source_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{name} source hash mismatch for {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        actual_hashes[relative] = actual_hash
    expected_patch = source_lock.get("patch_sha256")
    if expected_patch is not None:
        patch_marker = root / ".dressage-patch-sha256"
        if not patch_marker.is_file():
            raise RuntimeError(f"{name} patch evidence is missing: {patch_marker}")
        actual_patch = patch_marker.read_text(encoding="utf-8").strip()
        if actual_patch != expected_patch:
            raise RuntimeError(
                f"{name} patch mismatch: expected {expected_patch}, got {actual_patch}"
            )
    return revision, actual_hashes


def _probe_runtime_impl(lock: dict[str, Any]) -> dict[str, Any]:
    dressage_origin, dressage_root = _module_root("dressage")
    slime_origin, slime_root = _module_root("slime")
    megatron_training = importlib.import_module("megatron.training")
    megatron_origin = Path(megatron_training.__file__).resolve()
    adapter_origin, adapter_root = _module_root("dressage_distroll_gpu_adapter")
    contract_origin, contract_root = _module_root("agentic_rollout_contract")

    source_root = Path(os.environ["GPU_SOURCE_ROOT"]).resolve()
    expected_roots = {
        "dressage": Path(os.environ["DRESSAGE_ROOT"]).resolve(),
        "slime": Path(os.environ["SLIME_ROOT"]).resolve(),
        "megatron": Path(os.environ["MEGATRON_ROOT"]).resolve(),
    }
    expected_project_roots = {
        "dressage_distroll_gpu_adapter": source_root / "gpu_adapter" / "src",
        "agentic_rollout_contract": source_root / "contract" / "src",
    }
    for name, (actual, expected) in {
        "dressage_distroll_gpu_adapter": (
            adapter_root,
            expected_project_roots["dressage_distroll_gpu_adapter"],
        ),
        "agentic_rollout_contract": (
            contract_root,
            expected_project_roots["agentic_rollout_contract"],
        ),
    }.items():
        if actual != expected.resolve():
            raise RuntimeError(
                f"{name} must come from host source: expected {expected}, got {actual}"
            )
    dressage_revision, dressage_hashes = _verify_source(
        "dressage",
        dressage_root,
        lock,
        expected_root=expected_roots["dressage"],
    )
    slime_revision, slime_hashes = _verify_source(
        "slime",
        slime_root,
        lock,
        expected_root=expected_roots["slime"],
    )
    megatron_root = expected_roots["megatron"]
    if not megatron_origin.is_relative_to(megatron_root):
        raise RuntimeError(
            f"megatron.training must come from the host source: {megatron_origin}"
        )
    megatron_revision, megatron_hashes = _verify_source(
        "megatron",
        megatron_root,
        lock,
        expected_root=expected_roots["megatron"],
    )

    versions = {}
    for distribution, expected in lock["packages"].items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise RuntimeError(
                f"package version mismatch for {distribution}: "
                f"expected {expected}, got {actual}"
            )
        versions[distribution] = actual

    symbols = [
        "dressage.proxy.proxy_client:ProxyClient",
        "dressage.rollout.data_source:DressageDataSource",
        "dressage.training.tq_hydration:validate_tq_training_config",
        "dressage.training.tq_megatron_actor:TQMegatronTrainRayActor",
        "dressage.integrations.harbor.gateway:GatewayRuntime",
        "megatron.training.checkpointing:load_checkpoint",
        "dressage_distroll_gpu_adapter.rollout:generate_remote_harbor_sync",
        "dressage_distroll_gpu_adapter.persistent_train:train",
    ]
    resolved_symbols = {value: _check_symbol(value) for value in symbols}

    return {
        "source_root": str(source_root),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "origins": {
            "dressage": str(dressage_origin),
            "slime": str(slime_origin),
            "megatron.training": str(megatron_origin),
            "dressage_distroll_gpu_adapter": str(adapter_origin),
            "agentic_rollout_contract": str(contract_origin),
        },
        "versions": versions,
        "revisions": {
            "dressage": dressage_revision,
            "slime": slime_revision,
            "megatron": megatron_revision,
        },
        "source_hashes": {
            "dressage": dressage_hashes,
            "slime": slime_hashes,
            "megatron": megatron_hashes,
        },
        "symbols": resolved_symbols,
    }


def probe_runtime(lock: dict[str, Any]) -> dict[str, Any]:
    import_output = io.StringIO()
    with contextlib.redirect_stdout(import_output):
        result = _probe_runtime_impl(lock)
    captured = import_output.getvalue().strip()
    if captured:
        result["import_notices"] = captured.splitlines()
    return result


def _runtime_fingerprint(result: dict[str, Any]) -> str:
    stable = {
        key: result[key]
        for key in ("origins", "versions", "revisions", "source_hashes", "symbols")
    }
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _check_ray(
    lock: dict[str, Any], expected_nodes: int | None
) -> list[dict[str, Any]]:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=0)
    def remote_probe(runtime_lock: dict[str, Any]) -> dict[str, Any]:
        result = probe_runtime(runtime_lock)
        result["node_id"] = ray.get_runtime_context().get_node_id()
        return result

    ray.init(address="auto", logging_level="ERROR", log_to_driver=False)
    try:
        nodes = [node for node in ray.nodes() if node.get("Alive")]
        if expected_nodes is not None and len(nodes) != expected_nodes:
            raise RuntimeError(
                f"Ray node count mismatch: expected {expected_nodes}, got {len(nodes)}"
            )
        handles = []
        expected_ids = []
        for node in nodes:
            node_id = node["NodeID"]
            expected_ids.append(node_id)
            handles.append(
                remote_probe.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=False,
                    )
                ).remote(lock)
            )
        results = ray.get(handles)
    finally:
        ray.shutdown()

    actual_ids = [result["node_id"] for result in results]
    if set(actual_ids) != set(expected_ids) or len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError(
            f"Ray per-node probe coverage mismatch: expected {expected_ids}, got {actual_ids}"
        )
    fingerprints = {_runtime_fingerprint(result) for result in results}
    if len(fingerprints) != 1:
        raise RuntimeError("Ray nodes do not have one identical admitted GPU runtime")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--ray", action="store_true")
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(
            os.environ.get(
                "GPU_RUNTIME_LOCK",
                "/opt/dressage-distroll/configs/gpu-runtime-lock.json",
            )
        ),
    )
    parser.add_argument("--expected-nodes", type=int)
    args = parser.parse_args()

    lock = _load_lock(args.lock)
    if args.local:
        result: Any = probe_runtime(lock)
        result["fingerprint"] = _runtime_fingerprint(result)
    else:
        result = _check_ray(lock, args.expected_nodes)
        for node_result in result:
            node_result["fingerprint"] = _runtime_fingerprint(node_result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
