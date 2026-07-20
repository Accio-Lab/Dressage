#!/usr/bin/env python3
"""Materialize Dressage DAPO JSONL as an immutable local Harbor dataset.

This converter is an explicit dataset-generation tool.  Harbor runners consume
only the generated JobConfig and never import or invoke this module.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import errno
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Sequence


SCHEMA_VERSION = "dressage.harbor.dapo-dataset/v1"
SUPPORTED_HARBOR_VERSION = "0.18.0"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_CACHE_ROOT = Path("/root/dressage-harbor/datasets")
_SAFE_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ALLOWED_TEXT_CONTROLS = frozenset("\n\r\t")


class DapoDatasetError(ValueError):
    """The DAPO source or requested materialization is invalid."""


@dataclass(frozen=True, slots=True)
class DapoRecord:
    source_index: int
    instance_id: str
    prompt: str
    label: str
    source_blackbox_type: str
    task_type: str


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    root: Path
    tasks_dir: Path
    job_config_path: Path
    manifest_path: Path
    fingerprint: str
    task_count: int


_TEST_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

python3 /tests/verify.py \\
  --stream /logs/agent/claude-code.txt \\
  --expected /tests/expected.json \\
  --reward /logs/verifier/reward.json \\
  --details /logs/verifier/details.json
"""


_VERIFIER_SOURCE = r'''#!/usr/bin/env python3
"""Score the last Claude Code result with DAPO's contains-label rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def last_result(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    result = None
    for line in lines:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if (
            isinstance(event, dict)
            and event.get("type") == "result"
            and isinstance(event.get("result"), str)
        ):
            result = event["result"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    args = parser.parse_args()

    reward = 0
    found_result = False
    label = None
    result = None
    mismatch_reason = "verifier_error"
    error = None
    try:
        expected = json.loads(args.expected.read_text(encoding="utf-8"))
        label = expected.get("label") if isinstance(expected, dict) else None
        if not isinstance(label, str) or not label:
            mismatch_reason = "invalid_expected_label"
            raise ValueError("expected.json contains no non-empty label")
        result = last_result(args.stream)
        found_result = result is not None
        if result is None:
            mismatch_reason = "missing_result"
        elif label in result:
            reward = 1
            mismatch_reason = None
        else:
            mismatch_reason = "label_not_found"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    args.reward.parent.mkdir(parents=True, exist_ok=True)
    args.reward.write_text(json.dumps({"reward": reward}) + "\n", encoding="utf-8")
    details = {
        "expected_label": label,
        "found_result": found_result,
        "matched": bool(reward),
        "mismatch_reason": mismatch_reason,
        "result_length": len(result) if result is not None else None,
        "result_tail": result[-512:] if result is not None else None,
    }
    if error is not None:
        details["error"] = error
    args.details.write_text(json.dumps(details, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _error(path: Path, line_number: int, message: str) -> DapoDatasetError:
    return DapoDatasetError(f"{path}:{line_number}: {message}")


def _unexpected_control(value: str) -> int | None:
    for character in value:
        if ord(character) < 0x20 and character not in _ALLOWED_TEXT_CONTROLS:
            return ord(character)
    return None


def _parse_records(source: Path, source_bytes: bytes) -> tuple[DapoRecord, ...]:
    try:
        lines = source_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DapoDatasetError(f"DAPO source {source} is not valid UTF-8") from exc

    records: list[DapoRecord] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _error(source, line_number, f"invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise _error(source, line_number, "record must be a JSON object")

        prompt = payload.get("prompt")
        if not isinstance(prompt, list) or len(prompt) != 1:
            raise _error(source, line_number, "prompt must contain exactly one message")
        message = prompt[0]
        if not isinstance(message, dict) or message.get("role") != "user":
            raise _error(source, line_number, "prompt message must have role='user'")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise _error(source, line_number, "prompt content must be non-empty")

        label = payload.get("label")
        if not isinstance(label, str) or not label:
            raise _error(source, line_number, "label must be a non-empty string")
        if payload.get("reward_fn") != "contains_label":
            raise _error(source, line_number, "reward_fn must equal 'contains_label'")

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise _error(source, line_number, "metadata must be a JSON object")
        instance_id = metadata.get("instance_id")
        if not isinstance(instance_id, str) or not _SAFE_INSTANCE_ID.fullmatch(instance_id):
            raise _error(
                source,
                line_number,
                "metadata.instance_id must be a safe non-empty path component",
            )
        if instance_id in seen_ids:
            raise _error(source, line_number, f"duplicate instance_id {instance_id!r}")
        seen_ids.add(instance_id)

        for field, value in (("prompt", content), ("label", label)):
            codepoint = _unexpected_control(value)
            if codepoint is not None:
                raise _error(
                    source,
                    line_number,
                    f"{field} for instance_id {instance_id!r} contains "
                    f"unexpected control character U+{codepoint:04X}",
                )

        source_blackbox_type = payload.get("blackbox_type")
        if not isinstance(source_blackbox_type, str) or not source_blackbox_type:
            raise _error(source, line_number, "blackbox_type must be non-empty")
        task_type = payload.get("task_type")
        if not isinstance(task_type, str) or not task_type:
            raise _error(source, line_number, "task_type must be non-empty")
        records.append(
            DapoRecord(
                source_index=len(records),
                instance_id=instance_id,
                prompt=content,
                label=label,
                source_blackbox_type=source_blackbox_type,
                task_type=task_type,
            )
        )

    if not records:
        raise DapoDatasetError(f"DAPO source {source} contains no records")
    return tuple(records)


def load_records(path: str | Path) -> tuple[DapoRecord, ...]:
    """Parse and validate the complete DAPO source, independent of task limit."""

    source = Path(path).expanduser().resolve()
    try:
        source_bytes = source.read_bytes()
    except OSError as exc:
        raise DapoDatasetError(f"failed to read DAPO source {source}: {exc}") from exc
    return _parse_records(source, source_bytes)


def parse_limit(value: str | int | None, *, total: int) -> int:
    if value is None or (isinstance(value, str) and value.strip().lower() == "all"):
        return total
    if isinstance(value, bool):
        raise DapoDatasetError("task limit must be a positive integer or 'all'")
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise DapoDatasetError("task limit must be a positive integer or 'all'") from exc
    if limit < 1 or limit > total:
        raise DapoDatasetError(f"task limit must be between 1 and {total}, got {limit}")
    return limit


def _fingerprint(
    source_bytes: bytes,
    *,
    limit: int,
    model: str,
) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "limit": limit,
        "model": model,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _task_toml(record: DapoRecord) -> str:
    return f'''schema_version = "1.3"
artifacts = ["/logs/agent/claude-code.txt"]

[task]
name = {_toml_string(f"dressage/{record.instance_id}")}
authors = []
keywords = ["dapo", "math", "harbor"]

[metadata]
instance_id = {_toml_string(record.instance_id)}
source_index = {record.source_index}
source_blackbox_type = {_toml_string(record.source_blackbox_type)}
task_type = {_toml_string(record.task_type)}

[agent]
timeout_sec = 3600.0
network_mode = "allowlist"
allowed_hosts = ["127.0.0.1"]

[verifier]
timeout_sec = 120.0
environment_mode = "separate"

[verifier.environment]
network_mode = "no-network"
cpus = 1
memory_mb = 512
storage_mb = 1024
gpus = 0
workdir = "/app"

[environment]
network_mode = "allowlist"
allowed_hosts = ["127.0.0.1"]
cpus = 1
memory_mb = 1024
storage_mb = 2048
gpus = 0
workdir = "/app"
'''


def _write_text(path: Path, content: str, *, mode: int = 0o644) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)


def _write_task(tasks_dir: Path, record: DapoRecord) -> None:
    task_dir = tasks_dir / record.instance_id
    (task_dir / "environment").mkdir(parents=True, mode=0o755)
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(mode=0o755)
    _write_text(task_dir / "instruction.md", record.prompt.rstrip() + "\n")
    _write_text(task_dir / "task.toml", _task_toml(record))
    _write_text(
        tests_dir / "expected.json",
        json.dumps({"label": record.label}, ensure_ascii=False) + "\n",
    )
    _write_text(tests_dir / "verify.py", _VERIFIER_SOURCE, mode=0o755)
    _write_text(tests_dir / "test.sh", _TEST_SCRIPT, mode=0o755)


def _job_payload(
    tasks_dir: Path,
    *,
    model: str,
) -> dict[str, object]:
    return {
        "job_name": "dressage-harbor-dapo-bwrap",
        "jobs_dir": "/root/dressage-harbor/jobs/dapo",
        "n_attempts": 1,
        "quiet": True,
        "retry": {"max_retries": 0},
        "environment": {
            "import_path": (
                "dressage.integrations.harbor.environment:DressageEnvironment"
            )
        },
        "agents": [
            {
                "name": "claude-code",
                "model_name": model,
            }
        ],
        "datasets": [{"path": str(tasks_dir)}],
        "tasks": [],
    }


def _manifest_payload(
    *,
    source: Path,
    source_bytes: bytes,
    records: Sequence[DapoRecord],
    total_records: int,
    fingerprint: str,
    model: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "source": str(source),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "total_source_records": total_records,
        "selected_records": len(records),
        "agent": "claude-code",
        "model": model,
        "source_blackbox_types": dict(
            sorted(Counter(item.source_blackbox_type for item in records).items())
        ),
        "tasks": [
            {
                "instance_id": item.instance_id,
                "source_index": item.source_index,
                "source_blackbox_type": item.source_blackbox_type,
            }
            for item in records
        ],
    }


def _validate_cached(root: Path, *, fingerprint: str, task_count: int) -> PreparedDataset:
    manifest_path = root / "manifest.json"
    job_path = root / "job.json"
    tasks_dir = root / "tasks"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DapoDatasetError(f"invalid cached DAPO dataset at {root}: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("fingerprint") != fingerprint
        or manifest.get("selected_records") != task_count
        or not job_path.is_file()
        or not tasks_dir.is_dir()
        or sum(path.is_dir() for path in tasks_dir.iterdir()) != task_count
    ):
        raise DapoDatasetError(f"cached DAPO dataset is incomplete or mismatched: {root}")
    return PreparedDataset(
        root=root,
        tasks_dir=tasks_dir,
        job_config_path=job_path,
        manifest_path=manifest_path,
        fingerprint=fingerprint,
        task_count=task_count,
    )


def prepare_dataset(
    source: str | Path,
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    limit: str | int | None = 1,
    model: str = DEFAULT_MODEL,
) -> PreparedDataset:
    source_path = Path(source).expanduser().resolve()
    if not isinstance(model, str) or not model.strip():
        raise DapoDatasetError("model must be a non-empty string")
    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise DapoDatasetError(f"failed to read DAPO source {source_path}: {exc}") from exc
    records = _parse_records(source_path, source_bytes)
    selected_count = parse_limit(limit, total=len(records))
    selected = records[:selected_count]
    fingerprint = _fingerprint(
        source_bytes,
        limit=selected_count,
        model=model,
    )
    cache = Path(cache_root).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = cache / fingerprint
    if target.exists():
        return _validate_cached(
            target, fingerprint=fingerprint, task_count=selected_count
        )

    temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.", dir=cache))
    try:
        tasks_dir = temporary / "tasks"
        tasks_dir.mkdir(mode=0o755)
        for record in selected:
            _write_task(tasks_dir, record)
        job = _job_payload(
            target / "tasks",
            model=model,
        )
        _write_text(
            temporary / "job.json",
            json.dumps(job, indent=2, sort_keys=True) + "\n",
        )
        manifest = _manifest_payload(
            source=source_path,
            source_bytes=source_bytes,
            records=selected,
            total_records=len(records),
            fingerprint=fingerprint,
            model=model,
        )
        _write_text(
            temporary / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        try:
            temporary.rename(target)
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not target.exists():
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _validate_cached(target, fingerprint=fingerprint, task_count=selected_count)


def prepared_dataset_identity(job_config: str | Path) -> dict[str, str]:
    """Read and verify the preparation identity adjacent to a JobConfig."""

    job_path = Path(job_config).expanduser().resolve()
    manifest_path = job_path.with_name("manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DapoDatasetError(
            f"failed to read DAPO manifest next to {job_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise DapoDatasetError(f"invalid DAPO manifest at {manifest_path}")
    fingerprint = manifest.get("fingerprint")
    source_sha256 = manifest.get("source_sha256")
    source_value = manifest.get("source")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise DapoDatasetError(f"invalid Dataset fingerprint in {manifest_path}")
    if not isinstance(source_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_sha256
    ):
        raise DapoDatasetError(f"invalid source SHA-256 in {manifest_path}")
    if not isinstance(source_value, str) or not source_value:
        raise DapoDatasetError(f"invalid source path in {manifest_path}")

    source_path = Path(source_value).expanduser()
    if source_path.is_file():
        actual_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_sha256 != source_sha256:
            raise DapoDatasetError(
                f"DAPO source {source_path} no longer matches prepared Dataset "
                f"{fingerprint}; expected SHA-256 {source_sha256}, got {actual_sha256}. "
                "Run prepare_dataset.py again and update DRESSAGE_HARBOR_JOB_CONFIG."
            )
    return {
        "job_config": str(job_path),
        "manifest": str(manifest_path),
        "source": str(source_path),
        "source_sha256": source_sha256,
        "fingerprint": fingerprint,
    }


def check_runtime() -> None:
    if sys.version_info[:2] < (3, 12):
        raise DapoDatasetError(
            f"Harbor training requires Python >=3.12; got {sys.version_info.major}.{sys.version_info.minor}"
        )
    try:
        version = importlib.metadata.version("harbor")
    except importlib.metadata.PackageNotFoundError as exc:
        raise DapoDatasetError("Harbor is not installed in the active Python environment") from exc
    if version != SUPPORTED_HARBOR_VERSION:
        raise DapoDatasetError(
            f"Harbor {SUPPORTED_HARBOR_VERSION} is required; got {version}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a content-addressed Harbor dataset from Dressage DAPO JSONL"
    )
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--describe-job", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--limit", default="1", help="positive integer or 'all'")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check_runtime:
        check_runtime()
        return 0
    if args.describe_job is not None:
        identity = prepared_dataset_identity(args.describe_job)
        print(f"Harbor JobConfig: {identity['job_config']}")
        print(f"DAPO Dataset fingerprint: {identity['fingerprint']}")
        print(f"DAPO source SHA-256: {identity['source_sha256']}")
        print(f"DAPO source: {identity['source']}")
        return 0
    if args.input is None:
        raise DapoDatasetError("--input is required unless --check-runtime is used")
    prepared = prepare_dataset(
        args.input,
        cache_root=args.cache_root,
        limit=args.limit,
        model=args.model,
    )
    print(prepared.job_config_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DapoDatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
