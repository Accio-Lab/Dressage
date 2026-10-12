#!/usr/bin/env python3
"""Request a run-scoped, cooperative cancellation barrier."""

import argparse
import fcntl
import json
import os
import re
from pathlib import Path

from agentic_rollout_contract.canonical import canonical_digest
from dressage_distroll_gpu_adapter.execution import identity, write_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--diagnostics-dir", required=True, type=Path)
    parser.add_argument(
        "--after-step",
        action="store_true",
        help="Let this step finish, then stop the next step before any execution is submitted",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", args.run_id):
        parser.error("invalid run ID")
    if args.step < 0:
        parser.error("step must be non-negative")
    root = (
        args.diagnostics_dir / "rollout" / identity(args.run_id) / f"step-{args.step}"
    )
    if not root.is_dir():
        parser.error("durable RL step directory does not exist")
    descriptor = os.open(root / "commit.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        state_path = root / "state.json"
        if not state_path.is_file() or (root / "commit.json").exists():
            parser.error("step has no active durable state or is already committed")
        state = json.loads(state_path.read_bytes())
        payload = state.get("payload")
        if (
            state.get("run_id") != args.run_id
            or not isinstance(payload, dict)
            or state.get("digest") != canonical_digest(payload)
            or payload.get("step") != args.step
            or payload.get("committed")
        ):
            parser.error("durable step identity or digest does not match")
        target_step = args.step + 1 if args.after_step else args.step
        target = root.parent / f"step-{target_step}"
        if args.after_step and (target / "state.json").exists():
            parser.error(
                "next step already started; refuse to schedule a late boundary stop"
            )
        target.mkdir(exist_ok=True)
        marker = json.dumps({"run_id": args.run_id, "step": target_step}).encode()
        write_artifact(target / "stop-request.json", marker)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    print(
        f"cooperative stop requested: {args.run_id} step={target_step} after_step={args.after_step}"
    )


if __name__ == "__main__":
    main()
