#!/usr/bin/env python3
"""Inspect/release exact trajectory keys after a failed job; never resume it.

Run inside the GPU project container with the failed run's normal GPU env.
Default is read-only. This tool refuses unsettled CPU ownership.
"""

import argparse
import asyncio
import fcntl
import json
import urllib.parse
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-dir", type=Path, required=True)
    parser.add_argument("--failed-submission", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--settle-executions", action="store_true")
    args = parser.parse_args()
    if args.settle_executions and not args.apply:
        parser.error("--settle-executions requires --apply")

    import ray
    from agentic_rollout_contract.canonical import canonical_digest
    from dressage_distroll_gpu_adapter.config import remote_rollout_settings_from_env
    from dressage_distroll_gpu_adapter.execution import (
        ExecutionQueue,
        identity,
        write_artifact,
    )
    from dressage_distroll_gpu_adapter.trajectory_delivery import release_contexts
    from ray.util.state import list_actors

    settings = remote_rollout_settings_from_env()
    root = args.step_dir.resolve()
    state = json.loads((root / "state.json").read_text())
    payload = state["payload"]
    if (
        state["run_id"] != settings.run_id
        or state["digest"] != canonical_digest(payload)
        or root.parent.name != identity(settings.run_id)
    ):
        raise ValueError("state identity mismatch")
    records = [{"execution_id": key, **r} for key, r in payload["executions"].items()]
    if any(not r.get("settled") for r in records) and not args.settle_executions:
        raise RuntimeError(
            "CPU ownership not fully settled; use formal execution drain first"
        )
    submission = urllib.parse.quote(args.failed_submission, safe="")
    job = json.load(
        urllib.request.urlopen(
            f"http://127.0.0.1:8265/api/jobs/{submission}", timeout=15
        )
    )
    env = (job.get("runtime_env") or {}).get("env_vars", {})
    if (
        job["status"] not in {"FAILED", "STOPPED"}
        or not job.get("job_id")
        or env.get("DRESSAGE_DISTROLL_RUN_ID") != settings.run_id
    ):
        raise RuntimeError("job identity or terminal failure not proven")
    ray.init(address="auto", logging_level="ERROR")
    try:
        actors = list_actors(filters=[("job_id", "=", job["job_id"])], limit=1000)
        if not actors or len(actors) >= 1000 or any(a.state != "DEAD" for a in actors):
            raise RuntimeError("old readers are not all proven DEAD")
        report = {
            "run_id": settings.run_id,
            "step": payload["step"],
            "job_id": job["job_id"],
            "dead_actors": len(actors),
            "state": "checked",
        }

        async def clean():
            queue = ExecutionQueue(settings, root=root, proxy=None, materialize=None)
            try:
                # This is operational drain after all readers died, not trajectory
                # recovery. Reuse the same Control cancel/ACK and Policy proof.
                queue.payload = payload
                queue.policy_instance = payload["policy_instance"]
                for execution_id, record in payload["executions"].items():
                    if not record.get("settled"):
                        await queue._cancel_and_close(execution_id)
                await release_contexts(
                    queue.policy,
                    records,
                    policy_instance=payload["policy_instance"],
                    call=queue.call,
                )
            finally:
                await queue.close()

        if args.apply:
            with (root / "owner.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Refuse a stale snapshot read before acquiring the writer lock.
                if json.loads((root / "state.json").read_text()) != state:
                    raise RuntimeError("step changed before cleanup lock acquisition")
                path = root / "delivery-cleanup.json"
                write_artifact(path, json.dumps(report).encode())
                asyncio.run(clean())
                report["state"] = "released_after_failure"
                write_artifact(path, json.dumps(report).encode())
        print(json.dumps(report))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
