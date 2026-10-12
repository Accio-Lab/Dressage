"""Run upstream Proxy unchanged, with an out-of-event-loop diagnostic watchdog.

The watchdog only probes loopback health and records stacks, never restarts,
cancels, or repairs anything. Stack dumps exclude locals and request contents.
"""

import faulthandler
import importlib.util
import os
import runpy
import sys
import tempfile
import threading
import time
from http.client import HTTPException
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from .runtime_debug import event


def _health():
    with build_opener(ProxyHandler({})).open(
        "http://127.0.0.1:8800/health", timeout=5
    ) as response:
        return response.status == 200


def _watch(stop):
    failures = 0
    last_stack = float("-inf")
    while not stop.wait(30):
        started = time.monotonic()
        error = None
        try:
            healthy = _health()
        except (OSError, HTTPException) as exc:
            healthy, error = False, type(exc).__name__
        failures = 0 if healthy else failures + 1
        event(
            "proxy_health",
            healthy=healthy,
            consecutive_failures=failures,
            elapsed_sec=round(time.monotonic() - started, 3),
            error_type=error,
        )
        if failures >= 3 and started - last_stack >= 300:
            last_stack = started
            try:
                with tempfile.TemporaryFile(mode="w+b") as output:
                    faulthandler.dump_traceback(file=output, all_threads=True)
                    output.seek(0)
                    stack = output.read(1024 * 1024).decode("utf-8", errors="replace")
                event("proxy_unresponsive_stacks", stack=stack)
            except (OSError, RuntimeError, ValueError) as exc:
                event("proxy_stack_failed", error_type=type(exc).__name__)


def main():
    event(
        "proxy_watch_started", interval_sec=30, timeout_sec=5, failures_before_stack=3
    )
    stop = threading.Event()
    threading.Thread(
        target=_watch, args=(stop,), name="proxy-health-watch", daemon=True
    ).start()
    ray_client = None
    try:
        if os.environ.get("TQ_SOURCE_ROOT") and "--enable-transfer-queue" in sys.argv:
            import ray

            if ray.is_initialized():
                raise RuntimeError(
                    "Proxy requires its own reviewed TQ runtime environment"
                )
            ray.init(
                address="auto",
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": os.environ["PYTHONPATH"],
                    }
                },
            )
            ray_client = ray
            event("tq_private_runtime", source_root=os.environ["TQ_SOURCE_ROOT"])
        private_source = os.environ.get("DRESSAGE_PROXY_SOURCE")
        if private_source:
            import dressage.proxy.server as upstream

            from .proxy_source import patched_source

            path = Path(private_source)
            if path.read_bytes() != patched_source(
                Path(upstream.__file__).read_bytes()
            ):
                raise RuntimeError("private Proxy does not match the reviewed patch")
            name = "dressage.proxy._distroll_reviewed_server"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            event("proxy_private_source", path=str(path))
            module.main()
        else:
            runpy.run_module("dressage.proxy.server", run_name="__main__")
    finally:
        stop.set()
        if ray_client is not None:
            ray_client.shutdown()


if __name__ == "__main__":
    main()
