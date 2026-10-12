"""Copy the pinned TQ package into a new private directory and fix request linger.

Never edits the installed package. Must run before starting any new TQ readers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SOURCE_SHA = "dffbf81379b94d0718bf2677721ceb3050da98e9e10fdae77d8cabd7e15a8eb9"
RELATIVE = Path("transfer_queue/utils/zmq_utils.py")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(target):
    record = json.loads((target / "linger-overlay.json").read_text())
    data = (target / RELATIVE).read_bytes()
    restored = data.replace(b"sock.close(linger=0)", b"sock.close(linger=-1)")
    if record.get("source_sha256") != SOURCE_SHA or digest(restored) != SOURCE_SHA:
        raise ValueError("TQ overlay is not the reviewed source plus the linger fix")
    if (
        data.count(b"sock.close(linger=0)") != 1
        or digest(data) != record["patched_sha256"]
    ):
        raise ValueError("TQ overlay content mismatch")
    return record


def prepare(source, target):
    if target.exists():
        return check(target)
    data = (source / "utils/zmq_utils.py").read_bytes()
    if digest(data) != SOURCE_SHA or data.count(b"sock.close(linger=-1)") != 1:
        raise ValueError("unreviewed TransferQueue source; refusing to patch")
    target.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        source,
        target / "transfer_queue",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    patched = data.replace(b"sock.close(linger=-1)", b"sock.close(linger=0)")
    (target / RELATIVE).write_bytes(patched)
    record = {
        "source_sha256": SOURCE_SHA,
        "patched_sha256": digest(patched),
        "change": "request socket close linger -1 -> 0",
    }
    (target / "linger-overlay.json").write_text(json.dumps(record, indent=2) + "\n")
    return check(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and args.source is None:
        parser.error("--source is required for preparation")
    print(
        json.dumps(
            check(args.target) if args.check else prepare(args.source, args.target)
        )
    )


if __name__ == "__main__":
    main()
