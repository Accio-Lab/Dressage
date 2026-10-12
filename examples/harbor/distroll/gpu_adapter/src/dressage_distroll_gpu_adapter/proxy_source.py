"""Reviewed private Proxy source transformation; the original tree stays intact."""

import argparse
import hashlib
from pathlib import Path

SOURCE_SHA = "642a3de0057f4d21a107a6f63508c42b8bc86b474743798b09a58aa3687fd23f"
ANCHOR = """            if candidate is None:
                continue
            snapshot_rendered = _candidate_snapshot_rendered(
"""
REPLACEMENT = """            if candidate is None:
                continue
            if token_build_mode == "tito":
                from .tito.template_utils import assert_messages_append_only_with_allowed_roles

                try:
                    assert_messages_append_only_with_allowed_roles(
                        candidate.normalized_messages_snapshot,
                        normalized_request_messages,
                        tito_tokenizer.allowed_append_roles,
                    )
                except ValueError:
                    logger.info(
                        "TITO candidate rejected before incremental build: session=%s step=%s",
                        session.session_id, candidate.step_id,
                    )
                    continue
            snapshot_rendered = _candidate_snapshot_rendered(
"""


def patched_source(source):
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA:
        raise ValueError("unreviewed Dressage Proxy source")
    text = source.decode()
    if text.count(ANCHOR) != 1:
        raise ValueError("TITO guard patch anchor mismatch")
    text = text.replace(ANCHOR, REPLACEMENT)
    return text.encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    data = patched_source(args.source.read_bytes())
    compile(data, str(args.target), "exec")
    if args.target.exists():
        if args.target.read_bytes() != data:
            raise ValueError("existing private Proxy differs; refusing overwrite")
    else:
        args.target.parent.mkdir(parents=True, exist_ok=True)
        args.target.write_bytes(data)
    print(hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
