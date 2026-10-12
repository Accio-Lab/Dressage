"""Bounded, metadata-only JSONL diagnostics. Never pass request bodies here."""

import json
import logging
import os
from datetime import UTC, datetime
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path


@lru_cache(maxsize=1)
def _logger():
    logger = logging.getLogger("dressage.runtime_debug")
    logger.setLevel(logging.INFO)
    path = os.environ.get("DRESSAGE_DEBUG_LOG") or os.environ.get("POLICY_DEBUG_LOG")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        max_bytes = int(
            os.environ.get("DRESSAGE_DEBUG_LOG_MAX_BYTES", str(32 * 1024 * 1024))
        )
        backups = int(os.environ.get("DRESSAGE_DEBUG_LOG_BACKUPS", "4"))
        if max_bytes < 1024 or backups < 1:
            raise ValueError("debug log rotation requires positive bounded storage")
        handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def event(name, **fields):
    _logger().info(
        json.dumps(
            {
                "ts": datetime.now(UTC).isoformat(),
                "event": name,
                **fields,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
