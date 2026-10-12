from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def canonical_bytes(value: BaseModel | dict[str, Any] | list[Any]) -> bytes:
    if isinstance(value, BaseModel):
        payload: Any = value.model_dump(mode="json", exclude_none=True)
    else:
        payload = value
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_digest(value: BaseModel | dict[str, Any] | list[Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()
