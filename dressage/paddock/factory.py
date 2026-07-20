"""Factory for Dressage paddock modes."""

from __future__ import annotations

import importlib
import os
from typing import Any

from dressage.config import paddock_mode
from dressage.paddock.interface import Paddock


_PUBLIC_MODES = {"blackbox", "whitebox"}


def load_object(path: str) -> Any:
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"invalid object path: {path}")
    return getattr(importlib.import_module(module_path), attr)


def create_paddock_from_env(*, mode: str | None = None) -> Paddock:
    """Create the configured paddock.

    ``DRESSAGE_PADDOCK_CLASS`` remains the explicit advanced override.  The
    default path is controlled by ``DRESSAGE_PADDOCK_MODE`` and
    ``DRESSAGE_SANDBOX_PROVIDER``.
    """

    class_path = os.environ.get("DRESSAGE_PADDOCK_CLASS")
    if class_path:
        return load_object(class_path)()

    mode = (
        mode or os.environ.get("DRESSAGE_PADDOCK_MODE") or paddock_mode()
    ).strip().lower()
    if mode == "blackbox":
        from dressage.paddock.blackbox.paddock import BlackboxAgentPaddock

        return BlackboxAgentPaddock()
    if mode == "whitebox":
        from dressage.paddock.whitebox.paddock import WhiteboxToolPaddock

        return WhiteboxToolPaddock()
    expected = "|".join(sorted(_PUBLIC_MODES))
    raise ValueError(f"unsupported DRESSAGE_PADDOCK_MODE={mode!r}; expected {expected}")
