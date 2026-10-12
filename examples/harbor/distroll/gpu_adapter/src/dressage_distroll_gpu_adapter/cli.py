from __future__ import annotations

import uvicorn

from .config import policy_settings_from_env
from .policy_service import create_policy_app


def policy_main() -> None:
    settings = policy_settings_from_env()
    uvicorn.run(
        create_policy_app(settings),
        host=settings.listen_host,
        port=settings.listen_port,
        log_level="info",
    )
