"""Harbor integration primitives.

The package deliberately does not import Harbor at module import time.  This
keeps the base Dressage package importable on Python 3.10 and in environments
where the optional Harbor dependency is not installed.
"""

from dressage.integrations.harbor.config import (
    HARBOR_INTEGRATION_SCHEMA_VERSION,
    HarborIntegrationConfig,
    load_config,
)

__all__ = [
    "HARBOR_INTEGRATION_SCHEMA_VERSION",
    "HarborIntegrationConfig",
    "load_config",
]

