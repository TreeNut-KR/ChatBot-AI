from .character import(
    config as character_config,
    schema as character_schema,
)
from .office import(
    config as office_config,
    schema as office_schema,
)
from .shared import (
    base_config,
    error_tools,
    mongodb_client,
    queue_tools,
    search_adapter,
)
__all__ = [
    "character_config",
    "character_schema",

    "office_config",
    "office_schema",
    
    # shared
    "base_config",
    "error_tools",
    "mongodb_client",
    "queue_tools",
    "search_adapter",
]