import os
APP_MODE = os.getenv("APP_MODE", "")  # 기본값: 빈 문자열

from .shared import (
    base_config,
    error_tools,
    mongodb_client,
    queue_tools,
    search_adapter,
)

if APP_MODE == "character":
    from .character import (
        config as character_config,
        schema as character_schema,
    )
    __all__ = [
        "character_config",
        "character_schema",
        
        "base_config",
        "error_tools",
        "mongodb_client",
        "queue_tools",
        "search_adapter",
    ]
    
elif APP_MODE == "office":
    from .office import (
        config as office_config,
        schema as office_schema,
    )
    __all__ = [
        "office_config",
        "office_schema",
        
        "base_config",
        "error_tools",
        "mongodb_client",
        "queue_tools",
        "search_adapter",
    ]
    
else:
    __all__ = [
        "base_config",
        "error_tools",
        "mongodb_client",
        "queue_tools",
        "search_adapter",
    ]
