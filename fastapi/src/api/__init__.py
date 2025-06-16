import os
APP_MODE = os.getenv("APP_MODE", "")  # 기본값: 빈 문자열

if APP_MODE == "character":
    from .character import llm_controller as character_llm_router
    __all__ = [
        "character_llm_router"
    ]

elif APP_MODE == "office":
    from .office import llm_controller as office_llm_router
    __all__ = [
        "office_llm_router"
    ]

else:
    __all__ = []
