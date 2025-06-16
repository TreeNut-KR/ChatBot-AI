import os
APP_MODE = os.getenv("APP_MODE", "")  # 기본값: 빈 문자열

if APP_MODE == "character":
    from .character import app_state as character_app_state
    __all__ = [
        "character_app_state"
    ]
elif APP_MODE == "office":
    from .office import app_state as office_app_state
    __all__ = [
        "office_app_state"
    ]
else:
    __all__ = []