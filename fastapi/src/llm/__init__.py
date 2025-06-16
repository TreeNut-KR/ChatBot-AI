import os
APP_MODE = os.getenv("APP_MODE", "")  # 기본값: 빈 문자열

if APP_MODE == "character":
    from .llama import (
        character as character_llama,
    )
    from .openai import (
        character as character_openai,
    )
    __all__ = [
        "character_llama",
        "character_openai",
    ]

elif APP_MODE == "office":
    from .llama import (
        office as office_llama,
    )
    from .openai import (
        office as office_openai,
    )
    __all__ = [
        "office_llama",
        "office_openai",
    ]

else:
    __all__ = []