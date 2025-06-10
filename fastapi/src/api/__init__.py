from .character import llm_controller as character_llm_router
from .office import llm_controller as office_llm_router

__all__ = [
    "character_llm_router",
    "office_llm_router",
]
