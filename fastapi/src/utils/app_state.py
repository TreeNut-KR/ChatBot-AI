from typing import Optional
from .ai_models.bllossom_model import BllossomChatModel
from .ai_models.lumimaid_model import LumimaidChatModel
from .handlers.mongodb_handler import MongoDBHandler
from .handlers import error_handler as ChatError

RED = "\033[31m"
RESET = "\033[0m"

Bllossom_model: Optional[BllossomChatModel] = None
Lumimaid_model: Optional[LumimaidChatModel] = None

try:
    mongo_handler: Optional[MongoDBHandler] = MongoDBHandler()
except ChatError.InternalServerErrorException as e:
    mongo_handler = None
    print(f"{RED}ERROR{RESET}:    MongoDB 초기화 오류 발생: {str(e)}")