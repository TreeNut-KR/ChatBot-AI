from typing import Optional
from .ai_models.llama_office_model import LlamaOfficeModel
from .ai_models.llama_character_model import LlamaCharacterModel
from .handlers.mongodb_handler import MongoDBHandler
from .handlers import error_handler as ChatError

RED = "\033[31m"
RESET = "\033[0m"

LlamaOffice_model: Optional[LlamaOfficeModel] = None
LlamaCharacter_model: Optional[LlamaCharacterModel] = None

try:
    mongo_handler: Optional[MongoDBHandler] = MongoDBHandler()
except ChatError.InternalServerErrorException as e:
    mongo_handler = None
    print(f"{RED}ERROR{RESET}:    MongoDB 초기화 오류 발생: {str(e)}")