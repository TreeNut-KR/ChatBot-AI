from typing import Optional
from .ai_models.llama_character_model import LlamaCharacterModel
from .handlers.mongodb_handler import MongoDBHandler
from .handlers import error_handler as ChatError

RED = "\033[31m"
RESET = "\033[0m"

LlamaCharacter_model: Optional[LlamaCharacterModel] = None
mongo_handler: Optional[MongoDBHandler] = None

try:
    LlamaCharacter_model = LlamaCharacterModel()
    mongo_handler = MongoDBHandler()  # 비동기 초기화는 lifespan에서!
except ChatError.InternalServerErrorException as e:
    mongo_handler = None
    print(f"{RED}ERROR{RESET}:    MongoDB 초기화 오류 발생: {str(e)}")
except FileNotFoundError as e:
    LlamaCharacter_model = None
    print(f"{RED}ERROR{RESET}:    모델 파일 오류: {str(e)}")
except Exception as e:
    LlamaCharacter_model = None
    print(f"{RED}ERROR{RESET}:    예상치 못한 오류: {str(e)}")