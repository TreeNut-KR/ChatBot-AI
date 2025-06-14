# Desc: Package initializer for the utils module
"""
utils 패키지 초기화 모듈

이 모듈은 utils 패키지의 초기화를 담당하며, 다음과 같은 하위 모듈들을 포함합니다:

AI Models:
    - LlamaCharacter_model (LlamaCharacter): DarkIdol-Llama-3.1-8B 모델을 사용하는 대화 생성 모델
    - openai_character_model (OpenAiCharacter): OpenAI GPT 모델을 사용하는 대화 생성 모델

Handlers:
    - error_handler (ChatError): FastAPI 예외 처리
    - mongodb_handler (MongoDBHandler): MongoDB 데이터베이스 처리

Schemas:
    - chat_schema (ChatModel): FastAPI Pydantic 모델 정의

Services:
    - search_service(ChatSearch): 구글 검색 서비스

Routers:
    - character_controller (chearacter_router): 대화형 AI 캐릭터 관련 API 라우터
"""

# AI Models
from .ai_models.openai_character_model import OpenAICharacterModel as OpenAiCharacter

# Handlers
from .handlers import error_handler as ChatError
from .handlers.mongodb_handler import MongoDBHandler

# Schemas
from .schemas import chat_schema as ChatModel

# Services
from .services import search_service as ChatSearch

# Routers
from .routers import character_controller as CharacterController


__all__ = [
    # AI Models
    'OpenAiCharacter',
    
    # Handlers
    'ChatError',
    'MongoDBHandler',
    
    # Schemas
    'ChatModel',
    
    # Services
    'ChatSearch',

    # Routers
    'OfficeController',
    'CharacterController',
]