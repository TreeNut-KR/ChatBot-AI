# Desc: Package initializer for the utils module
"""
utils 패키지 초기화 모듈

이 모듈은 utils 패키지의 초기화를 담당하며, 다음과 같은 하위 모듈들을 포함합니다:

AI Models:
    - lumimaid_model (Lumimaid): Lumimaid 모델을 사용하는 대화 생성 모델
    - bllossom_model (Bllossom): Bllossom 모델을 사용하는 대화 생성 모델
    - openai_office_model (OpenAiOffice): OpenAI GPT 모델을 사용하는 대화 생성 모델
    - openai_character_model (OpenAiCharacter): OpenAI GPT 모델을 사용하는 대화 생성 모델

Handlers:
    - error_handler (ChatError): FastAPI 예외 처리
    - language_handler (LanguageProcessor): 자연어 처리
    - mongodb_handler (MongoDBHandler): MongoDB 데이터베이스 처리

Schemas:
    - chat_schema (ChatModel): FastAPI Pydantic 모델 정의

Routers:
    - office_controller (office_router): 대화형 AI 에이전트 관련 API 라우터
    - character_controller (chearacter_router): 대화형 AI 캐릭터 관련 API 라우터

Services:
    - search_service(ChatSearch): 구글 검색 서비스

App State:
    - app_state (AppState): FastAPI 애플리케이션 GPU 할당 클래스 전역 객체 상태 관리
"""

# AI Models
from .ai_models.lumimaid_model import LumimaidChatModel as Lumimaid
from .ai_models.bllossom_model import BllossomChatModel as Bllossom
from .ai_models.openai_office_model import OpenAIChatModel as OpenAiOffice
from .ai_models.openai_character_model import OpenAICharacterModel as OpenAiCharacter

# Handlers
from .handlers import error_handler as ChatError
from .handlers.language_handler import LanguageProcessor
from .handlers.mongodb_handler import MongoDBHandler

# Schemas
from .schemas import chat_schema as ChatModel

# Services
from .services import search_service as ChatSearch

# Routers
from .routers import office_controller as OfficeController
from .routers import character_controller as ChearacterController


__all__ = [
    # AI Models
    'Lumimaid',
    'Bllossom',
    'OpenAiOffice',
    'OpenAiCharacter',
    
    # Handlers
    'ChatError',
    'LanguageProcessor',
    'MongoDBHandler',
    
    # Schemas
    'ChatModel',
    
    # Services
    'ChatSearch',

    # Routers
    'OfficeController',
    'ChearacterController',
]