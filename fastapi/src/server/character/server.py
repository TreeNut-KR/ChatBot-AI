'''
파일은 Character FastAPI 서버를 구동하는 엔트리 포인트입니다.
'''
import os
import yaml
import sys
import uvicorn
import ipaddress

from pathlib import Path
from dotenv import load_dotenv
from pydantic import ValidationError
from contextlib import asynccontextmanager

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import character_app_state as AppState
from domain  import error_tools as ChatError
from api import character_llm_router

env_file_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_file_path)

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 애플리케이션의 수명 주기를 관리하는 컨텍스트 매니저입니다.
    
    Args:
        app (FastAPI): FastAPI 애플리케이션 인스턴스
        
    Yields:
        None: 애플리케이션 컨텍스트를 생성하고 종료할 때까지 대기
    """    
    try:
        assert AppState.llama_queue_handler is not None, "LlamaQueueHandler is not initialized"
        
        # 큐 핸들러 초기화 및 시작
        await AppState.llama_queue_handler.init()
        await AppState.llama_queue_handler.start()
        
        if AppState.mongo_handler is not None:
            await AppState.mongo_handler.init()
            
    except AssertionError as e:
        print(f"{RED}ERROR{RESET}:    {str(e)}")
    print(f"{GREEN}INFO{RESET}:     큐 핸들러 로드 완료")

    yield

    # 큐 핸들러 정지
    if AppState.llama_queue_handler:
        await AppState.llama_queue_handler.stop()
    AppState.llama_queue_handler = None
    print(f"{GREEN}INFO{RESET}:     큐 핸들러 정지 완료")

app = FastAPI(
    lifespan=lifespan,
)

def custom_openapi():
    """
    커스텀 OpenAPI 스키마를 생성하는 함수입니다.
    
    Returns:
        dict: OpenAPI 스키마 정의
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title = "ChatBot-AI FastAPI Character",
        version = "v1.6.*",
        routes = app.routes,
        description = (
            "이 API는 다음과 같은 기능을 제공합니다:\n\n"
            f"각 엔드포인트의 자세한 정보는 [📌 ChatBot-AI FastAPI 명세서](https://github.com/TreeNut-KR/ChatBot-AI/issues/4) 에서 확인할 수 있습니다."
        ),
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://drive.google.com/thumbnail?id=12PqUS6bj4eAO_fLDaWQmoq94-771xfim"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

ChatError.ExceptionManager.register(app) # 예외 핸들러 추가

app.add_middleware(
    SessionMiddleware,
    secret_key = os.getenv("SESSION_KEY", "default-secret")
)
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)
app.openapi = custom_openapi

@app.middleware("http")
async def ip_restrict_and_bot_blocking_middleware(request: Request, call_next):
    """
    IP 제한과 봇 차단을 처리하는 미들웨어입니다.
    
    Args:
        request (Request): 들어오는 HTTP 요청
        call_next (callable): 다음 미들웨어나 라우트 핸들러를 호출하는 함수
        
    Returns:
        Response: HTTP 응답 객체
        
    Raises:
        ChatError.IPRestrictedException: 허용되지 않은 IP 주소
        ChatError.BadRequestException: 봇 접근 시도
    """
    def load_bot_list(file_path: str) -> list:
        """
        YAML 파일에서 봇 리스트를 불러오는 함수입니다.
        
        Args:
            file_path (str): 봇 목록이 저장된 YAML 파일의 경로
            
        Returns:
            list: 소문자로 변환된 봇 이름 리스트
        """
        with open(file_path, 'r', encoding = 'utf-8') as file:
            data = yaml.safe_load(file)
            return [bot['name'].lower() for bot in data.get('bot_user_agents', [])]

    bot_user_agents = load_bot_list("/bot.yaml") # Docker 컨테이너 내 절대 경로로 수정
    user_agent = request.headers.get("User-Agent", "").lower()

    try:
        # 사용자 에이전트 기반 봇 차단
        if any(bot in user_agent for bot in bot_user_agents):
            raise ChatError.BadRequestException(detail = f"{user_agent} Bot access is not allowed.")

        response = await call_next(request)
        return response

    except ValidationError as e:
        raise ChatError.BadRequestException(detail = str(e))
    except ChatError.IPRestrictedException as e:
        return await ChatError.ExceptionHandlerFactory.generic_handler(request, e)
    except ChatError.BadRequestException as e:
        # 올바른 함수명 사용
        return await ChatError.ExceptionHandlerFactory.generic_handler(request, e)
    except HTTPException as e:
        if (e.status_code  ==  405):
            raise ChatError.MethodNotAllowedException(detail = "The method used is not allowed.")
        raise e
    except Exception as e:
        raise ChatError.InternalServerErrorException(detail = "Internal server error occurred.")

app.include_router(
    character_llm_router.character_router,
    prefix = "",
    tags = ["character Router"],
    responses = {500: {"description": "Internal Server Error"}}
)

if __name__  ==  "__main__":
    uvicorn.run(
        app,
        host = "0.0.0.0",
        port = 8003,
        http = "h11",
        loop="asyncio"
    )
