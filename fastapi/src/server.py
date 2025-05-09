'''
파일은 FastAPI 서버를 구동하는 엔트리 포인트입니다.
'''
import os
import yaml
import torch
import uvicorn
import ipaddress

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

import utils.app_state as AppState
from utils  import (
    ChatError,
    LlamaOffice,
    LlamaCharacter,
    OfficeController,
    ChearacterController,
)

load_dotenv()

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
    def get_cuda_device_info(device_id: int) -> str:
        """
        주어진 CUDA 장치 ID에 대한 정보를 반환합니다.
        """
        device_name = torch.cuda.get_device_name(device_id)
        device_properties = torch.cuda.get_device_properties(device_id)
        total_memory = device_properties.total_memory / (1024 ** 3)  # GB 단위로 변환
        return f"Device {device_id}: {device_name} (Total Memory: {total_memory:.2f} GB)"
    
    try:
        AppState.LlamaOffice_model = LlamaOffice()                 # cuda:1
        AppState.LlamaCharacter_model = LlamaCharacter()              # cuda:0
    except ChatError.InternalServerErrorException as e:
        component = "MongoDBHandler"
        print(f"{RED}ERROR{RESET}:    {component} 초기화 중 {e.__class__.__name__} 오류 발생: {str(e)}")
        exit(1)

    # 디버깅용 출력
    LlamaOffice_device_info = get_cuda_device_info(1)          # LlamaOffice 모델은 cuda:1
    LlamaCharacter_device_info = get_cuda_device_info(0)          # LlamaCharacter 모델은 cuda:0
    print(f"{GREEN}INFO{RESET}:     LlamaOffice 모델 로드 완료 ({LlamaOffice_device_info})")
    print(f"{GREEN}INFO{RESET}:     LlamaCharacter 모델 로드 완료 ({LlamaCharacter_device_info})")

    yield

    # 모델 메모리 해제
    AppState.LlamaOffice_model = None
    AppState.LlamaCharacter_model = None
    print(f"{GREEN}INFO{RESET}:     모델 해제 완료")

app = FastAPI(lifespan = lifespan)

ChatError.add_exception_handlers(app) # 예외 핸들러 추가
class ExceptionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        """
        HTTP 요청을 처리하고 예외를 처리하는 미들웨어입니다.
        
        Args:
            request (Request): 들어오는 HTTP 요청
            call_next (callable): 다음 미들웨어나 라우트 핸들러를 호출하는 함수
            
        Returns:
            Response: HTTP 응답 객체
        """
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            return await ChatError.generic_exception_handler(request, e)

def custom_openapi():
    """
    커스텀 OpenAPI 스키마를 생성하는 함수입니다.
    
    Returns:
        dict: OpenAPI 스키마 정의
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title = "ChatBot-AI FastAPI",
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


app.add_middleware(ExceptionMiddleware)
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

    def is_internal_ip(ip):
        """
        주어진 IP 주소가 내부 네트워크에 속하는지 확인합니다.
        
        Args:
            ip (str): 확인할 IP 주소 문자열
            
        Returns:
            bool: 내부 IP인 경우 True, 아닌 경우 False
        """
        try:
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj in ipaddress.ip_network(os.getenv("LOCAL_HOST"))
        except ValueError:
            return False

    ip_string = os.getenv("IP")
    allowed_ips = ip_string.split(", ") if ip_string else []
    client_ip = request.client.host

    bot_user_agents = load_bot_list("./fastapi/src/bot.yaml") # 경로 수정
    user_agent = request.headers.get("User-Agent", "").lower()

    try:
        # IP 및 내부 네트워크 범위에 따라 액세스 제한
        restricted_paths = [
            "/docs", "/redoc", "/openapi.json"
        ]
        # /office, /character 하위 모든 경로도 포함
        if (
            request.url.path in restricted_paths
            or request.url.path.startswith("/office")
            or request.url.path.startswith("/character")
        ) and (
            client_ip not in allowed_ips
            and not is_internal_ip(client_ip)
        ):
            raise ChatError.IPRestrictedException(detail = f"Unauthorized IP address: {client_ip}")

        # 사용자 에이전트 기반 봇 차단
        if any(bot in user_agent for bot in bot_user_agents):
            raise ChatError.BadRequestException(detail = f"{user_agent} Bot access is not allowed.")

        response = await call_next(request)
        return response

    except ValidationError as e:
        raise ChatError.BadRequestException(detail = str(e))
    except ChatError.IPRestrictedException as e:
        return await ChatError.generic_exception_handler(request, e)
    except ChatError.BadRequestException as e:
        return await ChatError.generic_exception_handler(request, e)
    except HTTPException as e:
        if (e.status_code  ==  405):
            raise ChatError.MethodNotAllowedException(detail = "The method used is not allowed.")
        raise e
    except Exception as e:
        raise ChatError.InternalServerErrorException(detail = "Internal server error occurred.")

@app.get("/")
async def root(request: Request):
    """
    API 루트 엔드포인트입니다.
    
    Returns:
        dict: 환영 메시지를 포함한 응답
    """
    return {
        "message": "Welcome to ChatBot-AI API. Access from IP: " + request.client.host
    }

app.include_router(
    OfficeController.office_router,
    prefix = "/office",
    tags = ["office Router"],
    responses = {500: {"description": "Internal Server Error"}}
)

app.include_router(
    ChearacterController.character_router,
    prefix = "/character",
    tags = ["character Router"],
    responses = {500: {"description": "Internal Server Error"}}
)

if __name__  ==  "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8001)
