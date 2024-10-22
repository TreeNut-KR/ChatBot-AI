import os
import uvicorn
import asyncio
from dotenv import load_dotenv
from asyncio import TimeoutError
from pydantic import ValidationError
from contextlib import asynccontextmanager

from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi import (APIRouter, Depends, FastAPI, HTTPException, Request)
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

import utils.Models as ChatModel
import utils.Error_handlers as ChatError
from utils.AI_Llama_8B import LlamaChatModel

llama_model = None  # 모델 전역 변수
load_dotenv()

# FastAPI 애플리케이션 초기화
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_model
    llama_model = LlamaChatModel()
    print("Llama 모델 로드 완료")
    yield
    llama_model = None
    print("Llama 모델 해제 완료")

app = FastAPI(lifespan=lifespan)  # 여기서 한 번만 app을 생성합니다.
ChatError.add_exception_handlers(app)  # 예외 핸들러 추가

class ExceptionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # 예외를 Error_handlers에서 정의한 generic_exception_handler로 위임
            return await ChatError.generic_exception_handler(request, e)

app.add_middleware(ExceptionMiddleware)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_KEY", "default-secret")  # 시크릿 키 대신 세션 키만 유지
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="ChatBot-AI FastAPI",
        version="v1.0.1",
        summary="AI 모델 관리 API",
        routes=app.routes,
        description=(
            "이 API는 다음과 같은 기능을 제공합니다:\n\n"
            "각 엔드포인트의 자세한 정보는 해당 엔드포인트의 문서에서 확인할 수 있습니다."
        ),
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://drive.google.com/thumbnail?id=12PqUS6bj4eAO_fLDaWQmoq94-771xfim"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.middleware("http")
async def ip_restrict_and_bot_blocking_middleware(request: Request, call_next):
    ip_string = os.getenv("IP")
    allowed_ips = ip_string.split(",") if ip_string else []
    client_ip = request.client.host

    # 봇의 User-Agent 패턴 목록 (필요에 따라 확장 가능)
    bot_user_agents = [
        "googlebot", "bingbot", "slurp", "duckduckbot", "baiduspider",
        "yandexbot", "sogou", "exabot", "facebot", "ia_archiver"
    ]

    user_agent = request.headers.get("User-Agent", "").lower()

    try:
        # IP 제한
        if request.url.path in ["/Llama", "/docs", "/redoc", "/openapi.json"] and client_ip not in allowed_ips:
            raise ChatError.IPRestrictedException(detail=f"Unauthorized IP address: {client_ip}")
        
        # 봇 차단
        if any(bot in user_agent for bot in bot_user_agents):
            raise ChatError.BadRequestException(detail=f"{user_agent} Bot access is not allowed.")

        # 봇이 아닌 경우에만 다음 처리를 진행 (Llama 호출 포함)
        response = await call_next(request)
        return response

    except ValidationError as e:
        raise ChatError.BadRequestException(detail=str(e))
    
    except ChatError.IPRestrictedException as e:
        return await ChatError.generic_exception_handler(request, e)
    
    except ChatError.BadRequestException as e:
        return await ChatError.generic_exception_handler(request, e)
    
    except HTTPException as e:
        if e.status_code == 405:
            raise ChatError.MethodNotAllowedException(detail="The method used is not allowed.")
        raise e
    
    except Exception as e:
        raise ChatError.InternalServerErrorException(detail="Internal server error occurred.")

@app.get("/")
async def root():
    return {"message": "Welcome to the API"}

@app.post("/Llama", response_model=ChatModel.Llama_Response, summary="Llama 모델 답변 생성")
async def Llama_(request: ChatModel.Llama_Request):
    '''
    Llama 모델에 질문 입력 시 답변 반환.
    '''
    try:
        # 20초 내에 응답이 생성되지 않으면 TimeoutError 발생
        tables = await asyncio.wait_for(run_in_threadpool(llama_model.generate_response, request.input_data), timeout=20.0)
        return {"output_data": tables}
    except TimeoutError:
        raise ChatError.InternalServerErrorException(detail="Llama model response timed out.")
    except ValidationError as e:
        raise ChatError.BadRequestException(detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise ChatError.InternalServerErrorException(detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
