import base64
import logging
import os
from contextlib import asynccontextmanager

import utils.Error_handlers as ChatError  # Error_handlers 추가
import utils.Models as ChatModel
import uvicorn
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse
from utils.AI_Llama_8B import LlamaChatModel

from fastapi import (APIRouter, Depends, FastAPI, HTTPException, Request,
                     Response)

llama_model = None  # 모델 전역 변수
load_dotenv()

# FastAPI 애플리케이션 초기화
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 시 실행할 로직 (예: 모델 로드)
    global llama_model
    llama_model = LlamaChatModel()
    print("Llama 모델 로드 완료")
    yield
    # 애플리케이션 종료 시 실행할 로직 (예: 리소스 해제)
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

# 유효하지 않은 요청에 대한 핸들러 추가
@app.middleware("http")
async def ip_restrict_and_catch_exceptions_middleware(request: Request, call_next):
    ip_string = os.getenv("IP")
    allowed_ips = ip_string.split(",") if ip_string else []
    client_ip = request.client.host

    try:
        if request.url.path in ["/Llama", "/docs", "/redoc", "/openapi.json"] and client_ip not in allowed_ips:
            raise ChatError.IPRestrictedException(detail=f"Unauthorized IP address: {client_ip}")

        response = await call_next(request)
        return response

    except ValidationError as e:
        raise ChatError.BadRequestException(detail=str(e))
    
    except ChatError.IPRestrictedException as e:
        return await ChatError.generic_exception_handler(request, e)
    
    except HTTPException as e:
        # 405 에러를 MethodNotAllowedException으로 변환
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
        tables = llama_model.generate_response(request.input_data)
        return {"output_data": tables}
    except ValidationError as e:
        raise ChatError.BadRequestException(detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise ChatError.InternalServerErrorException(detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)