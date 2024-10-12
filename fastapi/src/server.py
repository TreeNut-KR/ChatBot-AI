import os
from contextlib import asynccontextmanager

import utils.Error_handlers as ChatError
import utils.Models as ChatModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse
from utils.AI_Llama_8B import LlamaChatModel
# from utils.DB_mysql import MySQLDBHandler

from fastapi import APIRouter, FastAPI, Request
import uvicorn  # uvicorn 추가

# mysql_handler = MySQLDBHandler()  # MySQL 핸들러 초기화

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     '''
#     FastAPI 애플리케이션의 수명 주기를 관리하는 함수.
#     '''
#     await mysql_handler.connect()
#     try:
#         yield
#     finally:
#         await mysql_handler.disconnect()

app = FastAPI()
ChatError.add_exception_handlers(app)  # 예외 핸들러 추가

class ExceptionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # 예외 세부 사항을 보다 안전하게 처리
            error_detail = self._get_error_detail(e)
            return JSONResponse(
                status_code=500,
                content={"detail": error_detail}
            )
    
    def _get_error_detail(self, exception: Exception) -> str:
        if isinstance(exception, TypeError):
            return str(exception)
        try:
            return getattr(exception, 'detail', str(exception))
        except Exception as ex:
            return f"Unexpected error occurred: {str(ex)}"
        
app.add_middleware(ExceptionMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "your-secret-key")  # 기본 비밀 키 추가
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
        version="v1.0.0",
        summary="AI 모델 관리 API (개발 중인 버전)",
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

@app.post("/Llama", response_model=ChatModel.Llama_Response, summary="Llama 모델 답변 생성")
async def Llama_(request: ChatModel.Llama_Request):
    '''
    Llama 모델에 질문 입력 시 답변 반환.
    '''
    try:
        LCM = LlamaChatModel()
        tables = LCM.generate_response(request.input_data)
        return {"output_data": tables}
    except Exception as e:
        raise ChatError.InternalServerErrorException(detail=str(e))

# uvicorn 서버 실행
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
