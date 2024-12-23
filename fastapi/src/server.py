import os
import yaml
import torch
import uvicorn
import httpx
from dotenv import load_dotenv
from asyncio import TimeoutError
from pydantic import ValidationError
from contextlib import asynccontextmanager

from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi import (APIRouter, Depends, FastAPI, HTTPException, Request)
from starlette.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

import utils.Models as ChatModel
import utils.Error_handlers as ChatError
from utils.AI_Llama_8B import LlamaChatModel as Llama_8B
from utils.AI_Bllossom_8B import BllossomChatModel as Bllossom_8B

llama_model_8b = None  # Llama_8B 모델 전역 변수
bllossom_model_8b = None  # Bllossom_8B 모델 전역 변수

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

def load_bot_list(file_path: str) -> list:
    '''
    YAML 파일에서 봇 리스트를 불러오는 함수
    '''
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        return [bot['name'].lower() for bot in data.get('bot_user_agents', [])]

@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
    FastAPI AI 모델 애플리케이션 초기화
    '''
    global llama_model_8b, bllossom_model_8b

    # CUDA 디바이스 정보 가져오기 함수
    def get_cuda_device_info(device_id: int) -> str:
        device_name = torch.cuda.get_device_name(device_id)
        device_properties = torch.cuda.get_device_properties(device_id)
        total_memory = device_properties.total_memory / (1024 ** 3)  # GB 단위로 변환
        return f"Device {device_id}: {device_name} (Total Memory: {total_memory:.2f} GB)"

    # Llama 및 Bllossom 모델 로드
    llama_model_8b = Llama_8B()  # cuda:1
    bllossom_model_8b = Bllossom_8B()  # cuda:0

    # 디버깅용 출력
    llama_device_info = get_cuda_device_info(1)  # Llama 모델은 cuda:1
    bllossom_device_info = get_cuda_device_info(0)  # Bllossom 모델은 cuda:0

    print(f"Llama 모델 로드 완료 ({llama_device_info})")
    print(f"Bllossom 모델 로드 완료 ({bllossom_device_info})")

    yield

    # 모델 메모리 해제
    llama_model_8b = None
    bllossom_model_8b = None
    print("모델 해제 완료")

app = FastAPI(lifespan=lifespan)  # 여기서 한 번만 app을 생성합니다.
ChatError.add_exception_handlers(app)  # 예외 핸들러 추가

class ExceptionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        '''
        예외를 Error_handlers에서 정의한 generic_exception_handler로 위임
        '''
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            return await ChatError.generic_exception_handler(request, e)

app.add_middleware(ExceptionMiddleware)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_KEY", "default-secret")
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
        version="v1.0.2",
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
    allowed_ips = ip_string.split(", ") if ip_string else []
    client_ip = request.client.host
    
    bot_user_agents = load_bot_list("fastapi/src/bot.yaml")
    user_agent = request.headers.get("User-Agent", "").lower()

    try:
        if request.url.path in ["/Llama_stream","/Bllossom_stream", "/docs", "/redoc", "/openapi.json"] and client_ip not in allowed_ips:
            raise ChatError.IPRestrictedException(detail=f"Unauthorized IP address: {client_ip}")
        
        if any(bot in user_agent for bot in bot_user_agents):
            raise ChatError.BadRequestException(detail=f"{user_agent} Bot access is not allowed.")

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

@app.get("/search")
async def search(query: str):
    try:
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "q": query
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"HTTP 요청 오류: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

@app.post("/Llama_stream", summary="스트리밍 방식으로 Llama_8B 모델 답변 생성")
async def llama_stream(request: ChatModel.Llama_Request):
    '''
    Llama_8B 모델에 질문 입력 시 답변을 스트리밍 방식으로 반환
    '''
    try:
        response_stream = llama_model_8b.generate_response_stream(request.input_data)
        return StreamingResponse(response_stream, media_type="text/plain")
    except TimeoutError:
        raise ChatError.InternalServerErrorException(detail="Llama 모델 응답이 시간 초과되었습니다.")
    except ValidationError as e:
        raise ChatError.BadRequestException(detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unhandled Exception: {e}")
        raise ChatError.InternalServerErrorException(detail=str(e))

@app.post("/Bllossom_stream", summary="스트리밍 방식으로 Bllossom_8B 모델 답변 생성")
async def bllossom_stream(request: ChatModel.Bllossom_Request):
    '''
    Bllossom_8B 모델에 질문 입력 시 캐릭터 설정을 반영하여 답변을 스트리밍 방식으로 반환
    '''
    try:
        character_settings = {
            "character_name": request.character_name,
            "description": request.description,
            "greeting": request.greeting,
            "character_setting": request.character_setting,
            "tone": request.tone,
            "energy_level": request.energy_level,
            "politeness": request.politeness,
            "humor": request.humor,
            "assertiveness": request.assertiveness,
            "access_level": request.access_level
        }
        
        response_stream = bllossom_model_8b.generate_response_stream(
            input_text=request.input_data,
            character_settings=character_settings
        )
        return StreamingResponse(response_stream, media_type="text/plain")
    except TimeoutError:
        raise ChatError.InternalServerErrorException(detail="Bllossom 모델 응답이 시간 초과되었습니다.")
    except ValidationError as e:
        raise ChatError.BadRequestException(detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unhandled Exception: {e}")  # 디버깅 출력 추가
        raise ChatError.InternalServerErrorException(detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)