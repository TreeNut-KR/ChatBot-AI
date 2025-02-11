# server.py
'''
파일은 FastAPI 서버를 구동하는 엔트리 포인트입니다.
'''

import os
import yaml
import torch
import uvicorn
import httpx
import ipaddress
from dotenv import load_dotenv
from asyncio import TimeoutError
from pydantic import ValidationError
from contextlib import asynccontextmanager

from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi import (APIRouter,  Query, FastAPI, HTTPException, Request)
from starlette.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from utils  import ChatError, ChatModel, GoogleSearch, LanguageProcessor, MongoDBHandler, Llama_8B, Lumimaid_8B

llama_model_8b = None                   # Llama_8B 모델 전역 변수
Lumimaid_model_8b = None                # Lumimaid_8B 모델 전역 변수
mongo_handler = MongoDBHandler()        # MongoDB 핸들러 초기화
languageprocessor = LanguageProcessor() # LanguageProcessor 초기화
load_dotenv()

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
    global llama_model_8b, Lumimaid_model_8b

    # CUDA 디바이스 정보 가져오기 함수
    def get_cuda_device_info(device_id: int) -> str:
        device_name = torch.cuda.get_device_name(device_id)
        device_properties = torch.cuda.get_device_properties(device_id)
        total_memory = device_properties.total_memory / (1024 ** 3)  # GB 단위로 변환
        return f"Device {device_id}: {device_name} (Total Memory: {total_memory:.2f} GB)"

    # Llama 및 Lumimaid 모델 로드
    llama_model_8b = Llama_8B()  # cuda:0
    Lumimaid_model_8b = Lumimaid_8B()  # cuda:1

    # 디버깅용 출력
    llama_device_info = get_cuda_device_info(0)  # Llama 모델은 cuda:0
    Lumimaid_device_info = get_cuda_device_info(1)  # Lumimaid 모델은 cuda:1

    print(f"Llama 모델 로드 완료 ({llama_device_info})")
    print(f"Lumimaid 모델 로드 완료 ({Lumimaid_device_info})")

    yield

    # 모델 메모리 해제
    llama_model_8b = None
    Lumimaid_model_8b = None
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
        version="v1.2.0",
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
def is_internal_ip(ip):
    try:
        ip_obj = ipaddress.ip_address(ip)
        # IP가 내부 네트워크 범위(192.168.219.0/24)에 있는지 확인합니다
        return ip_obj in ipaddress.ip_network("192.168.219.0/24")
    except ValueError:
        return False

@app.middleware("http")
async def ip_restrict_and_bot_blocking_middleware(request: Request, call_next):
    ip_string = os.getenv("IP")
    allowed_ips = ip_string.split(", ") if ip_string else []
    client_ip = request.client.host

    bot_user_agents = load_bot_list("./fastapi/src/bot.yaml") # 경로 수정
    user_agent = request.headers.get("User-Agent", "").lower()

    try:
        # IP 및 내부 네트워크 범위에 따라 액세스 제한
        if (request.url.path in ["/Llama_stream", "/Lumimaid_stream", "/docs", "/redoc", "/openapi.json"]
                and client_ip not in allowed_ips
                and not is_internal_ip(client_ip)):
            raise ChatError.IPRestrictedException(detail=f"Unauthorized IP address: {client_ip}")

        # 사용자 에이전트 기반 봇 차단
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
        if (e.status_code == 405):
            raise ChatError.MethodNotAllowedException(detail="The method used is not allowed.")
        raise e
    except Exception as e:
        raise ChatError.InternalServerErrorException(detail="Internal server error occurred.")

@app.get("/")
async def root():
    return {"message": "Welcome to the API"}

mongo_router = APIRouter() # MySQL 관련 라우터 정의

@mongo_router.get("/db", summary="데이터베이스 목록 가져오기")
async def list_databases():
    '''
    데이터베이스 서버에 있는 모든 데이터베이스의 목록을 반환합니다.
    '''
    try:
        databases = await mongo_handler.get_db()
        return {"Database": databases}
    except Exception as e:
        raise ChatError.InternalServerErrorException(detail=str(e))

@mongo_router.get("/collections", summary="데이터베이스 컬렉션 목록 가져오기")
async def list_collections(db_name: str = Query(..., description="데이터베이스 이름")):
    '''
    현재 선택된 데이터베이스 내의 모든 컬렉션 이름을 반환합니다.
    '''
    try:
        collections = await mongo_handler.get_collection(database_name=db_name)
        return {"Collections": collections}
    except Exception as e:
        raise ChatError.InternalServerErrorException(detail=str(e))

app.include_router(
    mongo_router,
    prefix="/mongo",
    tags=["MongoDB Router"],
    responses={500: {"description": "Internal Server Error"}}
)

@app.get("/search")
async def search(query: str):
    try:
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GoogleSearch.GOOGLE_API_KEY,
            "cx": GoogleSearch.SEARCH_ENGINE_ID,
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

@app.post("/Llama_stream", summary="AI 모델이 검색 결과를 활용하여 답변 생성")
async def Llama_stream(request: ChatModel.Llama_Request):
    """
    Llama_8B 모델에 질문을 위키백과, 나무위키, 뉴스 등의 결과를 결합하여 AI 답변을 생성합니다.
    
    Args:
        request (ChatModel.Llama_Request): 사용자 질문과 인터넷 검색 옵션 포함
        
    Returns:
        StreamingResponse: 스트리밍 방식의 모델 응답
    """
    try:
        search_context = ""  # search_context를 초기화
        
        if request.google_access:
            # 위키백과, 나무위키, 뉴스 사이트 기반으로 검색
            search_results = await GoogleSearch.fetch_search_results(request.input_data, num_results=5)

            # 검색 결과를 텍스트로 통합
            if search_results:
                search_context = "\n".join([
                    f"제목: {item['title']}\n설명: {item['snippet']}\n링크: {item['link']}"
                    for item in search_results[:5] # 최대 5개만 사용
                ])
            else:
                search_context = ""

        # AI 모델에 입력 생성
        prompt = (
            f"사용자 질문은 {request.input_data}\n\n"
            f"참고 정보는 {search_context}\n\n"
        )

        # 응답 스트림 생성
        response_stream = llama_model_8b.generate_response_stream(
            input_text=prompt
        )
        
        return StreamingResponse(
            response_stream,
            media_type="text/plain",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except TimeoutError:
        raise ChatError.InternalServerErrorException(
            detail="Llama 모델 응답이 시간 초과되었습니다."
        )
    except ValidationError as e:
        raise ChatError.BadRequestException(detail=str(e))
    except Exception as e:
        print(f"처리되지 않은 예외: {e}")
        raise ChatError.InternalServerErrorException(detail=str(e))
    
@app.post("/Lumimaid_stream", summary="스트리밍 방식으로 Lumimaid_8B 모델 답변 생성")
async def Lumimaid_stream(request: ChatModel.Lumimaid_Request):
    """
    Lumimaid_8B 모델에 질문을 입력하고 캐릭터 설정을 반영하여 답변을 스트리밍 방식으로 반환합니다.

    Args:
        request (ChatModel.Lumimaid_Request): 사용자 요청 데이터 포함

    Returns:
        StreamingResponse: 스트리밍 방식의 모델 응답
    """
    try:
        # 캐릭터 설정 구성
        character_settings = {
            "character_name": request.character_name,
            "greeting": request.greeting,
            "context": request.context,
            "access_level": request.access_level
        }
        
        # 응답 스트림 생성
        response_stream = Lumimaid_model_8b.generate_response_stream(
            input_text=request.input_data,
            character_settings=character_settings
        )
        
        return StreamingResponse(
            response_stream,
            media_type="text/plain",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except TimeoutError:
        raise ChatError.InternalServerErrorException(
            detail="Lumimaid 모델 응답이 시간 초과되었습니다."
        )
    except ValidationError as e:
        raise ChatError.BadRequestException(detail=str(e))
    except Exception as e:
        print(f"처리되지 않은 예외: {e}")
        raise ChatError.InternalServerErrorException(detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)