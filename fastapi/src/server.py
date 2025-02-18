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

from utils  import ChatError, ChatModel, ChatSearch, LanguageProcessor, MongoDBHandler, Llama, Lumimaid, Bllossom

Bllossom_model = None                   # Bllossom 모델 전역 변수
Lumimaid_model = None                   # Lumimaid 모델 전역 변수
mongo_handler = MongoDBHandler()        # MongoDB 핸들러 초기화
languageprocessor = LanguageProcessor() # LanguageProcessor 초기화
load_dotenv()

def load_bot_list(file_path: str) -> list:
    """
    YAML 파일에서 봇 리스트를 불러오는 함수입니다.
    
    Args:
        file_path (str): 봇 목록이 저장된 YAML 파일의 경로
        
    Returns:
        list: 소문자로 변환된 봇 이름 리스트
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        return [bot['name'].lower() for bot in data.get('bot_user_agents', [])]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 애플리케이션의 수명 주기를 관리하는 컨텍스트 매니저입니다.
    
    Args:
        app (FastAPI): FastAPI 애플리케이션 인스턴스
        
    Yields:
        None: 애플리케이션 컨텍스트를 생성하고 종료할 때까지 대기
    """
    global Bllossom_model, Lumimaid_model

    # CUDA 디바이스 정보 가져오기 함수
    def get_cuda_device_info(device_id: int) -> str:
        device_name = torch.cuda.get_device_name(device_id)
        device_properties = torch.cuda.get_device_properties(device_id)
        total_memory = device_properties.total_memory / (1024 ** 3)  # GB 단위로 변환
        return f"Device {device_id}: {device_name} (Total Memory: {total_memory:.2f} GB)"

    # Bllossom 및 Lumimaid 모델 로드
    Bllossom_model = Bllossom()  # cuda:0
    Lumimaid_model = Lumimaid()  # cuda:1

    # 디버깅용 출력
    Bllossom_device_info = get_cuda_device_info(0)  # Bllossom 모델은 cuda:0
    Lumimaid_device_info = get_cuda_device_info(1)  # Lumimaid 모델은 cuda:1

    print(f"Bllossom 모델 로드 완료 ({Bllossom_device_info})")
    print(f"Lumimaid 모델 로드 완료 ({Lumimaid_device_info})")

    yield

    # 모델 메모리 해제
    Bllossom_model = None
    Lumimaid_model = None
    print("모델 해제 완료")

app = FastAPI(lifespan=lifespan)  # 여기서 한 번만 app을 생성합니다.
ChatError.add_exception_handlers(app)  # 예외 핸들러 추가

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
    """
    커스텀 OpenAPI 스키마를 생성하는 함수입니다.
    
    Returns:
        dict: OpenAPI 스키마 정의
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="ChatBot-AI FastAPI",
        version="v1.3.0",
        summary="AI 모델 관리 API",
        routes=app.routes,
        description=(
            "이 API는 다음과 같은 기능을 제공합니다:\n\n"
            f"각 엔드포인트의 자세한 정보는 [📌 ChatBot-AI FastAPI 명세서](https://github.com/TreeNut-KR/ChatBot-AI/issues/4) 에서 확인할 수 있습니다."
        ),
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://drive.google.com/thumbnail?id=12PqUS6bj4eAO_fLDaWQmoq94-771xfim"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
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
        # IP가 내부 네트워크 범위(192.168.219.0/24)에 있는지 확인합니다
        return ip_obj in ipaddress.ip_network("192.168.219.0/24")
    except ValueError:
        return False

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
    ip_string = os.getenv("IP")
    allowed_ips = ip_string.split(", ") if ip_string else []
    client_ip = request.client.host

    bot_user_agents = load_bot_list("./fastapi/src/bot.yaml") # 경로 수정
    user_agent = request.headers.get("User-Agent", "").lower()

    try:
        # # IP 및 내부 네트워크 범위에 따라 액세스 제한
        # if (request.url.path in ["/office_stream", "/character_stream", "/docs", "/redoc", "/openapi.json"]
        #         and client_ip not in allowed_ips
        #         and not is_internal_ip(client_ip)):
        #     raise ChatError.IPRestrictedException(detail=f"Unauthorized IP address: {client_ip}")

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
    """
    API 루트 엔드포인트입니다.
    
    Returns:
        dict: 환영 메시지를 포함한 응답
    """
    return {"message": "Welcome to the API"}

mongo_router = APIRouter() # MySQL 관련 라우터 정의

@mongo_router.get("/db", summary="데이터베이스 목록 가져오기")
async def list_databases():
    """
    데이터베이스 서버의 모든 데이터베이스 목록을 조회합니다.
    
    Returns:
        dict: 데이터베이스 목록을 포함한 응답
        
    Raises:
        ChatError.InternalServerErrorException: 데이터베이스 조회 실패 시
    """
    try:
        databases = await mongo_handler.get_db()
        return {"Database": databases}
    except Exception as e:
        raise ChatError.InternalServerErrorException(detail=str(e))

@mongo_router.get("/collections", summary="데이터베이스 컬렉션 목록 가져오기")
async def list_collections(db_name: str = Query(..., description="데이터베이스 이름")):
    """
    지정된 데이터베이스의 모든 컬렉션 목록을 조회합니다.
    
    Args:
        db_name (str): 조회할 데이터베이스 이름
        
    Returns:
        dict: 컬렉션 목록을 포함한 응답
        
    Raises:
        ChatError.InternalServerErrorException: 컬렉션 조회 실패 시
    """
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

'''현재 사용 중지된 코드
@app.get("/search")
async def search(query: str):
    try:
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": ChatSearch.GOOGLE_API_KEY,
            "cx": ChatSearch.SEARCH_ENGINE_ID,
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
'''

@app.post("/office_stream", summary="AI 모델이 검색 결과를 활용하여 답변 생성")
async def office_stream(request: ChatModel.Bllossom_Request):
    """
    Bllossom_8B 모델에 질문을 위키백과, 나무위키, 뉴스 등의 결과를 결합하여 AI 답변을 생성합니다.
    
    Args:
        request (ChatModel.Bllossom_Request): 사용자 질문과 인터넷 검색 옵션 포함
        
    Returns:
        StreamingResponse: 스트리밍 방식의 모델 응답
    """
    try:
        search_context = ""  # search_context를 초기화
        
        # DuckDuckGo 검색 결과 가져오기
        if request.google_access:  # 검색 옵션이 활성화된 경우
            duck_results = await ChatSearch.fetch_duck_search_results(query=request.input_data)
            
            if duck_results:
                # 검색 결과를 AI가 이해하기 쉬운 형식으로 변환
                formatted_results = []
                for idx, item in enumerate(duck_results[:20], 1):  # 상위 20개 결과만 사용
                    formatted_result = (
                        f"[검색결과 {idx}]\n"
                        f"제목: {item.get('title', '제목 없음')}\n"
                        f"내용: {item.get('snippet', '내용 없음')}\n"
                        f"출처: {item.get('link', '링크 없음')}\n"
                    )
                    formatted_results.append(formatted_result)
                
                # 모든 결과를 하나의 문자열로 결합
                search_context = (
                    "다음은 검색에서 가져온 관련 정보입니다:\n\n" +
                    "\n".join(formatted_results)
                )
                
        '''현재 DockDockGo 검색 기반으로 변경되어 사용 중지된 코드
        if request.google_access:
            # 위키백과, 나무위키, 뉴스 사이트 기반으로 검색
            search_results = await ChatSearch.fetch_google_filtered_results(query=request.input_data, num_results=9)
            # 검색 결과를 텍스트로 통합
            if search_results:
                search_context = "\n".join([
                    f"제목: {item['title']}\n설명: {item['snippet']}\n링크: {item['link']}"
                    for item in search_results[:5] # 최대 5개만 사용
                ])
            else:
                search_context = ""
            print(search_results)
        '''
        
        # 응답 스트림 생성
        response_stream = Bllossom_model.generate_response_stream(
            input_text=request.input_data,
            search_text=search_context
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
            detail="Bllossom 모델 응답이 시간 초과되었습니다."
        )
    except ValidationError as e:
        raise ChatError.BadRequestException(detail=str(e))
    except Exception as e:
        print(f"처리되지 않은 예외: {e}")
        raise ChatError.InternalServerErrorException(detail=str(e))
    
@app.post("/character_stream", summary="스트리밍 방식으로 Lumimaid_8B 모델 답변 생성")
async def character_stream(request: ChatModel.Lumimaid_Request):
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
        response_stream = Lumimaid_model.generate_response_stream(
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