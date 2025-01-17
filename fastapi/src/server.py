# server.py
# 파일은 FastAPI 서버를 구동하는 엔트리 포인트입니다.

import os
import yaml
import torch
import uvicorn
import httpx
import ipaddress
from bs4 import BeautifulSoup
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

from utils  import ChatError, ChatModel, LanguageProcessor, Llama_8B, Bllossom_8B

llama_model_8b = None  # Llama_8B 모델 전역 변수
bllossom_model_8b = None  # Bllossom_8B 모델 전역 변수
languageprocessor = LanguageProcessor()
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
    llama_model_8b = Llama_8B()  # cuda:0
    bllossom_model_8b = Bllossom_8B()  # cuda:1

    # 디버깅용 출력
    llama_device_info = get_cuda_device_info(0)  # Llama 모델은 cuda:1
    bllossom_device_info = get_cuda_device_info(1)  # Bllossom 모델은 cuda:0

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
        version="v1.1.0",
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
        # Check if the IP is in the internal network range (192.168.219.0/24)
        return ip_obj in ipaddress.ip_network("192.168.219.0/24")
    except ValueError:
        return False

@app.middleware("http")
async def ip_restrict_and_bot_blocking_middleware(request: Request, call_next):
    ip_string = os.getenv("IP")
    allowed_ips = ip_string.split(", ") if ip_string else []
    client_ip = request.client.host

    bot_user_agents = load_bot_list("/app/src/bot.yaml")  # 경로 수정
    user_agent = request.headers.get("User-Agent", "").lower()

    try:
        # Restrict access based on IP and internal network range
        # if (request.url.path in ["/Llama_stream", "/Bllossom_stream", "/docs", "/redoc", "/openapi.json"]
        #         and client_ip not in allowed_ips
        #         and not is_internal_ip(client_ip)):
        #     raise ChatError.IPRestrictedException(detail=f"Unauthorized IP address: {client_ip}")

        # Block bots based on user agent
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

def stream_search_results(search_results: dict):
    items = search_results.get("items", [])
    for item in items[:2]:  # 최대 3개의 결과만 스트리밍
        search_data_set = f"{item['title']}: {item['snippet']}\n"
        print(search_data_set)
        yield search_data_set

async def fetch_results(query: str, num: int, domain: str = "") -> list:
    """
    Google 검색 결과를 가져오는 함수
    :param query: 검색어
    :param num: 가져올 결과 수
    :param domain: 특정 도메인 필터 (없으면 전체 검색)
    :return: 검색 결과 리스트
    """
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": f"{query} {domain}".strip(),
        "num": min(num, 10)
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            search_results = response.json()

            return [
                {
                    "title": item.get("title", "제목 없음"),
                    "snippet": item.get("snippet", "설명 없음"),
                    "link": item.get("link", "링크 없음")
                }
                for item in search_results.get("items", [])
            ]
    except httpx.RequestError as e:
        print(f"HTTP 요청 오류: {str(e)}")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
    return []

async def fetch_search_results(query: str, num_results: int = 5) -> list:
    """
    위키백과, 나무위키, 다양한 뉴스 사이트 등, 사이트 기반으로 검색 결과를 가져옵니다.
    부족한 경우 최상단 검색 결과 추가.
    :param query: 검색어
    :param num_results: 가져올 각 도메인별 검색 결과 수
    :return: 검색 결과 리스트 (제목, 설명, 링크 포함)
    """
    domains = [
        "site:en.wikipedia.org",# 영어 위키백과
        "site:ko.wikipedia.org",# 한국어 위키백과
        "site:namu.wiki",       # 나무위키
        "site:news.naver.com",  # 네이버 뉴스
        "site:bbc.com",         # BBC
        "site:cnn.com",         # CNN
        "site:reuters.com",     # 로이터
        "site:nytimes.com",     # 다양한 뉴스 사이트
        "site:dcinside.com",    # 디시인사이드
        "site:reddit.com",      # 레딧
        "site:naver.com"        # 네이버
    ]

    all_results = []
    total_results_needed = num_results * len(domains)

    for domain in domains:
        domain_results = await fetch_results(query=query, num=num_results, domain=domain)
        all_results.extend(domain_results)

    if len(all_results) < total_results_needed:
        remaining_needed = total_results_needed - len(all_results)
        general_results = await fetch_results(query=query, num=remaining_needed)
        all_results.extend(general_results)

    return all_results[:total_results_needed]


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

@app.post("/Llama_stream", summary="AI 모델이 검색 결과를 활용하여 답변 생성")
async def Llama_stream(request: ChatModel.Llama_Request):
    """
    사용자 질문과 위키백과, 나무위키, 뉴스 결과를 결합하여 AI 답변을 생성합니다.
    :param request: 사용자 질문과 옵션 포함
    :return: AI 모델의 답변
    """
    print(f"Request: {request}")
    try:
        search_context = ""  # search_context를 초기화

        if request.google_access:
            # 위키백과, 나무위키, 뉴스 사이트 기반으로 검색
            search_results = await fetch_search_results(request.input_data, num_results=5)

            # 검색 결과를 텍스트로 통합
            if search_results:
                search_context = "\n".join([
                    f"제목: {item['title']}\n설명: {item['snippet']}\n링크: {item['link']}"
                    for item in search_results[:5]  # 최대 5개만 사용
                ])
            else:
                search_context = ""
        
        print(f"Search Context: {search_context}")

        # AI 모델에 입력 생성
        prompt = (
            f"사용자 질문은 {request.input_data}\n\n"
            f"참고 정보는 {search_context}\n\n"
        )

        # AI 모델로 답변 생성
        response_stream = llama_model_8b.generate_response_stream(input_text=prompt)
        return StreamingResponse(response_stream, media_type="text/plain")

    except TimeoutError:
        raise ChatError.InternalServerErrorException(detail="모델 응답이 시간 초과되었습니다.")
    except Exception as e:
        print(f"Unhandled Exception: {e}")
        raise ChatError.InternalServerErrorException(detail=str(e))


# @app.post("/Llama_stream", summary="스트리밍 방식으로 Llama_8B 모델 답변 생성")
# async def llama_stream(request: ChatModel.Llama_Request):
#     '''
#     Llama_8B 모델에 질문 입력 시 답변을 스트리밍 방식으로 반환
#     '''
#     try:
#         if not request.google_access_set:
#             # Google Access가 설정되지 않은 경우 바로 처리
#             response_stream = llama_model_8b.generate_response_stream(
#                 input_text=request.input_data,
#                 google_search=None
#             )
#             return StreamingResponse(response_stream, media_type="text/plain")

#         # 명사 추출 및 처리
#         search_nouns = languageprocessor.process_sentence(request.input_data)

#         # 기본값 설정
#         result = request.input_data  # 초기값은 입력 데이터로 설정

#         # 반환값 확인 및 디버깅
#         if not isinstance(search_nouns, dict):
#             print(f"Unexpected structure in process_sentence output: {search_nouns}")
#         else:
#             print(search_nouns)  # 정상적인 출력 확인

#         # 분석 결과 가져오기
#         analysis_result = search_nouns.get("분석 결과", {})
#         if not isinstance(analysis_result, dict):
#             print(f"Unexpected structure in '분석 결과': {analysis_result}")
#         else:
#             # 명사, 부사, 동사, 형용사 순으로 확인
#             for key in ["명사", "부사", "동사", "형용사"]:
#                 words = analysis_result.get(key, [])
#                 if isinstance(words, list) and words:  # 리스트이고 비어있지 않은 경우만 처리
#                     result = " ".join(words)
#                     break
#                 elif not isinstance(words, list):  # 예상 타입이 아닌 경우 디버깅 메시지 출력
#                     print(f"Unexpected type for '{key}': {type(words)}")

#         # 최종 결과 출력
#         print(f"Final result: {result}")

#         # Google 검색 API 호출
#         url = f"https://www.googleapis.com/customsearch/v1"
#         params = {"key": GOOGLE_API_KEY, "cx": SEARCH_ENGINE_ID, "q": result}
#         async with httpx.AsyncClient() as client:
#             response = await client.get(url, params=params)
#             response.raise_for_status()

#         # 검색 결과를 스트리밍 방식으로 전달
#         search_stream = stream_search_results(response.json())

#         response_stream = llama_model_8b.generate_response_stream(
#             input_text=request.input_data,
#             google_search="".join(search_stream)
#         )
#         return StreamingResponse(response_stream, media_type="text/plain")
    
#     except TimeoutError:
#         raise ChatError.InternalServerErrorException(detail="Llama 모델 응답이 시간 초과되었습니다.")
#     except ValidationError as e:
#         raise ChatError.BadRequestException(detail=str(e))
#     except httpx.RequestError as e:
#         raise HTTPException(status_code=500, detail=f"HTTP 요청 오류: {str(e)}")
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         print(f"Unhandled Exception: {e}")
#         raise ChatError.InternalServerErrorException(detail=str(e))
    
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
    uvicorn.run(app, host="0.0.0.0", port=8001)