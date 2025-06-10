from fastapi import Path, APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

import time
import utils.app_state as AppState
from .. import OpenAiCharacter, ChatModel, ChatError

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# 처리 시간 임계값 설정 수정
MAX_PROCESSING_TIME = 180  # 3분 (180초)로 줄임 - Nginx 타임아웃보다 충분히 짧게
RETRY_AFTER_MINUTES = 3    # 3분 후 재시도 권장

OPENAI_MODEL_MAP = {
    "gpt4.1": {
        "id": "gpt-4.1",
        "type": "post"
    },
    "gpt4.1_mini": {
        "id": "gpt-4.1-mini",
        "type": "post"
    },
}

def get_openai_links(base_url: str, path: str) -> dict:
    """
    OPENAI_MODEL_MAP을 기준으로 _links용 링크 딕셔너리 생성
    """
    return [
        {
            "href": f"{base_url}character/{k}",
            "rel": f"character_{k}",
            "type": OPENAI_MODEL_MAP[k]["type"],
        }
        for k in OPENAI_MODEL_MAP.keys() if k != path
    ]

def calculate_estimated_time(queue_size: int) -> int:
    """
    큐 크기와 워커 2개 기준으로 예상 처리 시간 계산 (초 단위)
    - 홀수 대기번호: 20초 처리 (워커1)
    - 짝수 대기번호: 10초 처리 (워커2)
    - 워커 2개 병렬 처리
    
    각 워커가 독립적으로 누적 처리:
    홀수 워커: 20초, 40초, 60초, 80초, 100초, 120초...
    짝수 워커: 210초, 50초, 710초, 100초, 1210초, 150초...
    """
    if queue_size == 0:
        # 큐가 비어있으면 첫 번째 요청 (Position 1: 20초)
        return 20
    
    current_queue_position = queue_size + 1  # 새로 추가될 요청의 위치
    
    # 홀수 번호 (워커1)
    if current_queue_position % 2 == 1:  
        worker_order = (current_queue_position + 1) // 2
        estimated_time = 20 * worker_order
    # 짝수 번호 (워커2)
    else:  
        worker_order = current_queue_position // 2
        estimated_time = 30 * worker_order
    
    return int(estimated_time)

character_router = APIRouter()

@character_router.post("/Llama", summary = "Llama 모델이 케릭터 정보를 기반으로 답변 생성")
async def character_llama(request: ChatModel.character_Request, req: Request):
    """
    DarkIdol-Llama-3.1-8B 모델에 질문을 입력하고 캐릭터 설정을 반영하여 답변을 JSON 방식으로 반환합니다.

    Args:
        request (ChatModel.character_Request): 사용자 요청 데이터 포함

    Returns:
        JSONResponse: JSON 방식으로 모델 응답
    """
    chat_list = []
    start_time = time.time()
    queue_status_before = None
    performance_stats = None

    # MongoDB에서 채팅 기록 가져오기
    if AppState.mongo_handler and request.db_id:
        try:
            chat_list = await AppState.mongo_handler.get_character_log(
                user_id = request.user_id,
                document_id = request.db_id,
                router = "character",
            )
        except Exception as e:
            print(f"{YELLOW}WARNING{RESET}:  채팅 기록을 가져오는 데 실패했습니다: {str(e)}")
            
    try:
        character_settings = {
            "character_name": request.character_name,
            "greeting": request.greeting,
            "context": request.context,
            "chat_list": chat_list,
        }
        
        # 큐 상태 확인 및 예상 대기 시간 계산
        queue_status_before = AppState.llama_queue_handler.get_queue_status()
        performance_stats = AppState.llama_queue_handler.get_performance_stats()

        # 큐 크기 (단일 큐로 변경)
        total_queue_size = queue_status_before['queue_size']
        
        # 예상 처리 시간 계산 (워커 2개, 홀수 20초/짝수 10초)
        estimated_time_seconds = calculate_estimated_time(total_queue_size)
        
        # 예상 처리 시간이 180초(3분) 이상이면 미리 429 반환
        if estimated_time_seconds > MAX_PROCESSING_TIME:
            estimated_minutes = estimated_time_seconds // 60
            retry_after_seconds = RETRY_AFTER_MINUTES * 60 + 10
            
            current_position = total_queue_size + 1
            worker_type = "홀수(20초)" if current_position % 2 == 1 else "짝수(10초)"
            
            print(
                f"{YELLOW}⏰ PRE-TIMEOUT{RESET}: Character: {request.character_name} | User: {request.user_id} | "
                f"Queue: {total_queue_size} | Position: {current_position}({worker_type}) | "
                f"Estimated: {estimated_time_seconds}s ({estimated_minutes}분) | "
                f"Threshold: {MAX_PROCESSING_TIME}s | HTTP: 429"
            )
        
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": f"현재 서버 이용자가 많아 예상 처리 시간이 {estimated_minutes}분입니다. {RETRY_AFTER_MINUTES}분 후 다시 요청해 주세요.",
                    "code": "QUEUE_OVERLOADED",
                    "retry_after": retry_after_seconds,
                    "queue_info": {
                        "current_queue_size": total_queue_size,
                        "your_position": current_position,
                        "worker_type": worker_type,
                        "estimated_wait_time_seconds": estimated_time_seconds,
                        "estimated_wait_time_minutes": estimated_minutes,
                        "max_allowed_time_seconds": MAX_PROCESSING_TIME,  # 180초로 업데이트
                        "processing_model": "dual_worker_odd_even"
                    }
                },
                headers={"Retry-After": str(retry_after_seconds)}
            )

        current_position = total_queue_size + 1
        worker_type = "홀수(20초)" if current_position % 2 == 1 else "짝수(10초)"
        
        print(
            f"🔄 큐에 요청 추가: Character: {request.character_name} | User: {request.user_id} | "
            f"Queue: {total_queue_size} | Position: {current_position}({worker_type}) | "
            f"예상 대기: {estimated_time_seconds}s ({estimated_time_seconds//60}분 {estimated_time_seconds%60}초) | "
            f"InputLen: {len(request.input_data)} chars"
        )
        
        full_response = await AppState.llama_queue_handler.add_request(
            input_text=request.input_data,
            character_settings=character_settings,
            user_id=request.user_id,
            character_name=request.character_name
        )
        
        processing_time = time.time() - start_time
        
        response_data = {
            "result": full_response,
            "processing_info": {
                "processing_time": f"{processing_time:.3f}s",
                "queue_position_before": total_queue_size,
                "your_position": total_queue_size + 1,
                "worker_type": "홀수(20초)" if (total_queue_size + 1) % 2 == 1 else "짝수(10초)",
                "estimated_wait_time": f"{estimated_time_seconds}s",
                "processing_mode": "dual_worker_odd_even"
            },
            "_links": [
                {   
                    "href": str(req.url),
                    "rel": "_self",
                    "type": str(req.method).lower(),
                },
                *get_openai_links(
                    base_url = str(req.base_url),
                    path = "Llama",
                ),
            ]
        }
        return JSONResponse(content=response_data)

    except TimeoutError as te:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too Many Requests", 
                "message": "서버 이용자가 많아 처리 시간이 초과되었습니다. 잠시 후 다시 요청해 주세요.",
                "code": "EXPLICIT_TIMEOUT_ERROR",
                "retry_after": retry_after_seconds,
                "queue_info": {
                    "current_queue_size": queue_status_before.get('queue_size', 0) if queue_status_before else 0,
                    "estimated_wait_time": performance_stats.get('estimated_wait_time', '알 수 없음') if performance_stats else '알 수 없음'
                }
            },
            headers={"Retry-After": str(retry_after_seconds)}
        )
    except ValidationError as ve:
        raise ChatError.BadRequestException(detail=str(ve))
    except Exception as e:
        error_str = str(e)
        timeout_keywords = [
            "시간 초과", "timeout", "요청 처리 시간 초과", "5분", "4분",
            "TimeoutError", "asyncio.TimeoutError", "concurrent.futures.TimeoutError",
            "처리 시간이 초과", "time limit exceeded", "request timeout", "500:", "504:"
        ]
        is_timeout = any(keyword in error_str.lower() for keyword in [kw.lower() for kw in timeout_keywords])
        
        if is_timeout or "시간 초과" in error_str:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": "서버 이용자가 많아 처리 시간이 초과되었습니다. 잠시 후 다시 요청해 주세요.",
                    "code": "TIMEOUT_DUE_TO_HIGH_LOAD",
                    "retry_after": retry_after_seconds,
                    "debug_info": {
                        "original_error": error_str,
                        "detected_as": "timeout_error",
                        "detection_method": "keyword_matching"
                    }
                },
                headers={"Retry-After": str(retry_after_seconds)}
            )
        else:
            raise ChatError.InternalServerErrorException(detail="내부 서버 오류가 발생했습니다.")

@character_router.post("/{gpt_set}", summary = "gpt 모델이 케릭터 정보를 기반으로 답변 생성")
async def character_gpt4o_mini(
        request: ChatModel.character_Request,
        req: Request,
        gpt_set: str = Path(
            ...,
            title="GPT 모델명",
            description= f"사용할 OpenAI GPT 모델의 별칭 (예: {list(OPENAI_MODEL_MAP.keys())})",
        )
    ):
    """
    gpt 모델에 질문을 입력하고 캐릭터 설정을 반영하여 답변을 JSON 방식으로 반환합니다.

    Args:
        request (ChatModel.character_Request): 사용자 요청 데이터 포함

    Returns:
        JSONResponse: JSON 방식으로 모델 응답
    """
    if gpt_set not in OPENAI_MODEL_MAP:
        raise HTTPException(status_code = 400, detail = "Invalid model name.")

    model_id = OPENAI_MODEL_MAP[gpt_set]["id"]
    chat_list = []
    
    # MongoDB에서 채팅 기록 가져오기
    if AppState.mongo_handler and request.db_id:
        try:
            chat_list = await AppState.mongo_handler.get_character_log(
                user_id = request.user_id,
                document_id = request.db_id,
                router = "character",
            )
        except Exception as e:
            print(f"{YELLOW}WARNING{RESET}:  채팅 기록을 가져오는 데 실패했습니다: {str(e)}")

    OpenAiCharacter_model = OpenAiCharacter(model_id = model_id)
    try:
        character_settings = {
            "character_name": request.character_name,
            "greeting": request.greeting,
            "context": request.context,
            "chat_list": chat_list,
        }
        full_response = OpenAiCharacter_model.generate_response(
            input_text =  request.input_data,
            character_settings = character_settings,
        )
        response_data = {
            "result": full_response,
            "_links": [
                {
                    "href": str(req.url),
                    "rel": "_self",
                    "type": str(req.method).lower(),
                },
                {
                    "href": str(req.base_url) + "character/Llama",
                    "rel": "character_llama",
                    "type": "post",
                },
                *get_openai_links(
                    base_url = str(req.base_url),
                    path = gpt_set,
                ),
            ]
        }
        return JSONResponse(content=response_data)

    except TimeoutError as te:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too Many Requests",
                "message": "서버 이용자가 많아 처리 시간이 초과되었습니다. 잠시 후 다시 요청해 주세요.",
                "code": "GPT_EXPLICIT_TIMEOUT_ERROR",
                "retry_after": 60
            },
            headers={"Retry-After": "60"}
        )
    except ValidationError as ve:
        raise ChatError.BadRequestException(detail=str(ve))
    except Exception as e:
        error_str = str(e)
        timeout_keywords = [
            "시간 초과", "timeout", "요청 처리 시간 초과", "5분", "4분",
            "TimeoutError", "asyncio.TimeoutError", "concurrent.futures.TimeoutError",
            "처리 시간이 초과", "time limit exceeded", "request timeout", "500:", "504:"
        ]
        
        is_timeout = any(keyword in error_str.lower() for keyword in [kw.lower() for kw in timeout_keywords])
        
        if is_timeout or "500: 요청 처리 시간 초과" in error_str:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": "서버 이용자가 많아 처리 시간이 초과되었습니다. 잠시 후 다시 요청해 주세요.",
                    "code": "GPT_TIMEOUT_DUE_TO_HIGH_LOAD",
                    "retry_after": 60,
                    "debug_info": {
                        "original_error": error_str,
                        "detected_as": "timeout_error",
                        "detection_method": "keyword_matching"
                    }
                },
                headers={"Retry-After": "60"}
            )
        else:
            raise ChatError.InternalServerErrorException(detail="내부 서버 오류가 발생했습니다.")
        
@character_router.get("/performance", summary="성능 통계 조회")
async def get_performance():
    """
    큐 핸들러의 성능 통계를 반환합니다.
    """
    if AppState.llama_queue_handler:
        stats = AppState.llama_queue_handler.get_performance_stats()
        return JSONResponse(content=stats)
    else:
        raise ChatError.InternalServerErrorException(detail="큐 핸들러가 초기화되지 않았습니다.")
