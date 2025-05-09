from fastapi import Path, APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

import utils.app_state as AppState
from .. import OpenAiOffice, ChatModel, ChatError, ChatSearch

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

OPENAI_MODEL_MAP = {
    "gpt4o_mini": {
        "id": "gpt-4o-mini",
        "type": "post"
    },
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
            "href": f"{base_url}office/{k}",
            "rel": f"office_{k}",
            "type": OPENAI_MODEL_MAP[k]["type"],
        }
        for k in OPENAI_MODEL_MAP.keys() if k != path
    ]

office_router = APIRouter()

@office_router.post("/Llama", summary = "Llama 모델이 검색 결과를 활용하여 답변 생성")
async def office_llama(request: ChatModel.office_Request, req: Request):
    """
    Bllossom_8B 모델에 질문을 위키백과, 나무위키, 뉴스 등의 결과를 결합하여 AI 답변을 JSON 방식으로 반환합니다.
    
    Args:
        request (ChatModel.office_Request): 사용자 질문과 인터넷 검색 옵션 포함
        
    Returns:
        JSONResponse: JSON 방식으로 모델 응답
    """
    chat_list = []
    search_context = ""

    # MongoDB에서 채팅 기록 가져오기
    if AppState.mongo_handler or request.db_id:
        try:
            chat_list = await AppState.mongo_handler.get_office_log(
                user_id = request.user_id,
                document_id = request.db_id,
                router = "office",
            )
        except Exception as e:
            print(f"{YELLOW}WARNING{RESET}:  채팅 기록을 가져오는 데 실패했습니다: {str(e)}")

    # DuckDuckGo 검색 결과 가져오기
    if request.google_access: # 검색 옵션이 활성화된 경우
        try:
            duck_results = await ChatSearch.fetch_duck_search_results(query = request.input_data)
        except Exception:
            print(f"{YELLOW}WARNING{RESET}:  검색의 한도 초과로 DuckDuckGo 검색 결과를 가져올 수 없습니다.")

        if duck_results:
            # 검색 결과를 AI가 이해하기 쉬운 형식으로 변환
            formatted_results = []
            for idx, item in enumerate(duck_results[:10], 1): # 상위 10개 결과만 사용
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
    try:        
        full_response = AppState.LlamaOffice_model.generate_response(
            input_text = request.input_data,
            search_text = search_context,
            chat_list = chat_list,
        )
        response_data = {
            "result": full_response,
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

    except TimeoutError:
        raise ChatError.InternalServerErrorException(
            detail = "Bllossom 모델 응답이 시간 초과되었습니다."
        )
    except ValidationError as e:
        raise ChatError.BadRequestException(detail = str(e))
    except Exception as e:
        print(f"처리되지 않은 예외: {e}")
        raise ChatError.InternalServerErrorException(detail = "내부 서버 오류가 발생했습니다.")


@office_router.post("/{gpt_set}", summary = "gpt 모델이 검색 결과를 활용하여 답변 생성")
async def office_gpt(
        request: ChatModel.office_Request,
        req: Request,
        gpt_set: str = Path(
            ...,
            title="GPT 모델명",
            description=f"사용할 OpenAI GPT 모델의 별칭 (예: {list(OPENAI_MODEL_MAP.keys())})",
            examples=list(OPENAI_MODEL_MAP.keys()),
        )
    ):
    """
    gpt 모델에 질문을 입력하고 응답을 JSON 방식으로 반환합니다.
    
    Args:
        request (ChatModel.office_Request): 사용자 질문과 인터넷 검색 옵션 포함
        
    Returns:
        JSONResponse: JSON 방식으로 모델 응답
    """
    if gpt_set not in OPENAI_MODEL_MAP:
        raise HTTPException(status_code = 400, detail = "Invalid model name.")

    model_id = OPENAI_MODEL_MAP[gpt_set]["id"]
    chat_list = []
    search_context = ""

    # MongoDB에서 채팅 기록 가져오기
    if AppState.mongo_handler or request.db_id:
        try:
            chat_list = await AppState.mongo_handler.get_office_log(
                user_id = request.user_id,
                document_id = request.db_id,
                router = "office",
            )
        except Exception as e:
            print(f"{YELLOW}WARNING{RESET}:  채팅 기록을 가져오는 데 실패했습니다: {str(e)}")

    # DuckDuckGo 검색 결과 가져오기
    if request.google_access: # 검색 옵션이 활성화된 경우
        try:
            duck_results = await ChatSearch.fetch_duck_search_results(query = request.input_data)
        except Exception:
            print(f"{YELLOW}WARNING{RESET}:  검색의 한도 초과로 DuckDuckGo 검색 결과를 가져올 수 없습니다.")

        if duck_results:
            # 검색 결과를 AI가 이해하기 쉬운 형식으로 변환
            formatted_results = []
            for idx, item in enumerate(duck_results[:10], 1): # 상위 10개 결과만 사용
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

    OpenAiOffice_model = OpenAiOffice(model_id = model_id)
    try:
        full_response = OpenAiOffice_model.generate_response(
            input_text = request.input_data,
            search_text = search_context,
            chat_list = chat_list,
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
                    "href": str(req.base_url) + "office/Llama",
                    "rel": "office_llama",
                    "type": "post",
                },
                *get_openai_links(
                    base_url = str(req.base_url),
                    path = gpt_set,
                ),
            ]
        }
        return JSONResponse(content=response_data)

    except TimeoutError:
        raise ChatError.InternalServerErrorException(detail = "OpenAI 모델 응답이 시간 초과되었습니다.")
    except ValidationError as e:
        raise ChatError.BadRequestException(detail = str(e))
    except Exception as e:
        print(f"처리되지 않은 예외: {e}")
        raise ChatError.InternalServerErrorException(detail = "내부 서버 오류가 발생했습니다.")
