from fastapi import Path, APIRouter, HTTPException
from pydantic import ValidationError

import utils.app_state as AppState
from .. import OpenAiCharacter, ChatModel, ChatError, MongoDBHandler

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

OPENAI_MODEL_MAP = {
    "gpt4.1": "gpt-4.1",
    "gpt4.1_mini": "gpt-4.1-mini",
}

try:
    mongo_handler = MongoDBHandler()
except ChatError.InternalServerErrorException as e:
    mongo_handler = None
    print(f"{RED}ERROR{RESET}:    MongoDB 초기화 오류 발생: {str(e)}")

character_router = APIRouter()

@character_router.post("/Llama", summary = "Llama 모델이 케릭터 정보를 기반으로 답변 생성")
async def character_llama(request: ChatModel.character_Request):
    """
    Lumimaid_8B 모델에 질문을 입력하고 캐릭터 설정을 반영하여 답변을 JSON 방식으로 반환합니다.

    Args:
        request (ChatModel.character_Request): 사용자 요청 데이터 포함

    Returns:
        JSONResponse: JSON 방식으로 모델 응답
    """
    chat_list = []

    # MongoDB에서 채팅 기록 가져오기
    if mongo_handler or request.db_id:
        try:
            chat_list = await mongo_handler.get_character_log(
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
        full_response = AppState.Lumimaid_model.generate_response(
            input_text =  request.input_data,
            character_settings = character_settings,
        )
        return full_response

    except TimeoutError:
        raise ChatError.InternalServerErrorException(
            detail = "Lumimaid 모델 응답이 시간 초과되었습니다."
        )
    except ValidationError as e:
        raise ChatError.BadRequestException(detail = str(e))
    except Exception as e:
        print(f"처리되지 않은 예외: {e}")
        raise ChatError.InternalServerErrorException(detail = "내부 서버 오류가 발생했습니다.")

@character_router.post("/{gpt_set}", summary = "gpt 모델이 케릭터 정보를 기반으로 답변 생성")
async def character_gpt4o_mini(
        request: ChatModel.character_Request,
        gpt_set: str = Path(
            ...,
            title="GPT 모델명",
            description="사용할 OpenAI GPT 모델의 별칭 (예: gpt4o_mini, gpt4.1, gpt4.1_mini)",
            examples=list(OPENAI_MODEL_MAP.keys()),
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

    model_id = OPENAI_MODEL_MAP[gpt_set]
    chat_list = []
    
    # MongoDB에서 채팅 기록 가져오기
    if mongo_handler or request.db_id:
        try:
            chat_list = await mongo_handler.get_character_log(
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
        return full_response

    except TimeoutError:
        raise ChatError.InternalServerErrorException(
            detail = "Lumimaid 모델 응답이 시간 초과되었습니다."
        )
    except ValidationError as e:
        raise ChatError.BadRequestException(detail = str(e))
    except Exception as e:
        print(f"처리되지 않은 예외: {e}")
        raise ChatError.InternalServerErrorException(detail = "내부 서버 오류가 발생했습니다.")
