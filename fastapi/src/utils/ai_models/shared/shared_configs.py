from dataclasses import dataclass
from typing import Optional, List, Dict, TypedDict

@dataclass(kw_only = True)
class OfficePrompt:
    """
    office 프롬프트를 정의하는 데이터 클래스입니다.
    
    Args:
        name (str): 캐릭터 이름
        context (str): 캐릭터 설정
        reference_data (str): 참고 데이터
        user_input (str): 사용자 입력
        chat_list (List[Dict], optional): 대화 기록 리스트, 기본값은 None
    """
    name: str
    context: str
    reference_data: str
    user_input: str
    chat_list: List[Dict] = None

@dataclass(kw_only = True)
class CharacterPrompt:
    """
    Character 프롬프트를 정의하는 데이터 클래스입니다.
    
    Args:
        name (str): 캐릭터 이름
        greeting (str): 캐릭터 인사말
        context (str): 캐릭터 설정
        user_input (str): 사용자 입력
        chat_list (List[Dict], optional): 대화 기록 리스트, 기본값은 None
    """
    name: str
    greeting: str
    context: str
    user_input: str
    chat_list: List[Dict] = None

@dataclass(kw_only = True)
class LlamaGenerationConfig:
    """
    Llama 모델의 생성 설정을 정의하는 데이터 클래스입니다.
    
    Args:
        prompt (str): 생성할 프롬프트
        max_tokens (int, optional): 최대 토큰 수, 기본값은 256
        temperature (float, optional): 온도 설정, 기본값은 0.8
        top_p (float, optional): 상위 p 설정, 기본값은 0.95
        stop (List[str], optional): 중지 문자열 리스트, 기본값은 None
    """
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    stop: Optional[List[str]] = None
    
@dataclass(kw_only = True)
class OpenAIGenerationConfig:
    """
    OpenAI 모델의 생성 설정을 정의하는 데이터 클래스입니다.
    
    Args:
        messages (List[dict]): 생성할 메시지 리스트
        max_tokens (int, optional): 최대 토큰 수, 기본값은 1000
        temperature (float, optional): 온도 설정, 기본값은 0.82
        top_p (float, optional): 상위 p 설정, 기본값은 0.95
    """
    messages: List[dict]
    max_tokens: int = 1000
    temperature: float = 0.82
    top_p: float = 0.95

class BaseConfig(TypedDict):
    """
    JSON 파일의 Prompt 기본 설정을 정의하는 TypedDict입니다.
    
    Args:
        character_name (str): 캐릭터 이름
        greeting (str): 캐릭터 인사말
        character_setting (List[str]): 캐릭터 설정 리스트
    """
    character_name: str
    greeting: str
    character_setting: List[str]