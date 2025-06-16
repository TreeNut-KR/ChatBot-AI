import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Dict

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
class LlamaGenerationConfig:
    """
    Llama 모델의 생성 설정을 정의하는 데이터 클래스입니다.
    
    Args:
        prompt (str): 생성할 프롬프트
        max_tokens (int, optional): 최대 토큰 수, 기본값은 1024
        temperature (float, optional): 온도 설정, 기본값은 1.3
        top_p (float, optional): 상위 p 설정, 기본값은 0.9
        min_p (float, optional): 최소 p 설정, 기본값은 0.1
        typical_p (float, optional): 전형적인 p 설정, 기본값은 1.0
        tfs_z (float, optional): TFS z 설정, 기본값은 1.1
        repeat_penalty (float, optional): 반복 패널티 설정, 기본값은 1.08
        frequency_penalty (float, optional): 빈도 패널티 설정, 기본값은 0.1
        presence_penalty (float, optional): 존재 패널티 설정, 기본값은 0.1
        stop (List[str], optional): 중지 문자열 리스트, 기본값은 None
        seed (int, optional): 랜덤 시드, 기본값은 None
    """
    prompt: str
    max_tokens: int = 1024
    temperature: float = 1.3
    top_p: float = 0.9
    min_p: float = 0.1
    typical_p: float = 1.0
    tfs_z: float = 1.1
    repeat_penalty: float = 1.08
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.1
    stop: Optional[List[str]] = field(default_factory=lambda: ["<|eot_id|>"])
    seed: Optional[int] = None

@dataclass(kw_only = True)
class ProcessingRequest:
    """
    처리 요청을 나타내는 데이터 클래스
    """
    id: str
    input_text: str
    search_text: str
    chat_list: list
    future: asyncio.Future
    created_at: float
    user_id: str