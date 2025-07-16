from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import asyncio

@dataclass(kw_only=True)
class VeniceGenerationConfig:
    """
    Venice API용 생성 파라미터 데이터 클래스
    """
    messages: List[Dict]
    model: str = "llama-3.3-70b"
    max_tokens: int = 1000
    temperature: float = 1.0
    top_p: float = 0.9
    stream: bool = False
    venice_parameters: Optional[Dict] = None

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
    user_name: str
    user_input: str
    chat_list: List[Dict] = None

@dataclass(kw_only = True)
class LlamaGenerationConfig:
    """
    Llama 모델의 생성 설정을 정의하는 데이터 클래스입니다.
    
    Args:
        prompt (str): 생성할 프롬프트
        max_tokens (int, optional): 최대 토큰 수, 기본값은 2048
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
    max_tokens: int = 2048
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
    top_k: int = 40

@dataclass(kw_only = True)
class ProcessingRequest:
    """
    처리 요청을 나타내는 데이터 클래스
    
    Args:
        id (str): 요청 ID
        input_text (str): 입력 텍스트
        character_settings (Dict[str, Any]): 캐릭터 설정
        future (asyncio.Future): 비동기 작업의 Future 객체
        created_at (float): 요청 생성 시간 (타임스탬프)
        user_id (str): 사용자 ID
        character_name (str): 캐릭터 이름
        user_name (str): 사용자 이름
    """
    id: str
    input_text: str
    character_settings: Dict[str, Any]
    future: asyncio.Future
    created_at: float
    user_id: str
    character_name: str
    user_name: str = ""
