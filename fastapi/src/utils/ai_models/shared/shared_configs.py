from dataclasses import dataclass
from typing import Optional, List, TypedDict

class OfficePrompt:
    def __init__(self, name: str, context: str, search_text: str) -> tuple:
        """
        초기화 메소드

        Args:
            name (str): 캐릭터 이름
            context (str): 캐릭터 설정
            search_text (str): 검색 텍스트
        """
        self.name=name
        self.context=context
        self.search_text=search_text

    def __str__(self) -> str:
        """
        문자열 출력 메소드
        
        Returns:
            str: 캐릭터 정보 문자열
        """
        return (
            f"Name: {self.name}\n"
            f"Context: {self.context}\n"
            f"Search Text: {self.search_text}"
        )
    
class CharacterPrompt:
    def __init__(self, name: str, greeting: str, context: str) -> tuple:
        """
        초기화 메소드

        Args:
            name (str): 캐릭터 이름
            greeting (str): 캐릭터의 초기 인사말
            context (str): 캐릭터 설정
        """
        self.name=name
        self.greeting=greeting
        self.context=context

    def __str__(self) -> str:
        """
        문자열 출력 메소드
        
        Returns:
            str: 캐릭터 정보 문자열
        """
        return (
            f"Name: {self.name}\n"
            f"greeting: {self.greeting}\n"
            f"Context: {self.context}"
        )

@dataclass
class GenerationConfig:
    prompt: str
    max_tokens: int=256
    temperature: float=0.8
    top_p: float=0.95
    stop: Optional[List[str]]=None
    
@dataclass
class OpenAICompletionConfig:
    messages: List[dict]
    max_tokens: int=1000
    temperature: float=0.82
    top_p: float=0.95

class BaseConfig(TypedDict):
    character_name: str
    greeting: str
    character_setting: List[str]