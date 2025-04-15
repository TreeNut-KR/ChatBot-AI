"""
파일은 GPT4oCharacterModel, CharacterPrompt 클래스를 정의하고  OpenAI API를 사용하여,
'AI 페르소나 서비스' 용도의 기능을 제공합니다.
"""
import os
import json
import warnings
from typing import Generator, List, Dict
from queue import Queue
from threading import Thread
from dotenv import load_dotenv
from openai import OpenAI

class CharacterPrompt:
    def __init__(self, name: str, greeting: str, context: str) -> None:
        """
        캐릭터 정보를 담는 클래스

        Args:
            name (str): 캐릭터 이름
            greeting (str): 캐릭터의 초기 인사말
            context (str): 캐릭터 설정
        """
        self.name = name
        self.greeting = greeting
        self.context = context

    def __str__(self) -> str:
        """
        문자열 출력 메서드
        
        Returns:
            str: 캐릭터 정보 문자열
        """
        return (
            f"Name: {self.name}\n"
            f"Greeting: {self.greeting}\n"
            f"Context: {self.context}"
        )

def build_openai_messages(character: CharacterPrompt, user_input: str, chat_list: List[Dict] = None) -> list:
    """
    캐릭터 정보와 대화 기록을 포함한 OpenAI API messages 형식 생성

    Args:
        character (CharacterPrompt): 캐릭터 정보
        user_input (str): 사용자 입력
        chat_list (List[Dict], optional): 이전 대화 기록

    Returns:
        list: OpenAI API 형식의 messages 리스트
    """
    system_prompt = (
        f"당신은 {character.name}입니다.\n"
        f"인사말: {character.greeting}\n"
        f"설정: {character.context}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    if chat_list and len(chat_list) > 0:
        for chat in chat_list:
            user_message = chat.get("input_data", "")
            assistant_message = chat.get("output_data", "")
            
            if user_message:
                messages.append({"role": "user", "content": user_message})
            if assistant_message:
                messages.append({"role": "assistant", "content": assistant_message})
    
    messages.append({"role": "user", "content": user_input})
    return messages

class OpenAICharacterModel:
    """
    [<img src="https://brandingstyleguides.com/wp-content/guidelines/2025/02/openAi-web.jpg" width="100" height="auto">](https://platform.openai.com/docs/models)
    
    OpenAI API를 사용하여 대화를 생성하는 클래스입니다.
    
    모델 정보:
    - 모델명: gpt-4o-mini, gpt-4.1, gpt-4.1-mini
    - 제작자: OpenAI
    - 소스: [OpenAI API](https://platform.openai.com/docs/models)
    """
    def __init__(self, model_id = 'gpt-4o-mini') -> None:
        self.model_id = model_id
        self.file_path = './models/config-OpenAI.json'
        
        # 환경파일 로드
        current_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_file_path = os.path.join(current_directory, '.env')
      
        if not os.path.exists(env_file_path):
            raise FileNotFoundError(f".env 파일을 찾을 수 없습니다: {env_file_path}")
        
        load_dotenv(env_file_path)
        
        # 기본 설정 로드
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
        except FileNotFoundError:
            self.data = {
                "character_name": "GPT 어시스턴트",
                "character_setting": "친절하고 도움이 되는 AI 어시스턴트입니다.",
                "greeting": "안녕하세요! 무엇을 도와드릴까요?"
            }
        
        # API 키 설정
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            
        # OpenAI 클라이언트 초기화
        self.client = self._init_client()
        self.response_queue = Queue()

    def _init_client(self) -> OpenAI:
        """
        OpenAI 클라이언트를 초기화합니다.
        
        Returns:
            OpenAI: 초기화된 OpenAI 클라이언트 인스턴스
        """
        try:
            return OpenAI(api_key=self.api_key)
        except Exception as e:
            print(f"OpenAI 클라이언트 초기화 중 오류: {e}")
            raise

    def _stream_completion(self, messages: list, **kwargs) -> None:
        """
        텍스트 생성을 위한 내부 스트리밍 메서드입니다.
        
        Args:
            messages (list): OpenAI API에 전달할 메시지 목록
            **kwargs: 생성 매개변수 (temperature, top_p 등)
            
        Effects:
            - response_queue에 생성된 텍스트 조각들을 순차적으로 추가
            - 스트림 종료 시 None을 큐에 추가
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                stream = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    stream=True,
                    **kwargs
                )
                
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        content = chunk.choices[0].delta.content
                        if content:
                            self.response_queue.put(content)
                            
                self.response_queue.put(None)
        except Exception as e:
            print(f"스트리밍 중 오류: {e}")
            self.response_queue.put(None)

    def create_streaming_completion(self,
                                    messages: list,
                                    max_tokens: int = 1000,
                                    temperature: float = 0.7,
                                    top_p: float = 0.95) -> Generator[str, None, None]:
        """
        스트리밍 방식으로 텍스트 응답을 생성하는 메서드입니다.
        
        Args:
            messages (list): OpenAI API에 전달할 메시지 목록
            max_tokens (int, optional): 생성할 최대 토큰 수. 기본값: 1000
            temperature (float, optional): 샘플링 온도 (0~2). 기본값: 0.7
            top_p (float, optional): 누적 확률 임계값 (0~1). 기본값: 0.95
            
        Returns:
            Generator[str, None, None]: 생성된 텍스트 조각들의 제너레이터
        """
        kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        thread = Thread(target=self._stream_completion, args=(messages,), kwargs=kwargs)
        thread.start()

        while True:
            text = self.response_queue.get()
            if text is None:
                break
            yield text

    def create_completion(self,
                          messages: list,
                          max_tokens: int = 1000,
                          temperature: float = 0.7,
                          top_p: float = 0.95) -> str:
        """
        주어진 프롬프트로부터 텍스트 응답 생성

        Args:
            prompt (str): 입력 프롬프트 (Llama3 형식)
            max_tokens (int, optional): 생성할 최대 토큰 수 (기본값 1000)
            temperature (float, optional): 생성 온도 (기본값 0.7)
            top_p (float, optional): top_p 샘플링 값 (기본값 0.95)

        Returns:
            str: 생성된 텍스트 응답
        """
        full_text = []
        for chunk in self.create_streaming_completion(messages, max_tokens, temperature, top_p):
            full_text.append(chunk)
        return "".join(full_text)

    def generate_response_stream(self, input_text: str, character_settings: Dict) -> Generator[str, None, None]:
        """
        Args:
            input_text (str): 사용자 입력 텍스트
            character_settings (dict): 캐릭터 설정 딕셔너리

        Returns:
            Generator[str, None, None]: 생성된 텍스트 조각들을 반환하는 제너레이터
        """
        try:
            character_info = CharacterPrompt(
                name=character_settings.get("character_name", self.data.get("character_name")),
                greeting=character_settings.get("greeting", self.data.get("greeting")),
                context=character_settings.get("character_setting", self.data.get("character_setting")),
            )
            chat_history = character_settings.get("chat_list", None)

            messages = build_openai_messages(character_info, input_text, chat_history)

            for text_chunk in self.create_streaming_completion(messages):
                yield text_chunk
        except Exception as e:
            print(f"응답 생성 중 오류: {e}")
            yield f"오류: {str(e)}"