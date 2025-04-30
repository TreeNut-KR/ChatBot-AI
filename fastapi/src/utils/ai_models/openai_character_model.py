"""
파일은 OpenAICharacterModel, CharacterPrompt 등등 클래스를 정의하고  OpenAI API를 사용하여,
'AI 페르소나 서비스' 용도의 기능을 제공합니다.
"""
import os
import json
import warnings
from typing import Optional, Generator, List, Dict
from queue import Queue
from threading import Thread
from dotenv import load_dotenv
from openai import OpenAI

from .shared.shared_configs import CharacterPrompt, OpenAICompletionConfig, BaseConfig
    
def build_openai_messages(character: CharacterPrompt, user_input: str, chat_list: List[Dict]=None) -> list:
    """
    캐릭터 정보와 대화 기록을 포함한 OpenAI API messages 형식 생성

    Args:
        character (CharacterPrompt): 캐릭터 정보
        user_input (str): 사용자 입력
        chat_list (List[Dict], optional): 이전 대화 기록

    Returns:
        list: OpenAI API 형식의 messages 리스트
    """
    system_prompt=(
        f"당신은 {character.name}입니다.\n"
        f"인사말: {character.greeting}\n"
        f"설정: {character.context}"
    )
    
    messages=[
        {"role": "system", "content": system_prompt}
    ]
    
    if chat_list and len(chat_list) > 0:
        for chat in chat_list:
            user_message=chat.get("input_data", "")
            assistant_message=chat.get("output_data", "")
            
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
    def __init__(self, model_id='gpt-4o-mini') -> None:
        self.model_id=model_id
        self.file_path='./prompt/config-OpenAI.json'
        
        # 환경파일 로드
        current_directory=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_file_path=os.path.join(current_directory, '.env')
        self.character_info: Optional[CharacterPrompt]=None
        
        if not os.path.exists(env_file_path):
            raise FileNotFoundError(f".env 파일을 찾을 수 없습니다: {env_file_path}")
        
        load_dotenv(env_file_path)
        
        # JSON 파일 읽기
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.data: BaseConfig=json.load(file)
        except FileNotFoundError:
            print(f"설정 파일을 찾을 수 없습니다: {self.file_path}")
            # 기본 설정 사용
            self.data: BaseConfig={
                "character_name": "GPT 어시스턴트",
                "character_setting": "친절하고 도움이 되는 AI 어시스턴트입니다.",
                "greeting": "안녕하세요! 무엇을 도와드릴까요?"
            }
        
        # API 키 설정
        self.api_key: str=os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            
        # OpenAI 클라이언트 초기화
        self.client=self._init_client()
        self.response_queue=Queue()

    def _init_client(self) -> OpenAI:
        """
        OpenAI 클라이언트를 초기화합니다.
        
        Returns:
            OpenAI: 초기화된 OpenAI 클라이언트 인스턴스
        """
        try:
            return OpenAI(api_key=self.api_key)
        except Exception as e:
            print(f"❌ OpenAI 클라이언트 초기화 중 오류: {e}")
            raise

    def _stream_completion(self, config: OpenAICompletionConfig) -> None:
        """
        텍스트 생성을 위한 내부 스트리밍 메서드입니다.

        Args:
            config (OpenAICompletionConfig): 생성 파라미터 객체

        Effects:
            - response_queue에 생성된 텍스트 조각들을 순차적으로 추가
            - 스트림 종료 시 None을 큐에 추가
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                stream=self.client.chat.completions.create(
                    model=self.model_id,
                    messages=config.messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        content=chunk.choices[0].delta.content
                        if content:
                            self.response_queue.put(content)
                            
                self.response_queue.put(None)
                
        except Exception as e:
            print(f"스트리밍 중 오류: {e}")
            self.response_queue.put(None)

    def create_streaming_completion(self, config: OpenAICompletionConfig) -> Generator[str, None, None]:
        """
        스트리밍 방식으로 텍스트 응답을 생성하는 메서드입니다.

        Args:
            config (OpenAICompletionConfig): 생성 파라미터 객체

        Returns:
            Generator[str, None, None]: 생성된 텍스트 조각들의 제너레이터
        """
        # 스트리밍 스레드 시작
        thread=Thread(
            target=self._stream_completion,
            args=(config,)
        )
        thread.start()

        # 응답 스트리밍
        while True:
            text=self.response_queue.get()
            if text is None:  # 스트림 종료
                break
            yield text

    def create_completion(self, config: OpenAICompletionConfig) -> str:
        """
        주어진 프롬프트로부터 텍스트 응답 생성

        Args:
            config (GenerationConfig): 생성 파라미터 객체

        Returns:
            str: 생성된 텍스트 응답
        """
        full_text=[]
        for chunk in self.create_streaming_completion(config=config):
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
            self.character_info=CharacterPrompt(
                name=character_settings.get("character_name", self.data.get("character_name")),
                greeting=character_settings.get("greeting", self.data.get("greeting")),
                context=character_settings.get("character_setting", self.data.get("character_setting")),
            )
            chat_list=character_settings.get("chat_list", None)
            
            # 이스케이프 문자 정규화
            normalized_chat_list=[]
            if chat_list and len(chat_list) > 0:
                for chat in chat_list:
                    normalized_chat={
                        "index": chat.get("index"),
                        "input_data": chat.get("input_data"),
                        "output_data": self._normalize_escape_chars(chat.get("output_data", ""))
                    }
                    normalized_chat_list.append(normalized_chat)
            else:
                normalized_chat_list=chat_list

            messages=build_openai_messages(
                character=self.character_info,
                user_input=input_text,
                chat_list=normalized_chat_list,
            )
            
            config=OpenAICompletionConfig(
                messages=messages,
                max_tokens=1000,
                temperature=0.82,
                top_p=0.95
            )

            for text_chunk in self.create_streaming_completion(config=config):
                yield text_chunk
                
        except Exception as e:
            print(f"응답 생성 중 오류: {e}")
            yield f"오류: {str(e)}"
        
    def _normalize_escape_chars(self, text: str) -> str:
        """
        이스케이프 문자가 중복된 문자열을 정규화합니다
        """
        if not text:
            return ""
            
        # 이스케이프된 개행문자 등을 정규화
        result=text.replace("\\n", "\n")
        result=result.replace("\\\\n", "\n")
        result=result.replace('\\"', '"')
        result=result.replace("\\\\", "\\")
        
        return result