'''
파일은 OpenAIOfficeModel, OfficePrompt 등등 클래스를 정의하고 OpenAI API를 사용하여,
'대화형 인공지능 서비스' 용도의 기능을 제공합니다.
'''
import os
import json
import warnings
from pathlib import Path
from typing import Optional, Generator, List, Dict
from queue import Queue
from threading import Thread
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

from .shared.shared_configs import OfficePrompt, OpenAIGenerationConfig, BaseConfig

def build_openai_messages(character_info: OfficePrompt) -> list:
    """
    캐릭터 정보와 대화 기록을 포함한 OpenAI API messages 형식 생성

    Args:
        character_info (OfficePrompt): 캐릭터 정보

    Returns:
        list: OpenAI API 형식의 messages 리스트
    """
    system_prompt = (
        f"당신은 {character_info.name}입니다.\n"
        f"설정: {character_info.context}\n"
        f"참고 정보(아래 정보는 참고만 하세요. 사용자의 질문과 직접 관련이 없으면 답변에 포함하지 마세요):{character_info.reference_data}"
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    if character_info.chat_list and len(character_info.chat_list) > 0:
        for chat in character_info.chat_list:
            user_message = chat.get("input_data", "")
            assistant_message = chat.get("output_data", "")
            
            if user_message:
                messages.append({"role": "user", "content": user_message})
            if assistant_message:
                messages.append({"role": "assistant", "content": assistant_message})
                
    messages.append({"role": "user", "content": character_info.user_input})
    return messages

class OpenAIOfficeModel:
    """
    [<img src = "https://brandingstyleguides.com/wp-content/guidelines/2025/02/openAi-web.jpg" width = "100" height = "auto">](https://platform.openai.com/docs/models)
    
    OpenAI API를 사용하여 대화를 생성하는 클래스입니다.
    
    모델 정보:
    - 제작자: OpenAI
    - 소스: [OpenAI API](https://platform.openai.com/docs/models)
    """
    def __init__(self, model_id = 'gpt-4o-mini') -> None:
        """
        OpenAIOfficeModel 클래스 초기화 메소드
        """
        self.model_id = model_id
        self.file_path = '/app/prompt/config-OpenAI.json'
        
        # 환경 변수 파일 경로 설정 수정
        env_file_path = Path(__file__).resolve().parents[3] / ".env"
        self.character_info: Optional[OfficePrompt] = None
        self.config: Optional[OpenAIGenerationConfig] = None

        if not os.path.exists(env_file_path):
            raise FileNotFoundError(f".env 파일을 찾을 수 없습니다: {env_file_path}")

        load_dotenv(env_file_path)
        
        # JSON 파일 읽기
        try:
            with open(self.file_path, 'r', encoding = 'utf-8') as file:
                self.data: BaseConfig = json.load(file)
        except FileNotFoundError:
            print(f"설정 파일을 찾을 수 없습니다: {self.file_path}")
            # 기본 설정 사용
            self.data: BaseConfig = {
                "character_name": "GPT 어시스턴트",
                "character_setting": "친절하고 도움이 되는 AI 어시스턴트입니다.",
                "greeting": "안녕하세요! 무엇을 도와드릴까요?"
            }
            
        # API 키 설정
        self.api_key: str = os.getenv("OPENAI_API_KEY")
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
            return OpenAI(api_key = self.api_key)
        except Exception as e:
            print(f"❌ OpenAI 클라이언트 초기화 중 오류 발생: {e}")
            raise
            
    def _stream_completion(self, config: OpenAIGenerationConfig) -> None:
        """
        텍스트 생성을 위한 내부 스트리밍 메서드입니다.

        Args:
            config (OpenAIGenerationConfig): 생성 파라미터 객체

        Effects:
            - response_queue에 생성된 텍스트 조각들을 순차적으로 추가
            - 스트림 종료 시 None을 큐에 추가
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                stream = self.client.chat.completions.create(
                    model = self.model_id,
                    messages = config.messages,
                    max_tokens = config.max_tokens,
                    temperature = config.temperature,
                    top_p = config.top_p,
                    stream = True
                )

                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        content = chunk.choices[0].delta.content
                        if content is not None:
                            self.response_queue.put(content)

                self.response_queue.put(None)

        except Exception as e:
            print(f"스트리밍 중 오류 발생: {e}")
            self.response_queue.put(None)

    def create_streaming_completion(self, config: OpenAIGenerationConfig) -> Generator[str, None, None]:
        """
        스트리밍 방식으로 텍스트 응답을 생성하는 메서드입니다.

        Args:
            config (OpenAIGenerationConfig): 생성 파라미터 객체

        Returns:
            Generator[str, None, None]: 생성된 텍스트 조각들의 제너레이터
        """
        # 스트리밍 스레드 시작
        thread = Thread(
            target = self._stream_completion,
            args = (config,)
        )
        thread.start()

        # 응답 스트리밍
        while True:
            text = self.response_queue.get()
            if text is None:  # 스트림 종료
                break
            yield text

    def generate_response(self, input_text: str, search_text: str, chat_list: List[Dict]) -> str:
        """
        API 호환을 위한 스트리밍 응답 생성 메서드

        Args:
            input_text (str): 사용자 입력 텍스트
            search_text (str): 검색 결과 텍스트
            chat_list (List[Dict]): 이전 대화 기록

        Returns:
            str: 생성된 텍스트을 반환하는 제너레이터
        """
        try:
            current_time = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
            time_info = f"현재 시간은 {current_time}입니다.\n\n"
            reference_text = time_info + (search_text if search_text else "")

            normalized_chat_list = []
            if chat_list and len(chat_list) > 0:
                for chat in chat_list:
                    normalized_chat = {
                        "index": chat.get("index"),
                        "input_data": chat.get("input_data"),
                        "output_data": self._normalize_escape_chars(chat.get("output_data", ""))
                    }
                    normalized_chat_list.append(normalized_chat)
            else:
                normalized_chat_list = chat_list

            self.character_info: OfficePrompt = OfficePrompt(
                name = self.data.get("character_name"),
                context = self.data.get("character_setting"),
                reference_data = reference_text,
                user_input = input_text,
                chat_list = normalized_chat_list,
            )

            # OpenAI API 메시지 형식 생성
            messages = build_openai_messages(character_info = self.character_info)

            self.config = OpenAIGenerationConfig(
                messages = messages,
                max_tokens = 1000,
                temperature = 0.82,
                top_p = 0.95
            )

            chunks = []
            for text_chunk in self.create_streaming_completion(config = self.config):
                chunks.append(text_chunk)
            return "".join(chunks)

        except Exception as e:
            print(f"응답 생성 중 오류 발생: {e}")
            return f"오류: {str(e)}"

    def _normalize_escape_chars(self, text: str) -> str:
        """
        이스케이프 문자가 중복된 문자열을 정규화합니다
        """
        if not text:
            return ""
            
        # 이스케이프된 개행문자 등을 정규화
        result = text.replace("\\n", "\n")
        result = result.replace("\\\\n", "\n")
        result = result.replace('\\"', '"')
        result = result.replace("\\\\", "\\")
        
        return result