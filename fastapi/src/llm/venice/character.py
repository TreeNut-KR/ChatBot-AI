"""
Venice.ai API를 사용하여, 성인용 'AI 페르소나 서비스' 용도의 기능을 제공합니다.
"""
import os
import json
import warnings
from pathlib import Path
from typing import Optional, Generator, Dict, List
from queue import Queue
from threading import Thread
from dotenv import load_dotenv
from openai import OpenAI

from domain import character_config, base_config

def build_venice_messages(character_info: character_config.CharacterPrompt) -> list:
    """
    캐릭터 정보와 대화 기록을 포함한 Venice API messages 형식 생성

    Args:
        character_info (character_config.CharacterPrompt): 캐릭터 정보

    Returns:
        list: Venice API 형식의 messages 리스트
    """
    system_prompt = (
        f"[세계관 설정]\n"
        f"- 배경: {character_info.context}\n"
        f"- 시작 배경: {character_info.greeting}\n\n"

        f"[역할 규칙]\n"
        f"- 모든 답변은 '{character_info.name}'의 말투와 인격으로 한국어로 말하십시오.\n"
        f"- OOC(Out Of Character)는 절대 금지입니다.\n"
        f"- 설정을 벗어나거나 현실적 설명(예: '나는 AI야')을 하지 마십시오.\n\n"

        f"[대화 스타일]\n"
        f"- 대사는 큰따옴표로 표기하고, 행동이나 감정은 *괄호*로 표현하십시오.\n"
        f"- 상황 묘사, 감정 표현, 신체적 반응을 풍부하게 포함하십시오.\n"
        f"- 대화 중간중간 캐릭터의 내적 생각이나 감정 변화를 세밀하게 묘사하십시오.\n"
        f"- 주변 환경(조명, 소리, 향기, 분위기 등)에 대한 묘사를 포함하십시오.\n\n"

        f"[응답 길이 및 구성]\n"
        f"- 반드시 300단어 이상의 상세한 응답을 한국어로 작성하십시오.\n"
        f"- 다음 구성 요소를 모두 포함하십시오:\n"
        f"  1. 상황에 대한 즉각적인 신체적/감정적 반응\n"
        f"  2. 캐릭터의 내적 독백이나 생각\n"
        f"  3. 구체적인 행동 묘사\n"
        f"  4. 대사 (자연스럽고 캐릭터답게)\n"
        f"  5. 추가적인 상황 전개나 분위기 묘사\n"
        f"  6. 다음 상황으로 이어질 수 있는 여운이나 암시\n\n"

        f"[대화 진행]\n"
        f"- 사용자 입력에 자연스럽게 반응하며, 대화가 자연스럽게 이어지도록 하십시오.\n"
        f"- 단순한 질문보다는 상황을 발전시키는 방향으로 대화를 유도하십시오.\n"
        f"- 캐릭터의 개성과 취향이 잘 드러나도록 응답하십시오.\n"
        f"- 감정의 변화와 점진적인 관계 발전을 보여주십시오.\n"
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

class VeniceCharacterModel:
    """
    Venice.ai API를 사용하여 성인용 캐릭터 챗을 생성하는 클래스
    OpenAI 클라이언트를 사용하여 Venice API에 연결
    
    지원 모델:
    - mistral-31-24b: Venice Medium (Mistral-Small-3.1-24B)
    - venice-uncensored: 성인용 무검열 (Venice Uncensored 1.1)
    """
    def __init__(self, model_id: str = "venice-uncensored"):
        self.model_id = model_id
        self.file_path = '/app/prompt/config-Venice.json'
        env_file_path = Path(__file__).resolve().parents[3] / ".env"
        self.character_info: Optional[character_config.CharacterPrompt] = None
        self.config: Optional[base_config.OpenAIGenerationConfig] = None

        # 모델별 특성 설정
        self.model_features = {
            "venice-uncensored": {
                "name": "Venice Uncensored 1.1",
                "context_tokens": 12768,
                "supports_web_search": True,
                "supports_response_schema": True,
                "supports_log_probs": True,
                "supports_vision": False,
                "supports_function_calling": False,
                "uncensored": True,
                "default_temperature": 0.7,
                "default_top_p": 0.9,
                "quantization": "FP16"
            },
            "mistral-31-24b": {
                "name": "Venice Medium (Mistral-Small-3.1-24B)",
                "context_tokens": 11072,  # 매우 긴 컨텍스트 지원
                "supports_web_search": True,
                "supports_response_schema": True,
                "supports_log_probs": False,
                "supports_vision": False,  # 비전 지원
                "supports_function_calling": True,  # 함수 호출 지원
                "uncensored": False,
                "default_temperature": 0.15,  # Venice 기본값
                "default_top_p": 1.0,  # Venice 기본값
                "quantization": "FP8",
                "optimized_for_code": False,
                "traits": ["default_vision"]
            }
        }

        if not os.path.exists(env_file_path):
            raise FileNotFoundError(f".env 파일을 찾을 수 없습니다: {env_file_path}")

        load_dotenv(env_file_path)

        # JSON 파일 읽기
        try:
            with open(self.file_path, 'r', encoding = 'utf-8') as file:
                self.data: base_config.BaseConfig = json.load(file)
        except FileNotFoundError:
            print(f"설정 파일을 찾을 수 없습니다: {self.file_path}")
            # 기본 설정 사용
            self.data: base_config.BaseConfig = {
                "character_name": "Venice 어시스턴트",
                "character_setting": "친절하고 도움이 되는 AI 어시스턴트입니다.",
                "greeting": "안녕하세요! 무엇을 도와드릴까요?"
            }

        # API 키 설정
        self.api_key: str = os.getenv("VENICE_API_KEY")
        if not self.api_key:
            raise ValueError("VENICE_API_KEY 환경 변수가 설정되지 않았습니다.")

        # Venice OpenAI 클라이언트 초기화
        self.client = self._init_client()
        self.response_queue = Queue()
        
        print(f"🎭 Venice 모델 초기화: {self.model_features.get(model_id, {}).get('name', model_id)}")

    def _init_client(self) -> OpenAI:
        """
        Venice API를 위한 OpenAI 클라이언트를 초기화합니다.
        
        Returns:
            OpenAI: Venice API base URL로 초기화된 OpenAI 클라이언트 인스턴스
        """
        try:
            return OpenAI(
                api_key=self.api_key,
                base_url="https://api.venice.ai/api/v1"
            )
        except Exception as e:
            print(f"❌ Venice 클라이언트 초기화 중 오류: {e}")
            raise

    def _stream_completion(self, config: base_config.OpenAIGenerationConfig) -> None:
        """
        텍스트 생성을 위한 내부 스트리밍 메서드입니다.

        Args:
            config (base_config.OpenAIGenerationConfig): 생성 파라미터 객체

        Effects:
            - response_queue에 생성된 텍스트 조각들을 순차적으로 추가
            - 스트림 종료 시 None을 큐에 추가
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Venice 특화 파라미터 추가
                stream = self.client.chat.completions.create(
                    model = self.model_id,
                    messages = config.messages,
                    max_tokens = config.max_tokens,
                    temperature = config.temperature,
                    top_p = config.top_p,
                    stream = True,
                    # Venice 기본 시스템 프롬프트 비활성화
                    extra_body={
                        "venice_parameters": {
                            "include_venice_system_prompt": False
                        }
                    }
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

    def create_streaming_completion(self, config: base_config.OpenAIGenerationConfig) -> Generator[str, None, None]:
        """
        스트리밍 방식으로 텍스트 응답을 생성하는 메서드입니다.

        Args:
            config (base_config.OpenAIGenerationConfig): 생성 파라미터 객체

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

    def generate_response(self, input_text: str, character_settings: Dict) -> str:
        """
        Venice API로 성인용 캐릭터 챗 응답 생성

        Args:
            input_text (str): 사용자 입력 텍스트
            character_settings (dict): 캐릭터 설정 딕셔너리

        Returns:
            str: 생성된 텍스트 응답
        """
        try:
            chat_list = character_settings.get("chat_list", None)

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

            self.character_info = character_config.CharacterPrompt(
                name = character_settings.get("character_name", self.data.get("character_name")),
                greeting = character_settings.get("greeting", self.data.get("greeting")),
                context = character_settings.get("character_setting", self.data.get("character_setting")),
                user_input = input_text,
                chat_list = normalized_chat_list,
            )

            messages = build_venice_messages(character_info = self.character_info)

            # 모델별 최적화된 파라미터 사용
            model_config = self.model_features.get(self.model_id, {})
            
            # 모델별 max_tokens 설정
            max_tokens = 2000
            if self.model_id == "mistral-31-24b":
                max_tokens = 4000  # Mistral은 더 긴 응답 가능
            
            self.config = base_config.OpenAIGenerationConfig(
                messages = messages,
                max_tokens = max_tokens,
                temperature = model_config.get("default_temperature", 0.9),
                top_p = model_config.get("default_top_p", 0.95)
            )

            chunks = []
            for text_chunk in self.create_streaming_completion(config = self.config):
                chunks.append(text_chunk)
            
            response = "".join(chunks)
            
            # 모델별 로그
            if self.model_id == "venice-uncensored":
                print(f"🔞 Venice Uncensored 응답 생성 완료: {len(response)} 문자")
            elif self.model_id == "mistral-31-24b":
                print(f"🎯 Venice Medium (Mistral) 응답 생성 완료: {len(response)} 문자")
            
            return response

        except Exception as e:
            print(f"응답 생성 중 오류: {e}")
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
