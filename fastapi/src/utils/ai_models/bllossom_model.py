'''
파일은 BllossomChatModel, OfficePrompt 클래스를 정의하고 llama_cpp_cuda를 사용하여,
Llama-3-Bllossom-8B.gguf 모델을 사용하여 대화를 생성하는 데 필요한 모든 기능을 제공합니다.
'''
from dataclasses import dataclass
from typing import TypedDict, Optional, Generator, List, Dict
from llama_cpp_cuda import (
    Llama,           # 기본 LLM 모델
    LlamaCache,      # 캐시 관리
    LlamaGrammar,    # 문법 제어
    LogitsProcessor  # 로짓 처리
)
import os
import sys
import json
import warnings
from queue import Queue
from threading import Thread
from contextlib import contextmanager
from transformers import AutoTokenizer
from datetime import datetime

from .shared.shared_configs import OfficePrompt, LlamaGenerationConfig, BaseConfig

BLUE = "\033[34m"
RESET = "\033[0m"

def build_llama3_messages(character_info: OfficePrompt) -> list:
    """
    캐릭터 정보와 대화 기록을 포함한 Llama3 messages 형식 생성

    Args:
        character_info (OfficePrompt): 캐릭터 정보

    Returns:
        list: Bllossom GGUF 형식의 messages 리스트
    """
    system_prompt = (
        f"system Name: {character_info.name}\n"
        f"system Context: {character_info.context}\n"
        f"참고 정보(아래 정보는 참고만 하세요. 사용자의 질문과 직접 관련이 없으면 답변에 포함하지 마세요):\n{character_info.reference_data}"
    )
    
    # 메시지 구성
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # 이전 대화 기록 추가
    if character_info.chat_list and len(character_info.chat_list) > 0:
        for chat in character_info.chat_list:
            # input_data와 output_data 직접 사용
            user_message = chat.get("input_data", "")
            assistant_message = chat.get("output_data", "")
            
            if user_message:
                messages.append({"role": "user", "content": user_message})
            if assistant_message:
                messages.append({"role": "assistant", "content": assistant_message})
    
    # 현재 사용자 입력 추가
    messages.append({"role": "user", "content": character_info.user_input})
    return messages

class BllossomChatModel:
    """
    [<img src = "https://cdn-avatars.huggingface.co/v1/production/uploads/63be962d4a2beec6555f46a3/CuJyXw6wwRj7oz2HxKoVq.png" width = "100" height = "auto">](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M)
    
    GGUF 포맷으로 경량화된 Llama-3-Bllossom-8B 모델을 로드하고, 주어진 입력 프롬프트에 대한 응답을 생성하는 클래스입니다.
    
    모델 정보:
    - 모델명: llama-3-Korean-Bllossom-8B
    - 유형: GGUF 포맷 (압축, 경량화)
    - 제작자: MLP-KTLim
    - 소스: [Hugging Face 모델 허브](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M)
    """
    def __init__(self) -> None:
        """
        [<img src = "https://cdn-avatars.huggingface.co/v1/production/uploads/63be962d4a2beec6555f46a3/CuJyXw6wwRj7oz2HxKoVq.png" width = "100" height = "auto">](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M)
    
        BllossomChatModel 클레스 초기화 메소드
        """
        self.model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M'
        self.model_path = "fastapi/ai_model/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf"
        self.file_path = './prompt/config-Llama.json'
        self.loading_text = f"{BLUE}LOADING{RESET}:    {self.model_id} 로드 중..."
        self.gpu_layers: int = 70
        self.character_info: Optional[OfficePrompt] = None
        self.config: Optional[LlamaGenerationConfig] = None

        print("\n"+ f"{BLUE}LOADING{RESET}:  " + "="*len(self.loading_text))
        print(f"{BLUE}LOADING{RESET}:    {__class__.__name__} 모델 초기화 시작...")

        # JSON 파일 읽기
        with open(self.file_path, 'r', encoding = 'utf-8') as file:
            self.data: BaseConfig = json.load(file)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # 진행 상태 표시
        print(f"{BLUE}LOADING{RESET}:    {__class__.__name__} 모델 초기화 중...")
        self.model: Llama = self._load_model()
        print(f"{BLUE}LOADING{RESET}:    모델 로드 완료!")
        print(f"{BLUE}LOADING{RESET}:  " + "="*len(self.loading_text) + "\n")
        
        self.response_queue: Queue = Queue()

    def _load_model(self) -> Llama:
        """
        GGUF 포맷의 Llama 모델을 로드하고 GPU 가속을 설정합니다.
        
        Args:
            gpu_layers (int): GPU에 오프로드할 레이어 수 (기본값: 50)
            
        Returns:
            Llama: 초기화된 Llama 모델 인스턴스
            
        Raises:
            RuntimeError: GPU 메모리 부족 또는 CUDA 초기화 실패 시
            OSError: 모델 파일을 찾을 수 없거나 손상된 경우
        """
        print(f"{self.loading_text}")
        try:
            # 경고 메시지 필터링
            warnings.filterwarnings("ignore")
            
            @contextmanager
            def suppress_stdout():
                # 표준 출력 리다이렉션
                with open(os.devnull, "w") as devnull:
                    old_stdout = sys.stdout
                    sys.stdout = devnull
                    try:
                        yield
                    finally:
                        sys.stdout = old_stdout

            # 모델 로드 시 로그 출력 억제
            with suppress_stdout():
                model = Llama(
                    model_path = self.model_path,
                    n_gpu_layers = self.gpu_layers,
                    main_gpu = 1,
                    n_ctx = 8191,
                    n_batch = 512,
                    verbose = False,
                    offload_kqv = True,
                    use_mmap = False,
                    use_mlock = True,
                    n_threads = 8
                )
            return model
        except Exception as e:
            print(f"❌ 모델 로드 중 오류 발생")
            
    def _stream_completion(self, config: LlamaGenerationConfig) -> None:
        """
        텍스트 생성을 위한 내부 스트리밍 메서드입니다.

        Args:
            config (LlamaGenerationConfig): 생성 파라미터 객체
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stream = self.model(
                    config.prompt,
                    stream = True,
                    echo = False,
                    max_tokens = config.max_tokens,
                    temperature = config.temperature,
                    top_p = config.top_p,
                    stop = config.stop,
                )
                for output in stream:
                    if 'choices' in output and len(output['choices']) > 0:
                        text = output['choices'][0].get('text', '')
                        if text:
                            self.response_queue.put(text)
                self.response_queue.put(None)
        except Exception as e:
            print(f"스트리밍 중 오류 발생: {e}")
            self.response_queue.put(None)

    def create_streaming_completion(self, config: LlamaGenerationConfig) -> Generator[str, None, None]:
        """
        스트리밍 방식으로 텍스트 응답을 생성하는 메서드입니다.

        Args:
            config (LlamaGenerationConfig): 생성 파라미터 객체

        Returns:
            Generator[str, None, None]: 생성된 텍스트 조각들의 제너레이터
        """
        thread = Thread(
            target = self._stream_completion,
            args = (config,)
        )
        thread.start()

        while True:
            text = self.response_queue.get()
            if text is None:
                break
            yield text

    def generate_response(self, input_text: str, search_text: str, chat_list: List[Dict]) -> str:
        """
        API 호환을 위한 스트리밍 응답 생성 메서드

        Args:
            input_text (str): 사용자 입력 텍스트
            search_text (str): 검색 텍스트
            chat_list (List[Dict]): 대화 기록

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

            messages = build_llama3_messages(character_info = self.character_info)

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True
            )

            self.config = LlamaGenerationConfig(
                prompt = prompt,
                max_tokens = 2048,
                temperature = 0.5,
                top_p = 0.80,
                stop = ["<|eot_id|>"]
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
