'''
파일은 LlamaCharacterModel, CharacterPrompt 클래스를 정의하고 llama_cpp_cuda를 사용하여,
DarkIdol-Llama-3.1-8B.gguf 모델을 사용하여 대화를 생성하는 데 필요한 모든 기능을 제공합니다.
'''
from typing import Optional, Generator
from llama_cpp_cuda import (
    Llama,           # 기본 LLM 모델
    LlamaCache,      # 캐시 관리
    LlamaGrammar,    # 문법 제어
    LogitsProcessor,  # 로짓 처리
)
import json
from queue import Queue
from threading import Thread
import os

from .shared.shared_configs import CharacterPrompt, LlamaGenerationConfig, BaseConfig

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

def build_llama3_prompt(character_info: CharacterPrompt) -> str:
    """
    캐릭터 정보와 대화 기록을 포함한 Llama3 프롬프트 형식 생성

    Args:
        character (CharacterPrompt): 캐릭터 정보

    Returns:
        str: Lumimaid GGUF 형식의 프롬프트 문자열
    """
    system_prompt = (
        f"[세계관 설정]\n"
        f"- 배경: {character_info.context}\n"
        f"- 첫 대사: {character_info.greeting}\n\n"

        f"[역할 규칙]\n"
        f"- 모든 답변은 '{character_info.name}'의 말투와 인격으로 말하십시오.\n"
        f"- OOC(Out Of Character)는 절대 금지입니다.\n"
        f"- 설정을 벗어나거나 현실적 설명(예: '나는 AI야')을 하지 마십시오.\n"
        f"- 대사는 큰따옴표로 표기하고, 행동이나 감정은 *괄호*로 표현하십시오.\n"
        f"- 사용자 입력에 자연스럽게 반응하며, 대화가 이어지도록 무분별한 질문은 배제한체 대화를 유도한다.\n"
        f"- 풍부한 상황 설명을 통해 1000words 유지하십시오.\n"
    )
    
    # 기본 프롬프트 시작
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>"
    )

    # 이전 대화 기록 추가
    if character_info.chat_list and len(character_info.chat_list) > 0:
        for chat in character_info.chat_list:
            if "dialogue" in chat:
                dialogue = chat["dialogue"]
                # 중복 방지: 대화 기록에서 <|begin_of_text|> 제거
                dialogue = dialogue.replace("<|begin_of_text|>", "")
                prompt += dialogue
    
    # 현재 사용자 입력 추가
    prompt +=  (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{character_info.user_input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt

class LlamaCharacterModel:
    """
    [<img src = "https://lh7-rt.googleusercontent.com/docsz/AD_4nXeiuCm7c8lEwEJuRey9kiVZsRn2W-b4pWlu3-X534V3YmVuVc2ZL-NXg2RkzSOOS2JXGHutDuyyNAUtdJI65jGTo8jT9Y99tMi4H4MqL44Uc5QKG77B0d6-JfIkZHFaUA71-RtjyYZWVIhqsNZcx8-OMaA?key=xt3VSDoCbmTY7o-cwwOFwQ" width = "290" height = "auto">](https://huggingface.co/Lewdiculous/DarkIdol-Llama-3.1-8B-v0.1-OAS-GGUF-IQ-Imatrix)
    
    GGUF 포맷으로 경량화된 DarkIdol-Llama-3.1-8B 모델을 로드하고, 주어진 입력 프롬프트에 대한 응답을 생성하는 클래스입니다.
    
    모델 정보: 
    - 모델명: DarkIdol-Llama-3.1-8B
    - 유형: GGUF 포맷 (압축, 경량화)
    - 제작자: Lewdiculous
    - 소스: [Hugging Face 모델 허브](https://huggingface.co/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF)
    """
    def __init__(self, main_gpu=1) -> None:  # 기본값을 1로 수정
        """
        LlamaCharacterModel 클레스 초기화 메소드
        """
        self.model_id = "QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q6_K"
        self.model_path = "/app/fastapi/ai_model/QuantFactory/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q6_K.gguf"
        self.file_path = '/app/prompt/config-Llama.json'
        self.loading_text = f"{BLUE}LOADING{RESET}:    {self.model_id} 로드 중..."
        self.gpu_layers: int = 40
        self.character_info: Optional[CharacterPrompt] = None
        self.config: Optional[LlamaGenerationConfig] = None
        self.main_gpu = main_gpu

        # 모델 파일 존재 확인 추가
        if not os.path.exists(self.model_path):
            print(f"{RED}ERROR{RESET}:    모델 파일을 찾을 수 없습니다: {self.model_path}")
            print(f"{RED}ERROR{RESET}:    현재 디렉토리: {os.getcwd()}")
            print(f"{RED}ERROR{RESET}:    볼륨 마운트를 확인해주세요.")
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")

        # GPU 가시성 강제 설정
        os.environ['CUDA_VISIBLE_DEVICES'] = str(main_gpu)
        
        print("\n"+ f"{BLUE}LOADING{RESET}:  " + "="*len(self.loading_text))
        print(f"{BLUE}LOADING{RESET}:    {__class__.__name__} 모델 초기화 시작...")
        print(f"{BLUE}LOADING{RESET}:    GPU {main_gpu} 전용으로 설정됨")

        # JSON 파일 읽기
        with open(self.file_path, 'r', encoding = 'utf-8') as file:
            self.data: BaseConfig = json.load(file)

        # 진행 상태 표시
        print(f"{BLUE}LOADING{RESET}:    {__class__.__name__} 모델 초기화 중...")
        self.model: Llama = self._load_model()
        print(f"{BLUE}LOADING{RESET}:    모델 로드 완료!")
        print(f"{BLUE}LOADING{RESET}:  " + "="*len(self.loading_text) + "\n")
        
        self.response_queue: Queue = Queue()

    def _load_model(self) -> Llama:
        """
        GGUF 포맷의 Llama 모델을 로드하고 GPU 가속을 설정합니다.
        """
        print(f"{self.loading_text}")
        try:
            # 경고 메시지 필터링
            import warnings
            import sys
            from contextlib import contextmanager
            
            warnings.filterwarnings("ignore")
            
            # GPU 제한 설정 강화
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.main_gpu)
            
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
                    main_gpu = 0,  # 항상 0 (CUDA_VISIBLE_DEVICES로 제한됨)
                    rope_scaling_type = 2,
                    rope_freq_scale = 2.0,
                    n_ctx = 16384,
                    n_batch = 512,
                    verbose = False,
                    offload_kqv = True,          # KQV 캐시를 GPU에 오프로드
                    use_mmap = False,            # 메모리 매핑 비활성화 (Office와 동일)
                    use_mlock = True,            # 메모리 잠금 활성화
                    n_threads = 6,               # 스레드 수 제한
                    # GPU 사용 제한 추가
                    tensor_split = None,         # 단일 GPU 사용
                )
            return model
        except Exception as e:
            print(f"❌ 모델 로드 중 오류 발생: {e}")
            raise e

    def _stream_completion(self, config: LlamaGenerationConfig) -> None:
        """
        별도 스레드에서 실행되어 응답을 큐에 넣는 메서드

        Args:
            config (LlamaGenerationConfig): 생성 파라미터 객체
        """
        try:
            stream = self.model.create_completion(
                prompt = config.prompt,
                max_tokens = config.max_tokens,
                temperature = config.temperature,
                top_p = config.top_p,
                stop = config.stop or ["<|eot_id|>"],
                stream = True
            )
            
            for output in stream:
                if 'choices' in output and len(output['choices']) > 0:
                    text = output['choices'][0].get('text', '')
                    if text:
                        self.response_queue.put(text)
            self.response_queue.put(None) # 스트림 종료를 알리는 None 추가
            
        except Exception as e:
            print(f"스트리밍 중 오류 발생: {e}")
            self.response_queue.put(None)

    def create_streaming_completion(self, config: LlamaGenerationConfig) -> Generator[str, None, None]:
        """
        스트리밍 방식으로 텍스트 응답 생성

        Args:
            config (LlamaGenerationConfig): 생성 파라미터 객체

        Returns:
            Generator[str, None, None]: 생성된 텍스트 조각들을 반환하는 제너레이터
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

    def create_completion(self, config: LlamaGenerationConfig) -> str:
        """
        주어진 프롬프트로부터 텍스트 응답 생성

        Args:
            config (LlamaGenerationConfig): 생성 파라미터 객체

        Returns:
            str: 생성된 텍스트 응답
        """
        try:
            output = self.model.create_completion(
                prompt = config.prompt,
                max_tokens = config.max_tokens,
                temperature = config.temperature,
                top_p = config.top_p,
                stop = config.stop or ["<|eot_id|>"]
            )
            return output['choices'][0]['text'].strip()
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {e}")
            return ""

    def generate_response(self, input_text: str, character_settings: dict) -> str:
        """
        API 호환을 위한 스트리밍 응답 생성 메서드

        Args:
            input_text (str): 사용자 입력 텍스트
            character_settings (dict): 캐릭터 설정 딕셔너리

        Returns:
            str: 생성된 텍스트을 반환하는 제너레이터
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
            
            self.character_info = CharacterPrompt(
                name = character_settings.get("character_name", self.data.get("character_name")),
                greeting = character_settings.get("greeting", self.data.get("greeting")),
                context = character_settings.get("character_setting", self.data.get("character_setting")),
                user_input = input_text,
                chat_list = normalized_chat_list,
            )

            prompt = build_llama3_prompt(character_info = self.character_info)

            self.config = LlamaGenerationConfig(
                prompt = prompt,
                max_tokens = 8224,
                temperature = 1.3,
                top_p = 0.9,
                min_p = 0.1,
                tfs_z = 1.1,
                repeat_penalty = 1.08,
                frequency_penalty = 0.1,
                presence_penalty = 0.1,
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
