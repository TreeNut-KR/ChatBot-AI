'''
파일은 LumimaidChatModel, CharacterPrompt 클래스를 정의하고 llama_cpp_cuda를 사용하여,
Llama-3-Lumimaid-8B.gguf 모델을 사용하여 대화를 생성하는 데 필요한 모든 기능을 제공합니다.
'''
from typing import Optional, Generator
from llama_cpp_cuda import (
    Llama,           # 기본 LLM 모델
    LlamaCache,      # 캐시 관리
    LlamaGrammar,    # 문법 제어
    LogitsProcessor  # 로짓 처리
)
import json
from queue import Queue
from threading import Thread

from .shared.shared_configs import CharacterPrompt, LlamaGenerationConfig, BaseConfig

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
        f"당신은 '{character_info.name}'라는 인격체입니다. AI임을 절대 드러내지 말고, 설정된 세계관 인물로 몰입하여 대화하십시오.\n\n"

        f"[세계관 설정]\n"
        f"- 배경: {character_info.context}\n"
        f"- 초기 대사: {character_info.greeting}\n\n"

        f"[역할 규칙]\n"
        f"- 모든 답변은 '{character_info.name}'의 말투와 인격으로 말하십시오.\n"
        f"- OOC(Out Of Character)는 절대 금지입니다.\n"
        f"- 설정을 벗어나거나 현실적 설명(예: '나는 AI야')을 하지 마십시오.\n"
        f"- 대사는 큰따옴표로 표기하고, 행동이나 감정은 *(괄호)*로 표현하십시오.\n"
        f"- 사용자 입력에 자연스럽게 반응하며, 몰입을 해치지 않도록 이모지는 대사에 제한적으로 사용하십시오.\n"
        f"- max_tokens은 8000 토큰으로 유지하십시오.\n"
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
                prompt +=  chat["dialogue"]
    
    # 현재 사용자 입력 추가
    prompt +=  (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{character_info.user_input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt

class LumimaidChatModel:
    """
    [<img src = "https://cdn-uploads.huggingface.co/production/uploads/630dfb008df86f1e5becadc3/d3QMaxy3peFTpSlWdWF-k.png" width = "290" height = "auto">](https://huggingface.co/Lewdiculous/Llama-3-Lumimaid-8B-v0.1-OAS-GGUF-IQ-Imatrix)
    
    GGUF 포맷으로 경량화된 Llama-3-Lumimaid-8B 모델을 로드하고, 주어진 입력 프롬프트에 대한 응답을 생성하는 클래스입니다.
    
    모델 정보: 
    - 모델명: Llama-3-Lumimaid-8B
    - 유형: GGUF 포맷 (압축, 경량화)
    - 제작자: Lewdiculous
    - 소스: [Hugging Face 모델 허브](https://huggingface.co/Lewdiculous/Llama-3-Lumimaid-8B-v0.1-OAS-GGUF-IQ-Imatrix)
    """
    def __init__(self) -> None:
        """
        [<img src = "https://cdn-uploads.huggingface.co/production/uploads/630dfb008df86f1e5becadc3/d3QMaxy3peFTpSlWdWF-k.png" width = "290" height = "auto">](https://huggingface.co/Lewdiculous/Llama-3-Lumimaid-8B-v0.1-OAS-GGUF-IQ-Imatrix)
    
        LumimaidChatModel 클레스 초기화 메소드
        """
        self.model_id = "v2-Llama-3-Lumimaid-8B-v0.1-OAS-Q5_K_S-imat"
        self.model_path = "fastapi/ai_model/v2-Llama-3-Lumimaid-8B-v0.1-OAS-Q5_K_S-imat.gguf"
        self.file_path = './prompt/config-Llama.json'
        self.loading_text = f"{BLUE}LOADING{RESET}:    {self.model_id} 로드 중..."
        self.gpu_layers: int = 70
        self.character_info: Optional[CharacterPrompt] = None
        self.config: Optional[LlamaGenerationConfig] = None

        print("\n"+ f"{BLUE}LOADING{RESET}:  " + "="*len(self.loading_text))
        print(f"{BLUE}LOADING{RESET}:    {__class__.__name__} 모델 초기화 시작...")

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
            model = Llama(
                model_path = self.model_path,
                n_gpu_layers = self.gpu_layers,
                main_gpu = 0,
                n_ctx = 8191,
                n_batch = 512,
                verbose = False,
                offload_kqv = True,          # KQV 캐시를 GPU에 오프로드
                use_mmap = False,            # 메모리 매핑 비활성화
                use_mlock = True,            # 메모리 잠금 활성화
                n_threads = 8,               # 스레드 수 제한
            )
            return model
        except Exception as e:
            print(f"❌ 모델 로드 중 오류 발생: {e}")
            raise

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
                max_tokens = 8191,
                temperature = 0.8,
                top_p = 0.95,
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
