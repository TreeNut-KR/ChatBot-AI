'''
파일은 LlamaOfficeModel, office_config.OfficePrompt 클래스를 정의하고 llama_cpp_cuda를 사용하여,
Meta-Llama-3.1-8B-Claude.Q4_0.gguf 모델을 사용하여 대화를 생성하는 데 필요한 모든 기능을 제공합니다.
'''
from typing import Optional, Generator, List, Dict
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
import time
from queue import Queue
from threading import Thread
from contextlib import contextmanager
from datetime import datetime

from domain import office_config, base_config

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

def build_llama3_prompt(character_info: office_config.OfficePrompt) -> str:
    """
    캐릭터 정보와 대화 기록을 기반으로 Llama3 GGUF 형식의 프롬프트 문자열을 생성합니다.

    Args:
        character_info (office_config.OfficePrompt): 캐릭터 기본 정보 및 대화 맥락 포함 객체

    Returns:
        str: Llama3 GGUF 포맷용 프롬프트 문자열
    """
    system_prompt = (
        f"당신은 AI 어시스턴트 {character_info.name}입니다.\n"
        f"당신의 역할: {character_info.context}\n\n"
        f"참고 정보 (사용자의 질문과 관련 있을 경우에만 활용하세요):\n"
        f"{character_info.reference_data}\n\n"
        f"지시 사항:\n"
        f"- 한국어로 답변하세요\n"
        f"- 친절하고 유익한 답변을 제공하세요\n"
        f"- 질문과 관련 없는 참고 정보는 언급하지 마세요\n"
        f"- 간결하면서도 핵심적인 정보를 포함하도록 하세요\n"
    )

    # 시스템 프롬프트 시작
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>"
    )

    # 대화 기록 추가
    if character_info.chat_list:
        for chat in character_info.chat_list:
            user_input = chat.get("input_data", "")
            assistant_output = chat.get("output_data", "")

            if user_input:
                prompt += (
                    "<|start_header_id|>user<|end_header_id|>\n"
                    f"{user_input}<|eot_id|>"
                )
            if assistant_output:
                prompt += (
                    "<|start_header_id|>assistant<|end_header_id|>\n"
                    f"{assistant_output}<|eot_id|>"
                )

    # 최신 사용자 입력 추가
    prompt += (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{character_info.user_input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    return prompt

class LlamaOfficeModel:
    """
    GGUF 포맷으로 경량화된 Meta-Llama-3.1-8B-Claude 모델을 로드하고, 주어진 입력 프롬프트에 대한 응답을 생성하는 클래스입니다.
    
    모델 정보: 
    - 모델명: Meta-Llama-3.1-8B-Claude
    - 유형: GGUF 포맷 (압축, 경량화)
    - 제작자: QuantFactory
    - 소스: Hugging Face 모델 허브
    """
    def __init__(self) -> None:
        """
        LlamaOfficeModel 클래스 초기화 메소드
        """
        self.model_id = 'Meta-Llama-3.1-8B-Claude.Q4_1'
        self.model_path = "/app/fastapi/ai_model/QuantFactory/Meta-Llama-3.1-8B-Claude.Q4_1.gguf"
        self.file_path = '/app/prompt/config-Llama.json'
        self.loading_text = f"{BLUE}LOADING{RESET}:    {self.model_id} 로드 중..."
        self.character_info: Optional[office_config.OfficePrompt] = None
        self.config: Optional[office_config.LlamaGenerationConfig] = None
        
        print("\n"+ f"{BLUE}LOADING{RESET}:  " + "="*len(self.loading_text))
        print(f"{BLUE}LOADING{RESET}:    {__class__.__name__} 모델 초기화 시작...")

        # JSON 파일 읽기
        with open(self.file_path, 'r', encoding = 'utf-8') as file:
            self.data: base_config.BaseConfig = json.load(file)

        # 진행 상태 표시
        print(f"{BLUE}LOADING{RESET}:    {__class__.__name__} 모델 초기화 중...")
        self.model: Llama = self._load_model()
        print(f"{BLUE}LOADING{RESET}:    모델 로드 완료!")
        print(f"{BLUE}LOADING{RESET}:  " + "="*len(self.loading_text) + "\n")
        
        self.response_queue: Queue = Queue()

    def _load_model(self) -> Llama:
        """
        GGUF 포맷의 Llama 모델을 로드하고 GPU 가속을 최대화합니다.
        """
        print(f"{self.loading_text}")
        try:
            warnings.filterwarnings("ignore")
            
            @contextmanager
            def suppress_stdout():
                with open(os.devnull, "w") as devnull:
                    old_stdout = sys.stdout
                    sys.stdout = devnull
                    try:
                        yield
                    finally:
                        sys.stdout = old_stdout

            # GPU 사용량 극대화를 위한 설정
            with suppress_stdout():
                model = Llama(
                    model_path = self.model_path,       # GGUF 모델 파일 경로
                    n_gpu_layers = -1,                  # 모든 레이어를 GPU에 로드
                    main_gpu = 1,                       # 1번 GPU 사용 (office 서비스용)
                    rope_scaling_type = 2,              # RoPE 스케일링 방식 (2 = linear) 
                    rope_freq_scale = 2.0,              # RoPE 주파수 스케일 → 긴 문맥 지원   
                    n_ctx = 8191,                       # 최대 context length
                    n_batch = 2048,                     # 배치 크기 (VRAM 제한 고려한 중간 값)
                    verbose = False,                    # 디버깅 로그 비활성화  
                    offload_kqv = True,                 # K/Q/V 캐시를 CPU로 오프로드하여 VRAM 절약
                    use_mmap = False,                   # 메모리 매핑 비활성화 
                    use_mlock = True,                   # 메모리 잠금으로 메모리 페이지 스왑 방지
                    n_threads = 12,                     # CPU 스레드 수 (코어 12개 기준 적절한 값)
                    tensor_split = [1.0],               # 단일 GPU에서 모든 텐서 로딩
                    split_mode = 1,                     # 텐서 분할 방식 (1 = 균등 분할)
                    flash_attn = True,                  # FlashAttention 사용 (속도 향상)
                    cont_batching = True,               # 연속 배칭 활성화 (멀티 사용자 처리에 효율적)
                    numa = False,                       # NUMA 비활성화 (단일 GPU 시스템에서 불필요)
                    f16_kv = True,                      # 16bit KV 캐시 사용
                    logits_all = False,                 # 마지막 토큰만 logits 계산
                    embedding = False,                  # 임베딩 비활성화
                )
            return model
        except Exception as e:
            print(f"❌ 모델 로드 중 오류 발생: {e}")
            raise e

    def _stream_completion(self, config: office_config.LlamaGenerationConfig) -> None:
        """
        별도 스레드에서 실행되어 응답을 큐에 넣는 메서드 (최적화)

        Args:
            config (office_config.LlamaGenerationConfig): 생성 파라미터 객체
        """
        try:
            stream = self.model.create_completion(
                prompt = config.prompt,
                max_tokens = config.max_tokens,
                temperature = config.temperature,
                top_p = config.top_p,
                min_p = config.min_p,
                typical_p = config.typical_p,
                tfs_z = config.tfs_z,
                repeat_penalty = config.repeat_penalty,
                frequency_penalty = config.frequency_penalty,
                presence_penalty = config.presence_penalty,
                stop = config.stop or ["<|eot_id|>"],
                stream = True,
                seed = config.seed,
            )
            
            token_count = 0
            for output in stream:
                if 'choices' in output and len(output['choices']) > 0:
                    text = output['choices'][0].get('text', '')
                    if text:
                        self.response_queue.put(text)
                        token_count += 1
                        
            print(f"    생성된 토큰 수: {token_count}")
            self.response_queue.put(None)  # 스트림 종료 신호
            
        except Exception as e:
            print(f"스트리밍 중 오류 발생: {e}")
            self.response_queue.put(None)

    def create_streaming_completion(self, config: office_config.LlamaGenerationConfig) -> Generator[str, None, None]:
        """
        스트리밍 방식으로 텍스트 응답 생성

        Args:
            config (office_config.LlamaGenerationConfig): 생성 파라미터 객체

        Returns:
            Generator[str, None, None]: 생성된 텍스트 조각들을 반환하는 제너레이터
        """
        # 큐 초기화 (이전 응답이 남아있을 수 있음)
        while not self.response_queue.empty():
            self.response_queue.get()
            
        # 스트리밍 스레드 시작
        thread = Thread(
            target = self._stream_completion,
            args = (config,)
        )
        thread.start()

        # 응답 스트리밍
        token_count = 0
        while True:
            text = self.response_queue.get()
            if text is None:  # 스트림 종료
                break
            token_count += 1
            yield text
            
        # 스레드가 완료될 때까지 대기
        thread.join()
        print(f"    스트리밍 완료: {token_count}개 토큰 수신")

    def create_completion(self, config: office_config.LlamaGenerationConfig) -> str:
        """
        주어진 프롬프트로부터 텍스트 응답 생성

        Args:
            config (office_config.LlamaGenerationConfig): 생성 파라미터 객체

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

    def generate_response(self, input_text: str, search_text: str, chat_list: List[Dict]) -> str:
        """
        API 호환을 위한 최적화된 응답 생성 메서드

        Args:
            input_text (str): 사용자 입력 텍스트
            search_text (str): 검색 텍스트
            chat_list (List[Dict]): 대화 기록

        Returns:
            str: 생성된 텍스트
        """
        start_time = time.time()
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

            self.character_info = office_config.OfficePrompt(
                name = self.data.get("character_name", "AI Assistant"),
                context = self.data.get("character_setting", "Helpful AI assistant"),
                reference_data = reference_text,
                user_input = input_text,
                chat_list = normalized_chat_list,
            )

            prompt = build_llama3_prompt(character_info = self.character_info)
            print(f"    프롬프트 길이: {len(prompt)} 문자")

            # 균형 잡힌 설정으로 수정
            self.config = office_config.LlamaGenerationConfig(
                prompt = prompt,
                max_tokens = 1024,                  # 적절한 토큰 수
                temperature = 0.7,                  # 온도 적절히 조정
                top_p = 0.9,                        # top_p 복원
                min_p = 0.1,                        # min_p 복원
                typical_p = 1.0,                    # typical_p 추가
                tfs_z = 1.1,                        # tfs_z 복원
                repeat_penalty = 1.08,              # repeat_penalty 복원
                frequency_penalty = 0.1,            # frequency_penalty 복원
                presence_penalty = 0.1,             # presence_penalty 복원
                seed = None,                        # 시드 없음 (다양성 확보)
            )
            
            print(f"    텍스트 생성 시작...")
            chunks = []
            for text_chunk in self.create_streaming_completion(config = self.config):
                chunks.append(text_chunk)
            
            result = "".join(chunks)
            generation_time = time.time() - start_time
            
            print(f"    생성 완료: {len(result)} 문자, {generation_time:.2f}초")
            
            if not result or len(result.strip()) < 10:
                print(f"    경고: 응답이 너무 짧습니다. 백업 방식 시도...")
                # 백업: 스트리밍 없이 직접 생성
                return self._generate_fallback_response(prompt)
            
            return result

        except Exception as e:
            generation_time = time.time() - start_time
            print(f"응답 생성 중 오류 발생: {e} (소요 시간: {generation_time:.2f}초)")
            return f"오류: {str(e)}"

    def _generate_fallback_response(self, prompt: str) -> str:
        """
        스트리밍 실패 시 백업 응답 생성 메서드
        
        Args:
            prompt (str): 생성할 프롬프트
            
        Returns:
            str: 생성된 응답
        """
        try:
            print(f"    백업 방식으로 응답 생성 중...")
            output = self.model.create_completion(
                prompt = prompt,
                max_tokens = 1024,
                temperature = 0.7,
                top_p = 0.9,
                repeat_penalty = 1.08,
                stop = ["<|eot_id|>"],
                stream = False  # 스트리밍 비활성화
            )
            
            if 'choices' in output and len(output['choices']) > 0:
                result = output['choices'][0].get('text', '').strip()
                print(f"    백업 방식 성공: {len(result)} 문자")
                return result
            else:
                return "응답을 생성할 수 없습니다. 다시 시도해 주세요."
                
        except Exception as e:
            print(f"    백업 방식도 실패: {e}")
            return "응답 생성에 실패했습니다. 잠시 후 다시 시도해 주세요."

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
