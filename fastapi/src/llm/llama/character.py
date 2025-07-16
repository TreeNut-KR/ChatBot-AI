'''
파일은 LlamaCharacterModel, character_config.CharacterPrompt 클래스를 정의하고 llama_cpp_cuda를 사용하여,
DarkIdol-Llama-3.1-8B.gguf 모델을 사용하여 대화를 생성하는 데 필요한 모든 기능을 제공합니다.
'''
from typing import Optional, Generator
from llama_cpp_cuda import (
    Llama,           # 기본 LLM 모델
    LlamaCache,      # 캐시 관리
    LlamaGrammar,    # 문법 제어
    LogitsProcessor,  # 로짓 처리
)
import warnings
import sys
import json
import time  # time 모듈 추가
from queue import Queue
from threading import Thread
import os

from domain import character_config, base_config

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

def build_llama3_prompt(character_info: character_config.CharacterPrompt) -> str:
    """
    캐릭터 정보와 대화 기록을 포함한 Llama3 프롬프트 형식 생성

    Args:
        character (character_config.CharacterPrompt): 캐릭터 정보

    Returns:
        str: Lumimaid GGUF 형식의 프롬프트 문자열
    """
    # 사용자 이름이 있는 경우 프롬프트에 포함
    user_info = f"- 사용자 이름: {character_info.user_name}\n" if character_info.user_name else ""
    
    system_prompt = (
        f"[대화 설정]\n"
        f"- 케릭터 또는 대화 출력 설정: {character_info.context}\n"
        f"- ![]() 형태의 이미지 출력 설정이 있다면, 무조건 대화에서 []의 상황에 알맞게 출력.\n"
        f"[세계관 설정]\n"
        f"- 시작 배경: {character_info.greeting}\n\n"
        
        f"[사용자 정보]\n"
        f"{user_info}"

        f"[역할 규칙]\n"
        f"- 모든 답변은 '{character_info.name}'의 말투와 인격으로 말하십시오.\n"
        f"- 사용자의 이름이 주어진 경우, 대화에서 자연스럽게 사용자의 이름을 불러주세요.\n"
        f"- OOC(Out Of Character)는 현실적 설명요구하지 않는 선에서만 허용입니다.\n"
        f"- 설정을 벗어나거나 현실적 설명(예: '나는 AI야')을 하지 마십시오.\n"
        f"- 대사는 큰따옴표로 표기하고, 행동이나 감정은 *괄호*로 표현하십시오.\n"
        f"- 사용자 입력에 자연스럽게 반응하며, 대화가 이어지도록 무분별한 질문은 배제한체 대화를 유도한다.\n"
        f"- 풍부한 상황 설명을 포함\n"
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
    - 모델명: Meta-Llama-3.1-8B-Claude
    - 유형: GGUF 포맷 (압축, 경량화)
    - 제작자: aashish1904
    - 소스: [Hugging Face 모델 허브](https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Claude-GGUF/blob/main/Meta-Llama-3.1-8B-Claude.Q4_1.gguf)
    """
    def __init__(self) -> None:  # 기본값을 1로 수정
        """
        LlamaCharacterModel 클레스 초기화 메소드
        """
        self.model_id = "Meta-Llama-3.1-8B-Claude.Q4_0"
        self.model_path = "/app/fastapi/ai_model/QuantFactory/Meta-Llama-3.1-8B-Claude.Q4_0.gguf"
        self.file_path = '/app/prompt/config-Llama.json'
        self.loading_text = f"{BLUE}LOADING{RESET}:    {self.model_id} 로드 중..."
        self.character_info: Optional[character_config.CharacterPrompt] = None
        self.config: Optional[character_config.LlamaGenerationConfig] = None
        
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
            from contextlib import contextmanager
            
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
                    n_gpu_layers = 50,                  # 모든 레이어를 GPU에 로드
                    main_gpu = 0,                       # 0번 GPU 사용
                    rope_scaling_type = 2,              # RoPE 스케일링 방식 (2 = linear) 
                    rope_freq_scale = 2.0,              # RoPE 주파수 스케일 → 긴 문맥 지원   
                    n_ctx = 8191,                       # 최대 context length (4096 토큰까지)
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

    def _stream_completion(self, config: character_config.LlamaGenerationConfig) -> None:
        """
        별도 스레드에서 실행되어 응답을 큐에 넣는 메서드 (최적화)

        Args:
            config (character_config.LlamaGenerationConfig): 생성 파라미터 객체
        """
        try:
            # mirostat 파라미터 제거하고 안정적인 설정 사용
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
                top_k = config.top_k,
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

    def create_streaming_completion(self, config: character_config.LlamaGenerationConfig) -> Generator[str, None, None]:
        """
        스트리밍 방식으로 텍스트 응답 생성

        Args:
            config (character_config.LlamaGenerationConfig): 생성 파라미터 객체

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

    def create_completion(self, config: character_config.LlamaGenerationConfig) -> str:
        """
        주어진 프롬프트로부터 텍스트 응답 생성

        Args:
            config (character_config.LlamaGenerationConfig): 생성 파라미터 객체

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

    def generate_response(self, input_text: str, user_name: str, character_settings: dict) -> str:
        """
        API 호환을 위한 최적화된 응답 생성 메서드

        Args:
            input_text (str): 사용자 입력 텍스트
            user_name (str): 사용자 이름
            character_settings (dict): 캐릭터 설정 딕셔너리

        Returns:
            str: 생성된 텍스트
        """
        start_time = time.time()
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
                context = character_settings.get("context", self.data.get("character_setting")),
                user_name = user_name,
                user_input = input_text,
                chat_list = normalized_chat_list,
            )

            prompt = build_llama3_prompt(character_info = self.character_info)
            print(f"    프롬프트 길이: {len(prompt)} 문자")

            # 균형 잡힌 설정으로 수정
            self.config = character_config.LlamaGenerationConfig(
                prompt = prompt,
                max_tokens = 256,                   # 적절한 토큰 수
                temperature = 0.8,                  # 온도 적절히 조정
                top_p = 0.9,                        # top_p 복원
                min_p = 0.1,                        # min_p 복원
                typical_p = 1.0,                    # typical_p 추가
                tfs_z = 1.1,                        # tfs_z 복원
                repeat_penalty = 1.05,              # repeat_penalty 복원
                frequency_penalty = 0,              # frequency_penalty 복원
                presence_penalty = 0,               # presence_penalty 복원
                seed = None,                        # 시드 없음 (다양성 확보)
                top_k = 40,                         # top_k 복원
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
                max_tokens = 512,
                temperature = 0.8,
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
