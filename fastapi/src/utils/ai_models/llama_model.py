'''
파일은 LlamaChatModel 클래스를 정의하고, 이 클래스는 Llama 8B 모델을 사용하여 대화를 생성하는 데 필요한 모든 기능을 제공합니다.
'''

import os
from threading import Thread

import torch
import transformers
from typing import Generator
from accelerate import Accelerator
from dotenv import load_dotenv
from torch.cuda.amp import GradScaler
from transformers import BitsAndBytesConfig, TextIteratorStreamer

class LlamaChatModel:
    """
    [<img src="https://cdn-avatars.huggingface.co/v1/production/uploads/646cf8084eefb026fb8fd8bc/oCTqufkdTkjyGodsx1vo1.png" width="100" height="auto">](https://huggingface.co/meta-llama/Llama-3.1-8B)
    
    Llama 모델을 로드하고 입력 프롬프트로부터 응답 텍스트를 생성하는 클래스입니다.
    
    모델 정보:
    - 모델명: Llama-3.1-8B-Instruct
    - 유형: 표준 Hugging Face Transformers 모델
    - 제작자: Meta (구 Facebook)
    - 소스: [Hugging Face 모델 허브](https://huggingface.co/meta-llama/Llama-3.1-8B)
    """
    def __init__(self):
        '''
        [<img src="https://cdn-avatars.huggingface.co/v1/production/uploads/646cf8084eefb026fb8fd8bc/oCTqufkdTkjyGodsx1vo1.png" width="100" height="auto">](https://huggingface.co/meta-llama/Llama-3.1-8B)

        
        LlamaChatModel 클래스 초기화 메소드
        '''
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        dotenv_path = os.path.join(parent_dir, '.env')
        load_dotenv(dotenv_path)
        self.cache_dir = "./fastapi/ai_model"
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        self.device = torch.device("cuda:0")  # 명확히 cuda:0로 지정

        self.model_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "device_map": {"": self.device},
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                double_quant=True,  # 추가 양자화
                compute_dtype=torch.float16
            )
        }

        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
        self.accelerator = Accelerator(mixed_precision="fp16", device_placement=False)
        self.scaler = GradScaler()

        print("토크나이저 로드 중...")
        self.tokenizer = self.load_tokenizer()
        print("모델 로드 중...")
        self.model = self.load_model()
        print("모델과 토크나이저 로드 완료!")
        
        self.model.gradient_checkpointing_enable()
        self.conversation_history = []

    def load_tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        """
        Llama 모델용 토크나이저를 로드하고 설정합니다.
        
        Returns:
            PreTrainedTokenizerBase: 설정된 토크나이저 인스턴스
            
        Raises:
            OSError: 토크나이저 로드 실패 시
            ValueError: 토큰 설정 실패 시
        """
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token
        )
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def load_model(self) -> transformers.PreTrainedModel:
        """
        Llama 모델을 로드하고 4비트 양자화를 적용합니다.
        
        Returns:
            PreTrainedModel: 양자화된 Llama 모델 인스턴스
            
        Raises:
            OSError: 모델 로드 실패 시
            RuntimeError: CUDA 메모리 부족 시
        """
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            **self.model_kwargs
        )
        return model

    def generate_response_stream(self, input_text: str) -> Generator[str, None, None]:
        """
        입력 텍스트에 대한 응답을 스트리밍 방식으로 생성합니다.
        
        Args:
            input_text (str): 모델에 입력할 프롬프트 텍스트

        Returns:
            Generator[str, None, None]: 생성된 텍스트 조각들을 반환하는 제너레이터
        """
        input_ids = self.tokenizer.encode(
            text=input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

        generation_kwargs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "min_new_tokens": 1,
            "max_new_tokens": 512,
            "do_sample": True,   # 샘플링 활성화
            "top_p": 0.9,        # Top-p Sampling 활성화 (0.9로 설정)
            "top_k": 0,          # Top-k 비활성화 (Top-p와 함께 사용하지 않음)
            "temperature": 0.7,  # 다양성을 위한 온도 조정
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
            "repetition_penalty": 1.2,  # 반복 방지 패널티
            "num_return_sequences": 1,  # 한 번에 하나의 시퀀스 생성
            "streamer": streamer        # 스트리밍 활성화
        }

        # Thread를 사용하여 생성 작업 비동기화
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Streamer를 통해 생성된 텍스트 반환
        for new_text in streamer:
            yield new_text
