import os
from threading import Thread

import torch
import transformers
from accelerate import Accelerator
from dotenv import load_dotenv
from torch.cuda.amp import GradScaler, autocast
from transformers import BitsAndBytesConfig, TextIteratorStreamer


class LlamaChatModel:
    def __init__(self):
        '''
        LlamaChatModel 클래스 초기화
        '''
        # 현재 파일의 경로를 기준으로 부모 디렉토리의 .env 파일 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        dotenv_path = os.path.join(parent_dir, '.env')
        load_dotenv(dotenv_path)
        self.cache_dir = "./fastapi/ai_model/"
        self.model_id = "meta-llama/Llama-3.2-1B"  # 원하는 모델 ID 설정
        self.model_kwargs = {
            "torch_dtype": torch.float16,  # float16으로 설정
            "trust_remote_code": True,
            "quantization_config": BitsAndBytesConfig(load_in_4bit=True)  # 양자화 적용
        }

        # Hugging Face Token 설정
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")

        # Accelerate 객체 초기화
        self.accelerator = Accelerator(mixed_precision="fp16")  # Mixed Precision 설정
        self.device_1050 = torch.device("cuda:1")  # GTX 1050 GPU에 할당

        print("토크나이저 로드 중...")
        self.tokenizer = self.load_tokenizer()
        print("모델 로드 중...")
        self.model, self.optimizer = self.load_model_with_accelerator()
        self.scaler = GradScaler()
        print("모델과 토크나이저 로드 완료!")

        # Gradient Checkpointing 활성화
        self.model.gradient_checkpointing_enable()

        self.conversation_history = []  # 대화 히스토리 초기화

    def load_tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        '''
        토크나이저를 로드합니다.
        :return: 로드된 토크나이저
        '''
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def load_model_with_accelerator(self) -> tuple:
        '''
        모델을 Accelerator를 사용하여 로드하고 옵티마이저를 준비합니다.
        :return: 모델과 옵티마이저
        '''
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            **self.model_kwargs
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # Accelerator로 모델과 옵티마이저 준비
        model, optimizer = self.accelerator.prepare(model, optimizer)
        
        # 1050 GPU에 모델 전송
        model.to(self.device_1050)
        return model, optimizer

    def generate_response_stream(self, input_text: str):
        '''
        입력 텍스트에 대한 응답을 스트리밍 방식으로 생성합니다.
        :param input_text: 입력 텍스트
        :yield: 생성된 텍스트의 스트림
        '''
        max_new_tokens = 400

        full_input = f"{input_text}"
        input_ids = self.tokenizer.encode(full_input, return_tensors="pt").to(torch.device("cpu"))
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        
        # 모델을 실행할 스레드를 생성합니다.
        generation_kwargs = {
            "input_ids": input_ids.to(self.device_1050),
            "attention_mask": attention_mask.to(self.device_1050),
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.64,
            "top_k": 51,
            "top_p": 0.63,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.21,
            "streamer": streamer
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 스트리머에서 텍스트가 생성될 때마다 이를 yield 합니다.
        for new_text in streamer:
            yield new_text
