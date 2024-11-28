import os
import re
from threading import Thread

import torch
import transformers
from accelerate import Accelerator
from dotenv import load_dotenv
from torch.cuda.amp import GradScaler
from transformers import BitsAndBytesConfig, TextIteratorStreamer


class BllossomChatModel:
    def __init__(self):
        '''
        BllossomChatModel 클래스 초기화
        '''
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        dotenv_path = os.path.join(parent_dir, '.env')
        load_dotenv(dotenv_path)
        self.cache_dir = "./fastapi/ai_model/"
        self.model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
        self.device = torch.device("cuda:0")  # 명확히 cuda:0로 지정

        self.model_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "device_map": {"": self.device},  # 명확히 모델과 GPU를 연결
            "quantization_config": BitsAndBytesConfig(load_in_4bit=True)
        }

        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
        self.accelerator = Accelerator(mixed_precision="fp16", device_placement=False)
        self.scaler = GradScaler()

        print("토크나이저 로드 중...")
        self.tokenizer = self.load_tokenizer()
        print("모델 로드 중...")
        self.model = self.load_model()
        print("모델과 토크나이저 로드 완료!")

        self.model.to(self.device)
        self.model.gradient_checkpointing_enable()
        self.conversation_history = []

    def load_tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def load_model(self) -> transformers.PreTrainedModel:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            **self.model_kwargs
        )
        return model

    def predict_response_length(self, input_text: str) -> int:
        """
        입력 텍스트를 기반으로 적절한 응답 길이를 예측합니다.
        """
        prompt = f"문장: {input_text}\n이 문장에 적절한 응답 길이를 예측하세요. 숫자로만 답하세요 (50~400):"
        
        # 모델 응답 생성
        predicted_length = self.generate_text(
            prompt,
            max_tokens=5,  # 간단한 숫자 예측만 필요
            temperature=0.2
        )

        # 모델 응답 디버깅 출력
        print(f"Generated prediction: {predicted_length}")

        # 숫자 추출 (정규식 활용)
        match = re.search(r'\b\d+\b', predicted_length)
        if match:
            predicted_length = int(match.group())
            print(f"Extracted length: {predicted_length}")
            return min(400, max(50, predicted_length))  # 범위 제한
        else:
            print("Prediction failed, returning default value: 200")
            return 200

    def generate_text(self, prompt: str, max_tokens: int, temperature: float = 0.7) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_k": 50,
            "top_p": 0.9,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text

    
    def generate_response_stream(self, input_text: str):
        max_new_tokens = self.predict_response_length(input_text)
        full_input = f"{input_text}"
        input_ids = self.tokenizer.encode(full_input, return_tensors="pt").to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

        generation_kwargs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.5,
            "top_k": 40,
            "top_p": 0.7,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.5,
            "streamer": streamer
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

