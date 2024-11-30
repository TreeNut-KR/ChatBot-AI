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
        
    def generate_response_stream(self, input_text: str, character_settings: dict):
        """
        캐릭터 설정과 부정 라벨을 활용하여 스트리밍 응답 생성
        """
        # 캐릭터 설정 구성 If there is no request for an answer to a specific language, I will answer in Korean
        prompt = (
            f"캐릭터 설정:\n"
            f"이름: {character_settings['character_name']}\n"
            f"설명: {character_settings['description']}\n"
            f"인사말: {character_settings['greeting']}\n"
            f"성격: {character_settings['character_setting']}\n"
            f"말투: {character_settings['tone']}\n"
            f"에너지 레벨: {character_settings['energy_level']}\n"
            f"공손함: {character_settings['politeness']}\n"
            f"유머 감각: {character_settings['humor']}\n"
            f"단호함: {character_settings['assertiveness']}\n"
            f"액세스 수준: {'허용됨' if character_settings['access_level'] else '제한됨'}\n\n"
            f"사용자 입력:\n\"{input_text}\"\n\n"
            f"부정 라벨:\n"
            f'''
            If there is no request for an answer to a specific language, I will answer in Korean.
            Avoid directly stating the character configuration.
            Do not mention setup details like personality traits, energy levels, or styles directly in the response.
            '''
            f"\n\n답변:"
        )

        max_new_tokens = 200
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

        generation_kwargs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.2,
            "streamer": streamer
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text


