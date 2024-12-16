import os
from threading import Thread

import torch
import transformers
from accelerate import Accelerator
from dotenv import load_dotenv
from torch.cuda.amp import GradScaler
from transformers import BitsAndBytesConfig, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

class LlamaChatModel:
    def __init__(self):
        '''
        LlamaChatModel 클래스 초기화
        '''
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        dotenv_path = os.path.join(parent_dir, '.env')
        load_dotenv(dotenv_path)
        self.cache_dir = "./fastapi/ai_model/"
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        self.device = torch.device("cuda:1")  # 명확히 cuda:1로 지정

        self.model_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "device_map": {"": self.device},
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

    class StopOnEOS(StoppingCriteria):
        def __init__(self, eos_token_id):
            super().__init__()
            self.eos_token_id = eos_token_id

        def __call__(self, input_ids, scores, **kwargs):
            # 자동 종료 조건: EOS 토큰이 생성되면 종료
            return self.eos_token_id in input_ids[0].tolist()

    def generate_response_stream(self, input_text: str):
        # prompt = self._build_prompt(input_text)
        input_ids = self.tokenizer.encode(
            text=input_text,
            # text_pair=prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

        # Stopping criteria 리스트 추가
        stopping_criteria = StoppingCriteriaList([
            self.StopOnEOS(eos_token_id=self.tokenizer.eos_token_id)
        ])

        generation_kwargs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "min_new_tokens": 10,  # 최소 토큰 수
            "max_new_tokens": 512,  # 최대 토큰 수 제한
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "streamer": streamer,
            "stopping_criteria": stopping_criteria
        }

        # 모델 생성 스레드 실행
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 스트리머를 통해 출력된 텍스트를 순차적으로 반환
        for new_text in streamer:
            yield new_text

            
    # def _build_prompt(self, user_input: str):
    #     """
    #     대화 기록 기반으로 프롬프트 생성
    #     """
    #     recent_history = self.conversation_history[-5:]  # 최근 5개의 대화만 유지
    #     history = "\n".join([f"{entry['role']}: {entry['content']}" for entry in recent_history])
    #     prompt = (
    #         "meta-llama/Llama-3.1-8B-Instruct(role:AI) prompt:\n"
    #         f"대화 기록: {history}\n"
    #         f"사용자 입력: {user_input}\n\n"
    #     )
    #     print(history)
    #     self.conversation_history.append({"role": "user", "content": user_input})
    #     return prompt
