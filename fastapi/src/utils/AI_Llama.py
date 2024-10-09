import os
import torch
import transformers
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator
import bitsandbytes as bnb
from dotenv import load_dotenv

# 현재 파일의 경로를 기준으로 부모 디렉토리의 .env 파일 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
dotenv_path = os.path.join(parent_dir, '.env')
load_dotenv(dotenv_path)

class LlamaChatModel:
    def __init__(self):
        '''
        LlamaChatModel 클래스 초기화
        '''
        self.model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.cache_dir = "./ai_model/"
        self.model_kwargs = {
            "torch_dtype": torch.float16,  # float16으로 변경
            "trust_remote_code": True,
        }

        # Hugging Face Token 설정
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")

        print("토크나이저 로드 중...")
        self.tokenizer = self.load_tokenizer()
        print("모델 로드 중...")

        # Accelerate 객체 초기화
        self.accelerator = Accelerator()
        self.model, self.optimizer = self.load_model_with_accelerator()
        self.scaler = GradScaler()
        print("모델과 토크나이저 로드 완료!")
        self.conversation_history = []  # 대화 히스토리 초기화

    def load_tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        '''
        토크나이저를 로드합니다.
        :return: 로드된 토크나이저
        '''
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
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
        return model, optimizer


    def allocate_pinned_memory(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        페이지 잠금 메모리를 활용하여 Tensor를 GPU로 전송합니다.
        :param tensor: CPU 상의 텐서
        :return: 페이지 잠금 메모리로 전송된 텐서
        '''
        pinned_tensor = tensor.pin_memory()
        return pinned_tensor

    def generate_response(self, input_text: str, max_new_tokens: int = 500) -> str:
        '''
        주어진 입력 텍스트에 대한 응답을 생성합니다.
        :param input_text: 입력 텍스트
        :param max_new_tokens: 생성할 최대 토큰 수
        :return: 생성된 응답 텍스트
        '''
        full_input = f"{input_text}"
        input_ids = self.tokenizer.encode(full_input, return_tensors="pt").to(torch.device("cpu"))
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Pinned Memory로 전송
        input_ids = self.allocate_pinned_memory(input_ids)
        attention_mask = self.allocate_pinned_memory(attention_mask)

        effective_max_tokens = min(max_new_tokens, 500)

        # Mixed Precision과 함께 generate 함수 직접 호출
        with autocast(dtype=torch.float16):  # float16으로 변경
            with torch.no_grad():
                output = self.model.generate(
                    input_ids.to(self.accelerator.device),
                    attention_mask=attention_mask.to(self.accelerator.device),
                    max_new_tokens=effective_max_tokens,
                    do_sample=True,
                    temperature=0.64,
                    top_k=51,
                    top_p=0.63,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.21,
                    stopping_criteria=transformers.StoppingCriteriaList([self.CustomStoppingCriteria()])
                )

        # GPU 메모리 비우기
        torch.cuda.empty_cache()

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.conversation_history.append(f"AI: {response.strip()}")
        return response.strip()


    class CustomStoppingCriteria(transformers.StoppingCriteria):
        def __init__(self, min_length: int = 10, min_ending_tokens: int = 2):
            self.min_length = min_length
            self.min_ending_tokens = min_ending_tokens

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            if input_ids.shape[1] > self.min_length and (
                input_ids[0, -1] == self.min_ending_tokens or input_ids[0, -2] == self.min_ending_tokens):
                return True
            return False
        
'''
테스트용 코드
'''
# if __name__ == "__main__":
#     LCM = LlamaChatModel()
#     while True:
#         print("입력: ")
#         user_input = input("")  # 사용자로부터 입력 받기
#         if user_input.lower() == "exit":
#             print("종료합니다.")
#             break
#         response = LCM.generate_response(user_input)
#         print(f"응답: {response}")
