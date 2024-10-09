import os
import torch
import transformers
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig, pipeline

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
        self.cache_dir = "./fastapi/ai_model/"
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        self.model_kwargs = {
            "torch_dtype": torch.float16,  # float16으로 설정
            "trust_remote_code": True,
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True)  # 양자화 적용
        }

        # Hugging Face Token 설정
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")

        # Accelerate 객체 초기화
        self.accelerator = Accelerator(mixed_precision="fp16")  # Mixed Precision 설정
        self.device_2080 = torch.device("cuda:0")  # 2080 GPU에 할당

        print("토크나이저 로드 중...")
        self.tokenizer = self.load_tokenizer()
        print("모델 로드 중...")
        self.model, self.optimizer = self.load_model_with_accelerator()
        # 복잡도 분석 모델 로드 중단
        # print("복잡도 분석 모델 로드 중...")
        # self.complexity_analyzer = self.load_complexity_analyzer()
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
            cache_dir=self.cache_dir,
            token=self.hf_token
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    # 복잡도 분석 모델 로드 메서드 주석 처리
    # def load_complexity_analyzer(self) -> transformers.Pipeline:
    #     '''
    #     복잡도 분석 모델을 로드합니다.
    #     :return: 로드된 복잡도 분석 파이프라인
    #     '''
    #     # 복잡도 분석 모델을 2080 GPU에서 로드하도록 설정
    #     return pipeline(
    #         "text-classification",
    #         model=self.model,  # 메인 모델을 재사용
    #         tokenizer=self.tokenizer  # 여기에 tokenizer 추가
    #     )

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
        
        # GPU에 모델 전송
        model.to(self.device_2080)  # 2080 GPU에 전송
        return model, optimizer

    def allocate_pinned_memory(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        페이지 잠금 메모리를 활용하여 Tensor를 GPU로 전송합니다.
        :param tensor: CPU 상의 텐서
        :return: 페이지 잠금 메모리로 전송된 텐서
        '''
        pinned_tensor = tensor.pin_memory()
        return pinned_tensor

    # 복잡도 분석 메서드 주석 처리
    # def analyze_complexity(self, input_text: str) -> str:
    #     '''
    #     입력 텍스트의 복잡성을 분석합니다.
    #     :param input_text: 입력 텍스트
    #     :return: 복잡도 결과 (low, medium, high)
    #     '''
    #     result = self.complexity_analyzer(input_text)
    #     label = result[0]['label']
    #     if "ENTAILMENT" in label:
    #         return "low"
    #     elif "CONTRADICTION" in label:
    #         return "high"
    #     return "medium"

    # 최대 토큰 수 조정 메서드 주석 처리
    # def adjust_max_tokens(self, complexity: str) -> int:
    #     '''
    #     복잡도에 따라 생성할 최대 토큰 수를 조정합니다.
    #     :param complexity: 입력 텍스트의 복잡도
    #     :return: 조정된 최대 토큰 수
    #     '''
    #     if complexity == "low":
    #         return 100  # 간단한 질문일 경우 100토큰 제한
    #     elif complexity == "high":
    #         return 500  # 복잡한 질문일 경우 500토큰 제한
    #     return 250  # 중간 수준의 복잡도일 경우 250토큰

    def generate_response(self, input_text: str) -> str:
        '''
        주어진 입력 텍스트에 대한 응답을 생성합니다.
        :param input_text: 입력 텍스트
        :return: 생성된 응답 텍스트
        '''
        # 2080 GPU에서 복잡도 분석 (주석 처리됨)
        # complexity = self.analyze_complexity(input_text)
        # max_new_tokens = self.adjust_max_tokens(complexity)

        max_new_tokens = 250  # 기본 최대 토큰 수 설정
        full_input = f"{input_text}"
        input_ids = self.tokenizer.encode(full_input, return_tensors="pt").to(torch.device("cpu"))
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Mixed Precision과 함께 generate 함수 직접 호출
        with autocast(dtype=torch.float16):  # Mixed Precision 사용
            with torch.no_grad():
                output = self.model.generate(
                    input_ids.to(self.device_2080),  # 2080 GPU에 전송
                    attention_mask=attention_mask.to(self.device_2080),  # 2080 GPU에 전송
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.64,
                    top_k=51,
                    top_p=0.63,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.21,
                )

        # GPU 메모리 비우기
        torch.cuda.empty_cache()

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.conversation_history.append(f"AI: {response.strip()}")
        return response.strip()


if __name__ == "__main__":
    LCM = LlamaChatModel()
    while True:
        print("입력: ")
        user_input = input("")  # 사용자로부터 입력 받기
        if user_input.lower() == "exit":
            print("종료합니다.")
            break
        response = LCM.generate_response(user_input)
        print(f"응답: {response}")
