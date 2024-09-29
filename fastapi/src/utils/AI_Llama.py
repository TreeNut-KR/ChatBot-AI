import torch
import os 
import transformers
import bitsandbytes as bnb
from dotenv import load_dotenv

# 현재 파일의 경로를 기준으로 부모 디렉토리의 .env 파일 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 부모 디렉토리
dotenv_path = os.path.join(parent_dir, '.env')
load_dotenv(dotenv_path)


class KoChatModel:
    def __init__(self, model_id: str, cache_dir: str):
        '''
        KoChatModel 클래스 초기화
        :param model_id: 사용할 모델 ID
        :param cache_dir: 모델을 저장할 경로
        '''
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.model_kwargs = {
            "torch_dtype": torch.float32,  # FP16 사용 (LLaMA 지원)
            "trust_remote_code": True,
        }

        # Hugging Face Token 설정
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")

        print("토크나이저 로드 중...")
        self.tokenizer = self.load_tokenizer()
        print("모델 로드 중...")
        self.model = self.load_model()
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
            use_auth_token=self.hf_token
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id  # pad_token_id 설정
        return tokenizer

    def load_model(self) -> transformers.PreTrainedModel:
        '''
        대화 모델을 로드합니다. 8-bit 양자화 및 메모리 최적화 적용.
        :return: 로드된 모델
        '''
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            load_in_8bit=True,  # 8-bit 양자화
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
            **self.model_kwargs
        )
        return model

    def generate_response(self, input_text: str, max_new_tokens: int = 100) -> str:
        '''
        주어진 입력 텍스트에 대한 응답을 생성합니다.
        :param input_text: 입력 텍스트
        :param max_new_tokens: 생성할 최대 토큰 수
        :return: 생성된 응답 텍스트
        '''
        # 대화 히스토리에 입력 텍스트 추가
        self.conversation_history.append(f"User: {input_text}")

        # 대화 히스토리를 하나의 문자열로 결합
        full_input = "\n".join(self.conversation_history) + "\nAI:"

        input_ids = self.tokenizer.encode(full_input, return_tensors="pt").to(self.model.device)

        # 텍스트 생성
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,  # 텍스트 다양성 조정
                top_k=50,         # top-k 샘플링 적용
                top_p=0.9         # top-p 샘플링 적용
            )

        # 생성된 텍스트 디코딩
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # AI 응답을 대화 히스토리에 추가
        self.conversation_history.append(f"AI: {response}")

        # AI의 응답만 반환
        return response

if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-3B-Instruct"  # LLaMA 대화 모델
    cache_dir = "./ai_model/"  # 모델을 저장할 경로

    # KoChatModel 인스턴스 생성
    ko_chat = KoChatModel(model_id, cache_dir)

    # 입력 텍스트
    input_text = "Python으로 MongoDB 연결하는 코드 짜줘."
    response = ko_chat.generate_response(input_text, max_new_tokens=450)

    print("생성된 응답:", response)  # 생성된 응답 출력
