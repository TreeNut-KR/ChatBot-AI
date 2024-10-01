
import torch
import transformers
import bitsandbytes as bnb
from dotenv import load_dotenv

class KoChatModel:
    def __init__(self):
        '''
        KoChatModel 클래스 초기화
        :param model_id: 사용할 모델 ID
        :param cache_dir: 모델을 저장할 경로
        '''
        self.model_id = "meta-llama/Llama-3.2-3B-Instruct"  # LLaMA 대화 모델
        self.cache_dir = "./ai_model/"  # 모델을 저장할 경로
        self.model_kwargs = {
            "torch_dtype": torch.float32,  # FP16 사용 (LLaMA 지원)
            "trust_remote_code": True,
        }

        # Hugging Face Token 설정
        self.hf_token = "your-huggingface-token"

        print("토크나이저 로드 중...")
        self.tokenizer = self.load_tokenizer()
        print("모델 로드 중...")
        self.model = self.load_model()
        print("모델과 토크나이저 로드 완료!")

    def load_tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        '''
        토크나이저를 로드합니다.
        :return: 로드된 토크나이저
        '''
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id, 
            cache_dir=self.cache_dir, 
            token=self.hf_token  # use_auth_token 대신 token 사용
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
            token=self.hf_token,  # use_auth_token 대신 token 사용
            **self.model_kwargs
        )
        return model

    def generate_response(self, input_text: str, max_new_tokens: int = 500) -> str:
        '''
        주어진 입력 텍스트에 대한 응답을 생성합니다.
        :param input_text: 입력 텍스트
        :param max_new_tokens: 생성할 최대 토큰 수 (기본값: 500)
        :return: 생성된 응답 텍스트
        '''
        full_input = f"{input_text}"

        input_ids = self.tokenizer.encode(full_input, return_tensors="pt").to(self.model.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()  # attention_mask 설정

        effective_max_tokens = min(max_new_tokens, 500)

        # 텍스트 생성
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=effective_max_tokens,
                do_sample=True,
                temperature=0.64,
                top_k=51,
                top_p=0.63,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.21,  # 반복 패널티 추가 (1.1 ~ 1.5 사이 권장)
                stopping_criteria=transformers.StoppingCriteriaList([
                    self.CustomStoppingCriteria()  # 사용자 정의 종료 기준 적용
                ])
            )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response.strip()


    class CustomStoppingCriteria(transformers.StoppingCriteria):
        def __init__(self, min_length: int = 10, min_ending_tokens: int = 2):
            '''
            사용자 정의 종료 기준. 응답이 필요 이상으로 길어지지 않도록 설정합니다.
            :param min_length: 최소 응답 길이
            :param min_ending_tokens: 최소 종료 토큰 수
            '''
            self.min_length = min_length
            self.min_ending_tokens = min_ending_tokens

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            '''
            입력 토큰을 분석하여 종료 여부를 결정합니다.
            '''
            # 생성된 토큰 수가 최소 길이를 넘고, 종료 토큰이 나왔을 때 중단
            if input_ids.shape[1] > self.min_length and (
                input_ids[0, -1] == self.min_ending_tokens or input_ids[0, -2] == self.min_ending_tokens):
                return True
            return False

if __name__ == "__main__":
    KCM = KoChatModel()
    response = KCM.generate_response("python으로 별 피라미드 출력하는 코드 작성해봐.")
    print(response)