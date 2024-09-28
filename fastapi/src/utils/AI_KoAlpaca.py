import torch
import transformers
import bitsandbytes as bnb

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
            "torch_dtype": torch.float16,  # FP16 사용
            "trust_remote_code": True,
        }
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
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        tokenizer.clean_up_tokenization_spaces = True  # FutureWarning 방지
        tokenizer.pad_token_id = tokenizer.eos_token_id  # pad_token_id 설정
        return tokenizer

    def load_model(self) -> transformers.PreTrainedModel:
        '''
        모델을 로드합니다. 8-bit 양자화 및 메모리 최적화 적용.
        
        :return: 로드된 모델
        '''
        # BitsAndBytesConfig 대신 기존 방식 유지
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            load_in_8bit=True,  # 8-bit 양자화
            cache_dir=self.cache_dir,
            **self.model_kwargs
        )
        return model

    def generate_response(self, input_text: str, max_new_tokens: int = 50) -> str:
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
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.bool).to(self.model.device)

        # 텍스트 생성
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.72,
                top_k=46,
                top_p=0.81
            )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # 생성된 응답을 대화 히스토리에 추가
        self.conversation_history.append(f"AI: {response}")

        # AI의 응답만 반환
        return response  # AI의 응답만 반환

if __name__ == "__main__":
    model_id = "beomi/KoAlpaca-Polyglot-5.8B"  # 한국어 대화 모델
    cache_dir = "./ai_model/"  # 모델을 저장할 경로

    # KoChatModel 인스턴스 생성
    ko_chat = KoChatModel(model_id, cache_dir)

    # 입력 텍스트
    input_text = "안녕? 반가워."
    response = ko_chat.generate_response(input_text, max_new_tokens=100)

    print("생성된 응답:", response)  # 생성된 응답 출력

    # 추가 대화 예시
    input_text = "네가 날 도와줄 수 있어?"
    response = ko_chat.generate_response(input_text, max_new_tokens=100)

    print("생성된 응답:", response)  # 생성된 응답 출력