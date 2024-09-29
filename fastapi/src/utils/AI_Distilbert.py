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
            "torch_dtype": torch.float32,  # FP16에서 FP32로 변경 (DistilBERT는 FP16을 지원하지 않음)
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
        분류 모델을 로드합니다.
        :return: 로드된 모델
        '''
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            **self.model_kwargs
        )
        return model

    def classify_text(self, input_text: str) -> str:
        '''
        주어진 입력 텍스트에 대한 감정 분류 결과를 생성합니다.
        :param input_text: 입력 텍스트
        :return: 생성된 분류 결과
        '''
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 분류 결과에서 가장 높은 점수를 가진 레이블을 반환
        predicted_label = torch.argmax(outputs.logits, dim=-1).item()
        return str(predicted_label)

if __name__ == "__main__":
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"  # 감정 분류를 위한 DistilBERT 모델
    cache_dir = "./ai_model/"  # 모델을 저장할 경로

    # KoChatModel 인스턴스 생성
    ko_chat = KoChatModel(model_id, cache_dir)

    # 입력 텍스트
    input_text = "I love programming with Python!"
    prediction = ko_chat.classify_text(input_text)

    print("분류 결과:", prediction)  # 분류 결과 출력
