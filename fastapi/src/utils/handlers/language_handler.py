'''
파일은 자연어 처리를 위한 LanguageProcessor 클래스를 정의하는 모듈입니다.
'''

import spacy  # 텍스트 분석을 위한 라이브러리
from googletrans import Translator  # 번역을 위한 라이브러리
from langdetect import detect, detect_langs  # 언어 감지 라이브러리

class LanguageProcessor:
    """
    자연어 처리를 수행하는 클래스입니다.
    
    이 클래스는 텍스트 분석, 언어 감지, 품사 태깅 등의 
    자연어 처리 기능을 제공합니다.
    
    Attributes:
        translator (Translator): Google 번역 API 클라이언트
        nlp_en (spacy.Language): 영어 처리를 위한 Spacy 모델
        nlp_ko (spacy.Language): 한국어 처리를 위한 Spacy 모델
    """

    def __init__(self):
        """
        LanguageProcessor 클래스를 초기화합니다.
        
        영어와 한국어 모델을 로드하고 번역기를 초기화합니다.
        
        Raises:
            OSError: Spacy 모델 로드 실패 시
            ImportError: 필요한 라이브러리 import 실패 시
        """
        self.translator = Translator()
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_ko = spacy.load("ko_core_news_sm")

    def detect_language(self, sentence: str) -> str:
        """
        입력된 문장의 언어를 감지합니다.
        
        Args:
            sentence (str): 언어를 감지할 입력 문장
            
        Returns:
            str: 감지된 언어의 ISO 639-1 코드 (예: 'en', 'ko')
            
        Raises:
            langdetect.LangDetectException: 언어 감지 실패 시
        """
        return detect(sentence)

    def analyze_with_spacy(self, sentence: str, lang: str) -> dict:
        """
        Spacy를 사용하여 문장의 품사를 분석합니다.
        
        Args:
            sentence (str): 분석할 입력 문장
            lang (str): 입력 문장의 언어 코드 ('en' 또는 'ko')
            
        Returns:
            dict: 품사별로 분류된 단어들을 포함하는 딕셔너리
                {
                    "명사": [str, ...],
                    "동사": [str, ...],
                    "형용사": [str, ...],
                    "부사": [str, ...]
                }
        """
        # 지원되지 않는 언어일 경우 영어 모델 사용
        if lang == "en":
            doc = self.nlp_en(sentence)
        elif lang == "ko":
            doc = self.nlp_ko(sentence)
        else:
            print(f"지원되지 않는 언어: {lang}, 기본값 영어로 분석.")
            doc = self.nlp_en(sentence)

        categories = {
            "명사": [],
            "동사": [],
            "형용사": [],
            "부사": []
        }

        for token in doc:
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                categories["명사"].append(token.text)
            elif token.pos_ == "VERB":
                categories["동사"].append(token.text)
            elif token.pos_ == "ADJ":
                categories["형용사"].append(token.text)
            elif token.pos_ == "ADV":
                categories["부사"].append(token.text)

        return categories

    def process_sentence(self, sentence: str) -> dict:
        """
        입력 문장에 대한 전체 자연어 처리를 수행합니다.
        
        이 메소드는 언어 감지와 품사 분석을 순차적으로 수행하여
        통합된 분석 결과를 제공합니다.
        
        Args:
            sentence (str): 처리할 입력 문장
            
        Returns:
            dict: 처리 결과를 포함하는 딕셔너리
                {
                    "입력 문장": str,
                    "언어": str,
                    "분석 결과": dict
                }
                
        Raises:
            Exception: 처리 과정에서 발생하는 모든 예외
        """
        try:
            # 언어 감지
            lang = self.detect_language(sentence)
            result = {"입력 문장": sentence, "언어": lang}

            # Spacy 분석
            analysis_result = self.analyze_with_spacy(sentence, lang)
            result["분석 결과"] = analysis_result

            return result

        except Exception as e:
            return {"오류": str(e)}


# # 모듈 테스트
# if __name__ == "__main__":
#     processor = LanguageProcessor()
#     test_sentence = "Llama 모델이 어떤 특징을 가지고 있는지 알려주세요."
#     output = processor.process_sentence(test_sentence)
#     print(output)