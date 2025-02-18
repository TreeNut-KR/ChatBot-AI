'''
파일은 자연어 처리를 위한 클래스를 정의하는 모듈입니다. 이 파일은 다음과 같은 기능을 제공합니다.
'''

import spacy  # 텍스트 분석을 위한 라이브러리
from googletrans import Translator  # 번역을 위한 라이브러리
from langdetect import detect, detect_langs  # 언어 감지 라이브러리

class LanguageProcessor:
    def __init__(self):
        """
        LanguageProcessor 클래스 초기화
        """
        self.translator = Translator()
        self.nlp_en = spacy.load("en_core_web_sm")  # 영어 모델 로드
        self.nlp_ko = spacy.load("ko_core_news_sm")  # 한국어 모델 로드

    def detect_language(self, sentence: str) -> str:
        """
        입력 문장의 언어를 감지
        :param sentence: 문장 (문자열)
        :return: 언어 코드 (예: 'en', 'ko')
        """
        return detect(sentence)

    def analyze_with_spacy(self, sentence: str, lang: str) -> dict:
        """
        문장을 Spacy를 사용하여 품사 분석 수행
        :param sentence: 문장 (문자열)
        :param lang: 언어 코드 ('en', 'ko')
        :return: 분석 결과 (딕셔너리)
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
        문장을 처리하여 딕셔너리 형식으로 결과 반환
        :param sentence: 입력 문장 (문자열)
        :return: 처리 결과 (딕셔너리)
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


# 모듈 테스트
if __name__ == "__main__":
    processor = LanguageProcessor()
    test_sentence = "스텔라이브한테 어떤 문제가 있는지 알려줄래?"
    output = processor.process_sentence(test_sentence)
    print(output)