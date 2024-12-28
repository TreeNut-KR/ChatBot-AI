# Lanaugae_handler.py
# 파일은 자연어 처리를 위한 클래스를 정의하는 모듈입니다. 이 파일은 다음과 같은 기능을 제공합니다.
import spacy  # 텍스트 분석을 위한 라이브러리
from googletrans import Translator  # 번역을 위한 라이브러리
from langdetect import detect, detect_langs  # 언어 감지 라이브러리

class LanguageProcessor:
    def __init__(self):
        """
        LanguageProcessor 클래스 초기화
        """
        self.translator = Translator()
        self.nlp = spacy.load("en_core_web_sm")

    def detect_language(self, sentence: str) -> str:
        """
        입력 문장의 언어를 감지
        :param sentence: 문장 (문자열)
        :return: 언어 코드 (예: 'en', 'ko')
        """
        detected_languages = detect_langs(sentence)
        for lang in detected_languages:
            if lang.lang == "ko" and lang.prob > 0.3:
                return "ko"
        return detect(sentence)

    def translate_sentence(self, sentence: str, src_lang: str = "ko", dest_lang: str = "en") -> str:
        """
        문장을 지정된 언어로 번역
        :param sentence: 번역할 문장 (문자열)
        :param src_lang: 원본 언어 (기본값: 한국어)
        :param dest_lang: 번역할 언어 (기본값: 영어)
        :return: 번역된 문장 (문자열)
        """
        return self.translator.translate(sentence, src=src_lang, dest=dest_lang).text

    def analyze_with_spacy(self, sentence: str) -> dict:
        """
        문장을 Spacy를 사용하여 주어, 동사, 품사 분석 수행
        :param sentence: 문장 (영어)
        :return: 분석 결과 (딕셔너리)
        """
        doc = self.nlp(sentence)
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

            # 번역 처리
            if lang == "en":
                result["번역된 문장"] = sentence
                result["번역 상태"] = "이미 영어 문장"
            elif lang == "ko":
                translated_text = self.translate_sentence(sentence)
                result["번역된 문장"] = translated_text
                result["번역 상태"] = "번역 완료"
            else:
                result["번역 상태"] = "지원되지 않는 언어"
                return result

            # Spacy 분석
            analysis_result = self.analyze_with_spacy(result["번역된 문장"])
            translated_analysis = {}
            for key, values in analysis_result.items():
                translated_key = self.translate_sentence(key, src_lang="en", dest_lang="ko")
                translated_values = [self.translate_sentence(value, src_lang="en", dest_lang="ko") for value in values]
                translated_analysis[translated_key] = translated_values

            # 번역된 분석 결과로 대체
            result["분석 결과"] = translated_analysis

            return result

        except Exception as e:
            return {"오류": str(e)}

# 모듈 테스트
# if __name__ == "__main__":
#     processor = LanguageProcessor()
#     test_sentence = "안녕하세요, 이 문장은 테스트를 위해 작성되었습니다."
#     output = processor.process_sentence(test_sentence)

#     # 명사 추출 및 공백으로 연결
#     nouns = output.get("분석 결과", {}).get("명사", [])
#     result = " ".join(nouns)

#     print(result)
