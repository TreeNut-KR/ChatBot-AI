'''
파일은 자연어 처리를 위한 LanguageProcessor 클래스를 정의하는 모듈입니다.
'''

import re
import time
import spacy                                    # 텍스트 분석을 위한 라이브러리
from googletrans import Translator              # 번역을 위한 라이브러리
from deep_translator import GoogleTranslator
from langdetect import detect                   # 언어 감지 라이브러리

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
        
    def _translate_text(self, text: str, target_lang: str) -> str:
        """
        텍스트를 지정된 언어로 번역합니다.

        Args:
            text (str): 번역할 텍스트
            target_lang (str): 대상 언어 코드 ('ko' 또는 'en')

        Returns:
            str: 번역된 텍스트

        Raises:
            Exception: 번역 실패 시 발생하는 예외
        """
        if not text.strip():
            return text
            
        try:
            # GoogleTranslator를 사용하여 번역 (더 안정적)
            translator = GoogleTranslator(target=target_lang)
            translated = translator.translate(text)
            return translated
        except Exception as e:
            # 첫 번째 방법 실패시 googletrans 라이브러리 사용
            try:
                translated = self.translator.translate(text, dest=target_lang)
                return translated.text
            except Exception as inner_e:
                print(f"번역 오류: {str(inner_e)}")
                return text
            
    def _split_and_translate(self, text: str, target_lang: str) -> str:
        """
        텍스트를 패턴에 따라 분할하고 각각 번역합니다.
        """
        patterns = [
            (r'\*\*(.*?)\*\*', '**', '**'),  # 볼드체
            (r'\*(.*?)\*', '*', '*'),         # 이탤릭체
            (r'```(.*?)```', '```', '```'),   # 코드 블록
            (r'`(.*?)`', '`', '`'),           # 인라인 코드
            (r'"(.*?)"', '"', '"'),           # 큰따옴표
            (r'\'(.*?)\'', "'", "'"),         # 작은따옴표
            (r'\((.*?)\)', '(', ')'),         # 괄호
            (r'\[(.*?)\]', '[', ']'),         # 대괄호
            (r'\{(.*?)\}', '{', '}'),         # 중괄호
        ]

        # 현재 처리해야 할 텍스트와 결과를 저장할 리스트
        current_text = text
        result_parts = []
        last_end = 0

        while current_text:
            # 모든 패턴에 대해 가장 먼저 나오는 매치 찾기
            earliest_match = None
            matched_pattern = None
            
            for pattern, start_delim, end_delim in patterns:
                match = re.search(pattern, current_text)
                if match and (earliest_match is None or match.start() < earliest_match.start()):
                    earliest_match = match
                    matched_pattern = (start_delim, end_delim)

            if earliest_match:
                # 특수 문자 이전의 일반 텍스트 처리
                if earliest_match.start() > 0:
                    normal_text = current_text[:earliest_match.start()]
                    translated_normal = self._translate_text(normal_text, target_lang)
                    result_parts.append(translated_normal)

                # 특수 문자로 감싸진 텍스트 처리
                special_text = earliest_match.group(1)
                translated_special = self._translate_text(special_text, target_lang)
                start_delim, end_delim = matched_pattern
                result_parts.append(f"{start_delim}{translated_special}{end_delim}")

                # 다음 처리를 위해 남은 텍스트 업데이트
                current_text = current_text[earliest_match.end():]
            else:
                # 남은 텍스트 전체 번역
                translated_remaining = self._translate_text(current_text, target_lang)
                result_parts.append(translated_remaining)
                break

        return ''.join(result_parts)

    def translate_to_korean(self, text: str) -> str:
        """
        주어진 텍스트를 한글로 번역합니다.
        """
        try:
            if not text or self.detect_language(text) == 'ko':
                return text
            return self._split_and_translate(text, 'ko')
        except Exception as e:
            print(f"번역 오류: {str(e)}")
            return text

    def translate_to_english(self, text: str) -> str:
        """
        주어진 텍스트를 영어로 번역합니다.
        """
        try:
            if not text or self.detect_language(text) == 'en':
                return text
            return self._split_and_translate(text, 'en')
        except Exception as e:
            print(f"번역 오류: {str(e)}")
            return text

# # 모듈 테스트
# if __name__ == "__main__":
#     processor = LanguageProcessor()
#     test_sentence = "Llama 모델이 어떤 특징을 가지고 있는지 알려주세요."
#     output = processor.process_sentence(test_sentence)
#     print(output)

# 번역 모듈 테스트
# if __name__ == "__main__":
#     processor = LanguageProcessor()
    
#     test_texts = [
#         "*Rachel smiles warmly* Oh, hello! Who might you be? **Important** This is bold text *Rachel smiles warmly* Oh, hello! Who might you be? **Important** This is bold text *Rachel smiles warmly* Oh, hello! Who might you be? **Important** This is bold text ",
#         "*이것은 이탤릭체입니다* 그리고 이것은 일반 텍스트입니다.",
#         "```This is a code block```"
#     ]
    
#     print("번역 테스트 시작...\n")
#     start_time = time.time()
    
#     for text in test_texts:
#         try:
#             translated = processor.translate_to_korean(text)
#             print(f"원문: {text}")
#             print(f"번역: {translated}")
#             print("-" * 50)
#         except Exception as e:
#             print(f"오류 발생: {e}")
#             print("-" * 50)
            
#     end_time = time.time()
#     print(f"\n번역 소요 시간: {end_time - start_time:.2f}초")