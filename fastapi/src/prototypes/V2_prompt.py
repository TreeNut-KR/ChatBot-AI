import json
import base64
from PIL import Image
from collections import OrderedDict

def extract_prompt_from_png(file_path: str):
    """
    V2 카드 사양을 준수하는 PNG 파일에서 프롬프트 값을 추출하는 함수
    
    :param file_path: PNG 이미지 파일 경로
    :return: 추출된 프롬프트 문자열 (없으면 빈 문자열 반환)
    """
    try:
        # 이미지 열기
        img = Image.open(file_path)

        # PNG의 메타데이터 확인
        metadata = img.info  # tEXt, iTXt 등 포함 가능
        if not metadata:
            print("❌ PNG 메타데이터가 존재하지 않습니다.")
            return None

        # "prompt" 또는 "description" 키 확인
        for key in metadata:
            if "chara" in key.lower():
                # base64 디코드 후 json 파싱
                decoded_data = base64.b64decode(metadata[key])
                json_data = json.loads(decoded_data)
                print("\n ==  =  추출된 프롬프트  ==  = ")
                
                # JSON 데이터 출력
                # 1. indent
                #     indent = 숫자: 지정된 숫자만큼 공백으로 들여쓰기
                    
                # 2. ensure_ascii
                #     ensure_ascii = True (기본값):
                #         모든 비-ASCII 문자를 \uXXXX 형식으로 이스케이프
                #         ex) "안녕" → "\uc548\ub155"
                #     ensure_ascii = False:
                #         비-ASCII 문자를 그대로 유지
                #         ex) "안녕" → "안녕"
                print(json.dumps(json_data, indent = 5, ensure_ascii = False), "\n")
                
    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")
        return None

if __name__  ==  "__main__":
    # 업로드된 PNG 파일 경로
    file_path = "C:/Users/treen/Downloads/main_rachel-29118321_spec_v2.png"

    # 프롬프트 추출
    extract_prompt_from_png(file_path)
    
