import base64
import json
from PIL import Image

def extract_prompt_from_png(file_path: str) -> str:
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
            return ""

        # "prompt" 또는 "description" 키 확인
        for key in metadata:
            if "chara" in key.lower():
                # base64 디코드 후 json 파싱
                decoded_data = base64.b64decode(metadata[key])
                json_data = json.loads(decoded_data)
                
                # 포맷에 따라 정보 추출 및 출력
                if json_data.get("spec") == "chara_card_v2":
                    print("\n=== CHARACTER INFORMATION ===")
                    print(f"Name: {json_data['data']['name']}")
                    print(f"Description: {json_data['data']['description']}")  # 긴 설명은 일부만 표시
                    
                    # 인사말이 있으면 출력
                    if 'alternate_greetings' in json_data['data']:
                        print("\nGreetings:")
                        for greeting in json_data['data']['alternate_greetings']:
                            print(f"- {greeting}")  # 긴 인사말은 일부만 표시
                    
                    return json_data
                
        print("❌ 지원되는 캐릭터 포맷을 찾을 수 없습니다.")
        return None

    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")
        return None

if __name__ == "__main__":
    # 업로드된 PNG 파일 경로
    file_path = "C:/Users/treen/Downloads/main_rachel-29118321_spec_v2.png"

    # 프롬프트 추출
    extracted_data = extract_prompt_from_png(file_path)
    
    if not extracted_data:
        print("❌ 캐릭터 정보를 추출할 수 없습니다.")
