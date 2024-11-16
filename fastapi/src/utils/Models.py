from pydantic import BaseModel, Field

input_data_set = Field(
    examples=["Llama AI 모델의 출시일과 버전들을 각각 알려줘."],
    title="사용자 입력 문장",
    min_length=1, max_length=500,
    description="사용자 입력 문장 길이 제약"
)

output_data_set = Field(
    examples=['''
    물론이죠! Llama AI 모델의 출시일과 버전들은 다음과 같습니다:

    1. Llama 1: 2023년 출시1

    2. Llama 2: 2024년 6월 1일 출시2

    3. Llama 3: 2024년 7월 23일 출시3

    4. Llama 3.1: 2024년 7월 24일 출시4

    이 모델들은 Meta (구 Facebook)에서 개발한 AI 모델입니다.
    각 버전마다 성능과 기능이 개선되었습니다. 더 궁금한 점이 있으신가요?
    '''],
    title="Llama 답변"
)

class Llama_Request(BaseModel):
    input_data: str = input_data_set
    
class Llama_Response(BaseModel):
    output_data: str = output_data_set
    
class Bllossom_Request(BaseModel):
    input_data: str = input_data_set
    
class Bllossom_Response(BaseModel):
    output_data: str = output_data_set