import re
import httpx
from pydantic import BaseModel, Field, field_validator, conint

class Validators:
    @staticmethod
    def validate_URL(v: str) -> str:
        """
        URL 형식 검증 함수
        """
        url_pattern = re.compile(
            r'''
            ^                     # 문자열의 시작
            https?://             # http:// 또는 https://
            (drive\.google\.com)  # Google Drive 도메인
            /thumbnail            # 경로의 일부
            \?id=([a-zA-Z0-9_-]+) # 쿼리 파라미터 id
            $                     # 문자열의 끝
            ''', re.VERBOSE
        )
        if not url_pattern.match(v):
            raise ValueError('유효한 URL 형식이 아닙니다.')
        return v

    @staticmethod
    async def check_img_url(img_url: str):
        '''
        URL의 연결 테스트 함수
        '''
        try:
            async with httpx.AsyncClient() as client:
                response = await client.head(img_url, follow_redirects=True)
            if response.status_code != 200:
                raise ValueError('이미지에 접근할 수 없습니다.')
        except httpx.RequestError:
            raise ValueError('이미지 URL에 접근하는 중 오류가 발생했습니다.')

# Llama Request Field
input_data_set = Field(
    examples=["Llama AI 모델의 출시일과 버전들을 각각 알려줘."],
    title="사용자 입력 문장",
    description="사용자 입력 문장 길이 제약",
    min_length=1, max_length=500
)

# Bllossom Request Field
## 숫자 타입 정의
NATURAL_NUM: int = conint(ge=1, le=10)  # 1~10 범위의 자연수

## 각 필드에 대한 설정
character_name_set = Field(
    examples=["KindBot"],
    title="케릭터 이름",
    description="캐릭터의 이름입니다. 봇의 정체성을 나타내며, 사용자가 이 이름으로 봇을 부릅니다.",
    min_length=1
)
description_set = Field(
    examples=["친절한 도우미 봇"],
    title="케릭터 설명",
    description="캐릭터의 짧은 설명입니다. 이 봇의 성격이나 역할을 간략히 표현하며, 사용자에게 첫인상을 제공합니다.",
    min_length=1
)
greeting_set = Field(
    examples=["안녕하세요! 무엇을 도와드릴까요?"],
    title="케릭터 인사말",
    description="사용자가 봇과 상호작용을 시작할 때 표시되는 인사말입니다. 봇의 성격과 의도를 반영합니다.",
    min_length=1
)
image_set = Field(
    examples=["https://drive.google.com/thumbnail?id=12PqUS6bj4eAO_fLDaWQmoq94-771xfim"],
    title="케릭터 이미지 URL",
    description="URL의 최대 길이는 일반적으로 2048자",
    min_length=1, max_length=2048
)
character_setting_set = Field(
    examples=["친절하고 공손한 봇"],
    title="케릭터 설정 값",
    description="캐릭터의 성격이나 태도를 나타냅니다. 이는 봇이 대화에서 어떻게 행동하고 응답할지를 정의합니다.",
    min_length=1
)
tone_set = Field(
    examples=["공손한"],
    title="케릭터 말투",
    description="대화의 어조를 나타냅니다. 봇이 대화에서 사용하는 언어 스타일이나 태도입니다.",
    min_length=1
)
energy_level_set = Field(
    examples=[8],
    title="케릭터 에너지 ",
    description="봇의 에너지 수준을 나타내는 숫자입니다. 높은 값일수록 활기차고 적극적인 대화를 할 수 있음을 의미합니다. 1(매우 느긋함) ~ 10(매우 활기참)."
)
politeness_set = Field(
    examples=[10],
    title="케릭터 공손함",
    description="봇의 공손함을 나타내는 숫자입니다. 높은 값일수록 공손하고 존중하는 언어를 사용할 가능성이 높습니다. 1(직설적임) ~ 10(매우 공손함)"
)
humor_set = Field(
    examples=[5],
    title="케릭터 유머 감각",
    description="봇의 유머 감각의 정도를 나타냅니다. 숫자가 높을수록 대화에서 유머러스한 요소를 추가하려고 시도합니다. 1(유머 없음) ~ 10(매우 유머러스함)."
)
assertiveness_set = Field(
    examples=[3],
    title="케릭터 단호함",
    description="봇의 단호함을 나타냅니다. 숫자가 높을수록 주장을 강하게 하거나 명확히 표현하는 경향이 있습니다. 1(매우 유연함) ~ 10(매우 단호함)."
)
access_level_set = Field(
    examples=[True, False],
    title="케릭터 액세스",
    description="봇의 액세스 수준을 나타냅니다. True: 특정 기능이나 영역에 대한 접근 권한이 허용됨. False: 제한됨."
)

# Common Response Field
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
    character_name: str = character_name_set
    description: str = description_set
    greeting: str = greeting_set
    image: str = image_set
    character_setting: str = character_setting_set
    tone: str = tone_set
    energy_level: NATURAL_NUM = energy_level_set    # type: ignore
    politeness: NATURAL_NUM = politeness_set        # type: ignore
    humor: NATURAL_NUM = humor_set                  # type: ignore
    assertiveness: NATURAL_NUM = assertiveness_set  # type: ignore
    access_level: bool = access_level_set
    
    @field_validator('image', mode='before')
    def check_img_url(cls, v):
        return Validators.validate_URL(v)
    
    def model_dump(self, **kwargs):
        """
        Pydantic BaseModel의 dict() 메서드를 대체하는 model_dump() 메서드입니다.
        필터링된 데이터만 반환하도록 수정할 수 있습니다.
        """
        return super().model_dump(**kwargs)
    
class Bllossom_Response(BaseModel):
    output_data: str = output_data_set