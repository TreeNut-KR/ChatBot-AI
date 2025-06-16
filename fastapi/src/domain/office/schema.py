'''
파일은 FastAPI 애플리케이션에서 사용되는 Pydantic 모델을 정의하는 모듈입니다.
'''
import uuid
from pydantic import BaseModel, Field, model_validator

class Validators:
    @staticmethod
    def validate_uuid(v: str) -> str:
        """
        UUID 형식 검증 함수
        """
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError('유효한 UUID 형식이 아닙니다.')
        return v

class CommonField:
    """
    공통 필드 정의 클래스입니다.

    이 클래스는 여러 Pydantic 모델에서 재사용할 수 있는 공통 필드들을 정의합니다.
    각 필드는 데이터의 유효성 검사를 위한 제약 조건과 메타데이터를 포함하며,
    모델 간의 일관성을 유지하고 중복 코드를 줄이는 데 사용됩니다.
    """
    db_id_set = Field(
        default = None,
        examples = ["123e4567-e89b-12d3-a456-426614174000"],
        title = "케릭터 DB ID",
        description = "캐릭터의 고유 식별자입니다. 데이터베이스에서 캐릭터를 식별하는 데 사용됩니다."
    )
    user_id_set = Field(
            examples = ["shaa97102"],
            title = "유저 id",
            min_length = 1, max_length = 50,
            description = "유저 id 길이 제약"
    )
    office_input_data_set = Field(
        examples = ["Llama AI 모델의 출시일과 버전들을 각각 알려줘."],
        title = "사용자 입력 문장",
        description = "사용자 입력 문장 길이 제약",
        min_length = 1
    )
    google_access_set = Field(
        examples = [False, True],
        default = False,
        title = "검색 기반 액세스",
        description = "검색 기반 액세스 수준을 나타냅니다. True: 검색 기반 활성화. False: 검색 기반 제한됨."
    )
    office_output_data_set = Field(
        examples = ['''
        물론이죠! Llama AI 모델의 출시일과 버전들은 다음과 같습니다:

        1. Llama 1: 2023년 출시1
        2. Llama 2: 2024년 6월 1일 출시2
        3. Llama 3: 2024년 7월 23일 출시3
        4. Llama 3.1: 2024년 7월 24일 출시4

        이 모델들은 Meta (구 Facebook)에서 개발한 AI 모델입니다.
        각 버전마다 성능과 기능이 개선되었습니다. 
        더 궁금한 점이 있으신가요?
        '''],
        title = "Llama 답변"
    )

class office_Request(BaseModel):
    """
    office 모델에 대한 요청 데이터를 정의하는 Pydantic 모델입니다.
    
    Attributes:
        input_data (str): 사용자의 입력 텍스트
        google_access (bool): 검색 기능 사용 여부
        db_id (uuid.UUID): 캐릭터의 DB ID
        user_id (str): 유저 id
    """
    input_data: str = CommonField.office_input_data_set
    google_access: bool = CommonField.google_access_set
    db_id: str | None = CommonField.db_id_set
    user_id: str | None = CommonField.user_id_set

    
    @model_validator(mode = "after")
    def validate_db_id_and_user_id(self):
        if self.db_id is None and self.user_id is None:
            return self

        if self.db_id is not None and self.user_id is not None:
            Validators.validate_uuid(self.db_id)
            return self

        missing_field  =  'db_id' if self.db_id is None else 'user_id'
        raise ValueError(f"{missing_field}가 누락되었습니다. 두 필드 모두 제공해야 합니다.")

class office_Response(BaseModel):
    """
    office 모델의 응답 데이터를 정의하는 Pydantic 모델입니다.
    
    Attributes:
        output_data (str): 모델이 생성한 응답 텍스트
    """
    output_data: str = CommonField.office_output_data_set
