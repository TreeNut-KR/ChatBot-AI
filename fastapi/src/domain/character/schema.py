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
            description = "유저 id 길이 제약",
            min_length = 1, max_length = 50
    )
    user_name_set = Field(
        examples = ["리트넛"],
        title = "유저 이름",
        description = "유저의 이름입니다. 봇과의 대화에서 사용자의 정체성을 나타냅니다.",
        min_length = 1, max_length = 50
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
    character_input_data_set = Field(
        examples = ["*엘리스에게 다가가서 말을 건다* 안녕?"],
        title = "character 사용자 입력 문장",
        description = "character 사용자 입력 문장 길이 제약",
        min_length = 1
    )
    character_name_set = Field(
        examples = ["엘리스"],
        title = "케릭터 이름",
        description = "캐릭터의 이름입니다. 봇의 정체성을 나타내며, 사용자가 이 이름으로 봇을 부릅니다.",
        min_length = 1
    )
    greeting_set = Field(
        examples=[
            (
                "*햇살이 부드럽게 비치는 정원에서 엘리스가 책을 읽으며 미소 짓고 있습니다.*"
                "*금발 머리카락이 리본 아래에서 반짝이며, 그녀의 파란 드레스는 바람에 살짝 흔들립니다. "
                "그녀의 곁에는 빨간 곰 인형이 놓여 있습니다.*"
                "\"안녕하세요! 오늘은 어떤 이야기를 함께 만들어볼까요? 상상 속에서라면 뭐든 가능하답니다!\""
                "*그녀의 눈동자는 호기심으로 반짝이며, 새로운 모험을 기대하는 듯합니다.*"
            )
        ],
        title="케릭터 인사말",
        description="사용자가 봇과 상호작용을 시작할 때 표시되는 인사말입니다. 봇의 성격과 의도를 반영합니다.",
        min_length=1
    )
    context_set = Field(
        examples=[
            (
                "엘리스는 17세의 호기심 많고 상상력이 풍부한 소녀입니다. 그녀는 동화 속에서 튀어나온 듯한 매력을 지니고 있으며, "
                "항상 새로운 모험을 꿈꿉니다."
                "키는 160cm이며, 금발의 긴 머리카락과 파란 눈동자를 가지고 있습니다. 그녀는 리본 장식이 달린 드레스를 즐겨 입으며, "
                "항상 곰 인형을 곁에 두고 다닙니다."
                "성격은 온화하고 다정하며, 누구와도 쉽게 친해질 수 있는 친화력을 가지고 있습니다. 그녀는 상상 속에서 새로운 이야기를 "
                "만들어내는 것을 좋아하며, 주변 사람들에게도 그 즐거움을 나누고 싶어 합니다."
                "취미는 독서, 정원에서 산책하기, 그리고 곰 인형과 함께 새로운 이야기를 상상하는 것입니다. 그녀는 특히 동화책을 좋아하며, "
                "그 속에서 영감을 얻어 자신만의 세계를 만들어갑니다."
                "엘리스는 주변 사람들에게 따뜻함과 희망을 전하며, 함께 있는 것만으로도 행복을 느끼게 하는 특별한 존재입니다."
            )
        ],
        title="케릭터 설정 값",
        description="캐릭터의 성격이나 태도를 나타냅니다. 이는 봇이 대화에서 어떻게 행동하고 응답할지를 정의합니다.",
        min_length=1
    )
    image_set = Field(
        examples = ["https://drive.google.com/thumbnail?id=1Ok6_Dq4R5aD1civ_BaFy5UvX3hepg7uS"],
        title = "케릭터 이미지 URL",
        description = "URL의 최대 길이는 일반적으로 2048자",
        min_length = 1, max_length = 2048
    )
    access_level_set = Field(
        examples = [True, False],
        default = True,
        title = "케릭터 액세스",
        description = "봇의 액세스 수준을 나타냅니다. True: 특정 기능이나 영역에 대한 접근 권한이 허용됨. False: 제한됨."
    )
    character_output_data_set = Field(
        examples = [
            (
                "*엘리스가 책을 덮고 환하게 웃으며 고개를 듭니다.*"
                "안녕! 이렇게 먼저 인사해줘서 정말 기뻐요 😊 오늘은 어떤 기분인가요?"
                "혹시 저랑 같이 이야기하고 싶은 게 있나요? 아니면 궁금한 게 있나요? 무엇이든 편하게 말해줘요!"
                "*곰 인형도 반갑다는 듯이 팔을 들어 인사합니다.*"
            )
        ],
        title = "character 답변"
    )

class character_Request(BaseModel):
    """
    character 모델에 대한 요청 데이터를 정의하는 Pydantic 모델입니다.
    
    Attributes:
        input_data (str): 사용자의 입력 텍스트
        character_name (str): 캐릭터의 이름
        greeting (str): 캐릭터의 인사말
        context (str): 캐릭터의 설정 정보
        db_id (uuid.UUID): 캐릭터의 DB ID
        user_id (str): 유저 id
    """
    input_data: str = CommonField.character_input_data_set
    character_name: str = CommonField.character_name_set
    greeting: str = CommonField.greeting_set
    context: str = CommonField.context_set
    db_id: str | None = CommonField.db_id_set
    user_id: str | None = CommonField.user_id_set
    user_name: str = CommonField.user_name_set

    @model_validator(mode = "after")
    def validate_db_id_and_user_id(self):
        if self.db_id is None and self.user_id is None:
            return self

        if self.db_id is not None and self.user_id is not None:
            Validators.validate_uuid(self.db_id)
            return self
        
        missing_field  =  'db_id' if self.db_id is None else 'user_id'
        raise ValueError(f"{missing_field}가 누락되었습니다. 두 필드 모두 제공해야 합니다.")

class character_Response(BaseModel):
    """
    character 모델의 응답 데이터를 정의하는 Pydantic 모델입니다.
    
    Attributes:
        output_data (str): 모델이 생성한 응답 텍스트
    """
    output_data: str = CommonField.character_output_data_set