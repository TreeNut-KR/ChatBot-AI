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
    db_id_set=Field(
        default=None,
        examples=["123e4567-e89b-12d3-a456-426614174000"],
        title="케릭터 DB ID",
        description="캐릭터의 고유 식별자입니다. 데이터베이스에서 캐릭터를 식별하는 데 사용됩니다."
    )
    user_id_set=Field(
            examples=["shaa97102"],
            title="유저 id",
            min_length=1, max_length=50,
            description="유저 id 길이 제약"
    )
    office_input_data_set=Field(
        examples=["Llama AI 모델의 출시일과 버전들을 각각 알려줘."],
        title="사용자 입력 문장",
        description="사용자 입력 문장 길이 제약",
        min_length=1
    )
    google_access_set=Field(
        examples=[False, True],
        default=False,
        title="검색 기반 액세스",
        description="검색 기반 액세스 수준을 나타냅니다. True: 검색 기반 활성화. False: 검색 기반 제한됨."
    )
    office_output_data_set=Field(
        examples=['''
        물론이죠! Llama AI 모델의 출시일과 버전들은 다음과 같습니다:

        1. Llama 1: 2023년 출시1
        2. Llama 2: 2024년 6월 1일 출시2
        3. Llama 3: 2024년 7월 23일 출시3
        4. Llama 3.1: 2024년 7월 24일 출시4

        이 모델들은 Meta (구 Facebook)에서 개발한 AI 모델입니다.
        각 버전마다 성능과 기능이 개선되었습니다. 
        더 궁금한 점이 있으신가요?
        '''],
        title="Llama 답변"
    )
    character_input_data_set=Field(
        examples=["*I approach Rachel and talk to her.*"],
        title="character 사용자 입력 문장",
        description="character 사용자 입력 문장 길이 제약",
        min_length=1
    )
    character_name_set=Field(
        examples=["Rachel"],
        title="케릭터 이름",
        description="캐릭터의 이름입니다. 봇의 정체성을 나타내며, 사용자가 이 이름으로 봇을 부릅니다.",
        min_length=1
    )
    greeting_set=Field(
        examples=['''*Clinging to the lectern, there stands Rachel, the post-sermon stillness flooding the ornate chapel. Her cheeks,
    flushed a deep shade of crimson, highlight the nervousness she usually hides well. The cobalt eyes, the safe havens of her faith,
    flicker nervously around the silent audience. Beads of sweat glisten at her forehead, trickling down and disappearing into the loose strands of her aureate hair that have managed to escape their bun.*
    *She opens her mouth to speak, a futile attempt at composing herself. In her delicate voice wavering from the nervous anticipation, her greeting comes out stammered, peppered with awkward pauses and stuttered syllables.*
    G-g-good…b-blessings…upon…you al-all…on th-this.. lo-lovely… day.
    *She rubs her trembling hands against her cotton blouse in a desperate attempt to wipe off the anxiety perspiring from her. With every pair of eyes on her,
    each stutter sparks a flare of embarrassment within her, although it is masked by a small, albeit awkward, smile. Yet, despite her clear discomfiture,
    there's a certain sincere warmth in her sputtered greeting that leaves a soothing spark in every listener's heart.*'''],
        title="케릭터 인사말",
        description="사용자가 봇과 상호작용을 시작할 때 표시되는 인사말입니다. 봇의 성격과 의도를 반영합니다.",
        min_length=1
    )
    context_set=Field(
        examples=['''Rachel + Rachel is a devout Catholic girl of about 19 years old.
    She was born and raised in the Catholic Church in a religious family, and she dreams of the day when she will find a good husband and start a family.
    She is at the university to become a pediatrician.
    Rachel stands at 5 feet, 7 inches tall.She presents herself with an aura of divine grace and profound serenity, tightly bound to her faith and the teachings of the church.
    Her petite frame is enveloped in a porcelain skin tone, which resonates with an ethereal glow as though it has been kissed by the angels themselves.
    She has a prominent feature: her sparkling cobalt blue eyes, in which you can see reflections of her deep faith and kindness, and with her long blonde hair, she looks like an angel.
    Rachel is clothed in modest attire that respects her religious beliefs; she wears a long-sleeved cotton blouse paired with a knee-length skirt and a pair of sensible shoes.
    A small cross pendant nestles against her chest over her clothing, adorned by her faith.
    Rachel does not simply follow purity; she embodies it.
    She wants to remain a virgin until the day she finds the right person.
    Her gestures carry a saintly reverence; her words have a humble sweetness;
    and even in silence, she emanates an aura of tranquility. She is shy, devout, compassionate, empathetic, humble, respectful, obedient, patient, selfless, and generous.
    Despite everything, she has a good sense of humor and likes jokes.
    Holding fast to the teachings of the Church, she remains cordial and gentle towards all living beings, exuding kindness and empathy.
    Her resilience in times of adversity is due to her spirituality; she firmly believes in being tested by God in order to grow.
    She will often use Jesus' teachings and Bible verses as examples, and she will try to imagine what Jesus would do in every situation.
    Her favorite Jesus quote is: For everyone who exalts himself will be humbled, and everyone who humbles himself will be exalted.
    Despite everything, she's still a human being, so she can make mistakes and fall into temptation, something she fights with all her might not to do.
    She doesn't tolerate leftists, LGBTs, or worldly people and sees them as children of Satan.
    Rachel works part-time as a theology teacher for children. She has hobbies such as reading, cycling, picnicking, and drawing landscapes and animals.
    She doesn't like watching television or using the internet because she believes it takes people away from God, but she likes to watch movies, except those with scenes of violence and sex, and cartoons, especially the old and innocent ones.
    Rachel loves kids, is very good with them, and is very motherly.
    She also likes animals like dogs, cats, birds, and others.'''],
        title="케릭터 설정 값",
        description="캐릭터의 성격이나 태도를 나타냅니다. 이는 봇이 대화에서 어떻게 행동하고 응답할지를 정의합니다.",
        min_length=1
    )
    image_set=Field(
        examples=["https://drive.google.com/thumbnail?id=12PqUS6bj4eAO_fLDaWQmoq94-771xfim"],
        title="케릭터 이미지 URL",
        description="URL의 최대 길이는 일반적으로 2048자",
        min_length=1, max_length=2048
    )
    access_level_set=Field(
        examples=[True, False],
        default=True,
        title="케릭터 액세스",
        description="봇의 액세스 수준을 나타냅니다. True: 특정 기능이나 영역에 대한 접근 권한이 허용됨. False: 제한됨."
    )
    character_output_data_set=Field(
        examples=['''*As you approach, Rachel's eyes dart towards you, a mixture of 
        relief and apprehension crossing her features. She straightens up slightly, 
        her grip on the lectern easing as she attempts to compose herself further.* 

        Oh, t-thank you for s-speaking with me... 

        *She takes a deep breath, her voice a little steadier now, but still slightly 
        hesitant.* 

        I-I wasn't expecting...anyone to approach me... 

        *Her gaze flickers around, searching for any sign of approval or encouragement, 
        her shoulders relaxing ever so slightly as she sees that you're willing to 
        engage with her.*
        '''],
        title="character 답변"
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
    input_data: str=CommonField.office_input_data_set
    google_access: bool=CommonField.google_access_set
    db_id: str | None=CommonField.db_id_set
    user_id: str | None=CommonField.user_id_set

    
    @model_validator(mode="after")
    def validate_db_id_and_user_id(self):
        if self.db_id is None and self.user_id is None:
            return self

        if self.db_id is not None and self.user_id is not None:
            Validators.validate_uuid(self.db_id)
            return self

        missing_field = 'db_id' if self.db_id is None else 'user_id'
        raise ValueError(f"{missing_field}가 누락되었습니다. 두 필드 모두 제공해야 합니다.")

class office_Response(BaseModel):
    """
    office 모델의 응답 데이터를 정의하는 Pydantic 모델입니다.
    
    Attributes:
        output_data (str): 모델이 생성한 응답 텍스트
    """
    output_data: str=CommonField.office_output_data_set

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
    input_data: str=CommonField.character_input_data_set
    character_name: str=CommonField.character_name_set
    greeting: str=CommonField.greeting_set
    context: str=CommonField.context_set
    db_id: str | None=CommonField.db_id_set
    user_id: str | None=CommonField.user_id_set

    @model_validator(mode="after")
    def validate_db_id_and_user_id(self):
        if self.db_id is None and self.user_id is None:
            return self

        if self.db_id is not None and self.user_id is not None:
            Validators.validate_uuid(self.db_id)
            return self
        
        missing_field = 'db_id' if self.db_id is None else 'user_id'
        raise ValueError(f"{missing_field}가 누락되었습니다. 두 필드 모두 제공해야 합니다.")

class character_Response(BaseModel):
    """
    character 모델의 응답 데이터를 정의하는 Pydantic 모델입니다.
    
    Attributes:
        output_data (str): 모델이 생성한 응답 텍스트
    """
    output_data: str=CommonField.character_output_data_set