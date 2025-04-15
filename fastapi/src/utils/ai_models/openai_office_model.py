'''
파일은 OpenAIChatModel, CharacterPrompt 클래스를 정의하고 OpenAI API를 사용하여,
'대화형 인공지능 서비스' 용도의 기능을 제공합니다.
'''
import os
import json
import warnings
from typing import Generator, List, Dict
from queue import Queue
from threading import Thread
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

class CharacterPrompt:
    def __init__(self, name: str, context: str, search_text: str) -> None:
        """
        초기화 메소드

        Args:
            name (str): 캐릭터 이름
            context (str): 캐릭터 설정
            search_text (str): 검색 텍스트
        """
        self.name = name
        self.context = context
        self.search_text = search_text

    def __str__(self) -> str:
        """
        문자열 출력 메소드
        
        Returns:
            str: 캐릭터 정보 문자열
        """
        return (
            f"Name: {self.name}\n"
            f"Context: {self.context}\n"
            f"Search Text: {self.search_text}"
        )

def build_openai_messages(character: CharacterPrompt, user_input: str, chat_list: List[Dict] = None) -> list:
    """
    캐릭터 정보와 대화 기록을 포함한 OpenAI API messages 형식 생성

    Args:
        character (CharacterPrompt): 캐릭터 정보
        user_input (str): 사용자 입력
        chat_list (List[Dict], optional): 이전 대화 기록

    Returns:
        list: OpenAI API 형식의 messages 리스트
    """
    system_prompt = (
        f"당신은 {character.name}입니다.\n"
        f"설정: {character.context}\n"
        f"다음 검색 정보를 참조하세요: {character.search_text}"
    )
    
    # 메시지 구성
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # 이전 대화 기록 추가
    if chat_list and len(chat_list) > 0:
        for chat in chat_list:
            user_message = chat.get("input_data", "")
            assistant_message = chat.get("output_data", "")
            
            if user_message:
                messages.append({"role": "user", "content": user_message})
            if assistant_message:
                messages.append({"role": "assistant", "content": assistant_message})
    
    # 현재 사용자 입력 추가
    messages.append({"role": "user", "content": user_input})
    
    return messages

class OpenAIChatModel:
    """
    [<img src="https://brandingstyleguides.com/wp-content/guidelines/2025/02/openAi-web.jpg" width="100" height="auto">](https://platform.openai.com/docs/models)
    
    OpenAI API를 사용하여 대화를 생성하는 클래스입니다.
    
    모델 정보:
    - 모델명: gpt-4o-mini, gpt-4.1, gpt-4.1-mini
    - 제작자: OpenAI
    - 소스: [OpenAI API](https://platform.openai.com/docs/models)
    """
    def __init__(self, model_id = 'gpt-4o-mini') -> None:
        """
        OpenAIChatModel 클래스 초기화 메소드
        """
        self.model_id = model_id
        self.file_path = './models/config-OpenAI.json'
        
        # 환경 변수 파일 경로 설정 수정
        current_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_file_path = os.path.join(current_directory, '.env')

        if not os.path.exists(env_file_path):
            raise FileNotFoundError(f".env 파일을 찾을 수 없습니다: {env_file_path}")

        load_dotenv(env_file_path)
        
        # JSON 파일 읽기
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
        except FileNotFoundError:
            print(f"설정 파일을 찾을 수 없습니다: {self.file_path}")
            # 기본 설정 사용
            self.data = {
                "character_name": "GPT 어시스턴트",
                "character_setting": "친절하고 도움이 되는 AI 어시스턴트입니다.",
                "greeting": "안녕하세요! 무엇을 도와드릴까요?"
            }
            
        # API 키 설정
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            
        # OpenAI 클라이언트 초기화
        self.client = self._init_client()
        self.response_queue = Queue()

    def _init_client(self) -> OpenAI:
        """
        OpenAI 클라이언트를 초기화합니다.
        
        Returns:
            OpenAI: 초기화된 OpenAI 클라이언트 인스턴스
        """
        try:
            return OpenAI(api_key=self.api_key)
        except Exception as e:
            print(f"❌ OpenAI 클라이언트 초기화 중 오류 발생: {e}")
            raise
            
    def _stream_completion(self, messages: list, **kwargs) -> None:
        """
        텍스트 생성을 위한 내부 스트리밍 메서드입니다.
        
        Args:
            messages (list): OpenAI API에 전달할 메시지 목록
            **kwargs: 생성 매개변수 (temperature, top_p 등)
            
        Effects:
            - response_queue에 생성된 텍스트 조각들을 순차적으로 추가
            - 스트림 종료 시 None을 큐에 추가
        """
        try:
            # 경고 메시지 필터링
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                stream = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    stream=True,
                    **kwargs
                )
                
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        content = chunk.choices[0].delta.content
                        if content is not None:
                            self.response_queue.put(content)
                
                self.response_queue.put(None)  # 스트림 종료 신호
                
        except Exception as e:
            print(f"스트리밍 중 오류 발생: {e}")
            self.response_queue.put(None)

    def create_streaming_completion(self,
                                   messages: list,
                                   max_tokens: int = 1000,
                                   temperature: float = 0.82,
                                   top_p: float = 0.95) -> Generator[str, None, None]:
        """
        스트리밍 방식으로 텍스트 응답을 생성하는 메서드입니다.
        
        Args:
            messages (list): OpenAI API에 전달할 메시지 목록
            max_tokens (int, optional): 생성할 최대 토큰 수. 기본값: 1000
            temperature (float, optional): 샘플링 온도 (0~2). 기본값: 0.82
            top_p (float, optional): 누적 확률 임계값 (0~1). 기본값: 0.95
            
        Returns:
            Generator[str, None, None]: 생성된 텍스트 조각들의 제너레이터
        """
        # kwargs 딕셔너리로 파라미터 전달
        kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        # 스트리밍 스레드 시작
        thread = Thread(
            target=self._stream_completion,
            args=(messages,),
            kwargs=kwargs
        )
        thread.start()

        # 응답 스트리밍
        while True:
            text = self.response_queue.get()
            if text is None:  # 스트림 종료
                break
            yield text

    def generate_response_stream(self, input_text: str, search_text: str, chat_list: List[Dict]) -> Generator[str, None, None]:
        """
        API 호환을 위한 스트리밍 응답 생성 메서드

        Args:
            input_text (str): 사용자 입력 텍스트
            search_text (str): 검색 결과 텍스트
            chat_list (List[Dict]): 이전 대화 기록

        Returns:
            Generator[str, None, None]: 생성된 텍스트 조각들을 반환하는 제너레이터
        """
        try:
            # 현재 시간 정보 추가
            current_time = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
            time_info = f"현재 시간은 {current_time}입니다.\n\n"
            
            # search_text가 비어있으면 시간 정보만 추가, 그렇지 않으면 시간 정보와 검색 결과 결합
            enhanced_search_text = time_info + (search_text if search_text else "")
            
            # 이스케이프 문자 정규화
            normalized_chat_list = []
            if chat_list and len(chat_list) > 0:
                for chat in chat_list:
                    normalized_chat = {
                        "index": chat.get("index"),
                        "input_data": chat.get("input_data"),
                        "output_data": self._normalize_escape_chars(chat.get("output_data", ""))
                    }
                    normalized_chat_list.append(normalized_chat)
            else:
                normalized_chat_list = chat_list
            
            character_info = CharacterPrompt(
                name=self.data.get("character_name"),
                context=self.data.get("character_setting"),
                search_text=enhanced_search_text,
            )

            # OpenAI API 메시지 형식 생성
            messages = build_openai_messages(
                character_info,
                input_text,
                normalized_chat_list
            )
            
            # 스트리밍 응답 생성
            for text_chunk in self.create_streaming_completion(
                messages=messages,
                max_tokens=1000,
                temperature=0.82,
                top_p=0.95
            ):
                yield text_chunk

        except Exception as e:
            print(f"응답 생성 중 오류 발생: {e}")
            yield f"오류: {str(e)}"

    def _normalize_escape_chars(self, text: str) -> str:
        """
        이스케이프 문자가 중복된 문자열을 정규화합니다
        """
        if not text:
            return ""
            
        # 이스케이프된 개행문자 등을 정규화
        result = text.replace("\\n", "\n")
        result = result.replace("\\\\n", "\n")
        result = result.replace('\\"', '"')
        result = result.replace("\\\\", "\\")
        
        return result

if __name__ == "__main__":
    model = OpenAIChatModel()
    
    try:
        def get_display_width(text: str) -> int:
            import wcwidth
            """주어진 문자열의 터미널 표시 너비를 계산"""
            return sum(wcwidth.wcwidth(char) for char in text)

        # 박스 크기 설정
        box_width = 50

        # 박스 생성
        print(f"╭{'─' * box_width}╮")

        # 환영 메시지 정렬
        title = "👋 환영합니다!"
        title_width = get_display_width(title)
        title_padding = (box_width - title_width) // 2
        print(f"│{' ' * title_padding}{title}{' ' * (box_width - title_width - title_padding)}│")

        # 인사말 가져오기 및 정렬
        greeting = f"🤖 : {model.data.get('greeting', '안녕하세요! 무엇을 도와드릴까요?')}"
        greeting_width = get_display_width(greeting)
        greeting_padding = (box_width - greeting_width) // 2
        print(f"│{' ' * greeting_padding}{greeting}{' ' * (box_width - greeting_width - greeting_padding)}│")

        print(f"╰{'─' * box_width}╯\n")
        
        while True:
            user_input = input("🗨️  user : ")
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("\n👋 대화를 종료합니다. 좋은 하루 되세요!")
                break
                
            print("🤖  bot : ", end='', flush=True)
            
            for text_chunk in model.generate_response_stream(user_input, search_text="", chat_list=[]):
                print(text_chunk, end='', flush=True)
            print("")
            print("\n" + "─"*50 + "\n")
            
    except KeyboardInterrupt:
        print("\n\n👋 프로그램이 안전하게 종료되었습니다.")
    except Exception as e:
        print(f"\n⚠️ 오류 발생: {e}")