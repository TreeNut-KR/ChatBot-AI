import json
import base64
from typing import Optional, Generator
from PIL import Image
from llama_cpp import Llama
from threading import Thread
from queue import Queue

class CharacterPrompt:
    def __init__(self, name: str, greeting: str, context: str):
        self.name = name
        self.greeting = greeting
        self.context = context

    def __str__(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Greeting: {self.greeting}\n"
            f"Context: {self.context}"
        )

class PNGPromptExtractor:
    """
    PNG 파일에서 캐릭터 프롬프트(예: "prompt" 또는 "description" 값)를 추출하는 클래스
    """
    def __init__(self, file_path: str) -> None:
        """
        초기화 메소드

        Args:
            file_path (str): PNG 이미지 파일 경로
        """
        self.file_path: str = file_path

    def extract(self) -> Optional[CharacterPrompt]:
        """
        PNG 파일의 메타데이터에서 캐릭터 정보 추출

        Returns:
            Optional[CharacterPrompt]: 추출된 캐릭터 정보, 실패 시 None 반환
        """
        try:
            img = Image.open(self.file_path)
            metadata = img.info
            
            if not metadata:
                print("❌ PNG 메타데이터가 존재하지 않습니다.")
                return None

            for key in metadata:
                if "chara" in key.lower():
                    decoded_data = base64.b64decode(metadata[key])
                    json_data = json.loads(decoded_data)

                    print("\n ==  =  추출된 JSON 데이터  ==  = ")
                    print(json.dumps(json_data, indent = 4, ensure_ascii = False), "\n")

                    # data 섹션에서 필요한 정보 추출
                    data = json_data.get("data", {})
                    name = data.get("name", "")
                    first_mes = data.get("first_mes", "")  # greeting으로 사용
                    description = data.get("description", "")  # context로 사용

                    if name and (first_mes or description):
                        return CharacterPrompt(
                            name = name,
                            greeting = first_mes,
                            context = description
                        )
                    else:
                        print("❌ JSON 내부에 필요한 캐릭터 정보가 부족합니다.")
                        return None

            print("❌ 'chara' 관련 메타데이터를 찾을 수 없습니다.")
            return None

        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
            return None

def build_llama3_prompt(character: CharacterPrompt, user_input: str) -> str:
    """
    캐릭터 정보를 포함한 Llama3 프롬프트 형식 생성

    Args:
        character (CharacterPrompt): 캐릭터 정보
        user_input (str): 사용자 입력

    Returns:
        str: Llama3 형식의 프롬프트 문자열
    """
    system_prompt = (
        f"Character Name: {character.name}\n"
        f"Character Context: {character.context}\n"
        f"Initial Greeting: {character.greeting}"
    )
    
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

class LlamaModelHandler:
    """
    GGUF 모델(Llama)을 로드하고 입력 프롬프트로부터 응답 텍스트를 생성하는 클래스
    """
    def __init__(self, model_path: str, verbose: bool = False, gpu_layers: int = 35) -> None:
        """
        초기화 메소드

        Args:
            model_path (str): GGUF 모델 파일 경로 또는 모델 ID
            verbose (bool, optional): 로드 시 로그 출력 여부 (기본값 False)
            gpu_layers (int, optional): GPU에 로드할 레이어 수 (기본값 35)
        """
        self.model_path: str = model_path
        self.verbose: bool = verbose
        self.gpu_layers: int = gpu_layers
        self.model: Llama = self._load_model()
        self.response_queue: Queue = Queue()

    def _load_model(self) -> Llama:
        """
        Llama 모델을 CUDA:1 디바이스에 로드

        Returns:
            Llama: 로드된 Llama 모델 객체
        """
        print("모델 로드 중...")
        try:
            model = Llama(
                model_path = self.model_path,
                verbose = self.verbose,
                n_ctx = 2048,
                n_threads = 4,
                n_gpu_layers = self.gpu_layers,  # GPU 레이어 활성화
                split_mode = 1,  # 1: 'nvidia cuda'
                main_gpu = 1,    # CUDA:1 디바이스 사용
                tensor_split = [0,1]  # 텐서를 GPU 0,1에 분할
            )
            print("✅ 모델이 CUDA:1에 성공적으로 로드되었습니다.")
            return model
        except Exception as e:
            print(f"❌ 모델 로드 중 오류 발생: {e}")
            raise

    def _stream_completion(self, prompt: str, max_tokens: int = 256,
                         temperature: float = 0.7, top_p: float = 0.95,
                         stop: Optional[list] = None) -> None:
        """
        별도 스레드에서 실행되어 응답을 큐에 넣는 메서드
        """
        try:
            stream = self.model.create_completion(
                prompt = prompt,
                max_tokens = max_tokens,
                temperature = temperature,
                top_p = top_p,
                stop = stop or ["<|eot_id|>"],
                stream = True
            )
            
            for output in stream:
                if 'choices' in output and len(output['choices']) > 0:
                    text = output['choices'][0].get('text', '')
                    if text:
                        self.response_queue.put(text)
            
            # 스트림 종료를 알리는 None 추가
            self.response_queue.put(None)
            
        except Exception as e:
            print(f"스트리밍 중 오류 발생: {e}")
            self.response_queue.put(None)

    def create_streaming_completion(self, prompt: str, max_tokens: int = 256,
                                 temperature: float = 0.7, top_p: float = 0.95,
                                 stop: Optional[list] = None) -> Generator[str, None, None]:
        """
        스트리밍 방식으로 텍스트 응답 생성

        Args:
            prompt (str): 입력 프롬프트
            max_tokens (int): 최대 토큰 수
            temperature (float): 생성 온도
            top_p (float): top_p 샘플링 값
            stop (Optional[list]): 중지 토큰 리스트

        Yields:
            str: 생성된 텍스트 조각들
        """
        # 스트리밍 스레드 시작
        thread = Thread(
            target = self._stream_completion,
            args = (prompt, max_tokens, temperature, top_p, stop)
        )
        thread.start()

        # 응답 스트리밍
        while True:
            text = self.response_queue.get()
            if text is None:  # 스트림 종료
                break
            yield text

    def create_completion(self, prompt: str, max_tokens: int = 256,
                          temperature: float = 0.7, top_p: float = 0.95,
                          stop: Optional[list] = None) -> str:
        """
        주어진 프롬프트로부터 텍스트 응답 생성

        Args:
            prompt (str): 입력 프롬프트 (Llama3 형식)
            max_tokens (int, optional): 생성할 최대 토큰 수 (기본값 256)
            temperature (float, optional): 생성 온도 (기본값 0.7)
            top_p (float, optional): top_p 샘플링 값 (기본값 0.95)
            stop (Optional[list], optional): 중지 토큰 리스트 (기본값 None)

        Returns:
            str: 생성된 텍스트 응답
        """
        try:
            output = self.model.create_completion(
                prompt = prompt,
                max_tokens = max_tokens,
                temperature = temperature,
                top_p = top_p,
                stop = stop or ["<|eot_id|>"]
            )
            return output['choices'][0]['text'].strip()
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {e}")
            return ""

if __name__  ==  "__main__":
    # PNG 파일 경로 (실제 파일 경로로 수정)
    png_file_path: str = "C:/Users/treen/Downloads/main_rachel-29118321_spec_v2.png"
    
    # PNG 파일에서 캐릭터 정보 추출
    extractor = PNGPromptExtractor(png_file_path)
    character_info: Optional[CharacterPrompt] = extractor.extract()
    
    if not character_info:
        print("PNG 파일에서 캐릭터 정보를 추출하지 못했습니다.")
        exit(1)
    
    print("✅ PNG에서 추출된 캐릭터 정보:")
    print(character_info)
    
    # Llama3 프롬프트 형식 적용
    user_input: str = "what your name?"  # 필요에 따라 시스템 프롬프트 수정
    llama3_prompt: str = build_llama3_prompt(character_info, user_input)
    
    # GGUF 모델 파일 경로 (실제 모델 파일 경로 또는 모델 ID로 수정)
    gguf_model_path: str = "fastapi/ai_model/v2-Llama-3-Lumimaid-8B-v0.1-OAS-Q5_K_S-imat.gguf"
    
    # 모델 로드 및 텍스트 생성
    model_handler = LlamaModelHandler(gguf_model_path, verbose = False)
    
    print("\n ==  =  모델 응답  ==  = ")
    for response_chunk in model_handler.create_streaming_completion(
        prompt = llama3_prompt,
        max_tokens = 2048,
        temperature = 0.7,
        top_p = 0.95,
        stop = ["<|eot_id|>"]
    ):
        print(response_chunk, end = '', flush = True)
    print("\n")
