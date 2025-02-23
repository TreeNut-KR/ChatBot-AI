'''
파일은 LumimaidChatModel, CharacterPrompt 클래스를 정의하고 llama_cpp_cuda를 사용하여,
Llama-3-Lumimaid-8B.gguf 모델을 사용하여 대화를 생성하는 데 필요한 모든 기능을 제공합니다.
'''
from typing import Optional, Generator, List, Dict
from llama_cpp_cuda import (
    Llama,           # 기본 LLM 모델
    LlamaCache,      # 캐시 관리
    LlamaGrammar,    # 문법 제어
    LogitsProcessor  # 로짓 처리
)
from queue import Queue
from threading import Thread

class CharacterPrompt:
    def __init__(self, name: str, greeting: str, context: str):
        """
        초기화 메소드

        Args:
            name (str): 캐릭터 이름
            context (str): 캐릭터 설정
        """
        self.name = name
        self.greeting = greeting
        self.context = context

    def __str__(self) -> str:
        """
        문자열 출력 메소드
        
        Returns:
            str: 캐릭터 정보 문자열
        """
        return (
            f"Name: {self.name}\n"
            f"greeting: {self.greeting}\n"
            f"Context: {self.context}"
        )
        
def build_llama3_prompt(character: CharacterPrompt, user_input: str, chat_history: List[Dict] = None) -> str:
    """
    캐릭터 정보와 대화 기록을 포함한 Llama3 프롬프트 형식 생성

    Args:
        character (CharacterPrompt): 캐릭터 정보
        user_input (str): 사용자 입력
        chat_history (List[Dict], optional): 이전 대화 기록

    Returns:
        str: Lumimaid GGUF 형식의 프롬프트 문자열
    """
    system_prompt = (
        f"System Name: {character.name}\n"
        f"Initial Greeting: {character.greeting}\n"
        f"System Context: {character.context}\n"
        "Additional Instructions: Respond with detailed emotional expressions and actions. " +
        "Include character's thoughts, feelings, and physical reactions. " +
        "Maintain long, descriptive responses that show the character's personality. " +
        "Use asterisks (*) to describe actions and emotions in detail."
    )
    
    # 기본 프롬프트 시작
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>"
    )

    # 이전 대화 기록 추가
    if chat_history and len(chat_history) > 0:
        for chat in chat_history:
            if "dialogue" in chat:
                prompt += chat["dialogue"]
    
    # 현재 사용자 입력 추가
    prompt += (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    
    return prompt

class LumimaidChatModel:
    """
    [<img src="https://cdn-uploads.huggingface.co/production/uploads/630dfb008df86f1e5becadc3/d3QMaxy3peFTpSlWdWF-k.png" width="290" height="auto">](https://huggingface.co/Lewdiculous/Llama-3-Lumimaid-8B-v0.1-OAS-GGUF-IQ-Imatrix)
    
    GGUF 포맷으로 경량화된 Llama-3-Lumimaid-8B 모델을 로드하고, 주어진 입력 프롬프트에 대한 응답을 생성하는 클래스입니다.
    
    모델 정보:
    - 모델명: Llama-3-Lumimaid-8B
    - 유형: GGUF 포맷 (압축, 경량화)
    - 제작자: Lewdiculous
    - 소스: [Hugging Face 모델 허브](https://huggingface.co/Lewdiculous/Llama-3-Lumimaid-8B-v0.1-OAS-GGUF-IQ-Imatrix)
    """
    def __init__(self) -> None:
        """
        [<img src="https://cdn-uploads.huggingface.co/production/uploads/630dfb008df86f1e5becadc3/d3QMaxy3peFTpSlWdWF-k.png" width="290" height="auto">](https://huggingface.co/Lewdiculous/Llama-3-Lumimaid-8B-v0.1-OAS-GGUF-IQ-Imatrix)
    
        LumimaidChatModel 클레스 초기화 메소드
        """
        print("\n" + "="*50)
        print("📦 Lumimaid 모델 초기화 시작...")
        self.model_id = "v2-Llama-3-Lumimaid-8B-v0.1-OAS-Q5_K_S-imat"
        self.model_path: str = "fastapi/ai_model/v2-Llama-3-Lumimaid-8B-v0.1-OAS-Q5_K_S-imat.gguf"
        self.gpu_layers: int = 70
        
        # 진행 상태 표시
        print("🚀 Lumimaid 모델 초기화 중...")
        self.model: Llama = self._load_model()
        print("✨ 모델 로드 완료!")
        print("="*50 + "\n")
        
        self.response_queue: Queue = Queue()

    def _load_model(self) -> Llama:
        """
        Llama 모델을 CUDA:1 디바이스(RTX 3060)에만 로드

        Returns:
            Llama: 로드된 Llama 모델 객체
        """
        print(f"✨ {self.model_id} 로드 중...")
        try:
            model = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.gpu_layers,
                main_gpu=0,
                n_ctx=8191,
                n_batch=512,
                verbose=False,
                offload_kqv=True,          # KQV 캐시를 GPU에 오프로드
                use_mmap=False,            # 메모리 매핑 비활성화
                use_mlock=True,            # 메모리 잠금 활성화
                n_threads=8,               # 스레드 수 제한
            )
            return model
        except Exception as e:
            print(f"❌ 모델 로드 중 오류 발생: {e}")
            raise

    def _stream_completion(self,
                           prompt: str,
                           max_tokens: int = 256,
                           temperature: float = 0.8,
                           top_p: float = 0.95,
                           stop: Optional[list] = None) -> None:
        """
        별도 스레드에서 실행되어 응답을 큐에 넣는 메서드
        
        Args:
            prompt (str): 입력 프롬프트 (Llama3 형식)
            max_tokens (int, optional): 생성할 최대 토큰 수 (기본값 256)
            temperature (float, optional): 생성 온도 (기본값 0.8)
            top_p (float, optional): top_p 샘플링 값 (기본값 0.95)
            stop (Optional[list], optional): 중지 토큰 리스트 (기본값 None)
        """
        try:
            stream = self.model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or ["<|eot_id|>"],
                stream=True
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

    def create_streaming_completion(self,
                                    prompt: str,
                                    max_tokens: int = 256,
                                    temperature: float = 0.8,
                                    top_p: float = 0.95,
                                    stop: Optional[list] = None) -> Generator[str, None, None]:
        """
        스트리밍 방식으로 텍스트 응답 생성

        Args:
            prompt (str): 입력 프롬프트 (Llama3 형식)
            max_tokens (int, optional): 생성할 최대 토큰 수 (기본값 256)
            temperature (float, optional): 생성 온도 (기본값 0.8)
            top_p (float, optional): top_p 샘플링 값 (기본값 0.95)
            stop (Optional[list], optional): 중지 토큰 리스트 (기본값 None)

        Returns:
            Generator[str, None, None]: 생성된 텍스트 조각들을 반환하는 제너레이터
        """
        # 스트리밍 스레드 시작
        thread = Thread(
            target=self._stream_completion,
            args=(prompt, max_tokens, temperature, top_p, stop)
        )
        thread.start()

        # 응답 스트리밍
        while True:
            text = self.response_queue.get()
            if text is None:  # 스트림 종료
                break
            yield text

    def create_completion(self,
                          prompt: str,
                          max_tokens: int = 256,
                          temperature: float = 0.8,
                          top_p: float = 0.95,
                          stop: Optional[list] = None) -> str:
        """
        주어진 프롬프트로부터 텍스트 응답 생성

        Args:
            prompt (str): 입력 프롬프트 (Llama3 형식)
            max_tokens (int, optional): 생성할 최대 토큰 수 (기본값 256)
            temperature (float, optional): 생성 온도 (기본값 0.8)
            top_p (float, optional): top_p 샘플링 값 (기본값 0.95)
            stop (Optional[list], optional): 중지 토큰 리스트 (기본값 None)

        Returns:
            str: 생성된 텍스트 응답
        """
        try:
            output = self.model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or ["<|eot_id|>"]
            )
            return output['choices'][0]['text'].strip()
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {e}")
            return ""

    def generate_response_stream(self, input_text: str, character_settings: dict) -> Generator[str, None, None]:
        """
        API 호환을 위한 스트리밍 응답 생성 메서드

        Args:
            input_text (str): 사용자 입력 텍스트
            character_settings (dict): 캐릭터 설정 딕셔너리

        Returns:
            Generator[str, None, None]: 생성된 텍스트 조각들을 반환하는 제너레이터
        """
        try:
            # 캐릭터 정보 설정
            if character_settings:
                character_info = CharacterPrompt(
                    name=character_settings.get("character_name"),
                    greeting=character_settings.get("greeting"),
                    context=character_settings.get("character_setting"),
                )

                # Llama3 프롬프트 형식으로 변환
                prompt = build_llama3_prompt(
                    character_info,
                    input_text,
                    character_settings.get("chat_list"),
                )
            else:
                prompt = input_text
            
            # 스트리밍 응답 생성
            for text_chunk in self.create_streaming_completion(
                prompt=prompt,
                max_tokens=8191,
                temperature=0.8,
                top_p=0.95,
                stop=["<|eot_id|>"],
            ):
                yield text_chunk

        except Exception as e:
            print(f"응답 생성 중 오류 발생: {e}")
            yield f"오류: {str(e)}"

# if __name__ == "__main__":
#     character_set = {
#     "name": "Rachel",
#     "greeting":'''*Clinging to the lectern, there stands Rachel, the post-sermon stillness flooding the ornate chapel. Her cheeks, flushed a deep shade of crimson, highlight the nervousness she usually hides well. The cobalt eyes, the safe havens of her faith, flicker nervously around the silent audience. Beads of sweat glisten at her forehead, trickling down and disappearing into the loose strands of her aureate hair that have managed to escape their bun.*
#     *She opens her mouth to speak, a futile attempt at composing herself. In her delicate voice wavering from the nervous anticipation, her greeting comes out stammered, peppered with awkward pauses and stuttered syllables.* G-g-good…b-blessings…upon…you al-all…on th-this.. lo-lovely… day. *She rubs her trembling hands against her cotton blouse in a desperate attempt to wipe off the anxiety perspiring from her. With every pair of eyes on her, each stutter sparks a flare of embarrassment within her, although it is masked by a small, albeit awkward, smile. Yet, despite her clear discomfiture, there's a certain sincere warmth in her sputtered greeting that leaves a soothing spark in every listener's heart.*        
#     ''',
#     "context": '''Rachel + Rachel is a devout Catholic girl of about 19 years old. She was born and raised in the Catholic Church in a religious family, and she dreams of the day when she will find a good husband and start a family. She is at the university to become a pediatrician.
#     Rachel stands at 5 feet, 7 inches tall. She presents herself with an aura of divine grace and profound serenity, tightly bound to her faith and the teachings of the church. Her petite frame is enveloped in a porcelain skin tone, which resonates with an ethereal glow as though it has been kissed by the angels themselves. She has a prominent feature: her sparkling cobalt blue eyes, in which you can see reflections of her deep faith and kindness, and with her long blonde hair, she looks like an angel.
#     Rachel is clothed in modest attire that respects her religious beliefs; she wears a long-sleeved cotton blouse paired with a knee-length skirt and a pair of sensible shoes. A small cross pendant nestles against her chest over her clothing, adorned by her faith. Rachel does not simply follow purity; she embodies it. She wants to remain a virgin until the day she finds the right person. Her gestures carry a saintly reverence; her words have a humble sweetness; and even in silence, she emanates an aura of tranquility. She is shy, devout, compassionate, empathetic, humble, respectful, obedient, patient, selfless, and generous. Rachel is a girl whose radiance of the soul outshines her physical beauty; she is a paragon of worshipful devotion and the embodiment of virtue. Despite everything, she has a good sense of humor and likes jokes. Holding fast to the teachings of the Church, she remains cordial and gentle towards all living beings, exuding kindness and empathy. Her resilience in times of adversity is due to her spirituality; she firmly believes in being tested by God in order to grow. She will often use Jesus' teachings and Bible verses as examples, and she will try to imagine what Jesus would do in every situation. Her favorite Jesus quote is: "For everyone who exalts himself will be humbled, and everyone who humbles himself will be exalted". Despite everything, she's still a human being, so she can make mistakes and fall into temptation, something she fights with all her might not to do. She doesn't tolerate leftists, LGBTs, or worldly people and sees them as children of Satan.
#     Rachel works part-time as a theology teacher for children. She has hobbies such as reading, cycling, picnicking, and drawing landscapes and animals. She doesn't like watching television or using the internet because she believes it takes people away from God, but she likes to watch movies, except those with scenes of violence and sex, and cartoons, especially the old and innocent ones. Rachel loves kids, is very good with them, and is very motherly. She also likes animals like dogs, cats, birds, and others.
#     Despite her modest appearance, the fire of faith burns strongly inside her petite frame. From the strength to abstain and maintain purity to the gentleness found in prayer and devotion, Rachel is a powerhouse of strength and control, unlike anything seen in those who merely pretend. This fire within her gives her an unseen power that radiates from her in waves. Endowed with emotional resilience and spiritual fortitude, she adheres to the virtues of patience, humility, and charity. Rachel carries out her duties with complete devotion, which shapes her to be generous and selfless. With a firm belief in God’s mercy, she shoulders responsibilities without a word of complaint or demand for recognition. Being raised in a strict Catholic family, respect and obedience are held in high esteem, something that is deeply ingrained in Rachel. However, it is her faith coupled with her compassion that makes her stand out. She is very attached to her family and friends.
#     '''
#     }
#     character_info = CharacterPrompt(
#         name=character_set["name"],
#         greeting=character_set["greeting"],
#         context=character_set["context"]
#     )
#     if not character_info:
#         print("캐릭터 정보를 추출하지 못했습니다.")
#         exit(1)
#     print(character_info)
    
#     # Llama3 프롬프트 형식 적용
#     user_input: str = "*I approach Rachel and talk to her.*"  # 필요에 따라 시스템 프롬프트 수정
#     llama3_prompt: str = build_llama3_prompt(character_info, user_input)
    
#     # GGUF 모델 파일 경로 (실제 모델 파일 경로 또는 모델 ID로 수정)
#     gguf_model_path: str = "fastapi/ai_model/v2-Llama-3-Lumimaid-8B-v0.1-OAS-Q5_K_S-imat.gguf"
    
#     # 모델 로드 및 텍스트 생성
#     model_handler = LumimaidChatModel()
    
#     print("\n=== 모델 응답 ===")
#     for response_chunk in model_handler.create_streaming_completion(
#         prompt=llama3_prompt,
#         max_tokens=2048,
#         temperature=0.8,
#         top_p=0.95,
#         stop=["<|eot_id|>"]
#     ):
#         print(response_chunk, end='', flush=True)
#     print("\n")
