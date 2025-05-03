import os
import warnings
import numpy as np
from queue import Queue
from openai import OpenAI
from threading import Thread
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


class ChatMemory:
    def __init__(self):
        self.chat_logs = []  # 대화 저장소
        self.embeddings = []  # 임베딩 저장소
        self.ai_model = OpenAIHandler()  # OpenAI 모델 통합
        self.embedding_cache = {}  # 임베딩 캐싱 추가
        
    def _get_embedding(self, text):
        """OpenAI 임베딩 API를 사용하여 텍스트를 벡터화 (캐싱 적용)"""
        # 캐시에 있으면 캐시에서 반환
        if text in self.embedding_cache:
            return self.embedding_cache[text]
            
        try:
            response = self.ai_model.client.embeddings.create(
                input = text,
                model = "text-embedding-3-small"
            )
            embedding = np.array(response.data[0].embedding)
            
            # 결과 캐싱
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"임베딩 중 오류 발생: {e}")
            return np.zeros(1536)
    
    def add_chat(self, text, response = None):
        """대화를 저장하고 벡터화하여 저장"""
        if response:
            # 사용자 메시지와 AI 응답을 함께 저장
            full_text = f"{text}, {response}"
        else:
            # 하나의 텍스트만 저장 (기존 방식)
            full_text = text
            
        # OpenAI 임베딩 모델을 사용하여 벡터화
        embedding = self._get_embedding(full_text)
        self.chat_logs.append(full_text)
        self.embeddings.append(embedding)

    def search_similar_chat(self, query, top_k = 3):
        """OpenAI 임베딩 기반 코사인 유사도 검색"""
        if not self.chat_logs:
            return []

        # 쿼리 임베딩 생성
        query_embedding = self._get_embedding(query)
        
        # 코사인 유사도 계산
        if len(self.embeddings) > 0:
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            
            # 유사도에 따라 인덱스 정렬
            sorted_indices = np.argsort(similarities)[::-1]  # 내림차순 정렬
            
            # 상위 결과 반환
            top_indices = sorted_indices[:top_k]
            candidates = [self.chat_logs[i] for i in top_indices]
            
            # 유사도 점수가 매우 비슷한 경우 재순위화 필요 여부 확인
            if len(top_indices) > 1 and self._needs_reranking(similarities[top_indices]):
                return self._gpt_rerank_candidates(query, candidates, top_k)
            
            return candidates
        return []

    def _needs_reranking(self, similarities):
        """재순위화가 필요한지 결정 (유사도가 비슷하면 재순위화 필요)"""
        if len(similarities) <=  1:
            return False
        
        # 상위 결과들의 유사도 차이가 작으면 재순위화 필요
        similarity_diff = similarities[0] - similarities[1]
        return similarity_diff < 0.05  # 임계값 조정 가능

    def _gpt_rerank_candidates(self, query, candidates, top_k = 3):
        """OpenAI API를 사용하여 후보 결과를 재순위화"""
        if not candidates:
            return []
        
        try:
            # 재순위화를 위한 프롬프트 구성
            system_prompt = "당신은 검색 결과 랭킹 전문가입니다. 주어진 쿼리에 가장 관련성이 높은 결과를 선택해주세요."
            
            # 후보들을 하나의 문자열로 결합
            candidates_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
            
            # 사용자 프롬프트 구성
            user_prompt = f"다음 쿼리에 가장 관련성이 높은 결과를 순위대로 나열해주세요:\n\n쿼리: {query}\n\n후보 결과:\n{candidates_text}\n\n결과 순위(숫자만 쉼표로 구분하여 나열):"
            
            # OpenAI API 호출
            response = self.ai_model.client.chat.completions.create(
                model = "gpt-4o-mini",
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens = 50,
                temperature = 0.2
            )
            
            # 응답 파싱
            content = response.choices[0].message.content.strip()
            
            # 숫자만 추출하기 위한 처리
            import re
            ranking = re.findall(r'\d+', content)
            
            # 숫자를 인덱스로 변환 (1-indexed -> 0-indexed)
            ranking = [int(r) - 1 for r in ranking if 0 < int(r) <=  len(candidates)]
            
            # 순위에 따라 결과 재정렬
            reranked_candidates = []
            for idx in ranking:
                if idx < len(candidates):
                    reranked_candidates.append(candidates[idx])
            
            # 누락된 항목 추가
            for i, candidate in enumerate(candidates):
                if i not in ranking and candidate not in reranked_candidates:
                    reranked_candidates.append(candidate)
            
            # top_k 개수만큼 반환
            return reranked_candidates[:top_k]
        
        except Exception as e:
            print(f"재순위화 중 오류 발생: {e}")
            # 오류 발생 시 원래 순서 반환
            return candidates[:top_k]

    def generate_response(self, user_query):
        """유사한 대화를 찾아 컨텍스트로 활용하여 응답 생성"""
        # 유사한 대화 검색
        similar_chats = self.search_similar_chat(user_query, top_k = 3)
        
        # 컨텍스트 구성
        context = "\n".join(similar_chats) if similar_chats else "관련 대화 기록이 없습니다."
        
        # OpenAI 모델을 통해 응답 생성
        response = self.ai_model.generate_response(user_query, context)
        
        return response


class OpenAIHandler:
    def __init__(self):
        """OpenAI API 핸들러 초기화"""
        # 환경 변수 파일 경로 설정
        current_directory = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(os.path.dirname(current_directory), '.env')
        
        load_dotenv(env_path)
        
        # API 키 설정
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key = self.api_key)
        self.model_id = 'gpt-4o-mini'
        self.response_queue = Queue()
        
    def _stream_completion(self, messages: list, **kwargs) -> None:
        """텍스트 생성을 위한 내부 스트리밍 메서드"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                stream = self.client.chat.completions.create(
                    model = self.model_id,
                    messages = messages,
                    stream = True,
                    **kwargs
                )
                
                full_response = ""
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        content = chunk.choices[0].delta.content
                        if content is not None:
                            full_response +=  content
                
                self.response_queue.put(full_response)
                
        except Exception as e:
            print(f"스트리밍 중 오류 발생: {e}")
            self.response_queue.put("오류가 발생했습니다. 다시 시도해주세요.")

    def generate_response(self, user_query, context):
        """컨텍스트 정보를 기반으로 응답 생성"""
        system_prompt = (
            "당신은 전문적이고 신뢰할 수 있는 비즈니스 파트너입니다. "
            "아래 제공된 기억 정보를 참고하여 자연스럽게 대화를 이어나가세요. "
            "비즈니스 맥락에 맞게 전문적이면서도 친절하게 응답하되, 제공된 기억을 적절히 언급하세요."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"다음은 우리의 이전 대화 기억입니다:\n{context}\n\n사용자 질문: {user_query}"}
        ]
        
        # 응답 생성 스레드 시작
        thread = Thread(
            target = self._stream_completion,
            args = (messages,),
            kwargs = {
                "max_tokens": 300,
                "temperature": 0.7
            }
        )
        thread.start()
        thread.join()  # 스레드 완료 대기
        
        # 응답 반환
        response = self.response_queue.get()
        return response
    
chat_memory = ChatMemory()

# 협상 대화
chat_memory.add_chat("*나는 회의실에서 거래처 담당자를 맞이하며 미소지으며 말한다* \"오늘 협상을 위해 시간 내주셔서 감사합니다.\", *상대방은 정중하게 고개를 끄덕이며 말한다* \"네, 좋은 결과가 있었으면 합니다.\"")
chat_memory.add_chat("*자료를 펼치며 진지한 표정으로 말한다* \"저희가 제안드리는 공급 단가는 개당 5,200원입니다. 대량 주문시 추가 할인도 가능합니다.\", *상대방은 자료를 검토하며 대답한다* \"현재 시장 가격보다 조금 높은데, 4,800원까지 가능할까요?\"")
chat_memory.add_chat("*잠시 생각하는 표정을 짓다가 대안을 제시한다* \"5,000원으로 하고, 3년 장기 계약을 체결하면 어떨까요? 안정적인 공급을 약속드립니다.\", *상대방이 동료와 잠시 상의한 후 고개를 끄덕인다* \"좋습니다. 장기 계약이라면 그 조건에 동의합니다.\"")
chat_memory.add_chat("*창가 쪽 테이블로 자리를 옮기며 계약서를 펼친다* \"이번 협상은 양사에 모두 윈윈이 될 것 같습니다. 특히 지속가능한 소재 사용에 합의한 점이 중요합니다.\", *상대방도 만족스러운 표정으로 말한다* \"맞습니다. 친환경 정책은 우리 회사의 핵심 가치이기도 하죠.\"")

# 일상 비즈니스 대화들
chat_memory.add_chat("*아침 회의에서 일정표를 확인하며 말한다* \"오늘 외부 미팅이 몇 건 있나요?\", *비서가 태블릿을 확인하며 대답한다* \"오전 10시 투자자 미팅, 오후 2시 파트너사 미팅이 있습니다.\"")
chat_memory.add_chat("*전화 통화 중에 물어본다* \"분기 보고서는 언제까지 제출하면 될까요?\", *팀장이 일정을 확인하며 대답한다* \"이번 주 금요일까지 초안을 보내주시면 검토하겠습니다.\"")
chat_memory.add_chat("*회의실에서 팀원들과 대화 중* \"새 프로젝트 일정은 어떻게 잡을까요?\", *프로젝트 매니저가 차트를 보여주며 대답한다* \"6개월 계획으로 진행하되, 첫 달은 기획에 집중하는 게 좋겠습니다.\"")
chat_memory.add_chat("*영상 회의에서 질문한다* \"해외 지사 성과는 어떤가요?\", *해외 담당자가 자료를 공유하며 설명한다* \"전년 대비 15% 성장했으나, 환율 영향으로 순이익은 5% 증가에 그쳤습니다.\"")
chat_memory.add_chat("*업무 메시지를 보낸다* \"이번 주 목표 달성률은 어떻게 되나요?\", *팀원이 곧바로 답장한다* \"현재 85%입니다. 내일까지 95% 달성이 가능할 것 같습니다.\"")
chat_memory.add_chat("*점심 식사 중 동료에게 묻는다* \"오전 이사회는 어땠어요?\", *동료가 커피를 마시며 대답한다* \"예상보다 순조롭게 진행됐어요. 신규 투자안이 승인되었습니다.\"")
chat_memory.add_chat("*사무실을 둘러보며 시설 담당자에게 묻는다* \"회의실 리모델링은 언제 완료되나요?\", *담당자가 일정표를 확인하며 대답한다* \"이번 달 말까지 모두 마무리될 예정입니다.\"")
chat_memory.add_chat("*서류를 검토하며 법무팀에 질문한다* \"이 계약서 검토 가능한가요?\", *법무팀 직원이 일정을 확인하며 답한다* \"네, 내일까지 검토의견 드리겠습니다.\"")
chat_memory.add_chat("*인사팀과 면담 중에 묻는다* \"신입 채용 일정은 어떻게 되나요?\", *인사 담당자가 설명한다* \"다음 달부터 서류 접수 시작하고, 두 달 내로 최종 합격자 발표 예정입니다.\"")
chat_memory.add_chat("*로비에서 동료를 만나 인사한다* \"오늘 중요한 미팅 있으신가요?\", *동료가 반갑게 대답한다* \"네, 11시에 신규 투자자와 미팅이 있습니다.\"")
chat_memory.add_chat("*화상회의 시작 전 질문한다* \"해외 지사에서는 몇 명이나 참석하나요?\", *비서가 참석자 명단을 보며 대답한다* \"일본에서 3명, 싱가포르에서 2명 참석 예정입니다.\"")
chat_memory.add_chat("*회사 식당에서 이야기한다* \"내일 경영 전략 회의에 참석하시나요?\", *상사가 일정을 확인하며 고개를 끄덕인다* \"네, 중요한 안건이 몇 가지 있어서 참석할 예정입니다.\"")
chat_memory.add_chat("*마케팅팀 미팅에서 물어본다* \"새 캠페인 예산은 얼마로 책정됐나요?\", *마케팅 팀장이 자료를 보여주며 대답한다* \"이번 분기에 3억원을 배정했습니다.\"")
chat_memory.add_chat("*인사고과 면담 중에 묻는다* \"올해 목표를 어떻게 설정하고 있나요?\", *직원이 자신의 계획을 설명한다* \"핵심 클라이언트 5개사 추가 확보와 매출 20% 증대를 목표로 하고 있습니다.\"")
chat_memory.add_chat("*회의 후 담당자에게 묻는다* \"프로젝트 진행 상황은 어떤가요?\", *담당자가 차트를 보여주며 설명한다* \"계획대로 진행 중입니다. 다음 주에는 베타 테스트를 시작할 예정입니다.\"")
chat_memory.add_chat("*해외 출장 전 준비 미팅에서 말한다* \"현지 파트너사와의 미팅 안건은 정리됐나요?\", *팀원이 자료를 건네며 답한다* \"네, 주요 논의 사항과 제안 내용을 모두 준비했습니다.\"")
chat_memory.add_chat("*퇴근 시간에 메시지를 보낸다* \"내일 아침 회의 자료 준비됐나요?\", *담당자가 답장한다* \"네, 방금 최종 검토 마쳤습니다. 공유드리겠습니다.\"")
chat_memory.add_chat("*전략 회의에서 질문한다* \"경쟁사 신제품 출시에 대한 대응 방안은 있나요?\", *전략팀장이 여러 가지 시나리오를 설명한다* \"세 가지 대응 전략을 준비했습니다. 가격 경쟁력 강화, 기능 개선, 마케팅 강화입니다.\"")
chat_memory.add_chat("*사내 교육 일정을 확인하며 묻는다* \"다음 리더십 교육은 언제인가요?\", *교육 담당자가 일정표를 확인하며 대답한다* \"다음 달 첫째 주 수요일에 예정되어 있습니다.\"")

# 협상에 대한 질문
user_query = "우리가 진행했던 협상에서 어떤 안건들이 논의되었지?"

# 유사한 대화 찾기
similar_chats = chat_memory.search_similar_chat(user_query, top_k = 3)

print("\n[유사도 기반 검색 결과]")
for i, chat in enumerate(similar_chats):
    print(f"{i+1}. {chat}")

# AI가 자동으로 응답 생성 (OpenAI 모델 사용)
ai_response = chat_memory.generate_response(user_query)
print("\n[협상 안건들 질문 테스트]")
print(f"사용자: {user_query}")
print(f"AI: {ai_response}")

# 새로운 대화 기록에 추가
chat_memory.add_chat(user_query, ai_response)

# 다른 질문 테스트
print("\n[협상 장소 질문 테스트]")
test_query = "협상이 진행된 장소는 어디였지?"
test_response = chat_memory.generate_response(test_query)
print(f"사용자: {test_query}")
print(f"AI: {test_response}")