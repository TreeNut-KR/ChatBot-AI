from locust import HttpUser, task, between, events
import json
import random
import string
import time
import csv

# 결과 저장용 리스트
test_results = []

class OfficeAPIUser(HttpUser):
    """
    Office API 전용 성능 테스트 (Llama 모델)
    """
    # AI 응답 후 다음 요청까지 충분한 대기 시간 (120-240초)
    wait_time = between(120, 240)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_run = False
        
    def on_start(self):
        """각 사용자가 시작할 때 실행"""
        self.user_id = f"office_user_{random.randint(1000, 9999)}"
        self.db_id = f"123e4567-e89b-12d3-a456-{random.randint(100000000000, 999999999999)}"
        
    @task(1)  # Office Llama API 테스트
    def test_office_llama(self):
        """Office Llama API 테스트"""
        if self.has_run:
            self.stop()
            return
            
        self.has_run = True
        
        # 다양한 업무용 질문들
        office_questions = [
            "Llama AI 모델의 출시일과 버전들을 각각 알려줘.",
            "파이썬 웹 프레임워크 중 가장 인기있는 것은? 각각의 특징과 장단점도 알려줘.",
            "Docker 컨테이너와 가상머신의 차이점을 자세히 설명해줘.",
            "FastAPI의 주요 특징과 장점을 구체적으로 알려줘.",
            "MongoDB와 MySQL의 차이점은 무엇인가요? 언제 어떤 것을 사용해야 할까요?",
            "REST API 설계 원칙과 모범 사례를 알려줘.",
            "마이크로서비스 아키텍처의 장단점과 구현 시 고려사항은?",
            "NoSQL과 SQL 데이터베이스의 차이점을 설명하고 사용 사례를 알려줘.",
            "클라우드 컴퓨팅 서비스 모델(IaaS, PaaS, SaaS)에 대해 알려줘.",
            "DevOps의 핵심 개념과 주요 도구들을 설명해줘.",
            "Git과 GitHub의 차이점과 협업 워크플로우를 알려줘.",
            "테스트 주도 개발(TDD)의 개념과 장점을 설명해줘.",
            "소프트웨어 아키텍처 패턴들(MVC, MVP, MVVM)의 차이점은?",
            "API 문서화의 중요성과 좋은 문서화 방법을 알려줘.",
            "데이터베이스 인덱스의 개념과 성능 최적화 방법은?",
            "캐싱 전략과 Redis, Memcached의 차이점을 알려줘.",
            "JWT 토큰 기반 인증의 동작 원리와 보안 고려사항은?",
            "GraphQL과 REST API의 차이점과 각각의 장단점은?",
            "함수형 프로그래밍과 객체지향 프로그래밍의 차이점을 설명해줘.",
            "CI/CD 파이프라인의 구성요소와 구현 방법을 알려줘.",
            "Kubernetes의 기본 개념과 Docker와의 관계를 설명해줘.",
            "웹 보안의 주요 위협(OWASP Top 10)과 대응 방법을 알려줘.",
            "로드 밸런싱의 종류와 각각의 특징을 설명해줘.",
            "데이터베이스 트랜잭션의 ACID 속성을 설명해줘.",
            "메시지 큐의 개념과 RabbitMQ, Apache Kafka의 차이점은?"
        ]
        
        # 검색 관련 질문들 (google_access=True일 때 사용)
        search_questions = [
            "2024년 최신 AI 기술 동향과 트렌드를 알려줘.",
            "현재 주식시장 상황과 경제 전망을 알려줘.",
            "최근 IT 업계 뉴스와 주요 이슈들을 정리해줘.",
            "2024년 프로그래밍 언어 인기 순위를 알려줘.",
            "최신 클라우드 서비스 동향과 AWS, Azure, GCP 비교를 해줘.",
            "현재 암호화폐 시장 상황과 블록체인 기술 동향을 알려줘.",
            "최근 사이버 보안 이슈와 대응 방법을 알려줘.",
            "2024년 모바일 앱 개발 트렌드를 정리해줘.",
            "최신 웹 개발 프레임워크 동향을 알려줘.",
            "현재 인공지능과 머신러닝 시장 동향을 알려줘."
        ]
        
        # google_access 설정 (50% 확률로 검색 활성화)
        use_search = random.choice([True, False])
        
        if use_search:
            question = random.choice(search_questions)
        else:
            question = random.choice(office_questions)
        
        payload = {
            "input_data": question,
            "google_access": use_search,
            "db_id": self.db_id,
            "user_id": self.user_id
        }
        
        self._make_request("/office/Llama", payload, "Office-Llama")

    def _make_request(self, endpoint, payload, test_type):
        """실제 HTTP 요청을 수행하고 결과를 기록"""
        start_time = time.time()
        
        try:
            # AI 응답을 위한 충분한 타임아웃 설정 (240초)
            with self.client.post(
                endpoint,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "OfficeLoadTest/1.0"
                },
                timeout=240,  # 240초 타임아웃
                catch_response=True
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                # 응답 분석
                success = response.status_code == 200
                response_data = None
                
                if success:
                    try:
                        response_data = response.json()
                        if "result" in response_data:
                            response.success()
                            # 실제 응답 내용 확인
                            result_text = response_data.get("result", "")
                            if len(result_text.strip()) < 10:  # 너무 짧은 응답은 실패로 간주
                                success = False
                                response.failure("응답이 너무 짧음")
                        else:
                            success = False
                            response.failure("응답에 'result' 필드가 없음")
                    except json.JSONDecodeError:
                        success = False
                        response.failure("JSON 파싱 실패")
                else:
                    response.failure(f"HTTP {response.status_code}")
                
                # 결과 저장
                result = {
                    "user_id": self.user_id,
                    "test_type": test_type,
                    "endpoint": endpoint,
                    "google_access": payload.get("google_access", False),
                    "question_length": len(payload.get("input_data", "")),
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "success": success,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "response_size": len(response.content) if response.content else 0
                }
                test_results.append(result)
                
                # 실시간 로그 - 응답 시간에 따른 색상 구분
                if response_time < 30:
                    time_emoji = "🟢"  # 30초 미만: 빠름
                elif response_time < 120:
                    time_emoji = "🟡"  # 30-120초: 보통
                else:
                    time_emoji = "🔴"  # 120초 이상: 느림
                    
                status_emoji = "✅" if success else "❌"
                search_emoji = "🔍" if payload.get("google_access", False) else "📚"
                print(
                    f"{status_emoji} {time_emoji} {search_emoji} {test_type} | User: {self.user_id} | "
                    f"Time: {response_time:.3f}s | Status: {response.status_code}"
                )
                
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            result = {
                "user_id": self.user_id,
                "test_type": test_type,
                "endpoint": endpoint,
                "google_access": payload.get("google_access", False),
                "question_length": len(payload.get("input_data", "")),
                "status_code": 0,
                "response_time": response_time,
                "success": False,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
                "response_size": 0
            }
            test_results.append(result)
            
            search_emoji = "🔍" if payload.get("google_access", False) else "📚"
            print(f"❌ 🔴 {search_emoji} {test_type} | User: {self.user_id} | 오류: {str(e)}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """테스트 완료 후 결과 분석 및 저장"""
    if not test_results:
        print("⚠️ 테스트 결과가 없습니다.")
        return
    
    # CSV 파일로 저장
    with open('office_performance_results.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "user_id", "test_type", "endpoint", "google_access", "question_length",
            "status_code", "response_time", "success", "timestamp", "response_size"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in test_results:
            # error 필드가 있으면 제외하고 저장
            filtered_result = {k: v for k, v in result.items() if k in fieldnames}
            writer.writerow(filtered_result)
    
    # 통계 분석
    successful_tests = [r for r in test_results if r['success']]
    failed_tests = [r for r in test_results if not r['success']]
    
    print(f"\n{'='*60}")
    print(f"🏢 Office API 성능 테스트 결과")
    print(f"{'='*60}")
    print(f"총 요청 수: {len(test_results)}")
    print(f"성공 요청 수: {len(successful_tests)}")
    print(f"실패 요청 수: {len(failed_tests)}")
    print(f"성공률: {len(successful_tests)/len(test_results)*100:.1f}%")
    
    if successful_tests:
        response_times = [r['response_time'] for r in successful_tests]
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        # 백분위수 계산
        sorted_times = sorted(response_times)
        p50 = sorted_times[int(len(sorted_times) * 0.5)]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
        
        print(f"\n⏱️ 응답 시간 통계:")
        print(f"  평균: {avg_time:.3f}초")
        print(f"  최소: {min_time:.3f}초")
        print(f"  최대: {max_time:.3f}초")
        print(f"  50%ile: {p50:.3f}초")
        print(f"  95%ile: {p95:.3f}초")
        print(f"  99%ile: {p99:.3f}초")
        
        # 응답 시간 분포
        fast_responses = len([t for t in response_times if t < 30])
        medium_responses = len([t for t in response_times if 30 <= t < 120])
        slow_responses = len([t for t in response_times if t >= 120])
        
        print(f"\n🚀 응답 시간 분포:")
        print(f"  🟢 빠름 (30초 미만): {fast_responses}회 ({fast_responses/len(response_times)*100:.1f}%)")
        print(f"  🟡 보통 (30-120초): {medium_responses}회 ({medium_responses/len(response_times)*100:.1f}%)")
        print(f"  🔴 느림 (120초 이상): {slow_responses}회 ({slow_responses/len(response_times)*100:.1f}%)")
        
        # 검색 기능별 성능 분석
        search_tests = [r for r in successful_tests if r.get('google_access', False)]
        no_search_tests = [r for r in successful_tests if not r.get('google_access', False)]
        
        print(f"\n🔍 검색 기능별 성능:")
        if search_tests:
            search_times = [r['response_time'] for r in search_tests]
            search_avg = sum(search_times) / len(search_times)
            print(f"  🔍 검색 활성화: {len(search_tests)}회, 평균 {search_avg:.3f}초")
            
        if no_search_tests:
            no_search_times = [r['response_time'] for r in no_search_tests]
            no_search_avg = sum(no_search_times) / len(no_search_times)
            print(f"  📚 검색 비활성화: {len(no_search_tests)}회, 평균 {no_search_avg:.3f}초")
        
        # 질문 길이별 성능 분석
        short_questions = [r for r in successful_tests if r.get('question_length', 0) < 50]
        medium_questions = [r for r in successful_tests if 50 <= r.get('question_length', 0) < 100]
        long_questions = [r for r in successful_tests if r.get('question_length', 0) >= 100]
        
        print(f"\n📝 질문 길이별 성능:")
        if short_questions:
            short_times = [r['response_time'] for r in short_questions]
            short_avg = sum(short_times) / len(short_times)
            print(f"  짧은 질문 (<50자): {len(short_questions)}회, 평균 {short_avg:.3f}초")
            
        if medium_questions:
            medium_times = [r['response_time'] for r in medium_questions]
            medium_avg = sum(medium_times) / len(medium_times)
            print(f"  중간 질문 (50-100자): {len(medium_questions)}회, 평균 {medium_avg:.3f}초")
            
        if long_questions:
            long_times = [r['response_time'] for r in long_questions]
            long_avg = sum(long_times) / len(long_times)
            print(f"  긴 질문 (100자+): {len(long_questions)}회, 평균 {long_avg:.3f}초")
    
    print(f"\n📁 결과 파일: office_performance_results.csv")
    print(f"{'='*60}")