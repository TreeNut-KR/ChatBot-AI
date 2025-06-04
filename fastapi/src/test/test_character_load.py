from locust import HttpUser, task, between, events
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json
import time
import random
import os
import csv

# 데이터클래스 정의
@dataclass
class TestResult:
    """테스트 결과를 저장하는 데이터클래스"""
    user_id: str
    test_type: str
    endpoint: str
    character_name: str
    status_code: int
    response_time: float
    success: bool
    failure_reason: str = ""
    retry_count: int = 0
    retry_after_seconds: int = 0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    response_size: int = 0
    is_retry_record: bool = False
    is_final_failure: bool = False
    is_final_success: bool = False
    error: str = ""
    process_time: float = 0.0  # 추가: 서버 처리 시간(초)

@dataclass
class CharacterScenario:
    """캐릭터 시나리오를 정의하는 데이터클래스"""
    character_name: str
    input_data: str
    greeting: str
    context: str

@dataclass
class TestStatistics:
    """테스트 통계를 저장하는 데이터클래스"""
    total_requests: int
    total_retries: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    server_error_failures: int
    retry_exceeded_failures: int
    other_failures: int
    user_retry_stats: Dict[str, int]
    response_times: List[float]

class TestResultManager:
    """테스트 결과를 관리하는 클래스"""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
    
    def add_result(self, result: TestResult):
        """테스트 결과 추가"""
        self.test_results.append(result)
    
    def get_final_results(self) -> List[TestResult]:
        """재시도 레코드를 제외한 최종 결과만 반환"""
        return [r for r in self.test_results if not r.is_retry_record]
    
    def get_retry_records(self) -> List[TestResult]:
        """재시도 레코드만 반환"""
        return [r for r in self.test_results if r.is_retry_record]
    
    def get_statistics(self) -> TestStatistics:
        """테스트 통계 계산"""
        final_results = self.get_final_results()
        retry_records = self.get_retry_records()
        
        successful_tests = [r for r in final_results if r.success]
        failed_tests = [r for r in final_results if not r.success]
        
        server_error_failures = [r for r in failed_tests if r.failure_reason.startswith('server_error')]
        retry_exceeded_failures = [r for r in failed_tests if r.failure_reason == '429_max_retries_exceeded']
        other_failures = [r for r in failed_tests if r not in server_error_failures and r not in retry_exceeded_failures]
        
        # 사용자별 재시도 통계
        user_retry_stats = {}
        for result in retry_records:
            user_id = result.user_id
            user_retry_stats[user_id] = user_retry_stats.get(user_id, 0) + 1
        
        return TestStatistics(
            total_requests=len(final_results),
            total_retries=len(retry_records),
            successful_requests=len(successful_tests),
            failed_requests=len(failed_tests),
            success_rate=len(successful_tests) / len(final_results) * 100 if final_results else 0,
            server_error_failures=len(server_error_failures),
            retry_exceeded_failures=len(retry_exceeded_failures),
            other_failures=len(other_failures),
            user_retry_stats=user_retry_stats,
            response_times=[r.response_time for r in successful_tests]
        )
    
    def save_to_csv(self, filename: str):
        """결과를 CSV 파일로 저장"""
        os.makedirs('performance_results', exist_ok=True)
        file_path = os.path.join('performance_results', filename)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                "user_id", "test_type", "endpoint", "character_name", "status_code", 
                "response_time", "success", "failure_reason", "retry_count", 
                "retry_after_seconds", "timestamp", "response_size", "is_retry_record",
                "is_final_failure", "is_final_success", "error", "process_time"  # ← 추가
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.test_results:
                writer.writerow(result.__dict__)
        
        return file_path

class ScenarioProvider:
    """테스트 시나리오를 제공하는 클래스"""
    
    @staticmethod
    def get_reina_scenario() -> CharacterScenario:
        """레이나 캐릭터 시나리오 반환"""
        return CharacterScenario(
            character_name="레이나",
            input_data="레이나와 함께 요리를 하며 오늘 뭘 만들어볼까? 새로운 레시피에 도전해보자!",
            greeting="밝은 주방에서 레이나가 앞치마를 두르고 요리 준비를 하고 있습니다. 그녀의 얼굴에는 요리에 대한 열정이 가득합니다.",
            context="레이나는 21세의 요리사 지망생으로, 요리에 대한 열정이 넘칩니다. 새로운 요리에 도전하는 것을 좋아하며, 사람들이 자신의 요리를 맛있게 먹는 모습을 보는 것이 가장 큰 기쁨입니다."
        )

class HttpRequestHandler:
    """HTTP 요청을 처리하는 클래스"""
    
    def __init__(self, client, user_id: str, db_id: str, result_manager: TestResultManager):
        self.client = client
        self.user_id = user_id
        self.db_id = db_id
        self.result_manager = result_manager
        self.max_retries = 10  # 429 재시도 제한 10번으로 상향
    
    def make_request(self, endpoint: str, scenario: CharacterScenario, test_type: str):
        """HTTP 요청 수행"""
        payload = {
            "input_data": scenario.input_data,
            "character_name": scenario.character_name,
            "greeting": scenario.greeting,
            "context": scenario.context,
            "db_id": self.db_id,
            "user_id": self.user_id
        }
        
        retry_count = 0
        while retry_count < self.max_retries:
            request_start_time = time.time()  # 각 시도마다 시간 측정 시작
            try:
                with self.client.post(
                    endpoint,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "CharacterLoadTest/1.0"
                    },
                    timeout=600,  # 10분으로 증가
                    catch_response=True
                ) as response:
                    request_end_time = time.time()
                    request_response_time = request_end_time - request_start_time
                    
                    # 429 응답 처리 - retry_after에 맞춰 대기 후 재시도
                    if response.status_code == 429:
                        if not self._handle_429_response(response, scenario, test_type, request_response_time, retry_count, endpoint):
                            retry_count += 1
                            if retry_count >= self.max_retries:
                                self._create_final_failure_result(429, "429_max_retries_exceeded", request_start_time, retry_count, scenario, test_type, endpoint)
                                return
                            self._wait_for_retry(response)
                            continue
                        return
                    
                    # 200 성공 응답 처리
                    elif response.status_code == 200:
                        self._handle_success_response(response, scenario, test_type, request_start_time, retry_count, endpoint)
                        return
                    
                    # 502 응답 처리 추가
                    elif response.status_code == 502:
                        self._handle_502_response(response, scenario, test_type, request_response_time, retry_count, endpoint)
                        return
                    
                    # 기타 모든 HTTP 오류는 재시도 없이 즉시 실패 처리
                    else:
                        self._create_final_failure_result(response.status_code, f"http_error_{response.status_code}", request_start_time, retry_count, scenario, test_type, endpoint)
                        return
            except Exception as e:
                self._handle_exception(e, scenario, test_type, request_start_time, retry_count, endpoint)
                return  # 예외 발생 시 루프 종료

    def _handle_429_response(self, response, scenario: CharacterScenario, test_type: str, response_time: float, retry_count: int, endpoint: str) -> bool:
        """429 응답 처리 - retry_after 정보 추출"""
        try:
            response_data = response.json()
            retry_after = response_data.get("retry_after", 60)
        except:
            retry_after = int(response.headers.get("Retry-After", 60))
        
        retry_result = TestResult(
            user_id=self.user_id,
            test_type=test_type,
            endpoint=endpoint,
            character_name=scenario.character_name,
            status_code=429,
            response_time=response_time,
            success=False,
            failure_reason=f"429_retry_attempt_{retry_count + 1}",
            retry_count=retry_count + 1,
            retry_after_seconds=retry_after,
            response_size=len(response.content) if response.content else 0,
            is_retry_record=True
        )
        self.result_manager.add_result(retry_result)
        
        print(f"🔄 429 재시도 {retry_count + 1}/{self.max_retries}회 | Character: {scenario.character_name} | User: {self.user_id} | {retry_after}초 대기 후 재시도")
        return False
    
    def _handle_success_response(self, response, scenario: CharacterScenario, test_type: str, start_time: float, retry_count: int, endpoint: str):
        """성공 응답 처리"""
        end_time = time.time()
        total_response_time = end_time - start_time

        success = True
        failure_reason = ""
        process_time = 0.0

        try:
            response_data = response.json()
            # 서버에서 process_time 파싱
            if "processing_info" in response_data and "processing_time" in response_data["processing_info"]:
                proc_time_str = response_data["processing_info"]["processing_time"]
                try:
                    process_time = float(proc_time_str.replace("s", ""))
                except Exception:
                    process_time = 0.0
            if "result" in response_data:
                response.success()
                result_text = response_data.get("result", "")
                if len(result_text.strip()) < 10:
                    success = False
                    failure_reason = "응답이 너무 짧음"
                    response.failure("응답이 너무 짧음")
            else:
                success = False
                failure_reason = "응답에 'result' 필드가 없음"
                response.failure("응답에 'result' 필드가 없음")
        except json.JSONDecodeError:
            success = False
            failure_reason = "JSON 파싱 실패"
            response.failure("JSON 파싱 실패")

        result = TestResult(
            user_id=self.user_id,
            test_type=test_type,
            endpoint=endpoint,
            character_name=scenario.character_name,
            status_code=response.status_code,
            response_time=total_response_time,
            success=success,
            failure_reason=failure_reason,
            retry_count=retry_count,
            response_size=len(response.content) if response.content else 0,
            is_final_success=success,
            process_time=process_time  # 서버에서 받은 값 저장
        )
        self.result_manager.add_result(result)

        self._log_response(success, total_response_time, test_type, scenario.character_name, response.status_code, retry_count)
    
    def _create_final_failure_result(self, status_code: int, failure_reason: str, start_time: float, retry_count: int, scenario: CharacterScenario, test_type: str, endpoint: str):
        """최종 실패 결과 생성"""
        end_time = time.time()
        total_response_time = end_time - start_time
        
        result = TestResult(
            user_id=self.user_id,
            test_type=test_type,
            endpoint=endpoint,
            character_name=scenario.character_name,
            status_code=status_code,
            response_time=total_response_time,
            success=False,
            failure_reason=failure_reason,
            retry_count=retry_count,
            response_size=0,
            is_final_failure=True
        )
        self.result_manager.add_result(result)
        
        print(f"❌ 🔴 {test_type} | Character: {scenario.character_name} | User: {self.user_id} | 실패: {failure_reason}")
    
    def _handle_exception(self, exception: Exception, scenario: CharacterScenario, test_type: str, start_time: float, retry_count: int, endpoint: str):
        """예외 처리"""
        end_time = time.time()
        total_response_time = end_time - start_time
        
        result = TestResult(
            user_id=self.user_id,
            test_type=test_type,
            endpoint=endpoint,
            character_name=scenario.character_name,
            status_code=0,
            response_time=total_response_time,
            success=False,
            failure_reason="exception",
            error=str(exception),
            retry_count=retry_count,
            response_size=0,
            is_final_failure=True
        )
        self.result_manager.add_result(result)
        
        print(f"❌ 🔴 {test_type} | Character: {scenario.character_name} | User: {self.user_id} | 예외 오류: {str(exception)}")
    
    def _wait_for_retry(self, response):
        """재시도 대기 - retry_after 값에 맞춰 정확히 대기"""
        try:
            response_data = response.json()
            retry_after = response_data.get("retry_after", 60)
        except:
            retry_after = int(response.headers.get("Retry-After", 60))
        
        # 서버에서 지정한 retry_after 시간을 정확히 준수
        print(f"⏳ {retry_after}초 대기 중... (서버 지정 retry_after)")
        time.sleep(retry_after)
    
    def _log_response(self, success: bool, response_time: float, test_type: str, character_name: str, status_code: int, retry_count: int):
        """응답 로깅"""
        if response_time < 30:
            time_emoji = "🟢"
        elif response_time < 120:
            time_emoji = "🟡"
        else:
            time_emoji = "🔴"
            
        status_emoji = "✅" if success else "❌"
        retry_info = f" (재시도: {retry_count}회)" if retry_count > 0 else ""
        
        print(
            f"{status_emoji} {time_emoji} {test_type} | Character: {character_name} | User: {self.user_id} | "
            f"Time: {response_time:.3f}s | Status: {status_code}{retry_info}"
        )
    
    def _handle_502_response(self, response, scenario: CharacterScenario, test_type: str, response_time: float, retry_count: int, endpoint: str):
        """502 응답 처리 - 더 자세한 디버깅 정보 수집"""
        
        # 응답 헤더 확인
        headers_info = dict(response.headers) if hasattr(response, 'headers') else {}
        
        # 응답 본문 확인 (있다면)
        try:
            content = response.text if hasattr(response, 'text') else ""
        except:
            content = ""
        
        print(f"🔍 502 디버깅 정보:")
        print(f"  Headers: {headers_info}")
        print(f"  Content: {content[:200]}...")  # 처음 200자만
        print(f"  Response Time: {response_time:.3f}s")
        
        result = TestResult(
            user_id=self.user_id,
            test_type=test_type,
            endpoint=endpoint,
            character_name=scenario.character_name,
            status_code=502,
            response_time=response_time,
            success=False,
            failure_reason="http_error_502_gateway",
            retry_count=retry_count,
            response_size=len(content) if content else 0,
            is_final_failure=True,
            error=f"Headers: {headers_info}, Content: {content[:100]}"
        )
        self.result_manager.add_result(result)

class TestReportGenerator:
    """테스트 보고서를 생성하는 클래스"""
    
    def __init__(self, result_manager: TestResultManager):
        self.result_manager = result_manager
    
    def generate_report(self, file_path: str):
        """테스트 보고서 생성"""
        stats = self.result_manager.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"👤 Character API 성능 테스트 결과")
        print(f"{'='*60}")
        
        self._print_basic_stats(stats)
        self._print_failure_analysis(stats)
        self._print_retry_stats(stats)
        self._print_response_time_stats(stats)
        self._print_character_stats()
        
        print(f"\n📁 결과 파일: {file_path}")
        print(f"{'='*60}")
    
    def _print_basic_stats(self, stats: TestStatistics):
        """기본 통계 출력"""
        print(f"총 최종 요청 수: {stats.total_requests}")
        print(f"총 재시도 발생 수: {stats.total_retries}")
        print(f"성공 요청 수: {stats.successful_requests}")
        print(f"실패 요청 수: {stats.failed_requests}")
        print(f"성공률: {stats.success_rate:.1f}%")
    
    def _print_failure_analysis(self, stats: TestStatistics):
        """실패 분석 출력"""
        print(f"\n📊 실패 원인별 분석:")
        print(f"  🚫 서버 오류: {stats.server_error_failures}회")
        print(f"  🔄 429 재시도 실패: {stats.retry_exceeded_failures}회")
        print(f"  ❓ 기타 실패: {stats.other_failures}회")
    
    def _print_retry_stats(self, stats: TestStatistics):
        """재시도 통계 출력"""
        if stats.user_retry_stats:
            print(f"\n🔄 사용자별 재시도 통계:")
            for user_id, retry_count in sorted(stats.user_retry_stats.items()):
                print(f"  {user_id}: {retry_count}회 재시도")
            
            print(f"\n📈 재시도 요약:")
            print(f"  재시도 발생 사용자 수: {len(stats.user_retry_stats)}명")
            print(f"  총 재시도 발생 횟수: {sum(stats.user_retry_stats.values())}회")
            print(f"  평균 재시도 횟수: {sum(stats.user_retry_stats.values())/len(stats.user_retry_stats):.1f}회")
    
    def _print_response_time_stats(self, stats: TestStatistics):
        """응답 시간 통계 출력"""
        if stats.response_times:
            avg_time = sum(stats.response_times) / len(stats.response_times)
            min_time = min(stats.response_times)
            max_time = max(stats.response_times)
            
            sorted_times = sorted(stats.response_times)
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
            
            fast_responses = len([t for t in stats.response_times if t < 30])
            medium_responses = len([t for t in stats.response_times if 30 <= t < 120])
            slow_responses = len([t for t in stats.response_times if t >= 120])
            
            print(f"\n🚀 응답 시간 분포:")
            print(f"  🟢 빠름 (30초 미만): {fast_responses}회 ({fast_responses/len(stats.response_times)*100:.1f}%)")
            print(f"  🟡 보통 (30-120초): {medium_responses}회 ({medium_responses/len(stats.response_times)*100:.1f}%)")
            print(f"  🔴 느림 (120초 이상): {slow_responses}회 ({slow_responses/len(stats.response_times)*100:.1f}%)")
    
    def _print_character_stats(self):
        """캐릭터별 통계 출력"""
        successful_results = [r for r in self.result_manager.get_final_results() if r.success]
        character_stats = {}
        
        for result in successful_results:
            char_name = result.character_name
            character_stats.setdefault(char_name, []).append(result.response_time)
        
        if character_stats:
            print(f"\n👥 캐릭터별 성능:")
            for char_name, times in character_stats.items():
                avg_time = sum(times) / len(times)
                print(f"  {char_name}: {len(times)}회, 평균 {avg_time:.3f}초")

# 전역 결과 관리자
result_manager = TestResultManager()

class CharacterAPIUser(HttpUser):
    """Character API 전용 성능 테스트 (Llama 모델) - 리팩토링된 버전"""
    
    wait_time = between(300, 600)  # 5-10분 대기
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_run = False
        
    def on_start(self):
        """각 사용자가 시작할 때 실행"""
        self.user_id = f"char_user_{random.randint(1000, 9999)}"
        self.db_id = f"123e4567-e89b-12d3-a456-{random.randint(100000000000, 999999999999)}"
        self.request_handler = HttpRequestHandler(self.client, self.user_id, self.db_id, result_manager)
        
    @task(1)
    def test_character_llama(self):
        """Character Llama API 테스트"""
        if self.has_run:
            self.stop()
            return
            
        self.has_run = True
        
        scenario = ScenarioProvider.get_reina_scenario()
        self.request_handler.make_request("/character/Llama", scenario, "Character-Llama")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """테스트 완료 후 결과 분석 및 저장"""
    if not result_manager.test_results:
        print("⚠️ 테스트 결과가 없습니다.")
        return

    # 결과 저장
    timestamp = datetime.now().strftime("%Y-%m-%d %H%M%S")
    filename = f'character_{timestamp}.csv'
    file_path = result_manager.save_to_csv(filename)

    # 보고서 생성
    report_generator = TestReportGenerator(result_manager)
    report_generator.generate_report(file_path)