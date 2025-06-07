"""
Llama 모델의 병렬 처리를 위한 단일 큐 핸들러 (2명씩 처리) - Office용
"""
import asyncio
import time
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ..ai_models.llama_office_model import LlamaOfficeModel
from .error_handler import InternalServerErrorException

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

@dataclass
class ProcessingRequest:
    """처리 요청을 나타내는 데이터 클래스"""
    id: str
    input_text: str
    search_text: str
    chat_list: list
    future: asyncio.Future
    created_at: float
    user_id: str

class LlamaQueueHandler:
    """
    Llama 모델의 병렬 처리를 관리하는 단일 큐 핸들러 (2명씩 처리) - Office용
    """
    def __init__(self, max_concurrent: int = 2):
        """
        LlamaQueueHandler 초기화 메서드

        Args:
            max_concurrent (int): 병렬로 처리할 워커의 수 (기본값: 2)
        """
        self.max_concurrent = max_concurrent
        self.request_queue: asyncio.Queue = asyncio.Queue()  # 단일 큐
        self.worker_models: list[Optional[LlamaOfficeModel]] = [None] * self.max_concurrent
        self.is_running = False
        self.total_processed = 0
        self.total_errors = 0
        self.worker_tasks: list[Optional[asyncio.Task]] = [None] * self.max_concurrent

        print(f"{BLUE}INFO{RESET}:     Office LlamaQueueHandler 초기화 (단일 큐, {self.max_concurrent} 워커)")

    async def init(self):
        """핸들러 초기화"""
        try:
            print(f"{GREEN}INFO{RESET}:     Office LlamaQueueHandler 초기화 완료")
        except Exception as e:
            print(f"{RED}ERROR{RESET}:    Office LlamaQueueHandler 초기화 실패: {str(e)}")
            raise InternalServerErrorException(detail=f"큐 핸들러 초기화 실패: {str(e)}")

    async def start(self):
        """큐 매니저 시작"""
        if not self.is_running:
            self.is_running = True
            # 각 워커 태스크 시작
            for i in range(self.max_concurrent):
                self.worker_tasks[i] = asyncio.create_task(self._worker(i))
            print(f"{GREEN}INFO{RESET}:     Office LlamaQueueHandler 워커 시작 (단일 큐)")

    async def stop(self):
        """큐 매니저 정지"""
        self.is_running = False

        # 워커 태스크 정리
        for task in self.worker_tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # 모델 메모리 해제
        for i in range(self.max_concurrent):
            self.worker_models[i] = None
        print(f"{YELLOW}INFO{RESET}:     Office LlamaQueueHandler 정지 완료")

    async def add_request(
        self,
        input_text: str,
        search_text: str,
        chat_list: list, 
        user_id: str = ""
    ) -> str:
        """
        새 요청을 큐에 추가하고 결과를 기다림
        
        Args:
            input_text: 사용자 입력
            search_text: 검색 결과 텍스트
            chat_list: 채팅 기록 리스트
            user_id: 사용자 ID (로깅용)
            
        Returns:
            생성된 응답 텍스트
        """
        if not self.is_running:
            raise InternalServerErrorException(detail="큐 핸들러가 실행 중이 아닙니다.")
        
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        
        request = ProcessingRequest(
            id=request_id,
            input_text=input_text,
            search_text=search_text,
            chat_list=chat_list,
            future=future,
            created_at=time.time(),
            user_id=user_id
        )
        
        # 단일 큐에 요청 추가
        await self.request_queue.put(request)
        
        queue_size = self.request_queue.qsize()
        print(
            f"🔄 Office 요청 추가: {request_id[:8]} | "
            f"User: {user_id} | Queue: {queue_size}"
        )
        
        # 결과 대기 (타임아웃 설정)
        try:
            result = await asyncio.wait_for(future, timeout=300.0)  # 5분 타임아웃
            return result
        except asyncio.TimeoutError:
            raise InternalServerErrorException(detail="요청 처리 시간 초과 (5분)")
        except Exception as e:
            raise InternalServerErrorException(detail=f"요청 처리 중 오류: {str(e)}")

    async def _worker(self, worker_id: int):
        """
        병렬 워커 - 각 워커가 단일 큐에서 요청을 가져와 처리
        """
        print(f"{BLUE}INFO{RESET}:     Office Worker-{worker_id} 시작 (단일 큐)")
        
        # 모델 인스턴스 생성 (각 워커별로)
        try:
            self.worker_models[worker_id] = LlamaOfficeModel()
            print(f"{GREEN}INFO{RESET}:     Office Worker-{worker_id} 모델 로드 완료")
        except Exception as e:
            print(f"{RED}ERROR{RESET}:    Office Worker-{worker_id} 모델 로드 실패: {str(e)}")
            return
        
        while self.is_running:
            try:
                # 단일 큐에서 요청 가져오기 (타임아웃 설정)
                request = await asyncio.wait_for(
                    self.request_queue.get(), 
                    timeout=1.0
                )
                
                actual_start_time = time.time()
                
                print(
                    f"🔄 Office Worker-{worker_id}: Processing {request.id[:8]} | "
                    f"User: {request.user_id}"
                )
                
                # 실제 처리 (CPU 집약적 작업을 스레드풀에서 실행)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self.worker_models[worker_id].generate_response,
                    request.input_text,
                    request.search_text,
                    request.chat_list
                )
                
                # 결과 검증
                if not result or len(result.strip()) < 10:
                    print(f"⚠️  Warning: 응답이 너무 짧습니다: '{result[:50]}...'")
                    result = "죄송합니다. 응답을 제대로 생성하지 못했습니다. 다시 시도해 주세요."
                
                # 결과 설정
                if not request.future.cancelled():
                    request.future.set_result(result)
                    self.total_processed += 1
                
                actual_processing_time = time.time() - actual_start_time
                total_time = time.time() - request.created_at
                
                print(
                    f"✅ Office Worker-{worker_id}: Completed {request.id[:8]} | "
                    f"User: {request.user_id} | "
                    f"ProcessTime: {actual_processing_time:.3f}s | "
                    f"TotalTime: {total_time:.3f}s | "
                    f"ResponseLen: {len(result)} chars"
                )
                
            except asyncio.TimeoutError:
                # 타임아웃 - 계속 대기
                continue
            except asyncio.CancelledError:
                # 정상적인 취소
                break
            except Exception as e:
                self.total_errors += 1
                if 'request' in locals() and not request.future.cancelled():
                    request.future.set_exception(e)
                print(f"❌ Office Worker-{worker_id}: Error processing request {request.id[:8]}: {e}")

    def get_queue_status(self) -> Dict[str, Any]:
        """현재 큐 상태 반환"""
        return {
            "queue_size": self.request_queue.qsize(),
            "processing_mode": "single_queue",
            "is_running": self.is_running,
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "model_loaded": all(self.worker_models)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        avg_processing_time = 25.0  # 예상 평균 처리 시간
        if self.total_processed > 0:
            avg_processing_time = 25.0  # 실제 처리 시간 기준으로 조정
        
        estimated_wait_time = self.request_queue.qsize() * avg_processing_time / self.max_concurrent
        
        return {
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "success_rate": (
                (self.total_processed / (self.total_processed + self.total_errors)) * 100
                if (self.total_processed + self.total_errors) > 0 else 0
            ),
            "queue_size": self.request_queue.qsize(),
            "estimated_wait_time": f"{estimated_wait_time:.1f}s",
            "avg_processing_time": f"{avg_processing_time:.1f}s",
            "processing_mode": "single_queue",
            "worker_active": any(self.worker_models)
        }