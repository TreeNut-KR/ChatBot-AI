import os
import logging
from logging.handlers import BaseRotatingHandler
from typing import Callable, Dict, Type, Optional
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request
from datetime import datetime, timedelta


# 현재 파일의 상위 디렉토리 경로
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)  # 상위 디렉토리

# 로그 디렉토리 및 파일 경로 설정
log_dir = os.path.join(parent_directory, "logs")

# 로그 디렉토리가 없는 경우 생성
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Logger 설정
logger = logging.getLogger("fastapi_error_handlers")
logger.setLevel(logging.DEBUG)

class DailyRotatingFileHandler(BaseRotatingHandler):
    """
    날짜별로 로그 파일을 회전시키는 핸들러.
    """
    def __init__(self, dir_path: str, date_format: str = "%Y%m%d", encoding=None):
        # 로그 파일 디렉토리와 날짜 형식을 저장
        self.dir_path = dir_path
        self.date_format = date_format
        self.current_date = datetime.now().strftime(self.date_format)
        log_file = os.path.join(self.dir_path, f"{self.current_date}.log")
        super().__init__(log_file, 'a', encoding)

    def shouldRollover(self, record):
        # 로그의 날짜가 변경되었는지 확인
        log_date = datetime.now().strftime(self.date_format)
        return log_date != self.current_date

    def doRollover(self):
        # 로그 파일의 날짜가 변경되었을 때 롤오버 수행
        self.current_date = datetime.now().strftime(self.date_format)
        self.baseFilename = os.path.join(self.dir_path, f"{self.current_date}.log")
        if self.stream:
            self.stream.close()
            self.stream = self._open()

# DailyRotatingFileHandler 설정
file_handler = DailyRotatingFileHandler(log_dir, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# StreamHandler 설정 (터미널 출력용)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

# 로그 포맷 설정
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Logger에 핸들러 추가
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# 예외 클래스 정의
class NotFoundException(HTTPException):
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(status_code=404, detail=detail)

class BadRequestException(HTTPException):
    def __init__(self, detail: str = "Bad request"):
        super().__init__(status_code=400, detail=detail)

class UnauthorizedException(HTTPException):
    def __init__(self, detail: str = "Unauthorized"):
        super().__init__(status_code=401, detail=detail)

class ForbiddenException(HTTPException):
    def __init__(self, detail: str = "Forbidden"):
        super().__init__(status_code=403, detail=detail)

class ValueErrorException(HTTPException):
    def __init__(self, detail: str = "Value Error"):
        super().__init__(status_code=422, detail=detail)

class InternalServerErrorException(HTTPException):
    def __init__(self, detail: Optional[str] = None):
        super().__init__(status_code=500, detail=detail)

class DatabaseErrorException(HTTPException):
    def __init__(self, detail: str = "Database Error"):
        super().__init__(status_code=503, detail=detail)

class IPRestrictedException(HTTPException):
    def __init__(self, detail: str = "Unauthorized IP address"):
        super().__init__(status_code=403, detail=detail)

class MethodNotAllowedException(HTTPException):
    def __init__(self, detail: str = "Method Not Allowed"):
        super().__init__(status_code=405, detail=detail)

# 예외와 핸들러 매핑
exception_handlers: Dict[Type[HTTPException], Callable[[Request, HTTPException], JSONResponse]] = {
    NotFoundException: lambda request, exc: JSONResponse(
        status_code=exc.status_code,
        content={"detail": "The requested resource could not be found."},
    ),
    BadRequestException: lambda request, exc: JSONResponse(
        status_code=exc.status_code,
        content={"detail": "The request was invalid."},
    ),
    UnauthorizedException: lambda request, exc: JSONResponse(
        status_code=exc.status_code,
        content={"detail": "Unauthorized access."},
    ),
    ForbiddenException: lambda request, exc: JSONResponse(
        status_code=exc.status_code,
        content={"detail": "Access to this resource is forbidden."},
    ),
    ValueErrorException: lambda request, exc: JSONResponse(
        status_code=exc.status_code,
        content={"detail": "The input data is invalid."},
    ),
    InternalServerErrorException: lambda request, exc: JSONResponse(
        status_code=exc.status_code,
        content={"detail": "An internal server error occurred."},
    ),
    DatabaseErrorException: lambda request, exc: JSONResponse(
        status_code=exc.status_code,
        content={"detail": "A database error occurred. Please contact the administrator."},
    ),
    IPRestrictedException: lambda request, exc: JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    ),
    MethodNotAllowedException: lambda request, exc: JSONResponse(
        status_code=exc.status_code,
        content={"detail": "The method used in the request is not allowed."},
    ),
}

# 기본 예외 처리기
async def generic_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    FastAPI 애플리케이션에서 발생한 HTTPException을 처리하며,
    요청 정보와 예외에 대한 세부 사항을 로그에 기록합니다.
    """
    handler = exception_handlers.get(type(exc), None)

    # 요청 본문 읽기
    body = await request.body()

    log_data = {
        "url": str(request.url),
        "method": request.method,
        "headers": dict(request.headers),
        "body": body.decode("utf-8") if body else "",
        "exception_class": exc.__class__.__name__,
        "detail": exc.detail
    }

    # 로그에 시간, 오류 코드, 자세한 내용을 기록
    error_message = f"{exc.status_code} - {exc.detail}"
    logger.error(f"{error_message} | URL: {log_data['url']} | Method: {log_data['method']}")

    # 정의된 핸들러가 있을 경우 호출, 없으면 기본 500 응답
    if handler:
        return handler(request, exc)
    else:
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred."},
        )

# FastAPI 애플리케이션에 예외 핸들러 추가
def add_exception_handlers(app: FastAPI):
    """
    FastAPI 애플리케이션에 예외 핸들러를 추가하는 함수.
    정의된 예외 타입과 관련된 핸들러를 FastAPI 애플리케이션에 등록합니다.
    """
    for exc_type in exception_handlers:
        app.add_exception_handler(exc_type, generic_exception_handler)
