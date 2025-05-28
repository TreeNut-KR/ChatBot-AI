@echo off
chcp 65001
SETLOCAL

REM 1. 환경변수 검증
echo [INFO] 환경변수 검증 중...
if not exist ".env" (
    echo [ERROR] .env 파일이 존재하지 않습니다.
    echo [INFO] .env 파일을 생성하거나 확인해주세요.
    pause
    exit /b 1
)

REM 필수 환경변수 로드
for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
    set %%a=%%b
)

REM MONGO_URL 확인 및 생성
if "%MONGO_URL%"=="" (
    echo [WARN] MONGO_URL이 설정되지 않았습니다. 기본값을 사용합니다.
    set MONGO_URL=mongodb://root:1234@192.168.3.145:27017/chatbot?authSource=admin
)

echo [INFO] 환경변수 검증 완료.
echo.

REM 2. Docker 데몬 상태 확인 및 실행
echo [INFO] Docker 데몬 상태 확인 중...
docker info >nul 2>&1
if errorlevel 1 (
    echo [INFO] Docker 데몬이 실행 중이 아닙니다. Docker Desktop을 시작합니다...
    start "" "D:\Docker\Docker\Docker Desktop.exe"
    echo [INFO] Docker 시작 대기 중...
    timeout /t 20 >nul
    echo [INFO] Docker 데몬 재확인 중...
    docker info >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Docker 데몬 시작 실패. 종료합니다.
        exit /b 1
    ) else (
        echo [INFO] Docker 데몬이 정상적으로 시작되었습니다.
    )
) else (
    echo [INFO] Docker 데몬이 이미 실행 중입니다.
)
echo.

REM 3. 기존 컨테이너 정리 (볼륨 보존)
echo [INFO] 기존 컨테이너 정리 중 (볼륨 보존)...
docker-compose down
if errorlevel 1 (
    echo [WARN] docker-compose down 경고가 있었지만 계속 진행합니다.
)
echo.

REM 4. 빌드 캐시 최적화를 위한 선택적 정리
echo [INFO] 필요시에만 이미지 정리...
set /p choice="기존 이미지를 삭제하시겠습니까? (y/N): "
if /i "%choice%"=="y" (
    echo [INFO] 기존 이미지 삭제 중...
    FOR /F "tokens=*" %%i IN ('docker images -q -f "dangling=true"') DO (
        docker rmi -f %%i
    )
    docker image prune -f
) else (
    echo [INFO] 기존 이미지 유지 (빌드 캐시 활용)
)
echo.

REM 5. Python 라이브러리 볼륨 초기화 여부 확인
echo [INFO] Python 라이브러리 볼륨 상태 확인...
set /p libs_choice="Python 라이브러리를 재설치하시겠습니까? (y/N): "
if /i "%libs_choice%"=="y" (
    echo [INFO] Python 라이브러리 볼륨 삭제 및 재생성...
    docker volume rm chatbot-ai_python-libs 2>nul
    echo [INFO] Python 라이브러리 초기화를 진행합니다...
    set REBUILD_LIBS=true
) else (
    echo [INFO] 기존 Python 라이브러리 볼륨 재사용...
    set REBUILD_LIBS=false
)
echo.

REM 6. 로그 폴더 정리
echo [INFO] 이전 로그 폴더 삭제...
IF EXIST .\fastapi\logs rmdir /s /q .\fastapi\logs
echo.

REM 7. __pycache__ 폴더 정리
echo [INFO] __pycache__ 폴더 삭제...
FOR /d /r .\fastapi\ %%i IN (__pycache__) DO (
    if exist "%%i" rmdir /s /q "%%i"
)
echo.

REM 8. 베이스 이미지 빌드 (캐시 옵션 제거)
echo [INFO] chatbotai-base 베이스 이미지 빌드...
docker build -f fastapi/src/Dockerfile.base -t chatbotai-base:latest .
if errorlevel 1 (
    echo [ERROR] chatbotai-base 빌드 실패. 종료합니다.
    exit /b 1
)
echo.

REM 9. Python 라이브러리 초기화 (필요시에만)
if "%REBUILD_LIBS%"=="true" (
    echo [INFO] Python 라이브러리 초기화 중...
    docker-compose up python-libs-init
    if errorlevel 1 (
        echo [ERROR] Python 라이브러리 초기화 실패. 종료합니다.
        exit /b 1
    )
    echo [INFO] Python 라이브러리 초기화 완료.
    echo.
) else (
    echo [INFO] 기존 Python 라이브러리 사용 - 초기화 건너뜀.
    echo.
)

REM 10. 애플리케이션 서비스 빌드 (병렬)
echo [INFO] 애플리케이션 서비스 빌드...
docker-compose build office character nginx --parallel
if errorlevel 1 (
    echo [ERROR] 애플리케이션 서비스 빌드 실패. 종료합니다.
    exit /b 1
)
echo.

REM 11. 서비스 실행
echo [INFO] 애플리케이션 서비스 실행...
docker-compose up -d office character nginx
if errorlevel 1 (
    echo [ERROR] 서비스 실행 실패. 종료합니다.
    exit /b 1
)
echo.

REM 12. 서비스 상태 확인
echo [INFO] 서비스 상태 확인...
docker-compose ps
echo.

echo [INFO] 모든 작업이 완료되었습니다.
echo [INFO] 서비스 접속 정보:
echo [INFO] - Office Server: http://localhost:8002
echo [INFO] - Character Server: http://localhost:8003  
echo [INFO] - Nginx Proxy: http://localhost:8001
echo.

ENDLOCAL
pause