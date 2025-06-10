@echo off
REM filepath: d:\github\ChatBot-AI\fastapi\src\test\test.bat
REM ChatBot-AI Llama 모델 성능 테스트를 위한 Locust 실행 스크립트

echo ============================================
echo   🦙 Llama 모델 API 성능 테스트 실행 스크립트
echo ============================================
echo.

REM 현재 디렉토리를 스크립트 위치로 변경
cd /d "%~dp0"

REM Python과 Locust 설치 확인
echo [INFO] Python 및 Locust 설치 확인 중...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python이 설치되지 않았거나 PATH에 등록되지 않았습니다.
    pause
    exit /b 1
)

pip show locust >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Locust가 설치되지 않았습니다. 설치를 시작합니다...
    pip install locust
    if errorlevel 1 (
        echo [ERROR] Locust 설치에 실패했습니다.
        pause
        exit /b 1
    )
    echo [INFO] Locust 설치가 완료되었습니다.
)

echo [INFO] 준비 완료!
echo.

REM 메뉴 선택
:MENU
echo ============================================
echo         🦙 Llama 모델 테스트 옵션 선택
echo ============================================
echo 1. Office Llama 테스트 (웹 UI)
echo 2. Character Llama 테스트 (웹 UI)
echo 3. 종료
echo ============================================
echo.

set /p choice="옵션을 선택하세요 (1-3): "

if "%choice%"=="1" goto OFFICE_WEB_UI
if "%choice%"=="2" goto CHARACTER_WEB_UI
if "%choice%"=="3" goto EXIT
echo [ERROR] 잘못된 선택입니다. 다시 선택해주세요.
goto MENU

:OFFICE_WEB_UI
echo [INFO] Office Llama 웹 UI 테스트를 시작합니다...
echo [INFO] 브라우저에서 http://localhost:8089 를 열어주세요.
echo.
echo ============================================
echo            📊 웹 UI 설정 가이드
echo ============================================
echo 권장 설정값:
echo   - Number of users: 10-100 (서버 성능에 따라 조절)
echo   - Spawn rate: 5 (초당 5명씩 생성)
echo   - Host: http://localhost:8001
echo.
echo 📈 테스트 특징:
echo   - Office용 업무 질문들 (프로그래밍, 기술 문서 등)
echo   - Google 검색 기능 50%% 확률로 활성화
echo   - 각 사용자당 1회씩만 요청 후 종료
echo   - AI 응답 대기 시간: 120-240초
echo.
echo [INFO] 종료하려면 Ctrl+C를 누르세요.
echo ============================================
echo.
locust -f test_office_load.py --host=http://localhost:8001
goto MENU

:CHARACTER_WEB_UI
echo [INFO] Character Llama 웹 UI 테스트를 시작합니다...
echo [INFO] 브라우저에서 http://localhost:8089 를 열어주세요.
echo.
echo ============================================
echo            📊 웹 UI 설정 가이드
echo ============================================
echo 권장 설정값:
echo   - Number of users: 10-100 (서버 성능에 따라 조절)
echo   - Spawn rate: 5 (초당 5명씩 생성)
echo   - Host: http://localhost:8001
echo.
echo 📈 테스트 특징:
echo   - 다양한 캐릭터와의 대화 (엘리스)
echo   - 캐릭터별 컨텍스트와 시나리오 포함
echo   - 각 사용자당 1회씩만 요청 후 종료
echo   - AI 응답 대기 시간: 120-240초
echo.
echo [INFO] 종료하려면 Ctrl+C를 누르세요.
echo ============================================
echo.
locust -f test_character_load.py --host=http://localhost:8001
goto MENU

:EXIT
echo [INFO] Llama 테스트 스크립트를 종료합니다.
goto END

:END
echo.
echo ============================================
echo              스크립트 종료
echo ============================================
echo.
echo [INFO] 테스트 결과 파일들:
if exist "office_performance_results.csv" echo   ✅ office_performance_results.csv (Office 테스트 결과)
if exist "character_performance_results.csv" echo   ✅ character_performance_results.csv (Character 테스트 결과)
echo.
echo [TIP] CSV 파일을 Excel로 열어서 상세한 성능 분석을 확인하세요!
pause