@echo off
chcp 65001
SETLOCAL

echo Checking Docker daemon status...
docker info >nul 2>&1
if errorlevel 1 (
    echo Docker daemon is not running. Starting Docker daemon...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    timeout /t 10
    docker info >nul 2>&1
    if errorlevel 1 (
        echo Failed to start Docker daemon. Exiting...
        exit /b 1
    ) else (
        echo Docker daemon started successfully.
    )
) else (
    echo Docker daemon is already running.
)
echo.

echo Docker Compose down...
docker compose down -v
if errorlevel 1 (
    echo Failed to execute docker compose down. Exiting...
    exit /b 1
)
echo.

echo Removing old Docker images...
FOR /F "tokens=*" %%i IN ('docker images -q --filter "dangling=false"') DO (
    docker rmi %%i
)
echo.

echo Removing old folders...
IF EXIST .\fastapi\src\logs rmdir /s /q .\fastapi\src\logs
echo.

echo Removing __pycache__ folders in ./fastapi...
FOR /d /r .\fastapi\ %%i IN (__pycache__) DO (
    if exist "%%i" rmdir /s /q "%%i"
)
echo.

echo Docker Compose build...
docker compose build --parallel
if errorlevel 1 (
    echo Failed to execute docker compose build. Exiting...
    exit /b 1
)
echo.

echo Starting Docker Compose...
docker compose up -d
if errorlevel 1 (
    echo Failed to execute docker compose up. Exiting...
    exit /b 1
)
echo.

echo Docker Compose started successfully.

echo Installing Python packages...
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html fastapi==0.112.0 uvicorn==0.30.5 databases aiomysql motor itsdangerous==2.2.0 python-dotenv==1.0.1 annotated-types==0.7.0 anyio==4.4.0 click==8.1.7 colorama==0.4.6 dnspython==2.6.1 h11==0.14.0 idna==3.7 pydantic==2.8.2 pydantic_core==2.20.1 setuptools==65.5.0 sniffio==1.3.1 starlette==0.37.2 typing_extensions==4.12.2 requests==2.32.3 httpx==0.27.0 pytest pytest-asyncio langchain-community transformers bitsandbytes==0.44.1 accelerate>=0.26.0 ua-parser jpype1 konlpy googletrans spacy langdetect beautifulsoup4
if errorlevel 1 (
    echo Failed to install Python packages. Exiting...
    exit /b 1
)
echo.

ENDLOCAL
