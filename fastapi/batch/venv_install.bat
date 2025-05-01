@echo off
chcp 65001
SETLOCAL

:: pip 최신 버전으로 업그레이드
python.exe -m pip install --upgrade pip

:: numpy 먼저 설치 (버전 제한)
pip install "numpy>= 1.22.4,<2.0.0"

:: CUDA 관련 패키지 설치
pip install torch == 2.3.1+cu118 torchvision == 0.18.1+cu118 torchaudio == 2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

:: CUDA llama-cpp 설치
set CMAKE_ARGS = "-DLLAMA_CUBLAS = on"
set FORCE_CMAKE = 1
pip install --no-cache-dir "https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.3.6+cu121-cp311-cp311-win_amd64.whl"

:: ExLlamaV2 설치 (최신 버전)
pip install exllamav2 == 0.2.8

:: Flash Attention 설치 (pre-built wheel 사용)
pip install --no-cache-dir --find-links https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.3/ flash-attn == 2.3.3

:: CUDA 빌드 도구 설치
pip install ninja

:: spaCy 설치 및 모델 다운로드
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download ko_core_news_sm

:: 나머지 requirements.txt 패키지 설치
pip install -r .\fastapi\requirements.txt
pip install -r .\fastapi\requirements_llama.txt

echo 가상 환경이 성공적으로 설정되었습니다.
ENDLOCAL
