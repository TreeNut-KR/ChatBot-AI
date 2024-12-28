@echo off
chcp 65001
SETLOCAL

:: pip 최신 버전으로 업그레이드 (가상 환경 내부)
python.exe -m pip install --upgrade pip

:: requirements.txt 파일에 있는 모든 패키지 설치
pip install -r .\fastapi\requirements.txt

:: spaCy 모델 설치
python -m spacy download en_core_web_sm

:: pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

echo 가상 환경이 성공적으로 설정되었습니다.
ENDLOCAL
