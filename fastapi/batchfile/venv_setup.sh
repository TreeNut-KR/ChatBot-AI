#!/bin/bash

# Python 설치 경로 설정
PYTHON_PATH=$(which python3)

sudo apt update
sudo apt install python3-venv

# 가상 환경 디렉토리 이름 설정
ENV_DIR=.venv

# Python 가상 환경 생성
$PYTHON_PATH -m venv $ENV_DIR

echo "가상 환경 활성화 중..."
# 가상 환경 활성화
source $ENV_DIR/bin/activate