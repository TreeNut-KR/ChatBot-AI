# 🤖 ChatBot-AI Project

> AI 기반 챗봇 API 프로젝트입니다.
> FastAPI를 활용한 백엔드 서버와 Llama 기반 AI 모델을 통합하여 구현되었습니다.

작업자 
| 구성원 | 업무 | 사용 기술 |  
|--------|--------|------------|  
| [서정훈 (CutTheWire)](https://github.com/CutTheWire) | AI API 구축 | FastAPI, llama_cpp_cuda, OpenAI, transformers |  


# 웹서버 리포지토리
[➡️ TreeNut-KR/ChatBot](https://github.com/TreeNut-KR/ChatBot)

## 📋 프로젝트 구조
```
ChatBot-AI/
├── fastapi/
│   ├── ai_model/         # AI 모델 관련 파일
│   ├── batch/            # 환경 설정 배치 파일
│   ├── certificates/     # http .pem 파일
│   ├── datasets/         # 학습 데이터셋
│   └── src/              # API 서버 코드 파일
│       ├── prototypes/   # 실험/프로토타입 코드 파일
│       ├── utils/        # 유틸리티, 핸들러, 서비스, 스키마 등 서버 기능 코드 파일
│       │   ├── ai_models/
|       |   |   ├── shared/
|       |   |   |   └──shared_configs.py
│       │   │   ├── bllossom_model.py
│       │   │   ├── llama_model.py    # ⚠️사용 안함(llama-cpp-cuda 도입 전 코드)
│       │   │   ├── lumimaid_model.py
│       │   │   ├── openai_character_model.py
│       │   │   └── openai_office_model.py
│       │   ├── handlers/
│       │   │   ├── error_handler.py
│       │   │   ├── language_handler.py
│       │   │   └── mongodb_handler.py
│       │   ├── schemas/
│       │   │   └── chat_schema.py
│       │   ├── services/
│       │   │   └── search_service.py
│       │   └── __init__.py
│       ├── .env
│       ├── bot.yaml
│       └── server.py     # 서버 구동 코드 파일
```

## 📋 UML 클래스 다이어그램 
### 📑 ChatBot-AI/fastapi/src/utils/ai_models 클래스 다이어그램 
![image](https://lh3.googleusercontent.com/d/11BO1kgmcn_I0N-gAegB8p36-PrAm4IHn)

### 📑 ChatBot-AI/fastapi/src/utils/handlers 클래스 다이어그램 
![image](https://lh3.googleusercontent.com/d/10s3xwUFxnmfKb8WBEvU3jqQhJgExNa28)

### 📑 ChatBot-AI/fastapi/src/utils/schemas 클래스 다이어그램
![image](https://lh3.googleusercontent.com/d/1Az97lKerSOJltMPWEMeAW6G72axCdIii)

## 📋 UML 패키지 다이어그램 
![image](https://lh3.googleusercontent.com/d/1_fifSzf7YFoEMQd80hUQGgF0rI0vsYtm)

## 🚀 주요 기능

- **AI 모델**:
  - Llama-3-Lumimaid-8B (GGUF 최적화)
  - Llama-3-Korean-Bllossom-8B (GGUF 최적화)
  - OpenAI 
    - GPT4o-mini
    - GPT4.1
    - GPT4.1-mini
- **데이터셋**:
  - ~~ko_wikidata_QA (137,505개 한국어 QA 쌍)~~ ⚠️ **사용 안함**

## ⚙️ 환경 설정

### 필수 요구사항
- Python 3.11
- CUDA 지원 GPU
- Windows 10 이상 운영체제

### 설치 방법
1. 환경 구성
    #### ① CUDA Toolkit

    - Version : 11.8
    - Download : [CUDA Toolkit 11.8 Downloads](https://developer.download.nvidia.com/compute/cuda/11.8.0/network_installers/cuda_11.8.0_windows_network.exe)

    - Version : 12.8
    - Download : [CUDA Toolkit 12.8 Downloads](https://developer.download.nvidia.com/compute/cuda/12.8.0/network_installers/cuda_12.8.0_windows_network.exe)

    #### ② cuDNN

    - Version : 8.7.0
    - Download : [Local Installers for Windows](https://developer.nvidia.com/downloads/c118-cudnn-windows-8664-87084cuda11-archivezip)
    - cuDNN directory location
        ```
        C:/tools/cuda/
        ```

    #### ③ Python

    - Version : 3.11.x
    - Download : [Python 3.11.4 - June 6, 2023](https://www.python.org/ftp/python/3.11.4/python-3.11.4-amd64.exe)


    #### ④ Visual C++ 재배포 가능 패키지 설치
    - Download : [ Latest Microsoft Visual C++ Downloads](https://download.visualstudio.microsoft.com/download/pr/1754ea58-11a6-44ab-a262-696e194ce543/3642E3F95D50CC193E4B5A0B0FFBF7FE2C08801517758B4C8AEB7105A091208A/VC_redist.x64.exe)
    - Download : [ Visual Studio 2013 (VC++ 12.0) Downloads](https://download.visualstudio.microsoft.com/download/pr/10912041/cee5d6bca2ddbcd039da727bf4acb48a/vcredist_x64.exe)
    - Download : [ Visual Studio 2012 (VC++ 11.0) Downloads](https://download.microsoft.com/download/1/6/B/16B06F60-3B20-4FF2-B699-5E9B7962F9AE/VSU_4/vcredist_x64.exe)
    - Download : [ Visual Studio 2010 (VC++ 10.0) Downloads](https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x64.exe)
    - Download : [ Visual Studio 2008 (VC++ 9.0) Downloads](https://download.microsoft.com/download/5/D/8/5D8C65CB-C849-4025-8E95-C3966CAFD8AE/vcredist_x64.exe)
    - Download : [ Visual Studio 2005 (VC++ 8.0) Downloads](https://download.microsoft.com/download/8/B/4/8B42259F-5D70-43F4-AC2E-4B208FD8D66A/vcredist_x64.EXE)

    #### ⑤ PyTorch

    - Run this Commandpip

        ```
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```

    #### ⑥ 환경 변수 설정
    
    - 시스템 변수 추가

    | 변수 이름 | 변수 값 |
    | --- | --- |
    | CUDA_HOME | C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8 |
    | CUDNN_HOME | C:/tools/cuda |

    - Path 환경 변수 추가

    | Set | | Path |
    | --- | --- | --- |
    |SET PATH |=|C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin|
    |SET PATH |=|C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/extras/CUPTI/lib64|
    |SET PATH |=|C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include|
    |SET PATH |=|C:/tools/cuda/bin|

2. 가상환경 생성

   - [venv_setup.bat](./fastapi/batch/venv_setup.bat)
   ```bash
   ./fastapi/batch/venv_setup.bat
   ```

3. 필요 패키지 설치

   - [venv_setup.bat](./fastapi/batch/venv_install.bat)
    ```bash
    ./fastapi/batch/venv_install.bat
    ```

4. 서버 실행
   - [server.py](./fastapi/src/server.py)
    ```bash
    ./.venv/Scripts/python.exe ./fastapi/src/server.py
    ``` 

## 📚 사용된 주요 CUDA 패키지

- torch (CUDA 11.8)
- llama-cpp-python (CUDA 12.8)

## 🔑 라이선스

- **AI 모델**: Meta AI 라이선스
- **데이터셋**: 비상업적 사용 (학습된 모델은 상업적 사용 가능, 현재는 사용하지 않음)

## 📌 참고사항

자세한 모델 및 데이터셋 정보는 각 폴더의 README.md를 참고해주세요:
- ⚠️ 중요 [AI 모델 정보](./fastapi/ai_model/README.md)
- [데이터셋 정보](./fastapi/datasets/README.md)
- [도메인 설정](./fastapi/certificates/DNS_README.md)
- [.pem 파일 생성](./fastapi/certificates/PEM_README.md)