# Windows에서 FastAPI의 HTTPS 설정 방법 (win-acme 사용)

이 문서는 **win-acme**(Windows용 ACMEv2 클라이언트)을 사용하여 Let's Encrypt로부터 SSL 인증서를 생성하고 FastAPI 애플리케이션에 HTTPS를 설정하는 방법을 설명합니다.

## 사전 준비 사항

- Windows 환경
- 관리자 권한 (일부 단계에서 필요)
- win-acme 클라이언트 설치 ([win-acme GitHub](https://github.com/win-acme/win-acme)에서 다운로드)
- 서버에 연결된 도메인 이름 (예: `example.com`)
- Python의 `python-dotenv` 라이브러리 설치 (.env 파일 관리를 위해 필요)

## HTTPS 설정 단계

### 1. win-acme 실행

1. win-acme이 설치된 디렉터리로 이동합니다.
2. win-acme 실행 파일을 실행합니다:
   ```bash
   ./wacs.exe
   ```

### 2. 인증서 생성 옵션 선택

다음과 같이 옵션을 선택합니다:

#### 메뉴 옵션
- **M:** 전체 옵션으로 인증서 생성

#### 도메인 선택

1. **수동 입력**: 도메인을 직접 입력합니다.
   ```
   Host: YOUR_DOMIN(ex -> your_domin.ddns.net)
   ```

#### 인증서 분할 옵션

- `4: 단일 인증서`를 선택하여 모든 도메인을 포함하는 단일 인증서를 생성합니다.

### 3. 소유권 검증 방법

도메인 소유권을 검증하기 위해 방법을 선택합니다:

1. **HTTP 검증 (http-01)**
   - `1: 네트워크 경로에 검증 파일 저장`을 선택합니다.
   - 사이트의 루트 경로를 설정합니다:
   - .well-known\acme-challenge에 경로로 네트워크 경로에 검증 테스트가 진행됨.
     ```
     Path: <프로젝트 디렉터리 ROOT 경로>
     ```
   - `web.config` 파일을 복사하라는 메시지가 표시되면:
     ```
     복사하시겠습니까? (y/n): y
     ```

### 4. 키 유형

인증서의 개인 키 유형을 선택합니다:

- **2: RSA 키** *(권장)*

### 5. 인증서 저장

인증서를 저장할 방법을 선택합니다:

1. **PEM 형식 파일** *(FastAPI 및 기타 Python 프레임워크에서 사용)*:

   - `.pem` 파일을 저장할 경로를 설정합니다:
     ```
     File path: <프로젝트 디렉터리 ROOT 경로>/fastapi/certificates
     ```
   - 비밀번호를 묻는 메시지에서:
     ```
     1: 비밀번호 없음
     ```

2. 추가 저장 옵션은:

   ```
   다른 저장 방식도 사용하시겠습니까?: 5
   ```

### 6. 인증서 설치 완료

인증서가 생성되면, 지정한 폴더(`./fastapi/certificates`)에 다음 파일이 생성됩니다:

- `fullchain.pem`: 인증서 체인
- `privkey.pem`: 개인 키

### 7. FastAPI에 HTTPS 구성

1. `.pem` 파일을 `./fastapi/certificates` 디렉터리로 이동합니다.

2. `.env` 파일을 생성하여 환경 변수를 저장합니다:
   - ./fastapi/certificates 경로에 추가된 .pem 파일들의 이름을 fastapi/src/.env 에 추가가
   ```env
   SSL_PW=
   KEY_PEM=YOUR_DOMIN-key.pem
   CRT_PEM=YOUR_DOMIN-crt.pem
   ```

   #### `.env` 파일 변수 설명
   - `SSL_PW`: SSL 인증서에 비밀번호가 설정된 경우 해당 비밀번호를 입력합니다. 비밀번호가 없다면 비워둡니다.
   - `KEY_PEM`: 개인 키 파일 이름입니다. 기본적으로 `YOUR_DOMIN-key.pem`으로 설정됩니다.
   - `CRT_PEM`: 인증서 파일 이름입니다. 기본적으로 `YOUR_DOMIN-crt.pem`으로 설정됩니다.
