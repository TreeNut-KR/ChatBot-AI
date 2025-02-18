## Hugging Face 캐시 데이터 정리

허깅페이스 모델을 사용하면서 C드라이브에 늘어난 데이터를 정리하려면 다음 명령어를 사용하세요:

```powershell
PS D:\github\ChatBot-AI> Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\"
```

다음은 `accelerate` 설정에 대한 **README.md** 예시입니다. 이 파일은 사용자가 여러 GPU를 활용하여 모델을 분산 학습할 수 있도록 `accelerate`를 설정하는 과정을 설명합니다.

---

# `accelerate` 설정 가이드

이 프로젝트에서는 **`accelerate`** 라이브러리를 사용하여 여러 GPU 환경에서 모델을 분산 학습할 수 있도록 설정합니다. 아래는 이 설정을 위한 **`accelerate config`** 명령어 실행 과정과 그에 대한 설명입니다.

## 1. `accelerate` 설치

먼저, `accelerate`를 설치해야 합니다. 프로젝트의 가상 환경에서 다음 명령어를 실행하여 `accelerate`를 설치합니다.

```bash
pip install accelerate
```

## 2. `accelerate` 설정

`accelerate` 설정을 위해 다음 명령어를 실행합니다:

```bash
accelerate config
```

### 설정 항목 설명

#### 1. **컴퓨터 환경**
- **질문**: "In which compute environment are you running?"
- **응답**: `This machine`

현재 로컬 머신에서 분산 학습을 설정하고 있음을 나타냅니다.

#### 2. **머신 유형**
- **질문**: "Which type of machine are you using?"
- **응답**: `multi-GPU`

여러 GPU를 사용하여 학습을 진행합니다.

#### 3. **다중 머신 사용 여부**
- **질문**: "How many different machines will you use (use more than 1 for multi-node training)?"
- **응답**: `2`

분산 학습을 위해 두 대의 머신을 사용합니다.

#### 4. **머신 순위 (Rank)**
- **질문**: "What is the rank of this machine?"
- **응답**: `0`

이 머신은 첫 번째 머신이며, 분산 학습 중 첫 번째 프로세스가 실행됩니다.

#### 5. **IP 주소 및 포트 설정**
- **질문**: "What is the IP address of the machine that will host the main process?"
- **응답**: `127.0.0.1`

메인 프로세스는 로컬 머신에서 실행됩니다.

- **질문**: "What is the port you will use to communicate with the main process?"
- **응답**: `29500`

포트는 기본값인 `29500`을 사용합니다.

#### 6. **로컬 네트워크 여부**
- **질문**: "Are all the machines on the same local network?"
- **응답**: `YES`

두 머신이 동일한 로컬 네트워크에 연결되어 있음을 나타냅니다.

#### 7. **분산 오류 체크**
- **질문**: "Should distributed operations be checked while running for errors?"
- **응답**: `yes`

분산 학습 중 오류를 확인하여 문제를 미리 방지합니다.

#### 8. **Torch Dynamo 최적화**
- **질문**: "Do you wish to optimize your script with torch dynamo?"
- **응답**: `NO`

Torch Dynamo 최적화는 사용하지 않겠습니다.

#### 9. **DeepSpeed 사용 여부**
- **질문**: "Do you want to use DeepSpeed?"
- **응답**: `NO`

DeepSpeed는 사용하지 않습니다.

#### 10. **FullyShardedDataParallel 사용 여부**
- **질문**: "Do you want to use FullyShardedDataParallel?"
- **응답**: `yes`

모델을 `FullyShardedDataParallel`로 분산하여 메모리 사용을 최적화합니다.

#### 11. **Sharding 전략**
- **질문**: "What should be your sharding strategy?"
- **응답**: `FULL_SHARD`

모든 파라미터를 샤딩하여 분산 학습을 진행합니다.

#### 12. **CPU로 파라미터 및 기울기 오프로드**
- **질문**: "Do you want to offload parameters and gradients to CPU?"
- **응답**: `yes`

메모리 최적화를 위해 파라미터와 기울기를 CPU로 오프로드합니다.

#### 13. **자동 랩 정책**
- **질문**: "What should be your auto wrap policy?"
- **응답**: `TRANSFORMER_BASED_WRAP`

Transformer 모델의 각 층을 자동으로 랩핑하여 분산 학습을 쉽게 설정합니다.

#### 14. **FSDP 설정**
- **질문**: "What should be your FSDP's backward prefetch policy?"
- **응답**: `BACKWARD_PRE`

역전파 단계에서 기울기를 미리 가져와 성능을 최적화합니다.

- **질문**: "Do you want to enable FSDP's forward prefetch policy?"
- **응답**: `yes`

앞으로의 계산을 미리 준비하여 성능을 최적화합니다.

- **질문**: "Do you want to enable FSDP's `use_orig_params` feature?"
- **응답**: `no`

원래 파라미터를 사용하지 않고 FSDP에서 최적화된 파라미터를 사용합니다.

- **질문**: "Do you want to enable CPU RAM efficient model loading?"
- **응답**: `YES`

모델을 효율적으로 로딩하여 CPU RAM을 절약합니다.

- **질문**: "Do you want to enable FSDP activation checkpointing?"
- **응답**: `yes`

활성화 체크포인팅을 사용하여 메모리 사용을 줄입니다.

#### 15. **GPU 사용 개수**
- **질문**: "How many GPU(s) should be used for distributed training?"
- **응답**: `2`

분산 학습을 위해 두 개의 GPU를 사용합니다.

#### 16. **혼합 정밀도 사용 여부**
- **질문**: "Do you wish to use mixed precision?"
- **응답**: `fp16`

16비트 부동소수점(`fp16`)을 사용하여 학습 속도를 최적화합니다.

## 3. 설정 저장

설정이 완료되면, **`accelerate`**는 `C:\Users\treen/.cache\huggingface\accelerate\default_config.yaml`에 설정 파일을 저장합니다. 이 파일을 통해 분산 학습 환경을 손쉽게 재사용할 수 있습니다.