# 📂 ./fastapi/ai_model **폴더**

> 해당 폴더는 사용 중인 모델에 대한 자료를 담는 폴더입니다.
>
> 현재 Character AI 서비스와 Office AI 서비스에서 Meta-Llama-3.1-8B-Claude 모델의 GGUF 포맷을 사용하고 있습니다. </br>
> ⚠️ AI 서버를 사용하기 위해선 GGUF 모델을 다운로드 받은 뒤, 해당 폴더(./fastapi/ai_model/QuantFactory/)에 배치해야 합니다.

## � **폴더 구조**

```
📦ai_model
 ┣ 📂QuantFactory
 ┃ ┣ 📜Meta-Llama-3.1-8B-Claude.Q4_0.gguf
 ┃ ┗ 📜Meta-Llama-3.1-8B-Claude.Q4_1.gguf
 ┗ 📜README.md
```

### 🟢 **모델 설명**

| 항목 | **LlamaCharacterModel** | **LlamaOfficeModel** | 
|------|----------------------|-----------------------|
| **기반 모델** | Meta-Llama-3.1-8B-Claude | Meta-Llama-3.1-8B-Claude |
| **모델 파일** | `Meta-Llama-3.1-8B-Claude.Q4_0.gguf` | `Meta-Llama-3.1-8B-Claude.Q4_1.gguf` |
| **제작자** | QuantFactory | QuantFactory |
| **포맷** | GGUF 포맷 (Q4_0 양자화) | GGUF 포맷 (Q4_1 양자화) |
| **GPU 할당** | GPU 0번 (`main_gpu = 0`) | GPU 1번 (`main_gpu = 1`) |
| **GPU 레이어** | `n_gpu_layers = 50` | `n_gpu_layers = -1` (모든 레이어) |
| **용도** | 캐릭터 롤플레이 대화 | 업무용 AI 어시스턴트 |
| **로딩 방식** | `llama_cpp_cuda` | `llama_cpp_cuda` |
| **컨텍스트 길이** | 8191 토큰 | 8191 토큰 |
| **배치 크기** | 2048 | 2048 |
| **사이트** | [QuantFactory/Meta-Llama-3.1-8B-Claude-GGUF](https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Claude-GGUF) | [QuantFactory/Meta-Llama-3.1-8B-Claude-GGUF](https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Claude-GGUF) |

### 🔧 **기술적 특징**

- **Flash Attention**: 메모리 효율성과 속도 향상을 위해 활성화
- **연속 배칭**: 멀티 사용자 처리를 위한 최적화
- **16bit KV 캐시**: 메모리 사용량 최적화
- **RoPE 스케일링**: 긴 문맥 지원을 위한 linear scaling (2x)
- **스트리밍 지원**: 실시간 텍스트 생성 및 응답

### 📥 **다운로드 링크**

- **Q4_0 모델**: [Meta-Llama-3.1-8B-Claude.Q4_0.gguf](https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Claude-GGUF/blob/main/Meta-Llama-3.1-8B-Claude.Q4_0.gguf)
- **Q4_1 모델**: [Meta-Llama-3.1-8B-Claude.Q4_1.gguf](https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Claude-GGUF/blob/main/Meta-Llama-3.1-8B-Claude.Q4_1.gguf)

