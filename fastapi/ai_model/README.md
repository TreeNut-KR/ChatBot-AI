# 📂 ./fastapi/ai_model **폴더**

> 해당 폴더는 사용 중인 모델에 대한 자료를 담는 폴더이다.
>
> model의 사용 방향이 다르기 때문에 기본 Llama와 Llama 기반의 Lumimaid, Bllossom GGUF 모델을 사용 중.
>

### 🟢 **모델 설명**

| 항목 | **LlamaChatModel** | **LumimaidChatModel** | **BllossomChatModel** |
|------|--------------------|----------------------|-----------------------|
| **기반 모델** | Llama-3.1-8B-Instruct | Llama-3-Lumimaid-8B | Llama-3-Korean-Bllossom-8B |
| **제작자** | Meta | Lewdiculous | MLP-KTLim |
| **포맷** | 표준 Hugging Face Transformers 모델 | GGUF 포맷 (압축, 경량화) | GGUF 포맷 (압축, 경량화) |
| **장치 활용** | `torch.device("cuda:0")` | `gpu_layers`를 이용해 GPU 할당 | `n_gpu_layers`를 이용해 GPU 할당 |
| **양자화 설정** | `BitsAndBytesConfig`로 4bit 양자화 | GGUF 자체의 양자화된 모델 사용 | GGUF 자체의 양자화된 모델 사용 |
| **로딩 방식** | `transformers`의 `AutoModelForCausalLM` | `llama_cpp_cuda` | `llama_cpp_cuda` |
| **사이트** | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B) | [Lewdiculous/Llama-3-Lumimaid-8B](https://huggingface.co/Lewdiculous/Llama-3-Lumimaid-8B-v0.1-OAS-GGUF-IQ-Imatrix) | [MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M) |

