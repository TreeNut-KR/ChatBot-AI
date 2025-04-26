from llama_cpp import Llama

# 모델 로드 (컨텍스트 길이 설정 추가)
llm=Llama(
    model_path="fastapi/ai_model/v2-Llama-3-Lumimaid-8B-v0.1-OAS-Q5_K_S-imat.gguf",
    n_ctx=8192  # 최대 컨텍스트 길이 설정
)

# 모델 실행
output=llm("What is the meaning of life?")
print(output)
