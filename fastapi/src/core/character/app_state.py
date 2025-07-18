from typing import Optional
from domain import (
    mongodb_client,
    error_tools,
    queue_tools,
    character_config,
)
# from llm import character_llama

RED = "\033[31m"
RESET = "\033[0m"

# llama_queue_handler: Optional[queue_tools.LlamaQueueHandler] = None
mongo_handler: Optional[mongodb_client.MongoDBHandler] = None

try:
    # # 큐 핸들러 초기화 (순차 처리 모드)
    # llama_queue_handler = queue_tools.LlamaQueueHandler(
    #     service_type=queue_tools.ServiceType.CHARACTER,
    #     model_class=character_llama.LlamaCharacterModel,
    #     processing_request_class=character_config.ProcessingRequest,
    #     max_concurrent=2,
    # )
    mongo_handler = mongodb_client.MongoDBHandler()  # 비동기 초기화는 lifespan
except error_tools.InternalServerErrorException as e:
    mongo_handler = None
    print(f"{RED}ERROR{RESET}:    MongoDB 초기화 오류 발생: {str(e)}")
except FileNotFoundError as e:
    llama_queue_handler = None
    print(f"{RED}ERROR{RESET}:    큐 핸들러 초기화 오류: {str(e)}")
except Exception as e:
    llama_queue_handler = None
    print(f"{RED}ERROR{RESET}:    예상치 못한 오류: {str(e)}")