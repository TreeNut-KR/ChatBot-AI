from typing import Optional
from .handlers.mongodb_handler import MongoDBHandler
from .handlers.queue_handler import LlamaQueueHandler
from .handlers import error_handler as ChatError

RED = "\033[31m"
RESET = "\033[0m"

llama_queue_handler: Optional[LlamaQueueHandler] = None
mongo_handler: Optional[MongoDBHandler] = None

try:
    # 큐 핸들러 초기화 (순차 처리 모드)
    llama_queue_handler = LlamaQueueHandler(max_concurrent=2)
    mongo_handler = MongoDBHandler()  # 비동기 초기화는 lifespan에서!
except ChatError.InternalServerErrorException as e:
    mongo_handler = None
    print(f"{RED}ERROR{RESET}:    MongoDB 초기화 오류 발생: {str(e)}")
except FileNotFoundError as e:
    llama_queue_handler = None
    print(f"{RED}ERROR{RESET}:    큐 핸들러 초기화 오류: {str(e)}")
except Exception as e:
    llama_queue_handler = None
    print(f"{RED}ERROR{RESET}:    예상치 못한 오류: {str(e)}")