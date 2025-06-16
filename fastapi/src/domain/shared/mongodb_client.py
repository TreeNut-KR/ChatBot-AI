'''
MongoDBHandler 클래스는 MongoDB에 연결하고 데이터베이스, 컬렉션 목록을 가져오는 클래스입니다.
'''

import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

from pymongo.errors import PyMongoError
from .error_tools import InternalServerErrorException

# ANSI 색상 코드
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

class MongoDBHandler:
    """
    MongoDB 데이터베이스 작업을 처리하는 핸들러 클래스입니다.
    
    Attributes:
        mongo_uri (str): MongoDB 연결 문자열
        client (AsyncIOMotorClient): MongoDB 비동기 클라이언트 인스턴스
        db (AsyncIOMotorDatabase): 기본 데이터베이스 인스턴스
    
    Raises:
        InternalServerErrorException: MongoDB 연결 또는 초기화 중 오류 발생 시
    """
    def __init__(self) -> None:
        """
        MongoDBHandler 클래스를 초기화합니다.
        
        환경 변수에서 MongoDB 연결 정보를 로드하고 데이터베이스에 연결합니다.
        
        Raises:
            InternalServerErrorException:
                - 환경 변수 로드 실패
                - MongoDB 연결 실패
                - 기타 초기화 오류
        """
        try:
            # 환경 변수 파일 경로 설정 수정
            env_file_path = Path(__file__).resolve().parents[3] / ".env"
            
            if not os.path.exists(env_file_path):
                raise FileNotFoundError(f".env 파일을 찾을 수 없습니다: {env_file_path}")
            
            load_dotenv(env_file_path)
            
            # 환경 변수에서 MongoDB 연결 정보 가져오기
            self.mongo_host = os.getenv("MONGO_HOST")
            self.mongo_port = os.getenv("MONGO_PORT")
            mongo_user = os.getenv("MONGO_ADMIN_USER")
            mongo_password = os.getenv("MONGO_ADMIN_PASSWORD")
            mongo_db = os.getenv("MONGO_DATABASE")
            
            # MongoDB URI 생성 - URI 옵션 형식 수정
            self.mongo_uri = (
                f"mongodb://{mongo_user}:{mongo_password}@{self.mongo_host}:{self.mongo_port}/{mongo_db}"
                "?authSource=admin"  # 첫 번째 옵션
                "&serverSelectionTimeoutMS=500"  # 두 번째 옵션
                "&connectTimeoutMS=1000"  # 세 번째 옵션
                "&socketTimeoutMS=6000"   # 네 번째 옵션
            )
            
            # 클라이언트와 데이터베이스 초기화
            self.client = None
            self.db = None
            
            print(f"{GREEN}INFO{RESET}:     MongoDBHandler 초기화 완료")
                
        except Exception as e:
            raise InternalServerErrorException(detail = f"MongoDBHandler 초기화 오류: {str(e)}")
    
    async def init(self):
        """
        MongoDB 클라이언트를 초기화하고 데이터베이스에 연결합니다.

        Raises:
            InternalServerErrorException: MongoDB 연결 중 오류 발생 시
        """
        try:
            # MongoDB 클라이언트 초기화
            self.client = AsyncIOMotorClient(self.mongo_uri)
            
            # 연결 테스트
            await self.client.admin.command('ping')
            
            # 데이터베이스 선택
            self.db = self.client[os.getenv("MONGO_DATABASE")]
            print(f"{GREEN}INFO{RESET}:     MongoDB 연결 성공")
        
        except PyMongoError as e:
            print(f"{RED}ERROR{RESET}:    MongoDB 연결 실패")
            raise InternalServerErrorException(detail = f"MongoDB 연결 오류 - 호스트: {self.mongo_host}, 포트: {self.mongo_port}")
        except Exception as e:
            raise InternalServerErrorException(detail = f"MongoDBHandler 초기화 오류: {str(e)}")
    
    async def get_office_log(self, user_id: str, document_id: str, router: str) -> List[Dict]:
        try:
            collection = self.db[f'{router}_log_{user_id}']

            document = await collection.find_one({"id": document_id})

            if document is None or not document.get("value", []):
                return []

            value_list = document.get("value", [])
            sorted_value_list = sorted(value_list, key = lambda x: x.get("index", 0))

            # 최신 8개만 선택
            latest_messages = sorted_value_list[-8:] if len(sorted_value_list) > 8 else sorted_value_list

            # 대화 기록 형식 수정
            formatted_chat_list = []
            for chat in latest_messages:
                formatted_chat = {
                    "index": chat.get("index"),
                    "input_data": chat.get("input_data"),
                    "output_data": chat.get("output_data")
                }
                formatted_chat_list.append(formatted_chat)

            return formatted_chat_list

        except PyMongoError as e:
            raise InternalServerErrorException(detail = f"Error retrieving chatlog value: {str(e)}")
        except Exception as e:
            raise InternalServerErrorException(detail = f"Unexpected error: {str(e)}")

    async def get_character_log(self, user_id: str, document_id: str, router: str) -> List[Dict]:
        """
        최신 10개의 대화 기록을 가져와서 Llama 프롬프트 형식으로 변환합니다.
        """
        try:
            collection = self.db[f'{router}_log_{user_id}']

            document = await collection.find_one({"id": document_id})

            if document is None or not document.get("value", []):
                return []

            value_list = document.get("value", [])
            sorted_value_list = sorted(value_list, key = lambda x: x.get("index", 0))
            latest_messages = sorted_value_list[-10:] if len(sorted_value_list) > 10 else sorted_value_list

            # 필요시 포맷 변환
            return latest_messages

        except PyMongoError as e:
            raise InternalServerErrorException(detail = f"Error retrieving chatlog value: {str(e)}")
        except Exception as e:
            raise InternalServerErrorException(detail = f"Unexpected error: {str(e)}")
