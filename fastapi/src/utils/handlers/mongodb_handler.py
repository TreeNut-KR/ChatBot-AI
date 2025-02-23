'''
MongoDBHandler 클래스는 MongoDB에 연결하고 데이터베이스, 컬렉션 목록을 가져오는 클래스입니다.
'''

import os
from typing import List
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError

from .error_handler import InternalServerErrorException, NotFoundException

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
            # 환경 변수 파일 경로 설정
            current_directory = os.path.dirname(os.path.abspath(__file__))
            env_file_path = os.path.join(current_directory, '../.env')
            load_dotenv(env_file_path)
            
            # 환경 변수에서 MongoDB 연결 URI 가져오기
            mongo_host = os.getenv("MONGO_HOST")
            mongo_port = os.getenv("MONGO_PORT", 27018)
            mongo_user = os.getenv("MONGO_ADMIN_USER")
            mongo_password = os.getenv("MONGO_ADMIN_PASSWORD")
            mongo_db = os.getenv("MONGO_DATABASE")
            mongo_auth = os.getenv("MONGO_AUTH")
            
            # MongoDB URI 생성
            self.mongo_uri = (
                f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/{mongo_db}?authSource={mongo_auth}"
            )
            
            # MongoDB 클라이언트 초기화
            self.client = AsyncIOMotorClient(self.mongo_uri)
            self.db = self.client[mongo_db]
        except PyMongoError as e:
            raise InternalServerErrorException(detail=f"MongoDB connection error: {str(e)}")
        except Exception as e:
            raise InternalServerErrorException(detail=f"Error initializing MongoDBHandler: {str(e)}")

    async def get_db(self) -> List[str]:
        """
        사용 가능한 모든 데이터베이스 이름 목록을 조회합니다.
        
        Returns:
            List[str]: 데이터베이스 이름 목록
            
        Raises:
            InternalServerErrorException: 
                - MongoDB 서버 연결 실패
                - 데이터베이스 목록 조회 실패
                - 예상치 못한 오류 발생
        """
        try:
            return await self.client.list_database_names()
        except PyMongoError as e:
            raise InternalServerErrorException(detail=f"Error retrieving database names: {str(e)}")
        except Exception as e:
            raise InternalServerErrorException(detail=f"Unexpected error: {str(e)}")

    async def get_collection(self, database_name: str) -> List[str]:
        """
        지정된 데이터베이스의 모든 컬렉션 이름 목록을 조회합니다.
        
        Args:
            database_name (str): 조회할 데이터베이스의 이름
            
        Returns:
            List[str]: 컬렉션 이름 목록
            
        Raises:
            NotFoundException:
                - 지정된 데이터베이스가 존재하지 않는 경우
            InternalServerErrorException:
                - MongoDB 서버 연결 실패
                - 컬렉션 목록 조회 실패
                - 예상치 못한 오류 발생
        """
        db_names = await self.get_db()
        if database_name not in db_names:
            raise NotFoundException(f"Database '{database_name}' not found.")
        try:
            return await self.client[database_name].list_collection_names()
        except PyMongoError as e:
            raise InternalServerErrorException(detail=f"Error retrieving collection names: {str(e)}")
        except Exception as e:
            raise InternalServerErrorException(detail=f"Unexpected error: {str(e)}")
