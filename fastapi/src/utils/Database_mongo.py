# Database_mongo.py
'''
MongoDBHandler 클래스는 MongoDB에 연결하고 데이터베이스, 컬렉션 목록을 가져오는 클래스입니다.
'''
import os
import json
from typing import Dict, List
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError

from .Error_handlers import InternalServerErrorException, NotFoundException

class MongoDBHandler:
    def __init__(self) -> None:
        """
        MongoDBHandler 클래스 초기화.
        MongoDB에 연결하고 필요한 환경 변수를 로드합니다.
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
        데이터베이스 이름 목록을 반환합니다.
        
        :return: 데이터베이스 이름 리스트
        :raises InternalServerErrorException: 데이터베이스 이름을 가져오는 도중 문제가 발생할 경우
        """
        try:
            return await self.client.list_database_names()
        except PyMongoError as e:
            raise InternalServerErrorException(detail=f"Error retrieving database names: {str(e)}")
        except Exception as e:
            raise InternalServerErrorException(detail=f"Unexpected error: {str(e)}")

    async def get_collection(self, database_name: str) -> List[str]:
        """
        데이터베이스의 컬렉션 이름 목록을 반환합니다.
        
        :param database_name: 데이터베이스 이름
        :return: 컬렉션 이름 리스트
        :raises NotFoundException: 데이터베이스가 존재하지 않을 경우
        :raises InternalServerErrorException: 컬렉션 이름을 가져오는 도중 문제가 발생할 경우
        """
        db_names = await self.get_db_names()
        if database_name not in db_names:
            raise NotFoundException(f"Database '{database_name}' not found.")
        try:
            return await self.client[database_name].list_collection_names()
        except PyMongoError as e:
            raise InternalServerErrorException(detail=f"Error retrieving collection names: {str(e)}")
        except Exception as e:
            raise InternalServerErrorException(detail=f"Unexpected error: {str(e)}")
