# Database_mysql.py
'''
파일은 MySQLDBHandler 클래스를 정의하고, 이 클래스는 MySQL 데이터베이스와 상호 작용하는 데 필요한 모든 기능을 제공합니다.
'''

import os
import asyncio
from dotenv import load_dotenv
from typing import NoReturn, List
from databases import Database

class MySQLDBHandler:
    def __init__(self) -> NoReturn:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        env_file_path = os.path.join(current_directory, '../.env')
        load_dotenv(env_file_path)

        self.database = Database(
            f"mysql://{os.getenv('MYSQL_ROOT_USER')}:" \
            f"{os.getenv('MYSQL_ROOT_PASSWORD')}@" \
            f"{os.getenv('MYSQL_ROOT_HOST')}:" \
            f"{os.getenv('MYSQL_ROOT_PORT')}/" \
            f"{os.getenv('MYSQL_DATABASE')}"
        )

    async def connect(self):
        await self.database.connect()

    async def disconnect(self):
        await self.database.disconnect()

    async def execute_query(self, query: str, params: tuple = None) -> List[dict]:
        result = await self.database.fetch_all(query=query, values=params)
        return [dict(row) for row in result]

    async def get_tables(self) -> List[str]:
        query = "SHOW TABLES"
        tables = await self.execute_query(query)
        return [table[f'Tables_in_{os.getenv("MYSQL_DATABASE")}'] for table in tables]