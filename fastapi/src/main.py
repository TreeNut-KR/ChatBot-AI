import asyncio
from utils.Database_mongo import MongoDBHandler

async def main():
    db_handler = MongoDBHandler()
    await db_handler.insert_dataset('fastapi/datasets/HAERAE-HUB_KOREAN-WEBTEXT_train.json', 'dataset')

if __name__ == "__main__":
    asyncio.run(main())