import httpx
import asyncio

async def check_http2():
    url = "https://localhost:8001/office_stream"
    payload = {
        "input_data": "Llama AI 모델의 출시일과 버전들을 각각 알려줘.",
        "google_access": False,
        "db_id": "123e4567-e89b-12d3-a456-426614174000",
        "user_id": "shaa97102"
    }
    headers = {
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(http2 = True, verify = False, timeout = 30.0) as client:
        response = await client.post(url, json = payload, headers = headers)
        if response.http_version  ==  "HTTP/2":
            print("HTTP/2 is supported")
        else:
            print("HTTP/2 is not supported")
        
        print("Response status code:", response.status_code)
        print("Response body:", response.text)

asyncio.run(check_http2())