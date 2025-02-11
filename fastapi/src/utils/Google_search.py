# Google_search.py
'''
파일은 Google 검색 결과를 가져오는 모듈입니다. 이 파일은 다음과 같은 기능을 제공합니다.
'''

import os
import httpx

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# def stream_search_results(search_results: dict):
#     items = search_results.get("items", [])
#     for item in items[:2]:  # 최대 3개의 결과만 스트리밍
#         search_data_set = f"{item['title']}: {item['snippet']}\n"
#         print(search_data_set)
#         yield search_data_set

async def fetch_results(query: str, num: int, domain: str = "") -> list:
    """
    Google 검색 결과를 가져오는 함수
    :param query: 검색어
    :param num: 가져올 결과 수
    :param domain: 특정 도메인 필터 (없으면 전체 검색)
    :return: 검색 결과 리스트
    """
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": f"{query} {domain}".strip(),
        "num": min(num, 10)
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            search_results = response.json()

            return [
                {
                    "title": item.get("title", "제목 없음"),
                    "snippet": item.get("snippet", "설명 없음"),
                    "link": item.get("link", "링크 없음")
                }
                for item in search_results.get("items", [])
            ]
    except httpx.RequestError as e:
        print(f"HTTP 요청 오류: {str(e)}")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
    return []

async def fetch_search_results(query: str, num_results: int = 5) -> list:
    """
    위키백과, 나무위키, 다양한 뉴스 사이트 등, 사이트 기반으로 검색 결과를 가져옵니다.
    부족한 경우 최상단 검색 결과 추가.
    :param query: 검색어
    :param num_results: 가져올 각 도메인별 검색 결과 수
    :return: 검색 결과 리스트 (제목, 설명, 링크 포함)
    """
    domains = [
        "site:en.wikipedia.org",# 영어 위키백과
        "site:ko.wikipedia.org",# 한국어 위키백과
        "site:namu.wiki",       # 나무위키
        "site:news.naver.com",  # 네이버 뉴스
        "site:bbc.com",         # BBC
        "site:cnn.com",         # CNN
        "site:reuters.com",     # 로이터
        "site:nytimes.com",     # 다양한 뉴스 사이트
        "site:dcinside.com",    # 디시인사이드
        "site:reddit.com",      # 레딧
        "site:naver.com"        # 네이버
    ]

    all_results = []
    total_results_needed = num_results * len(domains)

    for domain in domains:
        domain_results = await fetch_results(query=query, num=num_results, domain=domain)
        all_results.extend(domain_results)

    if len(all_results) < total_results_needed:
        remaining_needed = total_results_needed - len(all_results)
        general_results = await fetch_results(query=query, num=remaining_needed)
        all_results.extend(general_results)

    return all_results[:total_results_needed]

