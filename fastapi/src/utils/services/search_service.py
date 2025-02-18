"""
파일은 Google 및 DuckDuckGo 검색 API를 사용하여 웹 검색을 수행하는 서비스입니다.
"""

import os
import httpx
import json
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from urllib.parse import urlparse

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

def get_domain(url: str) -> str:
    """
    URL에서 메인 도메인 추출
    
    Args:
        url (str): 도메인을 추출할 URL 문자열
    
    Returns:
        str: 추출된 도메인 이름
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return ""

async def fetch_google_raw_results(query: str, num_results: int = 50) -> list:
    """
    Google API로부터 직접 검색 결과를 가져오는 함수
    
    Args:
        query (str): 검색할 키워드
        num_results (int, optional): 가져올 결과 수. 기본값 50
    
    Returns:
        list: Google 검색 결과 목록. 각 항목은 title, link, snippet 등을 포함하는 딕셔너리
    """
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": min(num_results, 50)  # Google API 최대 제한
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            return response.json().get("items", [])
    except Exception as e:
        print(f"검색 오류: {str(e)}")
        return []

async def fetch_google_filtered_results(query: str, num_results: int = 5) -> list:
    """
    Google 검색 결과를 도메인별로 필터링하여 가져오는 함수
    
    Args:
        query (str): 검색할 키워드
        num_results (int, optional): 각 도메인당 가져올 결과 수. 기본값 5
    
    Returns:
        list: 필터링된 검색 결과 목록. 각 항목은 title, snippet, link, source를 포함하는 딕셔너리
    """
    # 관심있는 도메인 목록
    target_domains = {
        "wikipedia.org": "위키백과",
        "namu.wiki": "나무위키",
        "news.naver.com": "네이버뉴스",
        "bbc.com": "BBC",
        "cnn.com": "CNN",
        "reuters.com": "로이터",
        "nytimes.com": "뉴욕타임스",
        "reddit.com": "레딧",
        "naver.com": "네이버"
    }

    # 한 번의 API 호출로 전체 결과 가져오기
    search_results = await fetch_google_raw_results(query, 50)
    
    # 도메인별로 결과 분류
    domain_results = {}
    other_results = []
    
    for item in search_results:
        domain = get_domain(item.get("link", ""))
        result = {
            "title": item.get("title", "제목 없음"),
            "snippet": item.get("snippet", "설명 없음"),
            "link": item.get("link", "링크 없음"),
            "source": None
        }
        
        # 관심 도메인 확인
        matched = False
        for target_domain, source_name in target_domains.items():
            if target_domain in domain:
                result["source"] = source_name
                if target_domain not in domain_results:
                    domain_results[target_domain] = []
                domain_results[target_domain].append(result)
                matched = True
                break
        
        if not matched:
            other_results.append(result)

    # 결과 병합 (각 도메인당 최대 num_results개)
    final_results = []
    for domain_items in domain_results.values():
        final_results.extend(domain_items[:num_results])
    
    # 부족한 경우 기타 결과로 보충
    if len(final_results) < num_results * len(target_domains):
        remaining = num_results * len(target_domains) - len(final_results)
        final_results.extend(other_results[:remaining])

    return final_results[:num_results * len(target_domains)]

async def fetch_duck_search_results(query: str) -> list:
    """
    DuckDuckGo에서 텍스트와 뉴스 검색 결과를 모두 가져오는 함수
    
    Args:
        query (str): 검색할 키워드
    
    Returns:
        list: DuckDuckGo 검색 결과 목록. 각 항목은 title, link, snippet를 포함하는 딕셔너리
    """
    # 검색 래퍼 설정 수정
    wrapper = DuckDuckGoSearchAPIWrapper(
        region="kr-kr",
        safesearch="moderate",
        max_results=100,  # 최대 결과 수 증가
        time="y",  # 검색 기간을 1년으로 설정
        backend="auto"  # 자동 백엔드 선택
    )

    # DuckDuckGoSearchResults 도구 설정
    search = DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        num_results=20,  # 반환할 결과 수 증가
        output_format="json",  # JSON 형식으로 출력 설정
        backend="text"  # 텍스트 검색 사용
    )

    # 검색 실행 및 결과 병합
    text_results = json.loads(search.invoke(query))
    
    return text_results