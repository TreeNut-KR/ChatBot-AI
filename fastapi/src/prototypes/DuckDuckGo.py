from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import json

# 검색 래퍼 설정 수정
wrapper = DuckDuckGoSearchAPIWrapper(
    region = "kr-kr",
    safesearch = "moderate",
    max_results = 100,  # 최대 결과 수 증가
    time = "y",  # 검색 기간을 1년으로 설정
    backend = "auto"  # 자동 백엔드 선택
)

# DuckDuckGoSearchResults 도구 설정
search = DuckDuckGoSearchResults(
    api_wrapper = wrapper,
    num_results = 20,  # 반환할 결과 수 증가
    output_format = "json",  # JSON 형식으로 출력 설정
    backend = "text"  # 텍스트 검색 사용
)

# 검색어를 사용하여 검색 수행
query = "Llama 3"
result = search.invoke(query)  # 수정된 부분

# JSON 결과 파싱 및 출력
print(f"검색어: {query}\n")
print("검색 결과:")

# JSON 문자열을 파이썬 객체로 변환
search_results = json.loads(result)

# 결과 출력
for idx, item in enumerate(search_results, 1):
    print(f"\n[결과 {idx}]")
    print(f"제목: {item.get('title', 'N/A')}")
    print(f"링크: {item.get('link', 'N/A')}")
    print(f"내용: {item.get('snippet', 'N/A')}")