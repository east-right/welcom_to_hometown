"""카카오 로컬(키워드 장소검색) 어댑터.

'검증된 장소 엔티티'(상호·주소·좌표·카테고리)를 넓게 가져오는 소스.
네이버 로컬(쿼리당 5개)과 달리 size 15 + 페이징이라 커버리지가 넓다.
단, 영업시간·기간·후기는 주지 않는다 — 그건 blog 크롤링(naver)의 몫.
"""
import httpx

from ..domain.models import Place

_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


class KakaoLocal:
    def __init__(self, api_key: str) -> None:
        self._headers = {"Authorization": f"KakaoAK {api_key}"}

    def search(self, query: str, category: str = "", size: int = 15) -> list[Place]:
        """키워드로 장소를 검색한다. 실패하면 빈 목록.

        category: '' | 'FD6'(음식점) | 'CE7'(카페) | 'AT4'(관광명소) 등.
        """
        params = {"query": query, "size": size, "sort": "accuracy"}
        if category:
            params["category_group_code"] = category
        try:
            response = httpx.get(_URL, headers=self._headers, params=params, timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        return [Place.from_kakao(doc) for doc in response.json().get("documents", [])]
