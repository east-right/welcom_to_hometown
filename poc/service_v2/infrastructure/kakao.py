"""카카오 로컬(키워드 장소검색) 어댑터.

'검증된 장소 엔티티'(상호·주소·좌표·카테고리)를 넓게 가져오는 소스.
페이지네이션으로 원하는 개수만큼(기본 20) 모은다(쿼리당 5개인 네이버 로컬과 달리).
단, 영업시간·기간·후기는 주지 않는다 — 그건 blog 크롤링(naver)의 몫.
"""
import httpx

from ..domain.models import Place

_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
_PAGE_SIZE = 15   # 카카오 최대


class KakaoLocal:
    def __init__(self, api_key: str) -> None:
        self._headers = {"Authorization": f"KakaoAK {api_key}"}

    def collect(self, query: str, category: str = "", limit: int = 20) -> list[Place]:
        """페이지를 넘겨가며 중복 없는 장소를 limit개까지 모은다.

        category: '' | 'FD6'(음식점) | 'CE7'(카페) | 'AT4'(관광명소) 등.
        """
        places: list[Place] = []
        seen: set[str] = set()
        for page in range(1, 4):   # 최대 3페이지(45개)까지
            documents, is_end = self._fetch(query, category, page)
            for doc in documents:
                place = Place.from_kakao(doc)
                if place.id and place.id not in seen:
                    seen.add(place.id)
                    places.append(place)
                    if len(places) >= limit:
                        return places
            if is_end:
                break
        return places

    def _fetch(self, query: str, category: str, page: int) -> tuple[list[dict], bool]:
        params = {"query": query, "size": _PAGE_SIZE, "page": page, "sort": "accuracy"}
        if category:
            params["category_group_code"] = category
        try:
            response = httpx.get(_URL, headers=self._headers, params=params, timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return [], True
        body = response.json()
        return body.get("documents", []), body.get("meta", {}).get("is_end", True)
