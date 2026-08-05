"""메인 에이전트가 쓰는 도구.

composition root(main)에서 인프라를 주입해 만든다(클로저) — 도구가 어떤 클라이언트를
쓰는지는 조립 시점에 정해지고, 에이전트는 함수 시그니처·docstring만 본다.

read_reviews가 '컨텍스트 방화벽'이다: blog 원문(대량)을 여기서 먹고 요약만 돌려주므로,
메인 에이전트의 컨텍스트엔 압축된 후기 요약만 들어간다(긴 루프여도 안 터짐).
"""
from ..domain.models import Review
from ..infrastructure.parsing import json_only


def make_find_places(kakao):
    def find_places(query: str, category: str = "") -> list:
        """카카오에서 실제 장소를 검색한다(상호·카테고리·주소).

        Args:
            query: 검색어. 지역명 포함. 예 '창원 귀산 카페'
            category: '' | 'FD6'(음식점) | 'CE7'(카페) | 'AT4'(관광명소). 축제·행사는 비워라.
        """
        return [
            {"name": p.name, "category": p.category, "address": p.address}
            for p in kakao.search(query, category)
        ]

    return find_places


def make_read_reviews(naver, runner, review_agent):
    async def read_reviews(place_name: str, region: str = "") -> dict:
        """장소명으로 블로그 후기를 읽어 운영시간·휴무·축제기간·광고여부·추천이유·근거URL을 돌려준다.

        카카오는 영업시간/기간을 주지 않으므로, 그 정보가 필요할 때 이 도구를 써라.

        Args:
            place_name: 장소/행사 이름
            region: 지역명(검색 정확도용). 예 '창원 귀산'
        """
        hits = naver.search(f"{region} {place_name}".strip(), n=3)
        bodies = await naver.fetch([h["url"] for h in hits]) if hits else []
        if not any(b for b in bodies if not b.startswith("[")):
            return Review().model_dump()
        raw = await runner.run(review_agent, _review_input(place_name, hits, bodies))
        return Review.model_validate_json(json_only(raw)).model_dump()

    return read_reviews


def _review_input(place_name: str, hits: list, bodies: list) -> str:
    lines = [f"장소: {place_name}", "", "[블로그 후기 원문]"]
    for hit, body in zip(hits, bodies):
        lines += [f"- {hit['title']}  ({hit['url']})", body[:1000], ""]
    lines.append("위 후기에서 이 장소의 정보를 규칙대로 요약해라.")
    return "\n".join(lines)
