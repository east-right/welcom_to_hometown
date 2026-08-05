"""장소 목록(뼈대)에 후기 요약(살)을 붙이는 단계.

카카오가 준 장소들을 '전부' 병렬로 blog 후기 검색·본문 취득한 뒤, 각 장소마다
review 서브에이전트로 요약한다. 여기서 미리 추리지 않는다 — 선택은 뒤(selector)에서.
후기 원문(대량)은 여기 갇히고, 밖으로는 압축된 요약만 나간다(컨텍스트 방화벽).

병렬로 도는 각 매장의 URL·fetch 성패는 gather가 끝난 뒤 flow 로그에 순차로 남긴다.
"""
import asyncio

from pydantic import ValidationError

from ..domain.models import Place, Review
from ..infrastructure.parsing import json_only


class PlaceEnricher:
    def __init__(self, naver, runner, review_agent, flow) -> None:
        self._naver = naver
        self._runner = runner              # 병렬이라 AgentLog만(무음 트레이스)
        self._review_agent = review_agent
        self._flow = flow                  # 파이프라인 데이터 흐름 로그

    async def enrich(self, places: list[Place], region: str) -> list[dict]:
        """모든 장소를 동시에 후기 요약해 [장소+요약] dict 목록으로 돌려준다."""
        pairs = await asyncio.gather(*(self._one(p, region) for p in places))
        self._flow.reviews_header()
        for candidate, url_status in pairs:
            self._flow.reviews(candidate["name"], url_status)
        return [candidate for candidate, _ in pairs]

    async def _one(self, place: Place, region: str) -> tuple[dict, list]:
        reviews = await self._naver.collect(f"{region} {place.name}".strip(), n=3)
        url_status = [(r["url"], _ok(r["text"]), len(r["text"])) for r in reviews]
        review = await self._summarize(place, reviews)
        candidate = {
            "name": place.name,
            "category": place.category,
            "address": place.address,
            **review.model_dump(),
        }
        return candidate, url_status

    async def _summarize(self, place: Place, reviews: list[dict]) -> Review:
        if not any(_ok(r["text"]) for r in reviews):
            return Review()
        raw = await self._runner.run(
            self._review_agent, _review_input(place.name, reviews), tag=place.name
        )
        try:
            return Review.model_validate_json(json_only(raw))
        except ValidationError:
            return Review()


def _ok(text: str) -> bool:
    """fetch가 실제 본문을 가져왔는지(실패 표식 '[...]'로 시작 안 하는지)."""
    return bool(text) and not text.startswith("[")


def _review_input(place_name: str, reviews: list[dict]) -> str:
    lines = [f"장소: {place_name}", "", "[블로그 후기 원문]"]
    for review in reviews:
        lines += [f"- {review['title']}  ({review['url']})", review["text"][:1000], ""]
    lines.append("위 후기에서 이 장소의 정보를 규칙대로 요약해라.")
    return "\n".join(lines)
