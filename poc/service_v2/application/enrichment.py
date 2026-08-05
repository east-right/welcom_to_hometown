"""장소 목록(뼈대)에 후기 요약(살)을 붙이는 단계.

카카오가 준 장소들을 '전부' 병렬로 blog 후기 검색·본문 취득한 뒤, 각 장소마다
review 서브에이전트로 요약한다. 여기서 미리 추리지 않는다 — 선택은 뒤(selector)에서.
후기 원문(대량)은 여기 갇히고, 밖으로는 압축된 요약만 나간다(컨텍스트 방화벽).
"""
import asyncio

from pydantic import ValidationError

from ..domain.models import Place, Review
from ..infrastructure.parsing import json_only


class PlaceEnricher:
    def __init__(self, naver, runner, review_agent, log=None) -> None:
        self._naver = naver
        self._runner = runner              # NullTracer 러너(병렬이라 무음)
        self._review_agent = review_agent
        self._log = log or (lambda _line: None)

    async def enrich(self, places: list[Place], region: str) -> list[dict]:
        """모든 장소를 동시에 후기 요약해 [장소+요약] dict 목록으로 돌려준다."""
        candidates = await asyncio.gather(*(self._one(p, region) for p in places))
        self._log(f"📄 {len(places)}곳 후기 병렬 요약 완료")
        return candidates

    async def _one(self, place: Place, region: str) -> dict:
        reviews = await self._naver.collect(f"{region} {place.name}".strip(), n=3)
        review = await self._summarize(place, reviews)
        return {
            "name": place.name,
            "category": place.category,
            "address": place.address,
            **review.model_dump(),
        }

    async def _summarize(self, place: Place, reviews: list[dict]) -> Review:
        if not any(r["text"] for r in reviews if not r["text"].startswith("[")):
            return Review()
        raw = await self._runner.run(self._review_agent, _review_input(place.name, reviews))
        try:
            return Review.model_validate_json(json_only(raw))
        except ValidationError:
            return Review()


def _review_input(place_name: str, reviews: list[dict]) -> str:
    lines = [f"장소: {place_name}", "", "[블로그 후기 원문]"]
    for review in reviews:
        lines += [f"- {review['title']}  ({review['url']})", review["text"][:1000], ""]
    lines.append("위 후기에서 이 장소의 정보를 규칙대로 요약해라.")
    return "\n".join(lines)
