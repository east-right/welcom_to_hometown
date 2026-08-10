"""후보 장소 '전부'에 리뷰 요약을 붙이는 단계(하네스의 핵심).

카카오가 찾은 장소들을 한 곳도 빼지 않고 '전부' 병렬로 blog 검색·본문 취득한 뒤,
각 장소마다 review 서브에이전트로 '질문 연관 3줄' 요약한다. 여기서 추리지 않는다 —
선택은 뒤(selector)에서. '전부 검색'을 코드가 보장하므로 모델의 변덕에 의존하지 않는다.
"""
import asyncio

from pydantic import ValidationError

from ..domain.models import Place, Review
from ..infrastructure.parsing import json_only


class PlaceEnricher:
    def __init__(self, naver, runner, review_agent, flow) -> None:
        self._naver = naver
        self._runner = runner
        self._review_agent = review_agent
        self._flow = flow

    async def enrich(self, places: list[Place], question: str) -> list[dict]:
        """모든 장소를 동시에 blog 검색·요약해 [장소+요약] dict 목록으로 돌려준다."""
        self._flow.enrich_header(len(places))
        pairs = await asyncio.gather(*(self._one(p, question) for p in places))
        for candidate, urls in pairs:
            self._flow.enrich_one(candidate["name"], urls)
        return [candidate for candidate, _ in pairs]

    async def _one(self, place: Place, question: str) -> tuple[dict, list]:
        reviews = await self._naver.collect(place.name, n=3)
        urls = [(r["url"], _ok(r["text"])) for r in reviews]
        review = await self._summarize(place, question, reviews)
        candidate = {
            "name": place.name,
            "category": place.category,
            "address": place.address,
            **review.model_dump(),
        }
        return candidate, urls

    async def _summarize(self, place: Place, question: str, reviews: list[dict]) -> Review:
        if not any(_ok(r["text"]) for r in reviews):
            return Review()
        raw = await self._runner.run(
            self._review_agent, _review_input(question, place, reviews), tag=place.name
        )
        try:
            return Review.model_validate_json(json_only(raw))
        except ValidationError:
            return Review()


def _ok(text: str) -> bool:
    return bool(text) and not text.startswith("[")


def _review_input(question: str, place: Place, reviews: list[dict]) -> str:
    lines = [
        f"[질문] {question}",
        f"[장소] {place.name} ({place.category}) {place.address}",
        "",
        "[블로그 후기 원문]",
    ]
    for review in reviews:
        lines += [f"- {review['title']}  ({review['url']})", review["text"][:1000], ""]
    lines.append("위 후기로 이 장소를 질문과 연관되게 요약해라.")
    return "\n".join(lines)
