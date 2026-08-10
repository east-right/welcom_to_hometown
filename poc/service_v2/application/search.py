"""검색 유스케이스(하네스 flow).

질문 → 짧은 쿼리 분해 → 카카오 전량 수집 → '전부' 리뷰 요약 → 10개 선택.
'전부 검색'을 코드가 보장한다. 모델은 판단(쿼리 분해)과 선택만 한다.
"""
from pydantic import ValidationError

from ..domain.models import Feed, Place, QueryPlan
from ..infrastructure.parsing import json_only

_PER_QUERY = 15    # 쿼리 하나당 카카오에서 모을 수(카카오 페이지 최대)
_MAX_PLACES = 30   # 합쳐서 요약에 태울 최대 장소 수
_REASON_CLIP = 300  # selector에 보여줄 요약 길이


class Search:
    def __init__(self, runner, query_planner, selector, kakao, enricher, flow) -> None:
        self._runner = runner
        self._query_planner = query_planner
        self._selector = selector
        self._kakao = kakao
        self._enricher = enricher
        self._flow = flow

    async def find(self, question: str) -> Feed:
        self._flow.question(question)
        plan = await self._plan(question)
        places = self._collect(plan)
        candidates = await self._enricher.enrich(places, question)
        feed = await self._select(question, candidates)
        self._flow.cards(feed)
        return feed

    async def _plan(self, question: str) -> QueryPlan:
        raw = await self._runner.run(self._query_planner, _plan_request(question))
        plan = QueryPlan.model_validate_json(json_only(raw))
        self._flow.queries(plan.queries)
        return plan

    def _collect(self, plan: QueryPlan) -> list[Place]:
        places: list[Place] = []
        seen: set[str] = set()
        for query in plan.queries:
            for place in self._kakao.collect(query.query, query.category, limit=_PER_QUERY):
                if place.id and place.id not in seen:
                    seen.add(place.id)
                    places.append(place)
        places = places[:_MAX_PLACES]
        self._flow.places(places)
        return places

    async def _select(self, question: str, candidates: list[dict]) -> Feed:
        raw = await self._runner.run(self._selector, _select_request(question, candidates))
        try:
            return Feed.model_validate_json(json_only(raw))
        except ValidationError:
            return Feed(cards=[])


def _plan_request(question: str) -> str:
    return f"질문: {question}\n이 질문을 카카오 검색용 짧은 키워드 3~6개로 쪼개라."


def _select_request(question: str, candidates: list[dict]) -> str:
    lines = [f"[질문] {question}", "", f"[후보 {len(candidates)}곳 — 장소 + 리뷰요약]"]
    for i, c in enumerate(candidates, 1):
        lines.append(f"{i}. {c['name']}  ({c['category']})  {c['address']}")
        meta = " · ".join(x for x in (
            f"운영 {c['hours']}" if c["hours"] else "",
            f"휴무 {c['closed']}" if c["closed"] else "",
            f"기간 {c['period']}" if c["period"] else "",
            "광고성" if c["ad"] else "",
        ) if x)
        if meta:
            lines.append(f"   {meta}")
        if c["reason"]:
            lines.append(f"   요약: {c['reason'][:_REASON_CLIP]}")
        if c["source"]:
            lines.append(f"   출처: {c['source']}")
    lines += ["", "위 후보에서 질문에 맞는 곳 최대 10개를 우선순위 순으로 골라 카드로 만들어라."]
    return "\n".join(lines)
