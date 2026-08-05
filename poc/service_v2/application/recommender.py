"""추천 유스케이스.

흐름이 한 단계 추상화로만 읽힌다:
  이해(understand) → [수집 → 전부 요약 → 선택](recommend)

수집(카카오)·요약(리뷰 병렬)은 코드가 지휘하고, main 모델은 '요약된 후보 전부'를
받아 '선택·랭킹'만 한다(이름만 보고 미리 추리지 않게).
"""
from pydantic import ValidationError

from ..domain.models import Feed, Place, QueryPlan, SlotExtraction, Slots
from ..infrastructure.parsing import json_only

_PER_QUERY = 10    # 쿼리 하나당 카카오에서 모을 수
_MAX_PLACES = 24   # 합쳐서 요약에 태울 최대 장소 수
_BODY_CLIP = 400   # selector에 보여줄 후기 이유 길이


class Recommender:
    def __init__(self, runner, slot_extractor, query_planner, selector, kakao, enricher) -> None:
        self._runner = runner
        self._slot_extractor = slot_extractor
        self._query_planner = query_planner
        self._selector = selector
        self._kakao = kakao
        self._enricher = enricher

    async def understand(self, conversation: str) -> SlotExtraction:
        """대화에서 슬롯을 추출하고, 부족하면 되물을 질문을 얻는다."""
        raw = await self._runner.run(self._slot_extractor, conversation)
        return SlotExtraction.model_validate_json(json_only(raw))

    async def recommend(self, slots: Slots) -> Feed:
        """쿼리를 쪼개 장소를 모으고 → 전부 후기 요약해 → main이 선택한 카드 피드를 만든다."""
        plan = await self._plan(slots)
        places = self._collect(plan)
        candidates = await self._enricher.enrich(places, slots.region)
        raw = await self._runner.run(self._selector, _select_request(slots, candidates))
        try:
            return Feed.model_validate_json(json_only(raw))
        except ValidationError:
            return Feed(cards=[])

    async def _plan(self, slots: Slots) -> QueryPlan:
        raw = await self._runner.run(self._query_planner, _plan_request(slots))
        return QueryPlan.model_validate_json(json_only(raw))

    def _collect(self, plan: QueryPlan) -> list[Place]:
        """여러 짧은 쿼리로 검색해 id 기준 중복을 제거한 장소 목록."""
        places: list[Place] = []
        seen: set[str] = set()
        for query in plan.queries:
            for place in self._kakao.collect(query.query, query.category, limit=_PER_QUERY):
                if place.id and place.id not in seen:
                    seen.add(place.id)
                    places.append(place)
        return places[:_MAX_PLACES]


def _plan_request(slots: Slots) -> str:
    return (
        f"조건: 지역={slots.region}, 목적={slots.purpose}.\n"
        "이 조건을 카카오 지역검색용 짧은 키워드 검색어 2~5개로 쪼개라."
    )


def _select_request(slots: Slots, candidates: list[dict]) -> str:
    lines = [f"조건: {slots.summary()}", "", f"[후보 {len(candidates)}곳 — 장소 + 후기요약]"]
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
            lines.append(f"   후기: {c['reason'][:_BODY_CLIP]}")
        if c["source"]:
            lines.append(f"   출처: {c['source']}")
    lines += ["", "위 후보에서 조건에 맞는 곳 최대 10개를 우선순위 순으로 골라 카드로 만들어라."]
    return "\n".join(lines)
