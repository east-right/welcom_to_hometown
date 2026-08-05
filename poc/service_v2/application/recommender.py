"""추천 유스케이스.

두 단계로만 읽힌다: 대화에서 조건을 이해하고(understand), 그 조건으로 추천한다(recommend).
recommend는 coordinator(agentic 에이전트)를 한 번 돌릴 뿐 — '어떻게 찾을지'(검색·후기 확인·
멈출 시점)는 그 에이전트가 도구를 써서 스스로 정한다.
"""
from ..domain.models import Feed, SlotExtraction, Slots
from ..infrastructure.parsing import json_only


class Recommender:
    def __init__(self, runner, slot_extractor, coordinator) -> None:
        self._runner = runner
        self._slot_extractor = slot_extractor
        self._coordinator = coordinator

    async def understand(self, conversation: str) -> SlotExtraction:
        """대화에서 슬롯을 추출하고, 부족하면 되물을 질문을 얻는다."""
        raw = await self._runner.run(self._slot_extractor, conversation)
        return SlotExtraction.model_validate_json(json_only(raw))

    async def recommend(self, slots: Slots) -> Feed:
        """조건을 coordinator에게 넘겨, 스스로 검색·확인해 만든 카드 피드를 받는다."""
        raw = await self._runner.run(self._coordinator, _request(slots))
        return Feed.model_validate_json(json_only(raw))


def _request(slots: Slots) -> str:
    return (
        f"조건: {slots.summary()}.\n"
        "이 조건에 맞는 곳을 찾아 추천 카드를 만들어라. "
        "find_places로 실제 장소를 찾고, 영업시간·기간 등이 필요하면 read_reviews로 확인해라."
    )
