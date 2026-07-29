"""추천 유스케이스.

대화 흐름을 한 단계 추상화로만 기술한다. 에이전트를 '어떻게' 실행하는지
(이벤트 루프·세션·로깅)는 AgentRunner가, 각 에이전트가 '무슨 말'을 하는지는
프롬프트가 감춘다. 여기서는 '무엇을 하는가'만 읽힌다.
"""
from ..domain.models import Feed, SlotExtraction, Slots


class Recommender:
    def __init__(self, runner, agents) -> None:
        self._runner = runner
        self._agents = agents

    async def understand(self, conversation: str) -> SlotExtraction:
        """지금까지의 대화에서 슬롯을 추출하고, 부족하면 되물을 질문을 얻는다."""
        raw = await self._runner.run(self._agents.slot_extractor, conversation)
        return SlotExtraction.model_validate_json(raw)

    async def recommend(self, slots: Slots) -> Feed:
        """조건을 검색·정리해 추천 카드 피드를 만든다."""
        summary = await self._runner.run(self._agents.coordinator, _search_request(slots))
        cards = await self._runner.run(self._agents.formatter, summary)
        return Feed.model_validate_json(cards)


def _search_request(slots: Slots) -> str:
    return (
        f"조건: 지역={slots.region}, 일자={slots.date}, "
        f"동반자={slots.companion}, 목적={slots.purpose}. "
        f"이 조건에 맞는 곳을 찾아줘."
    )
