"""추천 유스케이스.

대화 흐름을 한 단계 추상화로만 기술한다. 에이전트를 '어떻게' 실행하는지
(이벤트 루프·세션·로깅)는 AgentRunner가, 각 에이전트가 '무슨 말'을 하는지는
프롬프트가 감춘다. 여기서는 '무엇을 하는가'만 읽힌다.
"""
import json
import re

from ..domain.models import Feed, SlotExtraction, Slots

_DECODER = json.JSONDecoder()


class Recommender:
    def __init__(self, runner, agents) -> None:
        self._runner = runner
        self._agents = agents

    async def understand(self, conversation: str) -> SlotExtraction:
        """지금까지의 대화에서 슬롯을 추출하고, 부족하면 되물을 질문을 얻는다."""
        raw = await self._runner.run(self._agents.slot_extractor, conversation)
        return SlotExtraction.model_validate_json(_json_only(raw))

    async def recommend(self, slots: Slots) -> Feed:
        """조건을 검색·정리해 추천 카드 피드를 만든다."""
        summary = await self._runner.run(self._agents.coordinator, _search_request(slots))
        cards = await self._runner.run(self._agents.formatter, summary)
        return Feed.model_validate_json(_json_only(cards))


def _json_only(text: str) -> str:
    """thinking 모델이 추론 텍스트·<think> 블록을 붙여도 최종 JSON 객체만 뽑아낸다.

    <think>...</think>를 걷어낸 뒤, 파싱 가능한 JSON 객체를 스캔해 '마지막' 유효
    객체를 고른다(모델은 보통 최종 답을 끝에 둔다). 여분 텍스트가 앞뒤에 있어도 안전.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    best = None
    i = text.find("{")
    while i != -1:
        try:
            _, end = _DECODER.raw_decode(text[i:])
            best = text[i:i + end]
            i = text.find("{", i + end)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
    return best if best is not None else text


def _search_request(slots: Slots) -> str:
    return (
        f"조건: 지역={slots.region}, 일자={slots.date}, "
        f"동반자={slots.companion}, 목적={slots.purpose}. "
        f"이 조건에 맞는 곳을 찾아줘."
    )
