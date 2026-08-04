"""추천 유스케이스.

대화 흐름을 한 단계 추상화로만 기술한다. 에이전트를 '어떻게' 실행하는지
(이벤트 루프·세션·로깅)는 AgentRunner가, 각 에이전트가 '무슨 말'을 하는지는
프롬프트가 감춘다. 여기서는 '무엇을 하는가'만 읽힌다.
"""
import json
import re

from ..domain.models import Feed, QueryPlan, SlotExtraction, Slots

_DECODER = json.JSONDecoder()

_MAX_CANDIDATES = 24   # selector에게 보여줄 최대 후보 수
_FETCH_TOP = 12        # 그중 본문을 병렬로 읽을 상위 개수
_BODY_CLIP = 600       # 후보당 본문 미리보기 길이(자)


class Recommender:
    def __init__(self, runner, agents, places) -> None:
        self._runner = runner
        self._agents = agents
        self._places = places   # 검색+본문취득 게이트웨이(병렬 I/O)

    async def understand(self, conversation: str) -> SlotExtraction:
        """지금까지의 대화에서 슬롯을 추출하고, 부족하면 되물을 질문을 얻는다."""
        raw = await self._runner.run(self._agents.slot_extractor, conversation)
        return SlotExtraction.model_validate_json(_json_only(raw))

    async def recommend(self, slots: Slots) -> Feed:
        """조건을 검색어로 넓히고 → 병렬 검색·취득 → 선별해 카드 피드를 만든다."""
        plan = await self._expand(slots)
        candidates = await self._collect(plan)
        return await self._curate(slots, candidates)

    async def _expand(self, slots: Slots) -> QueryPlan:
        """슬롯을 여러 각도의 검색어 8~10개로 증강한다(모델은 이 판단만)."""
        raw = await self._runner.run(self._agents.query_expander, _expand_request(slots))
        return QueryPlan.model_validate_json(_json_only(raw))

    async def _collect(self, plan: QueryPlan) -> list[dict]:
        """검색어들을 병렬 검색하고, 상위 후보의 본문을 병렬로 붙인다."""
        hits = (await self._places.search(plan.queries))[:_MAX_CANDIDATES]
        bodies = await self._places.fetch([h["url"] for h in hits[:_FETCH_TOP]])
        return _attach_bodies(hits, bodies)

    async def _curate(self, slots: Slots, candidates: list[dict]) -> Feed:
        """후보를 보고 조건에 맞는 카드 10개를 골라 정렬한다(검색 없음)."""
        raw = await self._runner.run(self._agents.selector, _curate_request(slots, candidates))
        return Feed.model_validate_json(_json_only(raw))


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


def _expand_request(slots: Slots) -> str:
    return (
        f"조건: 지역={slots.region}, 일자={slots.date}, "
        f"동반자={slots.companion}, 목적={slots.purpose}.\n"
        "이 조건을 여러 각도로 넓게 커버하는 네이버 검색어 8~10개를 만들어라."
    )


def _attach_bodies(hits: list[dict], bodies: list[dict]) -> list[dict]:
    """검색 후보에 취득한 본문을 URL로 매칭해 붙인다(없으면 빈 문자열)."""
    body_by_url = {b["url"]: b["text"] for b in bodies}
    return [{**hit, "body": body_by_url.get(hit["url"], "")} for hit in hits]


def _curate_request(slots: Slots, candidates: list[dict]) -> str:
    """selector에게 넘길 후보 목록 텍스트. 본문이 있으면 일부만 미리보기로."""
    lines = [f"조건: {slots.summary()}", "", "[검색 후보]"]
    for i, c in enumerate(candidates, 1):
        lines.append(f"{i}. ({c.get('kind', '')}) {c['title']}")
        lines.append(f"   URL: {c['url']}")
        if c.get("desc"):
            lines.append(f"   요약: {c['desc']}")
        if c.get("body"):
            lines.append(f"   본문: {c['body'][:_BODY_CLIP]}")
    lines += ["", "위 후보에서 조건에 맞는 곳 최대 10개를 우선순위 순으로 골라 카드로 만들어라."]
    return "\n".join(lines)
