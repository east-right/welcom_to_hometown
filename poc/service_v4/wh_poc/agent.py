"""Layer 3 — 에이전트와 멀티턴 대화.

## 에이전트에게 남긴 판단

코드로 내릴 수 있는 것(페이지네이션, dedup, RRF 병합, 업종 확장 여부, 폴백 조건,
동시성 제어)은 전부 툴 안으로 내렸다. v2에서 어휘 확장(피자→이탈리안 등)과 다이닝코드
보강까지 collect_places 내부로 들어가면서, 에이전트가 실제로 판단하는 건 셋으로 줄었다.

1. 자연어 → 카카오가 먹는 검색어로 압축
2. region_prefix 지정 ('관천'→'신림' 같은 생활권 통칭 판단)
3. collect_info / collect_context 선택 (카페는 방문 경험, 축제는 일정 사실)
4. too_broad를 받았을 때 되물을지 진행할지

4번이 파이프라인에 되돌아가는 경로를 만든다. 되묻고 → 답을 받고 → 1단계부터 다시.

## 체크포인트를 collect_places 뒤에만 둔 이유

계측 결과: collect_places 0.3초 / collect_context 8초 / collect_info 28초.
0.3초짜리 뒤에서 물으면 재실행해도 0.3초지만, collect_info 뒤에서 물으면
이미 28초를 태운 뒤라 절약되는 게 없다. 그래서 후기가 부실할 때는 되묻지 않고
message로 한계만 밝히고 진행한다.
"""

import json
import os
import uuid

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from .schema import FeedOutput
from .tools import ALL_TOOLS
from .tracing import Tracer

APP_NAME = "wh_poc"
USER_ID = "user1"
AGENT_MODEL = os.getenv("WH_AGENT_MODEL", "gemini-3.1-flash-lite")


INSTRUCTION = """당신은 지역 장소를 찾아 추천하는 어시스턴트입니다.
사용자와 여러 번 주고받으며 원하는 것을 좁혀나갑니다.

# 응답 형식
모든 응답은 kind로 종류를 구분합니다. message는 항상 채웁니다.

- **chat** — 인사, 자기소개, 장소와 무관한 질문, 그리고 이미 보여준 추천에 대한 후속 질문.
  툴을 호출하지 말고 message에만 답하세요. places는 비웁니다.
- **question** — 정보가 부족해 사용자에게 되물을 때. places는 비웁니다.
- **recommendation** — 새로 찾은 추천 결과가 있을 때. places를 채웁니다.
  message에는 두 문장 이내의 짧은 요약만 씁니다. 각 장소 설명은 places에 들어가므로
  message에 목록을 나열하지 마세요.

# 되물어야 할 때 / 되묻지 말아야 할 때

되묻습니다(kind='question'):
- 지역이 특정되지 않은 경우. "카페 추천해줘"처럼 어디인지 없으면 검색 자체가 무의미합니다.
  → "어느 지역에서 찾아드릴까요?"
- collect_places가 'too_broad'를 반환했고, 사용자 요청에 좁히는 기준이 없는 경우.
  → "신림동 백반집이 30곳 넘게 있어요. 신림역 주변으로 좁혀드릴까요?"

되묻지 않습니다:
- 'too_broad'여도 요청에 이미 기준점이 있는 경우. 그대로 진행하세요.
    "신림역 백반집" → 역이 기준점 → 진행
    "성수동 카공카페" → 동네와 용도가 명확 → 진행
    "신림동 백반" → 기준점 없음 → 되묻기
    "서울 카페" → 너무 넓음 → 되묻기
- 한 대화에서 이미 두 번 되물었다면 더 묻지 말고 가진 결과로 진행하세요.
- 사용자가 "아무거나", "그냥 보여줘"라고 하면 즉시 진행하세요.

# 작업 순서

## 1단계 — 후보 수집
'지역 + 짧은 명사' 형태의 검색어를 만들어 collect_places를 호출합니다.
  예: "성수동에서 조용히 작업할 카페" → "성수동 카공카페"
  예: "이번 봄에 진해 벚꽃 축제 언제야" → "진해 군항제"

status별 행동:
- 'too_broad' → 위 기준으로 진행할지 되물을지 판단합니다.
- 'ok' → 다음 단계로. 축제처럼 대상이 하나뿐이면 결과가 적어도 정상입니다.
  collect_places는 내부적으로 업종어 확장과 다이닝코드 보강을 이미 수행했으므로,
  결과가 적어도 재시도할 필요가 없습니다. 그대로 다음 단계로 넘어가세요.

## 2단계 — 어떤 정보가 필요한지 판단
요청 성격에 따라 하나 또는 둘 다 선택합니다. 이 판단이 답변 품질을 좌우합니다.

- **collect_info** (블로그 후기 요약)
  방문자의 실제 경험이 근거인 경우. 맛, 분위기, 혼잡도, 좌석, 서비스 등.
  카페·음식점·술집 등 상설 매장에 적합합니다.

  region_prefix에는 그 동네를 사람들이 실제로 부르는 짧은 이름만 넣으세요.
  '역', '동' 같은 접미사와 도로명·행정동은 빼야 블로그 검색이 맞습니다.
    "신림역" → '신림'   (X: '신림역')
    "관악구 관천로" → '신림'
    "금천구 가산디지털2로" → '가산'

- **collect_context** (웹문서·뉴스 사실 정리)
  일정·기간·요금·예약처럼 확인된 사실이 필요한 경우.
  축제·행사·전시처럼 시기가 중요하거나, 관광지·캠핑장처럼 운영 정보가 필요할 때 씁니다.
  시점이 걸린 요청이면 먼저 get_today로 오늘 날짜를 확인하세요.

애매하면 둘 다 호출해도 됩니다. "캠핑장 추천"은 운영 정보와 이용 후기가 모두 도움이 됩니다.

## 3단계 — 근거 확인
read_file로 summary_file 또는 context_file을 읽습니다.
파일을 읽지 않고 답변을 만들면 안 됩니다.

## 4단계 — 추천 (kind='recommendation')
요청에 가장 잘 맞는 5~8곳을 골라 places에 담습니다.
- review_count 또는 source_count가 많을수록 신뢰도가 높습니다.
- freshness가 'outdated'면 지난 연도 정보이므로 caution에 그 사실을 적으세요.
- address와 url은 파일에 있는 값을 그대로 씁니다. 절대 지어내지 마세요.
- 파일에 없는 정보는 쓰지 마세요. 강점이 부족하면 그 장소를 빼는 편이 낫습니다.
- 후기를 확보한 곳이 적었다면 message에 그 사실을 밝히세요.

# 이어지는 대화

이미 추천을 보여준 뒤 사용자가 후속 질문을 하면, 대화에 남아있는 정보로 답할 수 있는지
먼저 확인하세요.
- "그중에 주차되는 곳은?" → 이미 읽은 파일 내용으로 답변 가능 → kind='chat', 툴 호출 없이 답변
- "다른 곳도 보여줘" / "강남으로 바꿔줘" → 새로 검색 필요 → 1단계부터 다시

같은 정보를 다시 수집하지 마세요. 수집은 시간이 오래 걸립니다.

# 실패 처리
collect_info나 collect_context가 error를 반환하면 collect_places부터 다시 실행하세요.
수집이 실패했는데 collect_places 결과만으로 평가를 지어내면 안 됩니다.
"""


def build_agent(model: str = AGENT_MODEL, tracer: Tracer | None = None) -> LlmAgent:
    """에이전트를 만든다. tracer를 주면 LLM·툴 호출에 콜백이 걸린다."""
    kwargs = {}
    if tracer is not None:
        kwargs = {
            "before_model_callback": tracer.before_model,
            "after_model_callback": tracer.after_model,
            "before_tool_callback": tracer.before_tool,
            "after_tool_callback": tracer.after_tool,
        }
    return LlmAgent(
        model=model,
        name="place_finder",
        description="지역 기반 장소 추천 에이전트",
        instruction=INSTRUCTION,
        tools=ALL_TOOLS,
        output_schema=FeedOutput,
        **kwargs,
    )


# 툴 호출을 사용자에게 보여줄 상태 문구로 번역한다.
# 모델이 매번 멘트를 지어내게 하는 것보다 일관되고, 토큰도 쓰지 않는다.
STATUS_TEXT = {
    "get_today": "날짜를 확인하고 있어요",
    "collect_places": "장소를 찾고 있어요",
    "collect_info": "방문 후기를 모으는 중이에요",
    "collect_context": "일정과 운영 정보를 확인하고 있어요",
    "read_file": "내용을 정리하고 있어요",
}

_FILE_KEYS = ("file", "url_file", "summary_file", "source_file", "context_file")


class Chat:
    """세션을 유지하며 여러 턴을 주고받는다.

    ADK 세션이 대화 이력을 들고 있으므로 에이전트가 이전 턴의 툴 결과를 참조할 수 있다.
    같은 인스턴스를 계속 쓰면 이어지는 대화가 되고, 새로 만들면 처음부터다.
    """

    def __init__(
        self,
        agent: LlmAgent | None = None,
        show_tools: bool = False,
        show_status: bool = True,
        trace: bool = True,
        capture_prompt: bool = False,
    ):
        self.tracer = Tracer(capture_prompt=capture_prompt) if trace else None
        self.runner = InMemoryRunner(
            agent=agent or build_agent(tracer=self.tracer), app_name=APP_NAME
        )
        self.session_id = str(uuid.uuid4())
        self.show_tools = show_tools
        self.show_status = show_status
        self.history: list[dict] = []
        self._ready = False

    async def _ensure_session(self):
        if not self._ready:
            await self.runner.session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=self.session_id
            )
            self._ready = True

    async def send(self, text: str) -> dict:
        await self._ensure_session()
        if self.tracer:
            self.tracer.start_turn(text)
        content = types.Content(role="user", parts=[types.Part(text=text)])

        out = {
            "query": text,
            "turn": len(self.history) + 1,
            "tool_calls": [],
            "files": {},
            "final_text": "",
            "result": None,
        }

        async for event in self.runner.run_async(
            user_id=USER_ID, session_id=self.session_id, new_message=content
        ):
            for call in event.get_function_calls():
                out["tool_calls"].append({"name": call.name, "args": dict(call.args)})
                if self.show_status and call.name in STATUS_TEXT:
                    print(f"… {STATUS_TEXT[call.name]}")
                if self.show_tools:
                    print(f"[CALL] {call.name}({dict(call.args)})")

            for fr in event.get_function_responses():
                r = fr.response
                if isinstance(r, dict):
                    for k in _FILE_KEYS:
                        if k in r:
                            out["files"][k] = r[k]
                    if self.show_tools:
                        brief = {k: v for k, v in r.items() if k != "content"}
                        if "content" in r:
                            brief["content"] = f"<{len(str(r['content']))}자>"
                        print(f"[RESP] {fr.name} → {brief}")

            if event.is_final_response() and event.content and event.content.parts:
                out["final_text"] = "".join(
                    p.text for p in event.content.parts if p.text
                ).strip()

        # output_schema가 있어도 모델이 스키마 툴을 반드시 부르는 건 아니다.
        # 특히 대화형 응답에서 자유 텍스트로 답하는 경우가 있어, 그때는 chat으로 감싼다.
        try:
            out["result"] = json.loads(out["final_text"])
        except (json.JSONDecodeError, TypeError):
            out["result"] = {
                "kind": "chat",
                "message": out["final_text"],
                "places": [],
            }

        if self.tracer:
            self.tracer.end_turn(out["result"])

        self.history.append(out)
        return out

    async def say(self, text: str) -> dict:
        """send + 사람이 읽기 좋은 출력."""
        print(f"\n{'=' * 60}\n👤 {text}\n{'=' * 60}")
        out = await self.send(text)
        r = out["result"]

        print(f"\n🤖 [{r.get('kind', '?')}] {r.get('message', '')}")
        for p in r.get("places", []):
            print(f"\n  ■ {p['name']}  ({p.get('address', '')})")
            print(f"    {p.get('reason', '')}")
            if p.get("highlights"):
                print(f"    {p['highlights']}")
            if p.get("period"):
                print(f"    기간: {p['period']}")
            if p.get("caution"):
                print(f"    주의: {p['caution']}")
        return out

    def trace(self, turn: int | None = None) -> None:
        """턴별 타임라인 출력. LLM 대기와 툴 실행이 언제 일어났는지 본다."""
        if self.tracer:
            self.tracer.report(turn)
        else:
            print("trace=False로 생성되어 기록이 없습니다.")

    def totals(self) -> None:
        """LLM 대기 / 툴 실행 비중과 토큰 합계."""
        if self.tracer:
            self.tracer.totals()

    def summary(self) -> None:
        """대화 이력 요약."""
        for h in self.history:
            r = h["result"] or {}
            print(
                f"[{h['turn']}] {h['query'][:28]:30} → {r.get('kind', '?'):15} "
                f"places={len(r.get('places', []))} tools={len(h['tool_calls'])}"
            )
