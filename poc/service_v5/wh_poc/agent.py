"""Layer 3 — 에이전트와 멀티턴 대화.

## 에이전트에게 남긴 판단

코드로 내릴 수 있는 것(수집·수확·필터·리랭크·동시성)은 전부 툴 안으로 내렸다.
에이전트가 실제로 판단하는 건 넷이다.

1. 자연어 → 카카오가 먹는 검색어로 압축 ('지역 + 짧은 명사', 막연한 맛집은 '식사 맛집')
2. region_prefix 지정 ('관천'→'신림' 같은 생활권 통칭 판단)
3. collect_info / collect_context 선택 (카페·식당은 방문 경험, 축제는 일정 사실)
4. too_broad를 받았을 때 되물을지 진행할지

4번이 파이프라인에 되돌아가는 경로를 만든다. 되묻고 → 답을 받고 → 1단계부터 다시.

## 체크포인트를 collect_places 뒤에만 둔 이유

계측 결과: collect_places는 빠르고 collect_info는 느리다(크롤·요약). 빠른 것 뒤에서
물으면 재실행이 싸지만, 느린 것 뒤에서 물으면 이미 비용을 태운 뒤라 절약이 없다.
그래서 후기가 부실할 때는 되묻지 않고 message로 한계만 밝히고 진행한다.
"""

import json
import os
import uuid

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.genai import types

from . import prompts
from .clients import LLM_BACKEND, LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_MODEL, llm_async
from .core import load_json
from .schema import FeedOutput
from .tools import ALL_TOOLS
from .tracing import Tracer

APP_NAME = "wh_poc"
USER_ID = "user1"

# 에이전트 백엔드는 툴 내부 LLM(clients.llm_async, LLM_BACKEND)과 독립적으로 스왑한다.
# WH_AGENT_BACKEND를 안 주면 WH_LLM_BACKEND를 따라가 현행과 동일하게 동작한다.
# 예: 툴 요약·HyDE는 Gemini로 두고(WH_LLM_BACKEND=gemini) 메인 에이전트만
# 로컬 vLLM(27B)로 돌리려면 WH_AGENT_BACKEND=local 만 켠다.
AGENT_BACKEND = os.getenv("WH_AGENT_BACKEND", LLM_BACKEND).lower()

# 한 요청에서 허용할 LLM 호출 수 상한. 정상 흐름은 4~5스텝(수집·요약선택·최종)이라
# 넉넉하다. 약한 로컬 모델이 get_today 등을 반복 호출하는 폭주 루프를 여기서 강제로
# 끊어(초과 시 ADK가 중단) 케이스가 수백 초로 늘어나는 걸 막는다.
MAX_LLM_CALLS = int(os.getenv("WH_MAX_LLM_CALLS", "30"))


def agent_model():
    """백엔드에 맞는 에이전트 모델을 만든다. gemini면 ADK 네이티브 문자열,
    local이면 vLLM(OpenAI 호환)을 가리키는 LiteLlm 객체."""
    if AGENT_BACKEND != "local":
        return os.getenv("WH_AGENT_MODEL", "gemini-3.1-flash-lite")
    # 원격 오픈모델을 쓸 때만 임포트 (litellm 미설치 환경에서도 gemini 경로는 동작)
    from google.adk.models.lite_llm import LiteLlm

    # Qwen3 등 추론 모델의 <think> 사고과정을 끈다. 에이전트 최종 응답은 FeedOutput
    # 스키마의 JSON이라 앞에 사고과정이 붙으면 파싱이 깨진다. thinking 미지원 모델은
    # 이 인자를 무시하므로 무해하다. LiteLlm의 kwargs는 litellm completion으로 전달된다.
    kwargs = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if LOCAL_API_BASE:
        kwargs["api_base"] = LOCAL_API_BASE
    if LOCAL_API_KEY:
        kwargs["api_key"] = LOCAL_API_KEY
    return LiteLlm(model=LOCAL_MODEL, **kwargs)


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
- 지역이 아예 없는 경우. "카페 추천해줘"처럼 어디인지 없으면 검색 자체가 무의미합니다.
  → "어느 지역에서 찾아드릴까요?"
- 지역이 광역시·특별시 전체처럼 지나치게 넓고 좁히는 기준이 전혀 없는 경우.
  → "서울 어디쯤에서 찾아드릴까요?"

되묻지 않습니다:
- 요청에 구체적 지역(동네·역·시/군 이름)이 있으면 too_broad여도 그대로 진행하세요.
  후보가 많으면 상위 결과로 추천하고, message에서 "더 좁혀줄 수 있다"고 안내만 합니다.
    "신림역 백반집" → 역이 기준점 → 진행
    "성수동 카공카페" → 동네와 용도가 명확 → 진행
    "창원 횟집" → 시(市) 이름이 있으니 진행 + message로 동네 좁히기 제안
    "서울 카페" → 너무 넓고 기준 없음 → 되묻기
    "카페 추천" → 지역 없음 → 되묻기
- 한 대화에서 이미 두 번 되물었다면 더 묻지 말고 가진 결과로 진행하세요.
- 사용자가 "아무거나", "그냥 보여줘"라고 하면 즉시 진행하세요.

# 작업 순서

## 1단계 — 후보 수집
'지역 + 짧은 명사' 형태의 검색어를 만들어 collect_places를 호출합니다.
  예: "성수동에서 조용히 작업할 카페" → "성수동 카공카페"
  예: "이번 봄에 진해 벚꽃 축제 언제야" → "진해 군항제"
  업종 없이 막연히 '맛집'만 찾으면 '지역 맛집'으로 검색하고 category_code='FD6'(음식점)를
  지정해 카페를 제외합니다. '식사 맛집' 같은 어색한 조합은 카카오가 문자 그대로 매칭해
  결과가 거의 안 나오니 쓰지 마세요.
  예: "가포 맛집 추천해줘" → query="가포 맛집", category_code="FD6"

status별 행동:
- 'too_broad' → 위 기준으로 진행할지 되물을지 판단합니다.
- 'ok' → 다음 단계로. 축제처럼 대상이 하나뿐이면 결과가 적어도 정상입니다.

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
  축제·행사처럼 **날짜가 결과를 바꾸는 요청에만** get_today로 오늘 날짜를 확인하세요.
  음식점·카페·술집 같은 상설 매장 추천에는 get_today를 호출하지 마세요.
  이미 호출한 툴은 같은 인자로 다시 호출하지 말고 다음 단계로 넘어가세요.

애매하면 둘 다 호출해도 됩니다. "캠핑장 추천"은 운영 정보와 이용 후기가 모두 도움이 됩니다.

## 3단계 — 근거 확인
최종 추천 매장을 확정할 때 read_file을 **한 번만** 호출해 근거를 읽습니다.
- collect_info를 썼으면 summary_file, collect_context를 썼으면 context_file **하나만** 읽습니다.
- 이 파일 하나에 판단에 필요한 모든 근거(장소별 요약·주소·url·강점)가 들어 있습니다.
- rerank_file·url_file·collect_places의 file 등 다른 파일은 읽지 마세요. 답변에 불필요하고
  맥락만 무겁게 만듭니다.
파일을 읽지 않고 답변을 만들면 안 됩니다.

**툴 결과에 `already_collected`나 `already_read`가 있으면 그 단계는 이미 끝난 것입니다.**
다시 수집·조회하지 말고, 이미 확보한 근거로 곧바로 4단계(최종 추천)로 넘어가세요.
같은 수집을 반복하면 답이 나빠지고 실패합니다.

## 4단계 — 추천 (kind='recommendation')
요청에 가장 잘 맞는 5~8곳을 골라 places에 담습니다.
- rerank_rank는 요청과의 적합도 순위입니다(작을수록 잘 맞음). 기본적으로 rerank_rank가
  낮은 순서대로 담고, 근거 파일에 강점이 전혀 없거나 요청과 명백히 무관한 곳만 건너뜁니다.
  놓치지 않도록 가능한 한 상위 8곳까지 채웁니다(상위를 건너뛰고 하위를 넣지 마세요).
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

# 로컬(vLLM) 최종 응답은 decode-bound(실측 ≈19 tok/s)라, 느린 워커가 장황한 산문을
# 통째로 뽑는 것 자체가 병목이다(827토큰 ≈ 45초). 로컬에서는 워커(27B)가 '어느 곳을
# 고르나 + 왜'만 짧게 내고, address·url·highlights·caution 등 근거 파일에 이미 있는
# 필드는 Chat.send의 _format_feed(Gemini)가 채워 완성한다. 무거운 생성을 느린 워커에서
# 빠른 포매터로 옮기는 것. gemini 경로는 output_schema로 직접 FeedOutput을 내므로 무변경.
if AGENT_BACKEND == "local":
    _STEP4_FULL = """## 4단계 — 추천 (kind='recommendation')
요청에 가장 잘 맞는 5~8곳을 골라 places에 담습니다.
- rerank_rank는 요청과의 적합도 순위입니다(작을수록 잘 맞음). 기본적으로 rerank_rank가
  낮은 순서대로 담고, 근거 파일에 강점이 전혀 없거나 요청과 명백히 무관한 곳만 건너뜁니다.
  놓치지 않도록 가능한 한 상위 8곳까지 채웁니다(상위를 건너뛰고 하위를 넣지 마세요).
- review_count 또는 source_count가 많을수록 신뢰도가 높습니다.
- freshness가 'outdated'면 지난 연도 정보이므로 caution에 그 사실을 적으세요.
- address와 url은 파일에 있는 값을 그대로 씁니다. 절대 지어내지 마세요.
- 파일에 없는 정보는 쓰지 마세요. 강점이 부족하면 그 장소를 빼는 편이 낫습니다.
- 후기를 확보한 곳이 적었다면 message에 그 사실을 밝히세요."""
    _STEP4_TERSE = """## 4단계 — 추천 (kind='recommendation')
요청에 가장 잘 맞는 5~8곳을 고릅니다. rerank_rank가 낮은(=적합) 곳을 우선 고르고,
review_count/source_count가 많을수록 신뢰도가 높습니다.
근거 파일에 없는 장소는 고르지 마세요. 강점이 부족한 곳은 빼는 편이 낫습니다.

최종 응답에는 고른 장소의 name과, 요청에 왜 맞는지 reason(한 문장)만 쓰세요.
address·url·highlights·caution·period·category는 쓰지 마세요 — 근거 파일에서 자동으로 채워집니다.
후기를 확보한 곳이 적었다면 message에 그 사실을 밝히세요."""
    INSTRUCTION = INSTRUCTION.replace(_STEP4_FULL, _STEP4_TERSE)


def build_agent(model=None, tracer: Tracer | None = None) -> LlmAgent:
    """에이전트를 만든다. tracer를 주면 LLM·툴 호출에 콜백이 걸린다.
    model을 안 주면 WH_LLM_BACKEND에 따라 gemini 문자열/LiteLlm을 자동 선택한다."""
    if model is None:
        model = agent_model()
    kwargs = {}
    if tracer is not None:
        kwargs = {
            "before_model_callback": tracer.before_model,
            "after_model_callback": tracer.after_model,
            "before_tool_callback": tracer.before_tool,
            "after_tool_callback": tracer.after_tool,
        }
    agent_kwargs = dict(
        model=model,
        name="place_finder",
        description="지역 기반 장소 추천 에이전트",
        instruction=INSTRUCTION,
        tools=ALL_TOOLS,
        **kwargs,
    )
    # output_schema는 gemini 경로에서만 건다. 로컬(vLLM)은 output_schema를 tools와 함께
    # 주면 vLLM이 스키마로 guided decoding을 걸어 tool 호출 자체를 막는다(ADK가 LiteLlm을
    # 네이티브 지원으로 오인해 set_model_response 우회를 안 씀). 그래서 로컬은 tools만으로
    # 돌리고, 최종 응답은 Chat.send에서 _format_feed로 FeedOutput으로 정형화한다.
    if AGENT_BACKEND != "local":
        agent_kwargs["output_schema"] = FeedOutput
    return LlmAgent(**agent_kwargs)


# 툴 호출을 사용자에게 보여줄 상태 문구로 번역한다.
# 모델이 매번 멘트를 지어내게 하는 것보다 일관되고, 토큰도 쓰지 않는다.
STATUS_TEXT = {
    "get_today": "날짜를 확인하고 있어요",
    "collect_places": "장소를 찾고 있어요",
    "collect_info": "방문 후기를 모으는 중이에요",
    "collect_context": "일정과 운영 정보를 확인하고 있어요",
    "read_file": "내용을 정리하고 있어요",
}

_FILE_KEYS = ("file", "url_file", "rerank_file", "summary_file", "source_file", "context_file")

# 최종 추천 개수. 리랭크 top-15 중 이 수만큼 스코어링으로 확정한다.
FINAL_K = int(os.getenv("WH_FINAL_K", "8"))


async def _select_by_score(user_query: str, summary_file: str, k: int = FINAL_K) -> list[dict] | None:
    """2차 선정 — 리랭크로 남은 매장들(summary_file)을 요청 적합도로 스코어링해 상위 k곳을
    코드가 확정한다. 임베딩 코사인은 같은 업종 후기끼리 뭉쳐(변별력 약함) 정답이 top-8 밖으로
    밀리는데, LLM이 요약을 읽고 점수를 매기면(cross-encoder처럼 질의·문서를 함께 봄) 유명·검증
    신호까지 반영해 변별이 산다. 선택을 LLM 자유판단이 아니라 '점수→코드 컷'으로 통제한다."""
    try:
        summ = load_json(summary_file)
    except (OSError, ValueError):
        return None
    if not summ:
        return None
    items = "\n".join(
        f"- {n}: {v.get('summary', '')} [강점] {' / '.join(v.get('strengths', []))}"
        for n, v in summ.items()
    )
    try:
        scored = json.loads(
            await llm_async(
                prompts.SELECT_SCORE.format(user_query=user_query, items=items),
                json_mode=True,
            )
        )
    except (json.JSONDecodeError, TypeError):
        return None

    def score_of(name: str) -> float:
        s = scored.get(name)
        return s.get("score", 0) if isinstance(s, dict) else (s or 0)

    places = []
    for name in sorted(summ, key=lambda n: -score_of(n))[:k]:
        v = summ[name]
        s = scored.get(name, {})
        reason = (s.get("reason") if isinstance(s, dict) else "") or v.get("summary", "")
        weaknesses = v.get("weaknesses", [])
        places.append({
            "name": name,
            "address": v.get("address", ""),
            "category": v.get("category", ""),
            "reason": reason,
            "highlights": v.get("strengths", [])[:3],
            "caution": weaknesses[0] if weaknesses else "",
            "period": "",
            "url": v.get("url", ""),
        })
    return places


async def _format_feed(user_query: str, agent_text: str, files: dict) -> dict:
    """로컬 백엔드 전용 포매터. output_schema 없이 돌린 워커의 자연어 답을 근거 파일과
    함께 FeedOutput dict로 정형화한다. 실패하면 자연어 답을 chat으로 감싼다."""
    evidence = "(없음)"
    for k in ("summary_file", "context_file"):
        if files.get(k):
            try:
                evidence = json.dumps(load_json(files[k]), ensure_ascii=False)[:20000]
                break
            except (OSError, ValueError):
                pass
    try:
        raw = await llm_async(
            prompts.FORMAT_FEED.format(
                user_query=user_query, agent_answer=agent_text or "", evidence=evidence
            ),
            json_mode=True,
        )
        data = json.loads(raw)
        data.setdefault("kind", "chat")
        data.setdefault("message", "")
        data.setdefault("places", [])
        return data
    except (json.JSONDecodeError, TypeError, KeyError):
        return {"kind": "chat", "message": agent_text or "", "places": []}


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
            user_id=USER_ID, session_id=self.session_id, new_message=content,
            run_config=RunConfig(max_llm_calls=MAX_LLM_CALLS),
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

        # 로컬(vLLM)은 output_schema 없이 tools만으로 돌리므로 최종이 자연어다 → 포매터로
        # FeedOutput 정형화. gemini는 스키마로 JSON을 직접 내지만, 자유 텍스트로 답하는
        # 경우가 있어 파싱 실패 시 chat으로 감싼다.
        if AGENT_BACKEND == "local":
            out["result"] = await _format_feed(text, out["final_text"], out["files"])
        else:
            try:
                out["result"] = json.loads(out["final_text"])
            except (json.JSONDecodeError, TypeError):
                out["result"] = {
                    "kind": "chat",
                    "message": out["final_text"],
                    "places": [],
                }

        # 2차 선정 — 추천이면 매장 선택을 에이전트 자유판단이 아니라 스코어링→코드 top-K로
        # 확정한다. summary_file(collect_info)이 있을 때만. 없으면(축제=context) 에이전트 답 유지.
        result = out["result"] or {}
        if result.get("kind") == "recommendation" and out["files"].get("summary_file"):
            picked = await _select_by_score(text, out["files"]["summary_file"])
            if picked:
                result["places"] = picked
                out["result"] = result

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
