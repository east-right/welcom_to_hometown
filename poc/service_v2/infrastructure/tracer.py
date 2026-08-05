"""에이전트 실행 과정 추적. 실행 로직과 '기록' 관심사를 분리한다.

AgentRunner는 이벤트를 흘려보내기만 하고, 무엇을 어떻게 남길지는 여기서 정한다.
agentic 루프를 사람이 그대로 따라갈 수 있게 스텝별로 남긴다:

    첫 입력 → [LLM이 툴 호출: find_places/read_reviews] → [툴 반환] → 반환값이 다음 턴 입력
    → (다음 스텝 반복) → 최종 출력(카드)

스텝 번호로 이 반복을 눈에 보이게 한다.
"""
import time

_PREVIEW = 800   # 툴 반환/원문 미리보기 길이(자)
_MAX_ITEMS = 12  # 리스트 반환 미리보기 최대 항목 수


class RunTracer:
    """에이전트 실행의 시작·툴 호출·응답·종료를 사람이 읽게 기록한다."""

    def __init__(self, logger) -> None:
        self._log = logger
        self._tool_started_at: dict[str, float] = {}
        self._step = 0

    def started(self, agent_name: str, model: str, text: str) -> None:
        self._step = 0
        self._log.write(f"┌─ [{agent_name}] 시작  (model={model})")
        self._log.write(f"│  ① 첫 입력 (전체 {len(text)}자):")
        self._write_block(text)   # 후보+본문까지 통째로 보이게 자르지 않는다

    def observe(self, event) -> None:
        """이벤트 하나(=루프 한 조각)를 보고 툴 호출/응답이면 기록한다."""
        if getattr(event, "partial", False):
            return  # 스트리밍 중간 조각은 건너뛴다(중복 방지)
        for part in _parts(event):
            if getattr(part, "function_call", None):
                self._on_tool_call(part.function_call)
            elif getattr(part, "function_response", None):
                self._on_tool_response(part.function_response)

    def finished(self, agent_name: str, final: str, elapsed: float, model_sec=None) -> None:
        self._log.write("│")
        self._log.write(f"│  ⑤ 최종 출력 (전체 {len(final)}자):")
        self._write_block(final)   # 최종 카드 JSON도 자르지 않고 전부 남긴다
        timing = f"{elapsed:.1f}초(벽시계)"
        if model_sec is not None:
            timing += f" / {model_sec:.2f}초(모델, 통신제외)"
        self._log.write(f"└─ [{agent_name}] 완료 ({timing}, 툴 {self._step}회)\n")

    # ── 루프 한 스텝: LLM이 툴을 부른다 ──────────────────
    def _on_tool_call(self, call) -> None:
        self._step += 1
        self._tool_started_at[call.name] = time.time()
        self._log.write("│")
        self._log.write(f"│  [스텝 {self._step}] ② LLM → 툴 호출: {call.name}")
        for key, value in dict(call.args).items():
            self._log.write(f"│      {key} = {value!r}")

    # ── 루프 한 스텝: 툴이 값을 돌려준다(= 다음 턴 LLM 입력) ──
    def _on_tool_response(self, response) -> None:
        elapsed = time.time() - self._tool_started_at.get(response.name, time.time())
        value = _unwrap(response.response)
        self._log.write(f"│      ← {response.name} 반환 ({elapsed:.1f}초)")
        self._write_block(_clip(_preview(value), _PREVIEW))

    def _write_block(self, text: str) -> None:
        for line in (text.splitlines() or [""]):
            self._log.write(f"│        {line}")


# ── 순수 헬퍼 ────────────────────────────────────────

def _parts(event) -> list:
    return (event.content.parts if event.content else []) or []


def _unwrap(response):
    """ADK는 dict가 아닌 툴 반환값을 {'result': 값}으로 감싼다. 그 값을 꺼낸다."""
    if isinstance(response, dict) and set(response) == {"result"}:
        return response["result"]
    return response


def _preview(value) -> str:
    """툴 반환값을 사람이 읽게 한 줄~여러 줄로. 리스트는 건수+항목, dict는 채워진 필드만."""
    if isinstance(value, list):
        head = "\n".join(f"- {_line(item)}" for item in value[:_MAX_ITEMS])
        more = f"\n… (+{len(value) - _MAX_ITEMS}건)" if len(value) > _MAX_ITEMS else ""
        return f"{len(value)}건:\n{head}{more}"
    if isinstance(value, dict):
        filled = ", ".join(f"{k}={v!r}" for k, v in value.items() if v not in ("", None, False))
        return filled or "(빈 결과)"
    return str(value)


def _line(item) -> str:
    if isinstance(item, dict):
        keys = ("name", "title", "category", "address", "hours", "url")
        return " | ".join(str(item[k]) for k in keys if item.get(k)) or str(item)
    return str(item)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]} …(+{len(text) - limit}자 생략)"
