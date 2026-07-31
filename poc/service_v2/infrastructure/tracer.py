"""에이전트 실행 과정 추적. 실행 로직과 '기록' 관심사를 분리한다.

AgentRunner는 이벤트를 흘려보내기만 하고, 무엇을 어떻게 남길지는 여기서 정한다.
파이프라인을 사람이 그대로 따라갈 수 있게 단계별로 남긴다:

    첫 입력  →  LLM이 만든 검색어(query 변환)  →  naver 결과(URL 목록)
    →  fetch_page가 bs4로 뽑은 원문  →  그 원문이 '다음 턴 LLM 입력'으로 들어감
    →  (다음 스텝 반복)  →  최종 출력

ADK 루프 한 바퀴 = [LLM이 툴 호출] → [툴 실행/반환] → 반환값이 다음 LLM 입력.
스텝 번호로 이 반복을 눈에 보이게 한다.
"""
import time

_BODY_PREVIEW = 800   # fetch_page 원문 미리보기 길이(자)
_INPUT_PREVIEW = 600  # 첫 입력 미리보기 길이(자)


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
        self._write_block(_clip(text, _INPUT_PREVIEW))

    def observe(self, event) -> None:
        """이벤트 하나(=루프 한 조각)를 보고 툴 호출/응답이면 기록한다."""
        if getattr(event, "partial", False):
            return  # 스트리밍 중간 조각은 건너뛴다(중복 방지)
        for part in _parts(event):
            if getattr(part, "function_call", None):
                self._on_tool_call(part.function_call)
            elif getattr(part, "function_response", None):
                self._on_tool_response(part.function_response)

    def finished(self, agent_name: str, final: str, elapsed: float) -> None:
        self._log.write("│")
        self._log.write(f"│  ⑤ 최종 출력 (전체 {len(final)}자):")
        self._write_block(_clip(final, _BODY_PREVIEW))
        self._log.write(f"└─ [{agent_name}] 완료 ({elapsed:.1f}초, 툴 {self._step}회)\n")

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
        self._log.write(
            f"│      ← 반환 ({elapsed:.1f}초) — 이 내용이 그대로 다음 턴 LLM 입력으로 들어간다"
        )
        if _is_fetched_bodies(value):
            self._write_fetched_bodies(value)   # ④ fetch_pages 병렬 원문
        elif isinstance(value, list):
            self._write_search_hits(value)      # ③ naver_search 결과(URL들)
        else:
            self._write_fetched_body(str(value))  # ④ fetch_page 원문 1건

    def _write_fetched_bodies(self, items: list) -> None:
        self._log.write(f"│      ④ fetch_pages 병렬 취득 {len(items)}건 (URL별 원문):")
        for i, item in enumerate(items, 1):
            body = str(item.get("text", ""))
            self._log.write(f"│        [{i}] {item.get('url', '')}  (원문 {len(body)}자)")
            self._write_block(_clip(body, _BODY_PREVIEW))

    def _write_search_hits(self, items: list) -> None:
        self._log.write(f"│      ③ naver 검색 결과 {len(items)}건 (제목 · URL):")
        for i, item in enumerate(items, 1):
            if isinstance(item, dict) and "error" in item:
                self._log.write(f"│        {i}. [검색실패] {item['error']}")
            elif isinstance(item, dict):
                self._log.write(f"│        {i}. {item.get('title', '')}")
                self._log.write(f"│           {item.get('url', '')}")
            else:
                self._log.write(f"│        {i}. {item}")

    def _write_fetched_body(self, body: str) -> None:
        self._log.write(f"│      ④ fetch_page 원문 (bs4 추출, 전체 {len(body)}자):")
        self._write_block(_clip(body, _BODY_PREVIEW))

    def _write_block(self, text: str) -> None:
        for line in (text.splitlines() or [""]):
            self._log.write(f"│        {line}")


# ── 순수 헬퍼 ────────────────────────────────────────

def _parts(event) -> list:
    return (event.content.parts if event.content else []) or []


def _is_fetched_bodies(value) -> bool:
    """fetch_pages 반환({url, text} 목록)인지 판별."""
    return (
        isinstance(value, list)
        and bool(value)
        and isinstance(value[0], dict)
        and "text" in value[0]
    )


def _unwrap(response):
    """ADK는 dict가 아닌 툴 반환값을 {'result': 값}으로 감싼다. 그 값을 꺼낸다."""
    if isinstance(response, dict) and set(response) == {"result"}:
        return response["result"]
    return response


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]} …(+{len(text) - limit}자 생략)"
