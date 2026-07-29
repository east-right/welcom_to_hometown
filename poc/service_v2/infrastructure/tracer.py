"""에이전트 실행 과정 추적. 실행 로직과 '기록' 관심사를 분리한다.

AgentRunner는 이벤트를 흘려보내기만 하고, 무엇을 어떻게 남길지는 여기서 정한다.
"""
import json
import time


class RunTracer:
    """에이전트 실행의 시작·툴 호출·응답·종료를 사람이 읽게 기록한다."""

    def __init__(self, logger) -> None:
        self._log = logger
        self._tool_started_at: dict[str, float] = {}

    def started(self, agent_name: str, text: str) -> None:
        self._log.write(f"┌─ [{agent_name}] 시작")
        self._log.write(f"│  입력: {text[:120]}")

    def observed(self, author: str, part) -> None:
        """이벤트 파트 하나를 보고 툴 호출/응답이면 기록한다."""
        if getattr(part, "function_call", None):
            self._on_tool_call(author, part.function_call)
        elif getattr(part, "function_response", None):
            self._on_tool_response(author, part.function_response)

    def finished(self, agent_name: str, final: str, elapsed: float) -> None:
        self._log.write(f"│  출력: {final[:150]}")
        self._log.write(f"└─ [{agent_name}] 완료 ({elapsed:.1f}초)\n")

    def _on_tool_call(self, author: str, call) -> None:
        self._tool_started_at[call.name] = time.time()
        args = json.dumps(dict(call.args), ensure_ascii=False)
        self._log.write(f"│  → [{author}] 호출 {call.name}({args[:100]})")

    def _on_tool_response(self, author: str, response) -> None:
        elapsed = time.time() - self._tool_started_at.get(response.name, time.time())
        body = str(response.response)
        self._log.write(
            f"│  ← [{author}] {response.name} 응답 ({elapsed:.1f}초, {len(body)}자)"
        )
        self._log.write(f"│      {body[:150]}")
