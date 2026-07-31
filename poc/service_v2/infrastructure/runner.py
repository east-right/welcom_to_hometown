"""ADK 에이전트를 1회 실행하고 최종 텍스트만 돌려주는 얇은 어댑터.

세션 생성·이벤트 루프·파트 파싱 같은 저수준 세부는 전부 여기 가둔다.
상위 계층은 `await runner.run(agent, text)` 한 줄만 본다.

부수적으로, 모델 비교용 지표(모델명·소요시간·토큰·출력)를 recorder에 남긴다.
"""
import time

from google.adk.runners import InMemoryRunner
from google.genai import types

from .model import model_label


class AgentRunner:
    def __init__(self, app_name: str, user_id: str, tracer, recorder=None) -> None:
        self._app = app_name
        self._user = user_id
        self._tracer = tracer
        self._recorder = recorder

    async def run(self, agent, text: str) -> str:
        """에이전트를 한 번 실행하고 최종 응답 텍스트를 반환한다."""
        runner = InMemoryRunner(agent=agent, app_name=self._app)
        session = await runner.session_service.create_session(
            app_name=self._app, user_id=self._user
        )

        self._tracer.started(agent.name, model_label(agent.model), text)
        started_at = time.time()

        final = ""
        tokens = _Tokens()
        async for event in runner.run_async(
            user_id=self._user,
            session_id=session.id,
            new_message=self._as_message(text),
        ):
            self._tracer.observe(event)
            tokens.add(getattr(event, "usage_metadata", None))
            if event.is_final_response() and event.content:
                final = self._join_text(event.content.parts)

        elapsed = time.time() - started_at
        self._tracer.finished(agent.name, final, elapsed)
        self._record(agent, elapsed, tokens, final)
        return final

    def _record(self, agent, elapsed: float, tokens: "_Tokens", final: str) -> None:
        if not self._recorder:
            return
        self._recorder.record(
            model=model_label(agent.model),
            agent=agent.name,
            elapsed_sec=round(elapsed, 2),
            tokens=tokens.as_dict(),
            chars=len(final),
            output=final,
        )

    @staticmethod
    def _as_message(text: str) -> types.Content:
        return types.Content(role="user", parts=[types.Part(text=text)])

    @staticmethod
    def _join_text(parts) -> str:
        return "".join(part.text or "" for part in parts)


class _Tokens:
    """실행 중 여러 모델 턴의 토큰 사용량을 누적한다(있을 때만; 없으면 0)."""

    def __init__(self) -> None:
        self.prompt = self.output = self.total = 0

    def add(self, usage) -> None:
        if not usage:
            return
        self.prompt += getattr(usage, "prompt_token_count", 0) or 0
        self.output += getattr(usage, "candidates_token_count", 0) or 0
        self.total += getattr(usage, "total_token_count", 0) or 0

    def as_dict(self) -> dict:
        return {"prompt": self.prompt, "output": self.output, "total": self.total}
