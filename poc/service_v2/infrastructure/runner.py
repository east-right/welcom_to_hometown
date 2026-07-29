"""ADK 에이전트를 1회 실행하고 최종 텍스트만 돌려주는 얇은 어댑터.

세션 생성·이벤트 루프·파트 파싱 같은 저수준 세부는 전부 여기 가둔다.
상위 계층은 `await runner.run(agent, text)` 한 줄만 본다.
"""
import time

from google.adk.runners import InMemoryRunner
from google.genai import types


class AgentRunner:
    def __init__(self, app_name: str, user_id: str, tracer) -> None:
        self._app = app_name
        self._user = user_id
        self._tracer = tracer

    async def run(self, agent, text: str) -> str:
        """에이전트를 한 번 실행하고 최종 응답 텍스트를 반환한다."""
        runner = InMemoryRunner(agent=agent, app_name=self._app)
        session = await runner.session_service.create_session(
            app_name=self._app, user_id=self._user
        )

        self._tracer.started(agent.name, text)
        started_at = time.time()

        final = ""
        async for event in runner.run_async(
            user_id=self._user,
            session_id=session.id,
            new_message=self._as_message(text),
        ):
            self._tracer.observe(event)
            if event.is_final_response() and event.content:
                final = self._join_text(event.content.parts)

        self._tracer.finished(agent.name, final, time.time() - started_at)
        return final

    @staticmethod
    def _as_message(text: str) -> types.Content:
        return types.Content(role="user", parts=[types.Part(text=text)])

    @staticmethod
    def _join_text(parts) -> str:
        return "".join(part.text or "" for part in parts)
