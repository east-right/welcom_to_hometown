"""에이전트 배선(registry). 어떤 에이전트가 어떤 툴을 갖는지 한눈에 보인다.

지시문(내용)은 prompts.py에, 실행 방법은 infrastructure/runner.py에 있다.
여기서는 '구조'만 조립한다.
"""
from google.adk.agents import LlmAgent

from ..config import MODEL
from ..domain.models import Feed, SlotExtraction
from ..infrastructure.naver import fetch_page, naver_search
from . import prompts


class AgentRegistry:
    """유스케이스가 직접 부르는 세 진입 에이전트를 보관한다."""

    def __init__(self) -> None:
        self.slot_extractor = _structured("slot_extractor", prompts.SLOT_EXTRACTOR, SlotExtraction)
        self.coordinator = _coordinator()
        self.formatter = _structured("formatter", prompts.FORMATTER, Feed)


def _coordinator() -> LlmAgent:
    """검색 도구를 직접 들고 검색·선별까지 하는 코디네이터.

    서브에이전트 없이 naver_search·fetch_page를 직접 호출한다(레벨 1단).
    모든 검색 원문이 이 에이전트 문맥에 쌓이는 대신, 홉이 적고 흐름이 단순하다.
    """
    return LlmAgent(
        name="coordinator",
        model=MODEL,
        instruction=prompts.COORDINATOR,
        tools=[naver_search, fetch_page],
    )


def _structured(name: str, instruction: str, schema) -> LlmAgent:
    """JSON 스키마로 출력을 강제하는 에이전트 (툴 없음)."""
    return LlmAgent(name=name, model=MODEL, instruction=instruction, output_schema=schema)
