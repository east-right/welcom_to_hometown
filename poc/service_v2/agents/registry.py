"""에이전트 배선(registry). 어떤 에이전트가 어떤 툴을 갖는지 한눈에 보인다.

지시문(내용)은 prompts.py에, 실행 방법은 infrastructure/runner.py에 있다.
여기서는 '구조'만 조립한다.
"""
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from ..config import MODEL
from ..domain.models import Feed, SlotExtraction
from ..infrastructure.naver import fetch_page, naver_search
from . import prompts

_SEARCH_TOOLS = [naver_search, fetch_page]


class AgentRegistry:
    """유스케이스가 직접 부르는 세 진입 에이전트를 보관한다."""

    def __init__(self) -> None:
        self.slot_extractor = _structured("slot_extractor", prompts.SLOT_EXTRACTOR, SlotExtraction)
        self.coordinator = _coordinator()
        self.formatter = _structured("formatter", prompts.FORMATTER, Feed)


def _coordinator() -> LlmAgent:
    """목적별 검색 서브 에이전트를 위임 호출하는 코디네이터."""
    subs = [
        _searcher("restaurant_agent", "식당·맛집"),
        _searcher("cafe_agent", "카페·디저트"),
        _festival(),
        _searcher("spot_agent", "관광지·명소"),
    ]
    return LlmAgent(
        name="coordinator",
        model=MODEL,
        instruction=prompts.COORDINATOR,
        tools=[AgentTool(agent=sub) for sub in subs],
    )


def _festival() -> LlmAgent:
    return LlmAgent(
        name="festival_agent",
        model=MODEL,
        description="축제·행사·공연 정보를 검색한다.",
        instruction=prompts.FESTIVAL,
        tools=_SEARCH_TOOLS,
    )


def _searcher(name: str, desc: str) -> LlmAgent:
    """장소(맛집/카페/명소) 계열 검색 에이전트."""
    return LlmAgent(
        name=name,
        model=MODEL,
        description=f"{desc}을(를) 네이버에서 검색한다.",
        instruction=prompts.PLACE.format(desc=desc),
        tools=_SEARCH_TOOLS,
    )


def _structured(name: str, instruction: str, schema) -> LlmAgent:
    """JSON 스키마로 출력을 강제하는 에이전트 (툴 없음)."""
    return LlmAgent(name=name, model=MODEL, instruction=instruction, output_schema=schema)
