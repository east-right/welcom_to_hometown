"""에이전트 배선(registry). 어떤 에이전트가 어떤 툴을 갖는지 한눈에 보인다.

지시문(내용)은 prompts.py에, 실행 방법은 infrastructure/runner.py에 있다.
여기서는 '구조'만 조립한다.
"""
from google.adk.agents import LlmAgent

from ..config import MODEL, MODEL_API_BASE, MODEL_API_KEY
from ..domain.models import Feed, QueryPlan, SlotExtraction
from ..infrastructure.model import build_model
from . import prompts

# WH_MODEL 하나로 모든 에이전트가 같은 모델을 쓴다. (원격 오픈모델이면 LiteLlm 객체)
_MODEL = build_model(MODEL, MODEL_API_BASE, MODEL_API_KEY)


class AgentRegistry:
    """유스케이스가 직접 부르는 세 에이전트. 모두 구조화 출력이고 툴이 없다.

    검색은 유스케이스가 병렬로 지휘하므로, 에이전트는 순수 추론만 한다:
      slot_extractor → 대화에서 슬롯 추출
      query_expander → 슬롯을 검색어 8~10개로 증강
      selector       → 검색 후보를 보고 추천 카드 선별·정렬
    """

    def __init__(self) -> None:
        self.slot_extractor = _structured("slot_extractor", prompts.SLOT_EXTRACTOR, SlotExtraction)
        self.query_expander = _structured("query_expander", prompts.QUERY_EXPANDER, QueryPlan)
        self.selector = _structured("selector", prompts.SELECTOR, Feed)


def _structured(name: str, instruction: str, schema) -> LlmAgent:
    """JSON 스키마로 출력을 강제하는 에이전트 (툴 없음)."""
    return LlmAgent(name=name, model=_MODEL, instruction=instruction, output_schema=schema)
