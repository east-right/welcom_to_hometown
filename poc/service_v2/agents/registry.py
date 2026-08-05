"""에이전트 배선. '구조'만 조립한다(내용은 prompts, 실행은 runner).

- slot_extractor : 대화 → 슬롯 (구조화 출력)
- query_planner  : 슬롯 → 카카오용 짧은 검색어 2~5개 (구조화 출력)
- review_agent   : 후기 원문 → 요약 (구조화 출력) — enricher가 장소마다 병렬로 돌린다
- selector       : 요약된 후보 전부 → 카드 선택·랭킹 (구조화 출력)

넷 다 툴 없는 구조화 출력 에이전트. 수집·요약은 코드가 지휘하고, 모델은 추론만 한다.
"""
from google.adk.agents import LlmAgent

from ..domain.models import Feed, QueryPlan, Review, SlotExtraction
from . import prompts


def slot_extractor(model) -> LlmAgent:
    return _structured(model, "slot_extractor", prompts.SLOT_EXTRACTOR, SlotExtraction)


def query_planner(model) -> LlmAgent:
    return _structured(model, "query_planner", prompts.QUERY_PLANNER, QueryPlan)


def review_agent(model) -> LlmAgent:
    return _structured(model, "review_agent", prompts.REVIEW, Review)


def selector(model) -> LlmAgent:
    return _structured(model, "selector", prompts.SELECTOR, Feed)


def _structured(model, name: str, instruction: str, schema) -> LlmAgent:
    return LlmAgent(name=name, model=model, instruction=instruction, output_schema=schema)
