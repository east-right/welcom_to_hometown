"""에이전트 배선. '구조'만 조립한다(내용은 prompts). 셋 다 툴 없는 구조화 출력.

- query_planner : 질문 → 짧은 카카오 검색어 여러 개  (메인 모델, 판단)
- review_agent  : 질문 + 후기 원문 → 질문 연관 3줄 요약  (서브 모델, 병렬)
- selector      : 요약된 후보 전부 → 카드 10개 선택      (메인 모델, 판단)
"""
from google.adk.agents import LlmAgent

from ..domain.models import Feed, QueryPlan, Review
from . import prompts


def query_planner(model) -> LlmAgent:
    return _structured(model, "query_planner", prompts.QUERY_PLANNER, QueryPlan)


def review_agent(model) -> LlmAgent:
    return _structured(model, "review_agent", prompts.REVIEW, Review)


def selector(model) -> LlmAgent:
    return _structured(model, "selector", prompts.SELECTOR, Feed)


def _structured(model, name: str, instruction: str, schema) -> LlmAgent:
    return LlmAgent(name=name, model=model, instruction=instruction, output_schema=schema)
