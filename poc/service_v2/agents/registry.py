"""에이전트 배선. '구조'만 조립한다(내용은 prompts, 실행은 runner).

- slot_extractor : 대화 → 슬롯 (구조화 출력, 툴 없음)
- review_agent   : 후기 원문 → 요약 (구조화 출력, 툴 없음) — read_reviews 도구 안에서 돈다
- coordinator    : 도구를 들고 스스로 검색·확인·선별하는 agentic 에이전트
                   (output_schema=Feed → 루프 중엔 툴, 최종 출력만 스키마 강제)
"""
from google.adk.agents import LlmAgent

from ..domain.models import Feed, Review, SlotExtraction
from . import prompts


def slot_extractor(model) -> LlmAgent:
    return _structured(model, "slot_extractor", prompts.SLOT_EXTRACTOR, SlotExtraction)


def review_agent(model) -> LlmAgent:
    return _structured(model, "review_agent", prompts.REVIEW, Review)


def coordinator(model, tools) -> LlmAgent:
    return LlmAgent(
        name="coordinator",
        model=model,
        instruction=prompts.COORDINATOR,
        tools=tools,
        output_schema=Feed,
    )


def _structured(model, name: str, instruction: str, schema) -> LlmAgent:
    return LlmAgent(name=name, model=model, instruction=instruction, output_schema=schema)
