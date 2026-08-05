"""CLI 진입점이자 조립 지점(composition root).

각 계층을 여기서 한 번만 연결한다:
  카카오/네이버 클라이언트 → 도구(클로저) → coordinator(agentic) → Recommender

실행: python -m poc.service_v2.main
"""
import asyncio

from .agents import registry
from .agents.tools import make_find_places, make_read_reviews
from .application.recommender import Recommender
from .config import (
    KAKAO_REST_API_KEY, MODEL, MODEL_API_BASE, MODEL_API_KEY,
    NAVER_CLIENT_ID, NAVER_CLIENT_SECRET,
)
from .infrastructure.kakao import KakaoLocal
from .infrastructure.logger import RunLogger
from .infrastructure.metrics import VllmMetrics
from .infrastructure.model import build_model
from .infrastructure.naver import NaverBlog
from .infrastructure.recorder import RunRecorder
from .infrastructure.runner import AgentRunner
from .infrastructure.tracer import RunTracer
from .interface import cli

APP = "wh"
USER = "u1"


def build_recommender(logger: RunLogger | None = None) -> Recommender:
    """의존성을 조립해 유스케이스 객체를 만든다.

    runner 두 개: 메인(coordinator/slot)과 리뷰(read_reviews 내부)용을 분리해
    로그 트리가 섞이지 않게 한다. logger를 주면(예: report) 그 로거로 기록한다.
    """
    logger = logger or RunLogger()
    recorder = RunRecorder()
    metrics = VllmMetrics(MODEL_API_BASE, MODEL_API_KEY) if MODEL_API_BASE else None
    main_runner = AgentRunner(APP, USER, RunTracer(logger), recorder, metrics)
    review_runner = AgentRunner(APP, USER, RunTracer(logger), recorder, metrics)

    model = build_model(MODEL, MODEL_API_BASE, MODEL_API_KEY)
    kakao = KakaoLocal(KAKAO_REST_API_KEY)
    naver = NaverBlog(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)

    review = registry.review_agent(model)
    tools = [
        make_find_places(kakao),
        make_read_reviews(naver, review_runner, review),
    ]
    coordinator = registry.coordinator(model, tools)
    return Recommender(main_runner, registry.slot_extractor(model), coordinator)


def main() -> None:
    try:
        asyncio.run(cli.run(build_recommender()))
    except (KeyboardInterrupt, EOFError):
        print("\n👋 종료합니다.")


if __name__ == "__main__":
    main()
