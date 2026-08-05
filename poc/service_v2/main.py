"""CLI 진입점이자 조립 지점(composition root).

각 계층을 여기서 한 번만 연결한다:
  카카오/네이버 → enricher(리뷰 병렬 요약) → selector(main 모델) → Recommender

실행: python -m poc.service_v2.main
"""
import asyncio

from .agents import registry
from .application.enrichment import PlaceEnricher
from .application.recommender import Recommender
from .config import (
    KAKAO_REST_API_KEY, MAIN_MODEL, MODEL, MODEL_API_BASE, MODEL_API_KEY,
    NAVER_CLIENT_ID, NAVER_CLIENT_SECRET,
)
from .infrastructure.kakao import KakaoLocal
from .infrastructure.logger import RunLogger
from .infrastructure.metrics import VllmMetrics
from .infrastructure.model import build_model
from .infrastructure.naver import NaverBlog
from .infrastructure.recorder import RunRecorder
from .infrastructure.runner import AgentRunner
from .infrastructure.tracer import NullTracer, RunTracer
from .interface import cli

APP = "wh"
USER = "u1"


def build_recommender(logger: RunLogger | None = None) -> Recommender:
    """의존성을 조립해 유스케이스 객체를 만든다.

    - main 러너: slot/selector(사람이 읽는 로그).
    - review 러너: 병렬로 도는 후기 요약 → NullTracer(로그 안 섞이게).
    - coordinator(선택) 모델은 MAIN_MODEL, 요약·슬롯은 MODEL(저렴).
    """
    logger = logger or RunLogger()
    recorder = RunRecorder()
    metrics = VllmMetrics(MODEL_API_BASE, MODEL_API_KEY) if MODEL_API_BASE else None
    main_runner = AgentRunner(APP, USER, RunTracer(logger), recorder, metrics)
    review_runner = AgentRunner(APP, USER, NullTracer(), recorder, metrics)

    main_model = build_model(MAIN_MODEL, MODEL_API_BASE, MODEL_API_KEY)
    sub_model = build_model(MODEL, MODEL_API_BASE, MODEL_API_KEY)
    kakao = KakaoLocal(KAKAO_REST_API_KEY)
    naver = NaverBlog(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)

    enricher = PlaceEnricher(naver, review_runner, registry.review_agent(sub_model), log=logger.write)
    return Recommender(
        main_runner,
        registry.slot_extractor(sub_model),
        registry.query_planner(main_model),   # 쿼리 쪼개기 = 판단이라 main 모델
        registry.selector(main_model),
        kakao,
        enricher,
    )


def main() -> None:
    try:
        asyncio.run(cli.run(build_recommender()))
    except (KeyboardInterrupt, EOFError):
        print("\n👋 종료합니다.")


if __name__ == "__main__":
    main()
