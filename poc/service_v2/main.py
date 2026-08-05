"""CLI 진입점이자 조립 지점(composition root).

각 계층을 여기서 한 번만 연결한다:
  카카오/네이버 → enricher(리뷰 병렬 요약) → selector(main 모델) → Recommender

로그는 실행마다 두 파일로 짝지어 남긴다:
  logs/<stem>.flow.log   — 파이프라인 데이터(쿼리·카카오·네이버 URL/fetch)  [콘솔+파일]
  logs/<stem>.agents.log — 에이전트 입출력(프롬프트+JSON)                     [파일만]

실행: python -m poc.service_v2.main
"""
import asyncio
from datetime import datetime

from .agents import registry
from .application.enrichment import PlaceEnricher
from .application.recommender import Recommender
from .config import (
    KAKAO_REST_API_KEY, MAIN_MODEL, MODEL, MODEL_API_BASE, MODEL_API_KEY,
    NAVER_CLIENT_ID, NAVER_CLIENT_SECRET,
)
from .infrastructure.kakao import KakaoLocal
from .infrastructure.logs import AgentLog, FileSink, FlowLog
from .infrastructure.metrics import VllmMetrics
from .infrastructure.model import build_model
from .infrastructure.naver import NaverBlog
from .infrastructure.recorder import RunRecorder
from .infrastructure.runner import AgentRunner
from .interface import cli

APP = "wh"
USER = "u1"


def build_recommender(log_name: str | None = None) -> Recommender:
    """의존성을 조립해 유스케이스 객체를 만든다.

    log_name을 주면(예: report의 실행별 이름) 그 이름으로 로그 파일을 만든다.
    coordinator(선택)·query_planner는 MAIN_MODEL, 요약·슬롯은 MODEL(저렴).
    """
    stem = log_name or f"run_{datetime.now():%Y%m%d}"
    flow = FlowLog(FileSink(f"logs/{stem}.flow.log", echo=True))
    agent_log = AgentLog(FileSink(f"logs/{stem}.agents.log"))

    recorder = RunRecorder()
    metrics = VllmMetrics(MODEL_API_BASE, MODEL_API_KEY) if MODEL_API_BASE else None
    main_runner = AgentRunner(APP, USER, agent_log, recorder, metrics)
    review_runner = AgentRunner(APP, USER, agent_log, recorder, metrics)

    main_model = build_model(MAIN_MODEL, MODEL_API_BASE, MODEL_API_KEY)
    sub_model = build_model(MODEL, MODEL_API_BASE, MODEL_API_KEY)
    kakao = KakaoLocal(KAKAO_REST_API_KEY)
    naver = NaverBlog(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)

    enricher = PlaceEnricher(naver, review_runner, registry.review_agent(sub_model), flow)
    return Recommender(
        main_runner,
        registry.slot_extractor(sub_model),
        registry.query_planner(main_model),   # 쿼리 쪼개기 = 판단이라 main 모델
        registry.selector(main_model),
        kakao,
        enricher,
        flow,
    )


def main() -> None:
    try:
        asyncio.run(cli.run(build_recommender()))
    except (KeyboardInterrupt, EOFError):
        print("\n👋 종료합니다.")


if __name__ == "__main__":
    main()
