"""진입점이자 조립 지점(composition root).

앱은 하나의 하네스 flow다:
  질문 → query_planner(짧은 쿼리) → 카카오 전량 수집 → '전부' 리뷰요약(서브) → selector(10개, 메인)
'전부 검색'을 코드가 보장한다(모델은 쿼리분해·선택만). CLI·웹·평가 모두 이 Search를 부른다.

로그(실행마다 두 파일):
  logs/<stem>.flow.log   — 흐름(질문→쿼리→카카오→리뷰URL→결과)  [콘솔+파일]
  logs/<stem>.agents.log — 에이전트 입출력(프롬프트+JSON)          [파일만]

실행: python -m poc.service_v2.main
"""
import asyncio
from datetime import datetime

from .agents import registry
from .application.enrichment import PlaceEnricher
from .application.search import Search
from .config import (
    KAKAO_REST_API_KEY, MAIN_MODEL, MODEL, MODEL_API_BASE, MODEL_API_KEY,
    NAVER_CLIENT_ID, NAVER_CLIENT_SECRET,
)
from .infrastructure.kakao import KakaoLocal
from .infrastructure.logs import AgentLog, FileSink, FlowLog
from .infrastructure.metrics import VllmMetrics
from .infrastructure.model import build_model
from .infrastructure.naver import NaverSearch
from .infrastructure.recorder import RunRecorder
from .infrastructure.runner import AgentRunner
from .interface import cli

APP = "wh"
USER = "u1"


def build_search(log_name: str | None = None) -> Search:
    """하네스 flow 엔진을 조립한다. 질문 → Feed(카드 10개)."""
    stem = log_name or f"run_{datetime.now():%Y%m%d}"
    flow = FlowLog(FileSink(f"logs/{stem}.flow.log", echo=True))
    agent_log = AgentLog(FileSink(f"logs/{stem}.agents.log"))
    metrics = VllmMetrics(MODEL_API_BASE, MODEL_API_KEY) if MODEL_API_BASE else None
    runner = AgentRunner(APP, USER, agent_log, RunRecorder(), metrics)

    main_model = build_model(MAIN_MODEL, MODEL_API_BASE, MODEL_API_KEY)   # 쿼리분해·선택
    sub_model = build_model(MODEL, MODEL_API_BASE, MODEL_API_KEY)         # 리뷰 요약(저렴)
    kakao = KakaoLocal(KAKAO_REST_API_KEY)
    naver = NaverSearch(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)

    enricher = PlaceEnricher(naver, runner, registry.review_agent(sub_model), flow)
    return Search(
        runner,
        registry.query_planner(main_model),
        registry.selector(main_model),
        kakao,
        enricher,
        flow,
    )


def main() -> None:
    try:
        asyncio.run(cli.run(build_search()))
    except (KeyboardInterrupt, EOFError):
        print("\n👋 종료합니다.")


if __name__ == "__main__":
    main()
