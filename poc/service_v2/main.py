"""CLI 진입점이자 조립 지점(composition root).

각 계층을 여기서 한 번만 연결한다. 실행:

    python -m service2.main
"""
import asyncio

from .agents.registry import AgentRegistry
from .application.recommender import Recommender
from .config import MODEL_API_BASE, MODEL_API_KEY
from .infrastructure.logger import RunLogger
from .infrastructure.metrics import VllmMetrics
from .infrastructure.naver import NaverPlaces
from .infrastructure.recorder import RunRecorder
from .infrastructure.runner import AgentRunner
from .infrastructure.tracer import RunTracer
from .interface import cli

APP = "wh"
USER = "u1"


def build_recommender(logger: RunLogger | None = None) -> Recommender:
    """의존성을 조립해 유스케이스 객체를 만든다.

    logger를 주면 그 로거로 기록한다(예: report가 실행별 로그 파일을 넘김).
    없으면 날짜별 기본 로거를 쓴다.
    """
    logger = logger or RunLogger()
    tracer = RunTracer(logger)
    recorder = RunRecorder()
    # 원격 vLLM이면 서버측 지표(통신시간 제외)를 붙인다. 네이티브(Gemini)면 생략.
    metrics = VllmMetrics(MODEL_API_BASE, MODEL_API_KEY) if MODEL_API_BASE else None
    runner = AgentRunner(APP, USER, tracer, recorder, metrics)
    places = NaverPlaces(log=logger.write)   # 검색·취득 진행상황을 파이프라인 로그에 남긴다
    return Recommender(runner, AgentRegistry(), places)


def main() -> None:
    try:
        asyncio.run(cli.run(build_recommender()))
    except (KeyboardInterrupt, EOFError):
        print("\n👋 종료합니다.")


if __name__ == "__main__":
    main()
