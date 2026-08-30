"""Layer 0 — 공용 유틸.

여러 곳에서 반복 사용되는 것만 모았다. 한 곳에서만 쓰는 로직은 호출부에 인라인으로 둔다.
"""

import asyncio
import functools
import html
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- 설정

TOOL_FILE_DIR = Path(os.getenv("WH_TOOL_FILE_DIR", "./tool_file"))
TOOL_FILE_DIR.mkdir(exist_ok=True, parents=True)

# 파이프라인 내부를 들여다볼 때 True. 에이전트 실행 중에는 끄는 편이 로그를 보기 좋다.
VERBOSE = True

# 동시성 제한 — 실측으로 정한 값
# 네이버 API. 키 1개면 10, 2번째 키(NAVER_CLIENT_ID_2)가 있으면 20(키당 10)으로 병렬↑.
# 429는 키(앱)당 걸리므로 키를 나눠 쏘면 동시성을 두 배로 가져갈 수 있다.
NAVER_SEM = asyncio.Semaphore(20 if os.getenv("NAVER_CLIENT_ID_2") else 10)
SCRAPE_SEM = asyncio.Semaphore(20)  # 본문 크롤링
LLM_SEM = asyncio.Semaphore(10)     # LLM 병렬 호출
KAKAO_SEM = asyncio.Semaphore(8)    # 카카오 로컬 API. 동기 호출을 to_thread로 감싸 병렬화

# 본문이 "[제목]" 형태로 시작하는 정상 글과 구분하기 위한 실패 마커.
# 대괄호 하나로 판정하면 "[봉천역] 털보네..." 같은 정상 글이 실패로 오탐된다.
FAIL_PREFIX = "__SCRAPE_FAIL__"


def set_verbose(flag: bool) -> None:
    global VERBOSE
    VERBOSE = flag


def log(*args) -> None:
    """VERBOSE일 때만 출력."""
    if VERBOSE:
        print(*args)


# ---------------------------------------------------------------- 계측

_STATS: dict[str, dict] = defaultdict(lambda: {"n": 0, "sec": 0.0})

# 직전 timed_tool이 남긴 카테고리별 누적 초 스냅샷. 트레이서가 툴 스텝에 붙여
# 'LLM 생성 vs 외부 I/O'를 분리해 보여줄 때 읽는다. timed_tool로 감싼 툴만 채운다.
_LAST_BREAKDOWN: dict[str, float] = {}


def take_breakdown() -> dict[str, float]:
    """직전 timed_tool의 {label: 초} 스냅샷을 꺼내며 비운다. 비계측 툴(get_today,
    read_file)은 갱신하지 않으므로 빈 dict를 받아 이전 툴 집계를 재사용하지 않는다."""
    snap = dict(_LAST_BREAKDOWN)
    _LAST_BREAKDOWN.clear()
    return snap


def track(label: str):
    """Layer 1 함수를 감싸 호출 횟수와 누적 시간을 집계한다.

    누적 시간은 벽시계가 아니라 '일감의 총량'이다.
    누적 >> 총합이면 병렬이 잘 먹고 있는 것이고,
    누적 ~= 총합이면 직렬화된 것이라 세마포어를 올려도 소용없다.
    """

    def deco(fn):
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*a, **kw):
                t0 = time.perf_counter()
                try:
                    return await fn(*a, **kw)
                finally:
                    _STATS[label]["n"] += 1
                    _STATS[label]["sec"] += time.perf_counter() - t0

            return awrapper

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                _STATS[label]["n"] += 1
                _STATS[label]["sec"] += time.perf_counter() - t0

        return wrapper

    return deco


def timed_tool(fn):
    """툴 총 소요시간과 내부 Layer 1 함수별 집계를 출력한다.

    functools.wraps 덕분에 inspect.signature가 원본을 따라가므로
    ADK의 툴 스키마 생성과 tool_context 주입에 영향이 없다.
    """

    @functools.wraps(fn)
    async def wrapper(*a, **kw):
        _STATS.clear()
        t0 = time.perf_counter()
        try:
            return await fn(*a, **kw)
        finally:
            log(f"[{fn.__name__}] 총 {time.perf_counter() - t0:.1f}초")
            for label, s in sorted(_STATS.items(), key=lambda x: -x[1]["sec"]):
                log(
                    f"   {label:<10} {s['sec']:7.1f}초  "
                    f"({s['n']}회, 평균 {s['sec'] / s['n']:.2f}초)"
                )
            # 트레이서가 이 툴 스텝에 붙일 수 있게 카테고리별 초를 남긴다.
            _LAST_BREAKDOWN.clear()
            _LAST_BREAKDOWN.update({label: s["sec"] for label, s in _STATS.items()})

    return wrapper


# ---------------------------------------------------------------- 파일

def dump(tool_name: str, data, run_id: str) -> str:
    """툴 결과를 파일로 떨군다. 같은 invocation의 산출물은 run_id로 묶인다."""
    path = TOOL_FILE_DIR / f"{run_id}_{tool_name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def load_json(file_path: str):
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def strip_tags(text: str) -> str:
    """네이버 응답의 <b> 하이라이트 태그와 HTML 엔티티 제거."""
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()
