"""레이어별 독립 테스트.

    python tests.py            전체
    python tests.py clients    Layer 1만
    python tests.py tools      Layer 2만
    python tests.py chat       Layer 3 단발
    python tests.py multiturn  멀티턴 시나리오
    python tests.py probe      되묻기 남발 점검

툴은 ADK가 tool_context를 주입하므로, 단독 호출에는 아래 FakeToolContext를 쓴다.
"""

import asyncio
import glob
import sys
import time

from wh_poc import Chat, set_verbose
from wh_poc.clients import kakao_keyword, llm_async, naver_search, scrape_urls
from wh_poc.core import FAIL_PREFIX, load_json
from wh_poc.tools import (
    collect_context,
    collect_info,
    collect_places,
    get_today,
    read_file,
)


class FakeToolContext:
    """ADK가 주입하는 ToolContext 대역.

    같은 인스턴스를 재사용하면 state가 유지되어 재시도 카운트나
    파일 경로 전달을 그대로 재현할 수 있다.
    """

    def __init__(self):
        self.state = {}


# ---------------------------------------------------------------- Layer 1

async def test_clients():
    print("\n=== 카카오: 상설 매장 vs 축제 ===")
    for q in ["신림역 밥집", "진해 군항제"]:
        places, is_end = kakao_keyword(q)
        print(f"{q}: {len(places)}개 (is_end={is_end})")
        for p in places[:3]:
            print(f"   {p['name']} | {p['category']} | {p['address']}")

    print("\n=== 네이버 3종 비교 ===")
    for kind in ["blog", "news", "webkr"]:
        items, status = await naver_search("진해 군항제", kind=kind, display=3, sort="date")
        print(f"--- {kind} ({status}) ---")
        for i in items:
            print(f"  {i['title'][:55]}  [{i['date'][:10]}]")

    print("\n=== 크롤링: 실패는 예외가 아니라 마커 문자열 ===")
    items, _ = await naver_search("봉천 도화불백", kind="blog", display=3, sort="date")
    scraped = await scrape_urls([i["link"] for i in items])
    for s in scraped:
        ok = not s["text"].startswith(FAIL_PREFIX)
        print(f"[{'OK  ' if ok else 'FAIL'}] {len(s['text']):5}자 | {s['text'][:60]}")

    print("\n=== LLM 병렬 (5개가 1개와 비슷하면 정상) ===")
    t0 = time.time()
    await asyncio.gather(*[llm_async("1+1은? 숫자만") for _ in range(5)])
    print(f"5개 병렬: {time.time() - t0:.1f}초")


# ---------------------------------------------------------------- Layer 2

async def test_tools():
    print("\n=== get_today ===")
    print(get_today())

    ctx = FakeToolContext()
    print("\n=== collect_places: 부족하면 retry, 소진되면 바로 ok ===")
    print(await collect_places("신림역 밥집", ctx))

    print("\n=== retry 흐름: 어휘를 바꿔가며 3회 ===")
    ctx2 = FakeToolContext()
    for q in ["신림역 조용한 작업 백반집", "신림역 백반집", "신림역 밥집"]:
        res = await collect_places(q, ctx2)
        print(f"  {res.get('status'):10} | {q}")

    print("\n=== collect_info: 위 결과를 이어받는다 ===")
    info = await collect_info(ctx, "신림역 혼밥하기 좋은 백반집", "신림")
    print(info)
    for name, s in list(load_json(info["summary_file"]).items())[:3]:
        print(f"\n  {name} ({s['review_count']}건) | {s['address']}")
        print(f"    강점: {s['strengths']}")
        print(f"    약점: {s['weaknesses']}")

    print("\n=== collect_context: 축제 케이스 ===")
    ctx3 = FakeToolContext()
    print(await collect_places("진해 군항제", ctx3))
    context = await collect_context(ctx3, "진해 군항제 언제 열려?")
    print(context)
    for name, c in list(load_json(context["context_file"]).items())[:3]:
        print(f"\n  {name} ({c['source_count']}건, {c['freshness']})")
        print(f"    기간: {c['period']}")
        print(f"    사실: {c['facts']}")

    print("\n=== read_file: 허용 경로 밖은 막힌다 ===")
    latest = sorted(glob.glob("./tool_file/*_info_summary.json"))[-1]
    print(list(read_file(latest)["content"].keys())[:5])
    print(read_file("./run.py"))


# ---------------------------------------------------------------- Layer 3

async def test_chat():
    set_verbose(False)

    print("\n### 기준점이 명확한 요청 — 되묻지 않고 진행해야 정상")
    c1 = Chat()
    await c1.say("신림역에서 혼밥하기 좋은 백반집 추천해줘")

    print("\n### 행사형 — get_today + collect_context를 고르는지")
    c2 = Chat()
    await c2.say("이번에 진해 군항제 가려는데 언제 하는지랑 볼거리 알려줘")
    print("호출된 툴:", [t["name"] for t in c2.history[-1]["tool_calls"]])

    print("\n### 애매한 쿼리 — 어느 소스를 고르는지")
    c3 = Chat()
    await c3.say("가평 캠핑장 추천해줘. 시설 좋고 예약 쉬운 곳으로")
    print("호출된 툴:", [t["name"] for t in c3.history[-1]["tool_calls"]])


async def test_multiturn():
    set_verbose(False)

    print("\n### 시나리오 1 — 되묻기 → 답변 → 후속질문 → 새 검색")
    mt = Chat()
    await mt.say("신림동 백반")               # too_broad → question 기대
    await mt.say("신림역 근처로 좁혀줘")        # 좁혀서 재수집 → recommendation
    await mt.say("그중에 웨이팅 없는 곳은?")     # 툴 없이 chat 기대
    print("호출된 툴:", [t["name"] for t in mt.history[-1]["tool_calls"]])
    await mt.say("아니다 강남역 쪽으로 다시 찾아줘")  # 새 검색

    print("\n--- 이력 ---")
    mt.summary()

    print("\n### 시나리오 2 — 지역이 없으면 툴 없이 되물어야 한다")
    mt2 = Chat()
    await mt2.say("카페 추천해줘")
    print("호출된 툴:", [t["name"] for t in mt2.history[-1]["tool_calls"]])
    await mt2.say("성수동")

    print("\n### 시나리오 3 — 잡담")
    mt3 = Chat()
    await mt3.say("안녕 너 뭐하는 애야?")
    r = mt3.history[-1]["result"]
    print(f"kind={r['kind']} places={len(r['places'])} "
          f"tools={[t['name'] for t in mt3.history[-1]['tool_calls']]}")


async def test_trace():
    """오케스트레이션 트레이싱. 턴 안에서 LLM과 툴이 언제 무엇을 했는지 본다."""
    set_verbose(False)   # 툴 내부 로그를 끄면 타임라인이 잘 보인다

    c = Chat(show_status=False)
    await c.say("신림동 백반")
    await c.say("신림역 근처로 좁혀줘")
    await c.say("그중에 웨이팅 없는 곳은?")

    print("\n### 턴별 타임라인")
    c.trace()

    print("\n### 합계")
    c.totals()


async def test_probe():
    """되묻기 남발 점검. 기준점이 명확한 요청에 question이 나오면 과한 것."""
    set_verbose(False)
    for q in ["신림역 백반집", "성수동 카공카페", "진해 군항제", "가평 캠핑장"]:
        out = await Chat().send(q)
        kind = (out["result"] or {}).get("kind", "?")
        flag = "   ← 되물음(과함)" if kind == "question" else ""
        print(f"{q:18} → {kind}{flag}")


# ----------------------------------------------------------------

SUITES = {
    "clients": test_clients,
    "tools": test_tools,
    "chat": test_chat,
    "multiturn": test_multiturn,
    "probe": test_probe,
    "trace": test_trace,
}


async def main():
    names = sys.argv[1:] or list(SUITES)
    for n in names:
        if n not in SUITES:
            print(f"알 수 없는 스위트: {n}. 가능: {', '.join(SUITES)}")
            continue
        print(f"\n{'#' * 60}\n# {n}\n{'#' * 60}")
        await SUITES[n]()


if __name__ == "__main__":
    asyncio.run(main())
