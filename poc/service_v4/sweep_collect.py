"""초기 수집 recall 스윕.

웹수확은 고정하고 카카오 수집 개수(10/20/30/45)만 늘려가며 Recall@수집이 몇에서
100%가 되는지 본다. 검색어는 LLM이 압축한 것을 쓴다(에이전트의 1단계 쿼리 결정 대역).

    python sweep_collect.py            # test.json 전체
"""

import asyncio
import json

from eval import _hit
from wh_poc import prompts
from wh_poc.clients import (
    in_same_region,
    kakao_all,
    llm_async,
    naver_search,
    verify_names,
)
from wh_poc.core import load_json, set_verbose

CAPS = [10, 20, 30, 45]

# 에이전트 1단계(자연어→카카오 검색어) 대역. 실제 에이전트도 이 압축만 한다.
CQ_PROMPT = """사용자 요청을 카카오맵 지역 검색어로 압축하라.
- '지역 + 짧은 명사' 형태. 서술형 문장 금지.
- 예: "성수동에서 조용히 작업할 카페" → "성수동 카공카페"
- 예: "창원 가포 맛집 추천" → "창원 가포 맛집"
검색어 한 줄만 출력하라.
요청: "{q}"
"""


async def compress(question: str) -> str:
    raw = await llm_async(CQ_PROMPT.format(q=question))
    return raw.strip().strip('"').splitlines()[0]


async def harvest(query: str, base: list[dict]) -> list[dict]:
    """collect_places의 웹수확과 동일 — 스니펫에서 상호명 뽑아 카카오 역조회 후 같은 구만."""
    web, _ = await naver_search(f"{query} Top", kind="webkr", display=30)
    snip = "\n".join(f"{w['title']} {w['description']}" for w in web)
    try:
        names = json.loads(await llm_async(f"{prompts.EXTRACT_NAMES}\n{snip}", json_mode=True))
        names = [n.strip() for n in names if isinstance(n, str) and n.strip()]
    except (json.JSONDecodeError, TypeError):
        names = []
    out, seen = [], {p["id"] for p in base}
    for p in verify_names(names, region_hint=(query.split() or [""])[0]):
        if p["id"] not in seen and in_same_region(p, base):
            out.append(p)
            seen.add(p["id"])
    return out


async def main():
    set_verbose(False)
    cases = load_json("test.json")
    nans = 0
    tot = {c: 0 for c in CAPS}
    tot_kakao_only = {c: 0 for c in CAPS}  # 수확 없이 카카오만 (기여도 비교용)
    tot_harvest_only = 0

    for c in cases:
        q, answers = c["question"], c["answer"]
        nans += len(answers)
        query = await compress(q)
        base = kakao_all(query)  # 카카오 전량(≤45)
        harv = await harvest(query, base)

        cells = []
        for cap in CAPS:
            kk = base[:cap]
            names_k = [p["name"] for p in kk]
            names_kh = names_k + [p["name"] for p in harv]
            hk = sum(_hit(a, names_k) >= 0 for a in answers)
            hkh = sum(_hit(a, names_kh) >= 0 for a in answers)
            tot_kakao_only[cap] += hk
            tot[cap] += hkh
            cells.append(f"@{cap}:{hkh}/{len(answers)}")
        tot_harvest_only += sum(_hit(a, [p["name"] for p in harv]) >= 0 for a in answers)
        print(f"{q[:22]:24} '{query}'  base={len(base)} harv={len(harv)}  " + " ".join(cells))

    print(f"\n{'=' * 60}\n정답 총 {nans}개 | 카카오 수집개수별 Recall@수집 (웹수확 포함)\n")
    print(f"{'카카오컷':>8} {'카카오만':>12} {'카카오+웹수확':>16}")
    for cap in CAPS:
        print(f"{cap:>6}개  {tot_kakao_only[cap]:>4}/{nans} ({tot_kakao_only[cap]/nans*100:4.0f}%)"
              f"   {tot[cap]:>4}/{nans} ({tot[cap]/nans*100:4.0f}%)")
    print(f"\n(참고) 웹수확만으로 잡히는 정답: {tot_harvest_only}/{nans} ({tot_harvest_only/nans*100:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
