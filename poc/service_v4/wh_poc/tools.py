"""Layer 2 — 툴.

에이전트에 노출되는 함수들. docstring이 곧 LLM에게 주는 명세다.

설계 기준: 코드가 관측 가능한 조건(결과 개수, is_end, API 응답 상태)은 함수 안으로
내리고, 의미론적 판단(검색어 압축, 어휘 교체, 소스 선택, 되물을지)만 에이전트에게 남긴다.
툴은 신호를 주고 판단은 에이전트가 한다.
"""

import asyncio
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from google.adk.tools import ToolContext

from . import prompts
from .clients import (
    in_same_region,
    kakao_all,
    llm_async,
    naver_search,
    scrape_urls,
    verify_names,
)
from .core import (
    FAIL_PREFIX,
    TOOL_FILE_DIR,
    dump,
    load_json,
    log,
    timed_tool,
)

# collect_places 수집 기준
MAX_COUNT = 45  # 카카오가 노출하는 상한. 이만큼 채우면 후보가 넘친다는 신호(too_broad).

INFO_PER_PLACE = 3  # 매장당 채택할 블로그 글 수
INFO_BODY_CHARS = 2000
CONTEXT_BODY_CHARS = 4000


def _run_id(tool_context: ToolContext) -> str:
    """현재 invocation 식별자. 같은 요청에서 나온 파일들을 한 덩어리로 묶는다."""
    rid = tool_context.state.get("temp:run_id")
    if rid is None:
        rid = str(int(time.time() * 1000))
        tool_context.state["temp:run_id"] = rid
    return rid


# ---------------------------------------------------------------- get_today

def get_today() -> dict:
    """오늘 날짜와 요일, 현재 시각을 알려준다.

    '오늘', '이번 주말', '지금 하는' 같은 상대적 시점이 요청에 포함되거나,
    축제·행사처럼 기간이 있는 대상을 찾을 때 먼저 호출해 현재 시점을 확인한다.

    Returns:
        {"date": "2026-08-13", "weekday": "목요일", "time": "14:30", "year": 2026, "month": 8}
    """
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    return {
        "date": now.strftime("%Y-%m-%d"),
        "weekday": weekdays[now.weekday()],
        "time": now.strftime("%H:%M"),
        "year": now.year,
        "month": now.month,
    }


# ---------------------------------------------------------------- collect_places

@timed_tool
async def collect_places(
    query: str,
    tool_context: ToolContext,
    category_code: str = "",
) -> dict:
    """지역 키워드로 장소 후보 목록을 수집한다. 모든 작업의 첫 단계다.

    카카오 로컬 검색만으로는 '맛집' 태그가 없는 로컬 노포(동네 순대집, 국숫집 등)를
    놓친다. 그래서 두 갈래로 모은다.
      1) 카카오 로컬 키워드 검색 전량 (최대 45개, is_end면 조기 종료)
      2) 수확 보강 — 네이버 웹검색 스니펫에서 상호명을 뽑아(LLM) 카카오로 역조회.
         카카오가 놓친 로컬 매장이 여기서 잡힌다.
    수집한 후보를 자르지 않고 전량 파일에 남긴다.

    검색어는 '지역 + 짧은 명사' 형태여야 한다. 서술형 문장은 결과가 거의 나오지 않는다.
      좋은 예: '성수동 카페', '진해 군항제', '가평 캠핑장'
      나쁜 예: '성수동에서 조용히 작업하기 좋은 카페'

    Args:
        query: 검색어. '지역 + 짧은 명사' 조합.
        category_code: 카테고리 필터. 'CE7'=카페, 'FD6'=음식점, 'AT4'=관광명소, 'CT1'=문화시설.
                       확실하지 않으면 빈 문자열로 두는 편이 결과가 많다.

    Returns:
        status가 'too_broad'면 카카오 상한(45개)을 채운 것이다. message를 읽고
        그대로 진행할지 사용자에게 좁힐 기준을 되물을지 판단한다.
        status가 'ok'면 수집이 끝난 것이므로 다음 단계로 넘어간다.
        축제나 특정 시설처럼 대상이 하나뿐인 경우 결과가 적어도 'ok'가 나온다. 정상이다.
    """
    tool_context.state["temp:collect_attempts"] = (
        tool_context.state.get("temp:collect_attempts", 0) + 1
    )

    # 1) 카카오 로컬 전량 — 기준점이자 수확분의 지역 필터 근거
    places = kakao_all(query, category_code=category_code)
    base = list(places)                 # 지역 필터 기준 스냅샷
    base_n = len(base)                  # too_broad 판정은 카카오 base 기준 (수확분 제외)
    log(f"[collect_places] '{query}' 카카오 {base_n}개")

    # 2) 수확 보강 — 웹검색 스니펫에서 상호명만 LLM으로 뽑아 카카오로 역조회.
    #    카카오 '맛집' 태그가 없는 로컬 노포는 카카오가 놓치지만 웹 스니펫엔 리스트로
    #    잡힌다. 역조회로 실존·주소를 확보하고, 같은 구가 아니면 버린다.
    #    'Top'을 붙여 랭킹 리스트 페이지를 노린다 — 한 스니펫에 로컬 매장이 촘촘히
    #    나열돼 수확 밀도가 높다(지역 무관한 일반 바이어스라 하드코딩이 아니다).
    web_items, _ = await naver_search(f"{query} Top", kind="webkr", display=30)
    snippets = "\n".join(f"{w['title']} | {w['description']}" for w in web_items)
    try:
        names = json.loads(
            await llm_async(f"{prompts.EXTRACT_NAMES}\n{snippets}", json_mode=True)
        )
        names = [n.strip() for n in names if isinstance(n, str) and n.strip()]
    except (json.JSONDecodeError, TypeError):
        names = []

    seen = {p["id"] for p in places}
    n_harvest = 0
    for p in verify_names(names, region_hint=(query.split() or [""])[0]):
        if p["id"] not in seen and in_same_region(p, base):
            p["source"] = "web_harvest"
            places.append(p)
            seen.add(p["id"])
            n_harvest += 1
    log(f"[collect_places] 수확 후보 {len(names)}개 → 신규 확보 {n_harvest}개")

    file_path = dump("collect_places", places, _run_id(tool_context))

    # 파일 경로는 temp: 없이 저장한다. temp:는 invocation 스코프라 다음 턴에 사라지는데,
    # 멀티턴에서는 되묻고 답을 받은 뒤에도 이 경로가 필요하다.
    tool_context.state["last_places_file"] = file_path
    tool_context.state["last_query"] = query

    payload = {
        "count": len(places),
        "source_breakdown": {"kakao": base_n, "web_harvest": n_harvest},
        "file": file_path,
    }

    # base(카카오)가 45개 상한을 채웠다는 건 is_end 전에 상한에 닿았다는 뜻 — 아직 더 있다.
    # 후보가 넘친다는 사실만 알려주고, 좁힐지 진행할지는 에이전트가 판단한다. 발화에
    # 이미 기준점이 있는지(예: "역"이 붙었는지)는 코드가 알 수 없다.
    if base_n >= MAX_COUNT:
        payload["status"] = "too_broad"
        payload["message"] = (
            f"'{query}' 후보가 {MAX_COUNT}개 이상입니다. "
            "사용자 요청에 이미 좁히는 기준(역 이름, 구체적 동네 등)이 있으면 "
            "그대로 진행하세요. 기준이 없어 범위가 막연하다면 "
            "사용자에게 좁힐 기준을 되물어도 됩니다."
        )
    else:
        payload["status"] = "ok"

    return payload


# ---------------------------------------------------------------- collect_info

@timed_tool
async def collect_info(
    tool_context: ToolContext, user_query: str, region_prefix: str
) -> dict:
    """수집된 장소들의 블로그 후기를 모아 장점·단점 중심으로 요약한다.

    실제 방문자 경험(맛, 분위기, 혼잡도, 1인석 유무 등)이 판단 근거로 필요할 때 쓴다.
    카페·음식점처럼 방문 후기가 많은 대상에 적합하다.
    축제·행사처럼 '언제 열리는지'가 중요한 대상은 collect_context를 대신 쓴다.

    Args:
        user_query: 사용자의 원래 요청 문장. 어떤 관점에서 장단점을 판단할지 결정한다.
                    예: '신림역 혼밥하기 좋은 백반집'
        region_prefix: 블로그 검색어에 붙일 지역명. 사람들이 실제로 그 동네를 부르는
                    짧은 통칭을 쓴다. '역', '동' 접미사와 도로명·행정동은 뺀다.
                    예: '신림역' → '신림', '관악구 관천로' → '신림',
                        '금천구 가산디지털2로' → '가산'

    Returns:
        {"count": int, "no_info": int, "url_file": str, "summary_file": str}
        count는 요약이 만들어진 장소 수, no_info는 후기를 못 찾은 장소 수.
        판단 근거는 summary_file에 있다. read_file로 읽어라.
    """
    places_file = tool_context.state.get("last_places_file")
    if not places_file:
        return {
            "status": "error",
            "message": "collect_places를 먼저 호출하세요. 이 툴을 다시 호출해도 같은 결과입니다.",
        }

    places = load_json(places_file)
    run_id = _run_id(tool_context)

    # --- 1) 매장별 블로그 URL 수집 ---
    # 도로명은 행정 표기라 실제 리뷰에서 쓰는 생활권 이름과 다르다("관천" vs "신림").
    # 그 매핑은 지역 상식이라 코드가 못 하므로 에이전트가 준 region_prefix를 쓴다.
    # sim/date는 매장마다 유리한 쪽이 달라 사전 판단이 불가능하므로 둘 다 던지고 합친다.
    queries = [f"{region_prefix} {p['name']}".strip() for p in places]

    raw = await asyncio.gather(
        *[
            naver_search(q, kind="blog", display=10, sort=s)
            for q in queries
            for s in ("sim", "date")
        ]
    )

    urls, reasons = {}, {}
    for i, p in enumerate(places):
        merged = raw[i * 2][0] + raw[i * 2 + 1][0]
        statuses = (raw[i * 2][1], raw[i * 2 + 1][1])

        seen, hits = set(), []
        for b in merged:
            if b["link"] in seen:
                continue
            seen.add(b["link"])
            # 검색어가 통해도 무관한 글이 섞이므로 상호명 실제 등장 여부로 한 번 더 거른다
            if p["name"] in b["title"] or p["name"] in b["description"]:
                hits.append(b)

        urls[p["name"]] = hits[:INFO_PER_PLACE]
        if not hits:
            reasons[p["name"]] = (
                "empty" if all(s == "empty" for s in statuses) else "no_match_after_filter"
            )

    url_file = dump("info_urls", urls, run_id)
    if reasons:
        log(f"[collect_info] 후기 미확보 {len(reasons)}개 → {dict(Counter(reasons.values()))}")

    # --- 2) 본문 크롤링 (파일로 남기지 않음: 요약이 나오면 원문은 용도 끝) ---
    links, owner = [], []
    for name, items in urls.items():
        for item in items:
            links.append(item["link"])
            owner.append(name)

    scraped = await scrape_urls(links)
    bodies: dict[str, list[str]] = {name: [] for name in urls}
    for name, s in zip(owner, scraped):
        if not s["text"].startswith(FAIL_PREFIX):
            bodies[name].append(s["text"])
    log(
        f"[collect_info] 크롤링 {len(scraped)}건 중 "
        f"성공 {sum(len(v) for v in bodies.values())}건"
    )

    # --- 3) 매장당 후기 전체를 한 번에 넣어 통합 요약 ---
    # 후기를 개별 요약하면 반복 빈도(=신뢰도) 정보가 사라진다.
    async def summarize_one(name: str, texts: list[str]) -> dict:
        joined = "\n\n---\n\n".join(
            f"[후기 {i + 1}]\n{t[:INFO_BODY_CHARS]}" for i, t in enumerate(texts)
        )
        prompt = (
            prompts.SUMMARIZE_INFO.format(
                user_query=user_query, place_name=name, n=len(texts)
            )
            + "\n"
            + joined
        )
        try:
            return json.loads(await llm_async(prompt, json_mode=True))
        except json.JSONDecodeError:
            return {"strengths": [], "weaknesses": [], "summary": "정보 부족"}

    names = [n for n, t in bodies.items() if t]
    results = await asyncio.gather(*[summarize_one(n, bodies[n]) for n in names])

    # 주소·URL을 요약에 병합 — 이게 없으면 최종 답변에서 주소를 지어낸다
    meta = {p["name"]: p for p in places}
    summaries = {
        n: {
            **r,
            "review_count": len(bodies[n]),
            "address": meta[n].get("address", ""),
            "category": meta[n].get("category", ""),
            "url": meta[n].get("url", ""),
        }
        for n, r in zip(names, results)
    }
    summary_file = dump("info_summary", summaries, run_id)

    return {
        "count": len(summaries),
        "no_info": len(places) - len(summaries),
        "url_file": url_file,
        "summary_file": summary_file,
    }


# ---------------------------------------------------------------- collect_context

@timed_tool
async def collect_context(
    tool_context: ToolContext, user_query: str, top_n: int = 10
) -> dict:
    """수집된 장소들의 일정·운영정보를 웹문서와 뉴스에서 찾아 정리한다.

    축제·행사·전시처럼 '언제 열리는지'가 핵심이거나, 관광지·캠핑장처럼 예약·요금·시설
    정보가 필요할 때 쓴다. 뉴스는 날짜와 장소가 사실로 명시되어 블로그보다 신뢰도가 높다.

    카페·음식점처럼 방문자 경험이 중요한 대상은 collect_info를 대신 쓴다.

    Args:
        user_query: 사용자의 원래 요청 문장.
        top_n: 상위 몇 개 장소를 조사할지. 축제·행사는 후보가 적으므로 기본값으로 충분하다.

    Returns:
        {"count": int, "no_info": int, "source_file": str, "context_file": str}
        정리된 사실 정보는 context_file에 있다. read_file로 읽어라.
        freshness가 'outdated'인 항목은 지난 연도 정보이므로 주의해서 다뤄라.
    """
    places_file = tool_context.state.get("last_places_file")
    if not places_file:
        return {
            "status": "error",
            "message": "collect_places를 먼저 호출하세요. 이 툴을 다시 호출해도 같은 결과입니다.",
        }

    places = load_json(places_file)[:top_n]
    run_id = _run_id(tool_context)
    today = get_today()

    # --- 1) 웹문서 + 뉴스 동시 검색 ---
    # 웹문서는 공식 페이지·관공서 자료에, 뉴스는 일정·규모 사실에 강하다. 둘을 합쳐야 커버가 된다.
    tasks = []
    for p in places:
        region = p.get("address", "").split()[0] if p.get("address") else ""
        tasks.append(naver_search(f"{region} {p['name']}".strip(), kind="webkr", display=5))
        tasks.append(
            naver_search(f"{p['name']} {today['year']}", kind="news", display=5, sort="date")
        )
    raw = await asyncio.gather(*tasks)

    sources = {}
    for i, p in enumerate(places):
        web_items, _ = raw[i * 2]
        news_items, _ = raw[i * 2 + 1]
        merged = [{**w, "kind": "web"} for w in web_items] + [
            {**n, "kind": "news"} for n in news_items
        ]
        sources[p["name"]] = [
            d for d in merged if p["name"] in d["title"] or p["name"] in d["description"]
        ]
    source_file = dump("context_sources", sources, run_id)
    log(
        f"[collect_context] {len(places)}곳 중 문서 확보 "
        f"{sum(1 for v in sources.values() if v)}곳"
    )

    # --- 2) 본문 크롤링 ---
    links, owner = [], []
    for name, items in sources.items():
        for item in items[:4]:
            links.append(item["link"])
            owner.append(name)

    scraped = await scrape_urls(links, max_chars=CONTEXT_BODY_CHARS)
    bodies: dict[str, list[str]] = {name: [] for name in sources}
    for name, s in zip(owner, scraped):
        if not s["text"].startswith(FAIL_PREFIX):
            bodies[name].append(s["text"])

    # 본문 크롤링이 다 실패해도 검색 스니펫만으로 요약을 시도한다.
    # 공식 페이지는 봇 차단이 잦은데, 스니펫에 일정이 그대로 담긴 경우가 많다.
    for name, items in sources.items():
        if not bodies[name] and items:
            bodies[name] = [
                "\n".join(
                    f"{d['title']} | {d['description']} ({d['date']})" for d in items
                )
            ]
    log(
        f"[collect_context] 크롤링 {len(scraped)}건 중 "
        f"성공 {sum(1 for s in scraped if not s['text'].startswith(FAIL_PREFIX))}건"
    )

    # --- 3) 장소별 사실 정리 ---
    async def summarize_one(name: str, texts: list[str]) -> dict:
        joined = "\n\n---\n\n".join(f"[자료 {i + 1}]\n{t}" for i, t in enumerate(texts))
        prompt = (
            prompts.SUMMARIZE_CONTEXT.format(
                user_query=user_query, today=today["date"], name=name, n=len(texts)
            )
            + "\n"
            + joined
        )
        try:
            return json.loads(await llm_async(prompt, json_mode=True))
        except json.JSONDecodeError:
            return {
                "period": "",
                "facts": [],
                "cautions": [],
                "summary": "정보 부족",
                "freshness": "unknown",
            }

    names = [n for n, t in bodies.items() if t]
    results = await asyncio.gather(*[summarize_one(n, bodies[n]) for n in names])

    meta = {p["name"]: p for p in places}
    contexts = {
        n: {
            **r,
            "source_count": len(bodies[n]),
            "address": meta[n].get("address", ""),
            "category": meta[n].get("category", ""),
            "url": meta[n].get("url", ""),
        }
        for n, r in zip(names, results)
    }
    context_file = dump("context_summary", contexts, run_id)

    return {
        "count": len(contexts),
        "no_info": len(places) - len(contexts),
        "source_file": source_file,
        "context_file": context_file,
    }


# ---------------------------------------------------------------- read_file

def read_file(file_path: str) -> dict:
    """툴이 만든 결과 파일을 읽는다. 최종 답변을 쓰기 전에 반드시 호출한다.

    collect_info의 summary_file, collect_context의 context_file에 판단 근거가 들어있다.
    파일을 읽지 않고 답변을 지어내면 안 된다.

    Args:
        file_path: 이전 툴이 반환한 file / summary_file / context_file 값.

    Returns:
        {"content": ...} 파일 내용.
    """
    path = Path(file_path)
    if not path.exists():
        return {"status": "error", "message": f"파일을 찾을 수 없습니다: {file_path}"}
    if not str(path.resolve()).startswith(str(TOOL_FILE_DIR.resolve())):
        return {"status": "error", "message": "허용되지 않은 경로입니다."}
    try:
        return {"content": load_json(str(path))}
    except json.JSONDecodeError:
        return {"status": "error", "message": "파일을 읽을 수 없습니다."}


ALL_TOOLS = [get_today, collect_places, collect_info, collect_context, read_file]
