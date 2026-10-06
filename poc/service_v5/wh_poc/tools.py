"""Layer 2 — 툴.

에이전트에 노출되는 함수들. docstring이 곧 LLM에게 주는 명세다.

설계 기준: 코드가 관측 가능한 조건(결과 개수, is_end, API 응답 상태)은 함수 안으로
내리고, 의미론적 판단(검색어 압축, 소스 선택, 되물을지)만 에이전트에게 남긴다.
툴은 신호를 주고 판단은 에이전트가 한다.
"""

import asyncio
import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from google.adk.tools import ToolContext

from . import prompts
from .clients import (
    daum_blog,
    embed_texts,
    in_same_region,
    kakao_all,
    kakao_keyword,
    llm_async,
    naver_search,
    scrape_urls,
)
from .core import (
    FAIL_PREFIX,
    KAKAO_SEM,
    TOOL_FILE_DIR,
    dump,
    load_json,
    log,
    timed_tool,
)

# collect_places 수집 기준
MAX_COUNT = 45  # 카카오가 노출하는 상한. 이만큼 채우면 후보가 넘친다는 신호(too_broad).

RERANK_KEEP = int(os.getenv("WH_RERANK_KEEP", "15"))  # HyDE 리랭크로 남길 매장 수(크롤·요약·
                        # 2차 스코어링 대상). 약한 cos 필터가 정답을 미리 버리지 않게 넉넉히 남겨
                        # 강한 2차 스코어러로 넘긴다. KEEP 스윕 실험용으로 env로도 조정 가능.
RERANK_SNIPPETS = 10    # 매장당 확보할 블로그 스니펫 수 (많을수록 매장 신호가 풍부)
INFO_PER_PLACE = 3      # 남긴 매장당 요약에 쓸 블로그 본문 수
INFO_BODY_CHARS = 2000  # 요약에 넣는 본문 최대 길이
CONTEXT_BODY_CHARS = 4000


def _run_id(tool_context: ToolContext) -> str:
    """현재 invocation 식별자. 같은 요청에서 나온 파일들을 한 덩어리로 묶는다."""
    rid = tool_context.state.get("temp:run_id")
    if rid is None:
        rid = str(int(time.time() * 1000))
        tool_context.state["temp:run_id"] = rid
    return rid


def _cos(a: list[float], b: list[float]) -> float:
    """코사인 유사도. 임베딩 벡터 두 개의 방향 일치도(-1~1)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _norm_name(s: str) -> str:
    """상호명 비교용 정규화 — 공백·기호를 없애 표기 차이를 흡수한다.
    '서울밥집 신림점'과 '서울밥집'을 부분포함으로 같은 매장으로 보게 하기 위한 것."""
    return re.sub(r"[\s\-_()·.,'\"]", "", s).lower()


def _is_franchise(place: dict) -> bool:
    """음식점 프랜차이즈 판정. 카카오는 프랜차이즈의 브랜드명을 카테고리 끝에 붙인다
    ('음식점 > 카페 > 커피전문점 > 스타벅스'). 그래서 상호명이 그 마지막 조각으로
    '시작'하면 프랜차이즈다. 일반 매장은 마지막 조각이 업종명(갈비·칼국수)이라
    이름이 그걸로 시작하지 않는다(서울'갈비', '손'칼국수). 음식점 카테고리에만 적용."""
    cat = place.get("category", "")
    segs = [s.strip() for s in cat.split(">")]
    # 브랜드는 기본 업종보다 깊은 단계에 붙는다(음식점>카페>커피전문점>스타벅스). 얕으면
    # 마지막 조각이 넓은 업종어(카페)라 '카페모조'처럼 오탐 난다 → 3단계 이상만 본다.
    if segs[0] != "음식점" or len(segs) < 3:
        return False
    return len(segs[-1]) >= 2 and place["name"].startswith(segs[-1])


def _domain(cat: str) -> str:
    """장소의 대분류 도메인 — 카페 / 관광·문화(sight) / 음식점(food, 그 외 식당 전체).
    카카오는 카페도 '음식점 > 카페 …'로 분류해 1단계(음식점)로는 식당과 안 갈린다.
    수확 노이즈 제거는 '한식 vs 일식' 같은 장르가 아니라 이 도메인 단위로만 판단한다 —
    장르로 거르면 한식 base에 딸려온 우동(일식)·칼국수(분식) 같은 정답까지 잘려나간다."""
    segs = [s.strip() for s in cat.split(">")]
    if "카페" in segs:
        return "cafe"
    if segs and segs[0] in ("관광명소", "문화시설", "여행"):
        return "sight"
    return "food"


def _dominant_domain(base: list[dict]) -> str | None:
    """base가 한 도메인에 확실히 몰려 있으면(60%+) 그 도메인, 섞였으면 None(필터 안 함).
    구체 쿼리(카페·횟집 등)면 한 도메인이 대부분이라 다른 도메인 수확은 노이즈다."""
    if not base:
        return None
    top, n = Counter(_domain(p["category"]) for p in base).most_common(1)[0]
    return top if n / len(base) >= 0.6 else None


# ---------------------------------------------------------------- get_today

def get_today(tool_context: ToolContext | None = None) -> dict:
    """오늘 날짜와 요일, 현재 시각을 알려준다.

    '오늘', '이번 주말', '지금 하는' 같은 상대적 시점이 요청에 포함되거나,
    축제·행사처럼 기간이 있는 대상을 찾을 때 먼저 호출해 현재 시점을 확인한다.

    Returns:
        {"date": "2026-08-13", "weekday": "목요일", "time": "14:30", "year": 2026, "month": 8}
    """
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    today = {
        "date": now.strftime("%Y-%m-%d"),
        "weekday": weekdays[now.weekday()],
        "time": now.strftime("%H:%M"),
        "year": now.year,
        "month": now.month,
    }
    # 재호출 가드: 날짜는 한 요청 안에서 안 바뀐다. 이미 확인했으면 같은 값에 "그만 부르라"는
    # 신호를 붙여 반복 호출 루프를 누른다. tool_context 없이 부르는 내부 호출은 그대로 값만 준다.
    if tool_context is not None:
        if tool_context.state.get("today_checked"):
            return {**today, "already_checked": True,
                    "note": "오늘 날짜는 이미 확인했습니다. 다시 호출하지 말고 지금 정보로 "
                            "다음 단계를 진행하거나 최종 답을 내세요."}
        tool_context.state["today_checked"] = True
    return today


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
    # 재호출 가드: 같은 검색어로 이미 수집했으면 재수집하지 않고 이전 결과를 돌려준다.
    # 27B가 최종 답 대신 파이프라인을 반복하는 걸 막아 시간·컨텍스트 낭비를 없앤다.
    # 검색어가 바뀌면(되묻고 좁힌 경우 등) 정당한 재검색이므로 그대로 수행한다.
    if (tool_context.state.get("last_query") == query
            and tool_context.state.get("last_places_file")):
        return {
            "status": "ok",
            "already_collected": True,
            "file": tool_context.state["last_places_file"],
            "message": "이미 이 검색어로 수집을 마쳤습니다. 다시 수집하지 말고 다음 "
                       "단계로 넘어가세요 (필요하면 collect_info, 이미 했으면 최종 추천).",
        }

    # 1) 카카오 로컬 전량 — 지역 필터 기준이자, 수확 뒤 부족분을 채울 후보 풀
    base = kakao_all(query, category_code=category_code)
    base_n = len(base)                  # too_broad 판정은 카카오 base 기준 (수확분 제외)
    log(f"[collect_places] '{query}' 카카오 {base_n}개")

    # 2) 수확 — 웹 'Top' 리스트(diningcode 등)의 스니펫에서 상호명만 LLM으로 뽑는다.
    #    카카오 '맛집' 태그가 없는 로컬 노포는 카카오가 놓치지만 웹 스니펫엔 리스트로
    #    잡힌다. 한 스니펫에 매장이 촘촘히 나열돼 수확 밀도가 높다. 블로그는 노이즈가 커
    #    여기선 안 쓴다 — 블로그는 collect_info의 후기 요약에서만 쓴다.
    web_items, _ = await naver_search(f"{query} Top", kind="webkr", display=30)
    snippets = "\n".join(f"{w['title']} | {w['description']}" for w in web_items)
    try:
        names = json.loads(
            await llm_async(f"{prompts.EXTRACT_NAMES}\n{snippets}", json_mode=True)
        )
        names = [n.strip() for n in names if isinstance(n, str) and n.strip()]
    except (json.JSONDecodeError, TypeError):
        names = []

    # 카카오 base에 이미 있는 이름은 문자열 유사도로 먼저 걸러 카카오 역조회를 아낀다.
    # 표기가 조금 달라도('서울밥집' vs '서울밥집 신림점') 정규화 후 부분포함이면 같은 매장.
    base_norms = [_norm_name(p["name"]) for p in base]
    new_names = [
        n for n in names
        if len(nn := _norm_name(n)) >= 2
        and not any(nn in b or b in nn for b in base_norms)
    ]

    # base에 없는 '진짜 새 이름'만 카카오로 역조회해 실존·주소·좌표를 확보한다.
    # kakao_keyword가 동기(blocking)라 to_thread로 감싸 병렬화한다(직렬이면 이름당 순차).
    region_hint = (query.split() or [""])[0]

    async def _verify(n: str) -> dict | None:
        async with KAKAO_SEM:
            ps, _ = await asyncio.to_thread(kakao_keyword, f"{region_hint} {n}".strip())
            if not ps:
                ps, _ = await asyncio.to_thread(kakao_keyword, n)
        return ps[0] if ps else None

    seen = {p["id"] for p in base}
    harvest = []
    for p in await asyncio.gather(*[_verify(n) for n in new_names]):
        if p and p["id"] not in seen and in_same_region(p, base):
            p["source"] = "web_harvest"
            harvest.append(p)
            seen.add(p["id"])
    log(
        f"[collect_places] 수확 이름 {len(names)}개 → base중복 제외 후 {len(new_names)}개 검증 "
        f"→ 신규 확보 {len(harvest)}개"
    )

    # 프랜차이즈 제거 — 로컬 맛집 추천에 프랜차이즈는 노이즈(카페 체인 포함). base·수확 모두.
    harvest = [p for p in harvest if not _is_franchise(p)]
    base = [p for p in base if not _is_franchise(p)]

    # 도메인 필터 — base가 한 도메인에 몰린 구체 쿼리면 다른 도메인 수확은 노이즈다(음식점
    # 검색에 딸려온 카페 등) → 제거. 막연한 요청이면 도메인이 퍼져 있어 필터하지 않는다.
    # base는 쿼리 직접 결과라 건드리지 않고 수확분에만 적용한다.
    dominant = _dominant_domain(base)
    if dominant:
        before = len(harvest)
        harvest = [p for p in harvest if _domain(p["category"]) == dominant]
        log(f"[collect_places] 도메인 필터({dominant}) 수확 {before - len(harvest)}개 제거")

    # 3) 후보 조립 — 수확(로컬 노포)을 앞에 두고 카카오 base로 CAP까지 채운다. CAP을 넘는
    #    꼬리는 자르는데, 수확을 앞에 둬야 CAP에 밀려 잘리는 게 base 하위(노이즈)지 힘들게
    #    건진 노포가 아니게 된다(전 질의 검증 결과 잘린 것은 하위 노이즈뿐, 정답 손실 없음).
    n_harvest = len(harvest)
    fill = [p for p in base if p["id"] not in {h["id"] for h in harvest}]
    places = (harvest + fill)[: max(RERANK_KEEP * 3, n_harvest)]
    log(f"[collect_places] 웹-우선 조립: 수확 {n_harvest} + 카카오 {len(places) - n_harvest} = {len(places)}")

    file_path = dump("collect_places", places, _run_id(tool_context))

    # 파일 경로는 temp: 없이 저장한다. temp:는 invocation 스코프라 다음 턴에 사라지는데,
    # 멀티턴에서는 되묻고 답을 받은 뒤에도 이 경로가 필요하다.
    tool_context.state["last_places_file"] = file_path
    tool_context.state["last_query"] = query
    # 새 수집 라운드 시작 — 이전 라운드의 요약·읽음 기록을 비워 재호출 가드가 꼬이지 않게.
    tool_context.state["info_result"] = None
    tool_context.state["read_files"] = []

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

    내부적으로 후보가 많으면 HyDE 리랭크로 상위 RERANK_KEEP개만 골라 요약한다. 요청에
    맞는 '이상적 리뷰'를 생성해 임베딩하고, 각 매장의 블로그 스니펫과 max 유사도로 전체를
    한 번에 정렬한다(사전 필터 없이 — 중간 컷이 정답을 미리 버리지 않게). 본문은 스니펫과
    임베딩이 거의 같아 랭킹엔 스니펫만 쓴다. 남긴 매장만 본문을 크롤해 요약하되,
    협찬/체험단 리뷰는 제외한다.

    Returns:
        {"count", "no_info", "reranked_from", "url_file", "rerank_file", "summary_file"}
        count는 요약이 만들어진 장소 수, no_info는 후기를 못 찾은 장소 수.
        판단 근거는 summary_file에 있다. read_file로 읽어라.
    """
    places_file = tool_context.state.get("last_places_file")
    if not places_file:
        return {
            "status": "error",
            "message": "collect_places를 먼저 호출하세요. 이 툴을 다시 호출해도 같은 결과입니다.",
        }

    # 재호출 가드: 이번 수집 라운드에서 이미 요약했으면 재수집(크롤·요약)하지 않고 이전
    # 결과를 돌려준다. 무거운 재작업을 막고, summary_file을 읽어 최종 답으로 가도록 유도한다.
    cached = tool_context.state.get("info_result")
    if cached:
        return {
            **cached,
            "already_collected": True,
            "message": "이미 요약을 마쳤습니다. 재수집하지 말고 summary_file을 read_file로 "
                       "읽어(아직 안 읽었다면) 최종 추천을 완성하세요.",
        }

    places = load_json(places_file)
    run_id = _run_id(tool_context)
    t0 = time.perf_counter()  # 단계별 벽시계 계측용

    # --- 1) 매장별 블로그 스니펫 수집 (크롤링은 리랭크 뒤 상위만) ---
    # 도로명은 행정 표기라 실제 리뷰에서 쓰는 생활권 이름과 다르다("관천" vs "신림").
    # 그 매핑은 지역 상식이라 코드가 못 하므로 에이전트가 준 region_prefix를 쓴다.
    # 카카오(다음) 블로그를 쓴다 — 로컬 노포 후기가 네이버보다 풍부한 경우가 많다.
    # accuracy/recency는 매장마다 유리한 쪽이 달라 둘 다 던지고 합친다.
    queries = [f"{region_prefix} {p['name']}".strip() for p in places]
    raw = await asyncio.gather(
        *[
            daum_blog(q, display=10, sort=s)
            for q in queries
            for s in ("accuracy", "recency")
        ]
    )

    hits, reasons = {}, {}
    for i, p in enumerate(places):
        merged = raw[i * 2][0] + raw[i * 2 + 1][0]
        statuses = (raw[i * 2][1], raw[i * 2 + 1][1])
        seen, kept_hits = set(), []
        for b in merged:
            if b["link"] in seen:
                continue
            seen.add(b["link"])
            # 검색어가 통해도 무관한 글이 섞이므로 상호명 실제 등장 여부로 한 번 더 거른다
            if p["name"] in b["title"] or p["name"] in b["description"]:
                kept_hits.append(b)
        hits[p["name"]] = kept_hits[:RERANK_SNIPPETS]
        if not kept_hits:
            reasons[p["name"]] = (
                "empty" if all(s == "empty" for s in statuses) else "no_match_after_filter"
            )
    url_file = dump("info_urls", hits, run_id)
    if reasons:
        log(f"[collect_info] 후기 미확보 {len(reasons)}개 → {dict(Counter(reasons.values()))}")
    t_snip = time.perf_counter()

    # --- 2) HyDE 리랭크 — 요청을 서로 다른 3개 각도(전통노포 / 인기웨이팅 / 미디어검증)의
    # '기준 문서'로 생성해 임베딩한다. 단일 HyDE는 프레이밍 하나에 정답이 묻히지만(예:
    # '노포' 톤에 웨이팅 인기집이 밀림), 여러 각도의 max로 보면 어느 한 각도만 맞아도
    # 상위로 올라온다 — 11개 질문 검증에서 정답 top20 recall 77%→91%로 올랐다.
    try:
        facets = json.loads(
            await llm_async(prompts.HYDE_MULTI.format(user_query=user_query), json_mode=True)
        )
        facets = [f.strip() for f in facets if isinstance(f, str) and f.strip()]
    except (json.JSONDecodeError, TypeError):
        facets = []
    if not facets:  # facet 생성 실패 시 단일 HyDE로 폴백
        single = (await llm_async(prompts.HYDE_REVIEW.format(user_query=user_query))).strip()
        facets = [single or user_query]

    # 매장별로 스니펫을 한 문서로 concat해 매장당 1벡터만 임베딩한다. 스니펫을 개별 임베딩하면
    # (매장당 최대 RERANK_SNIPPETS개, 케이스당 수백 개) 임베딩 API 배치가 여러 번 도는데,
    # concat하면 매장 수(~50)만 임베딩해 케이스당 1배치로 끝난다. 매장 점수는 그 매장 문서
    # 벡터와 facet 3개의 max cos — 어느 한 각도만 맞아도 상위로. 후기 없는 매장은 -1(최하위).
    snip_texts, snip_owner = [], []
    for p in places:
        joined = " ".join(f"{b['title']} {b['description']}" for b in hits[p["name"]]).strip()
        if joined:
            snip_texts.append(joined)
            snip_owner.append(p["name"])
    vecs = await embed_texts(facets + snip_texts)
    facet_vecs, snip_vecs = vecs[: len(facets)], vecs[len(facets):]

    cos_score = {p["name"]: -1.0 for p in places}
    for name, v in zip(snip_owner, snip_vecs):
        cos_score[name] = max(_cos(fv, v) for fv in facet_vecs)

    # 순수 HyDE 코사인으로 정렬한다. 수집 순서(웹-우선 조립)를 RRF로 섞어봤지만, 그건
    # 유의미한 랭킹이 아니라 조립 순서일 뿐이라 노이즈에 가까워 뺐다.
    ranked = sorted(places, key=lambda p: -cos_score[p["name"]])
    kept = ranked[:RERANK_KEEP]
    rank_of = {p["name"]: r for r, p in enumerate(kept, 1)}

    # 콘솔엔 남긴 것만, 전체 순위는 rerank_file에.
    log(f"[collect_info] HyDE facet {len(facets)}개, 대표: {facets[0][:100]}")
    log(f"[collect_info] 리랭크 {len(places)}→{len(kept)}")
    for r, p in enumerate(kept, 1):
        log(f"   [{r:2}] cos={cos_score[p['name']]:+.3f} {p['name']}")
    ranking = [
        {"rank": r, "name": p["name"], "kept": rank_of.get(p["name"]) is not None,
         "cos": round(cos_score[p["name"]], 4), "source": p.get("source", "")}
        for r, p in enumerate(ranked, 1)
    ]
    rerank_file = dump(
        "info_rerank",
        {"user_query": user_query, "hypo_facets": facets,
         "keep": RERANK_KEEP, "ranking": ranking},
        run_id,
    )
    t_rank = time.perf_counter()

    # --- 3) 본문 크롤 + 요약 (남긴 매장만) — 협찬/체험단 리뷰는 제외 ---
    links, owner = [], []
    for p in kept:
        for b in hits[p["name"]][:INFO_PER_PLACE]:
            links.append(b["link"])
            owner.append(p["name"])

    scraped = await scrape_urls(links)
    bodies: dict[str, list[str]] = {p["name"]: [] for p in kept}
    n_sponsored = 0
    for name, s in zip(owner, scraped):
        if s["text"].startswith(FAIL_PREFIX):
            continue
        if s.get("sponsored"):
            n_sponsored += 1
            continue
        bodies[name].append(s["text"])
    log(
        f"[collect_info] 크롤 {len(scraped)}건 중 성공 {sum(len(v) for v in bodies.values())}건, "
        f"협찬 제외 {n_sponsored}건"
    )
    t_scrape = time.perf_counter()

    # 매장당 후기를 한 번에 넣어 통합 요약 (개별 요약하면 반복 빈도 정보가 사라진다)
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

    names = [p["name"] for p in kept if bodies[p["name"]]]
    results = await asyncio.gather(*[summarize_one(n, bodies[n]) for n in names])

    # 주소·URL·리랭크 점수를 요약에 병합 — 주소가 없으면 최종 답변에서 지어낸다
    meta = {p["name"]: p for p in places}
    summaries = {
        n: {
            **r,
            "review_count": len(bodies[n]),
            "rerank_rank": rank_of.get(n),
            "address": meta[n].get("address", ""),
            "category": meta[n].get("category", ""),
            "url": meta[n].get("url", ""),
        }
        for n, r in zip(names, results)
    }

    # 2차 스코어링 — 요약된 후보를 요청 적합도로 '전부' 점수화(LLM cross-encoder)해 최종 순위를
    # 코드가 확정한다. 리랭크(임베딩)는 top RERANK_KEEP까지 넉넉히 남기고, 여기서 강한 LLM 점수로
    # 재정렬한다. 에이전트에 맡기면 긴 목록을 게을리 평가(상위 몇 곳만)해 하위 정답이 평가조차 못
    # 받고 탈락하므로(재현성도 낮음) 코드가 전수 스코어링을 강제한다. score 순으로 정렬해 남긴다.
    if summaries:
        items = "\n".join(
            f"- {n}: {v.get('summary', '')} [강점] {' / '.join(v.get('strengths', []))}"
            for n, v in summaries.items()
        )
        try:
            scored = json.loads(
                await llm_async(
                    prompts.SELECT_SCORE.format(user_query=user_query, items=items),
                    json_mode=True,
                )
            )
        except (json.JSONDecodeError, TypeError):
            scored = {}
        for n, v in summaries.items():
            s = scored.get(n)
            raw = s.get("score", 0) if isinstance(s, dict) else (s or 0)
            v["score"] = int(raw) if isinstance(raw, (int, float)) else 0
            if isinstance(s, dict) and s.get("reason"):
                v["select_reason"] = s["reason"]
        summaries = dict(sorted(summaries.items(), key=lambda kv: -(kv[1].get("score") or 0)))
    summary_file = dump("info_summary", summaries, run_id)
    log(
        f"[collect_info] 단계별 벽시계: 스니펫수집 {t_snip - t0:.1f}s | "
        f"리랭크 {t_rank - t_snip:.1f}s | 크롤 {t_scrape - t_rank:.1f}s | "
        f"요약 {time.perf_counter() - t_scrape:.1f}s"
    )

    result = {
        "count": len(summaries),
        "no_info": len(kept) - len(summaries),
        "reranked_from": len(places),
        "url_file": url_file,
        "rerank_file": rerank_file,
        "summary_file": summary_file,
    }
    tool_context.state["info_result"] = result   # 재호출 가드용 캐시
    return result


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

def read_file(tool_context: ToolContext, file_path: str) -> dict:
    """툴이 만든 결과 파일을 읽는다. 최종 추천 매장을 확정하기 직전에 한 번만 호출한다.

    collect_info의 summary_file, collect_context의 context_file에 판단 근거가 들어있다.
    파일을 읽지 않고 답변을 지어내면 안 된다.

    같은 파일을 이미 읽었으면 내용을 다시 싣지 않는다. 근거는 앞서 읽은 그대로이니
    다시 읽지 말고 최종 답을 내라(반복해서 읽으면 컨텍스트만 불어난다).

    Args:
        file_path: 이전 툴이 반환한 file / summary_file / context_file 값.

    Returns:
        {"content": ...} 파일 내용. 이미 읽은 파일이면 {"already_read": True, "note": ...}.
    """
    path = Path(file_path)
    if not path.exists():
        return {"status": "error", "message": f"파일을 찾을 수 없습니다: {file_path}"}
    if not str(path.resolve()).startswith(str(TOOL_FILE_DIR.resolve())):
        return {"status": "error", "message": "허용되지 않은 경로입니다."}

    # 재호출 가드: 이미 읽은 파일이면 5k짜리 내용을 이력에 또 붓지 않는다. 27B가 최종
    # 판단 대신 파이프라인을 반복하면 read_file도 반복돼 컨텍스트가 32k를 넘겨 터진다.
    read_files = tool_context.state.get("read_files", [])
    if file_path in read_files:
        return {
            "already_read": True,
            "note": "이미 읽은 파일입니다. 근거는 앞서 읽은 그대로이니 다시 수집·조회하지 "
                    "말고 그 내용으로 최종 추천을 완성하세요.",
        }
    try:
        content = load_json(str(path))
    except json.JSONDecodeError:
        return {"status": "error", "message": "파일을 읽을 수 없습니다."}
    tool_context.state["read_files"] = read_files + [file_path]
    return {"content": content}


ALL_TOOLS = [get_today, collect_places, collect_info, collect_context, read_file]
