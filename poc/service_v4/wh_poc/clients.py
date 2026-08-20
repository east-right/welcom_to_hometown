"""Layer 1 — 외부 클라이언트.

각 API를 감싸는 얇은 래퍼. 툴로 노출하지 않는다.

v2에서 추가된 것:
    kakao_all           카카오 45개 전량 수집
    diningcode_names    네이버 웹검색 스니펫에서 다이닝코드 매장명 수확
    verify_names        매장명 문자열을 카카오로 역조회해 실존·좌표 확보
    rrf_merge           여러 검색 결과를 Reciprocal Rank Fusion으로 병합
    fanout_queries      업종어 확장 쿼리 생성 (LLM)
    in_same_region      구(區) 단위 지역 일치 판정
"""

import asyncio
import json
import os
import re

import httpx
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

from .core import (
    FAIL_PREFIX,
    LLM_SEM,
    NAVER_SEM,
    SCRAPE_SEM,
    strip_tags,
    track,
)

UTIL_MODEL = os.getenv("WH_UTIL_MODEL", "gemini-3.1-flash-lite")

_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

_BODY_SELECTORS = [
    "div.se-main-container",  # 네이버 스마트에디터 ONE
    "#postViewArea",          # 네이버 구버전 에디터
    "div#post-view",
    "article",                # 일반 웹페이지
    "main",
]


# ---------------------------------------------------------------- LLM
# (fanout_queries가 아래에서 참조하므로 파일 앞쪽에 배치)

def llm(prompt: str, *, json_mode: bool = False, temperature: float = 0.0) -> str:
    """유틸 LLM 동기 호출. 단발 처리용."""
    config = types.GenerateContentConfig(temperature=temperature)
    if json_mode:
        config.response_mime_type = "application/json"
    resp = _client.models.generate_content(
        model=UTIL_MODEL, contents=prompt, config=config
    )
    return resp.text


@track("llm")
async def llm_async(
    prompt: str, *, json_mode: bool = False, temperature: float = 0.0
) -> str:
    """유틸 LLM 비동기 호출. 병렬 처리용."""
    config = types.GenerateContentConfig(temperature=temperature)
    if json_mode:
        config.response_mime_type = "application/json"
    async with LLM_SEM:
        resp = await _client.aio.models.generate_content(
            model=UTIL_MODEL, contents=prompt, config=config
        )
    return resp.text


# ---------------------------------------------------------------- 카카오

@track("kakao")
def kakao_keyword(
    query: str,
    page: int = 1,
    category_code: str = "",
    x: str | float | None = None,
    y: str | float | None = None,
    radius: int | None = None,
) -> tuple[list[dict], bool]:
    """카카오 로컬 키워드 검색 1페이지. (결과, is_end) 반환.

    카카오는 상호명·카테고리·태그 문자열 매칭이라 서술형 검색어에 약하다.
    페이지당 15개, 전체 노출 가능 문서는 45개가 상한.

    x, y를 주면 그 좌표 기준으로 검색하고 응답에 distance(m)가 붙는다.
    """
    params = {"query": query, "size": 15, "page": page}
    if category_code:
        # 카카오는 단일 코드만 받는다. 여러 개가 오면 첫 번째만.
        params["category_group_code"] = category_code.split(",")[0].strip()
    if x is not None and y is not None:
        params["x"] = str(x)
        params["y"] = str(y)
        if radius:
            params["radius"] = min(int(radius), 20000)

    resp = requests.get(
        "https://dapi.kakao.com/v2/local/search/keyword.json",
        headers={"Authorization": f"KakaoAK {os.environ['KAKAO_REST_API_KEY']}"},
        params=params,
        timeout=5,
    )
    resp.raise_for_status()
    body = resp.json()

    places = [
        {
            "id": d["id"],
            "name": d["place_name"],
            "category": d["category_name"],
            "address": d["road_address_name"] or d["address_name"],
            "phone": d.get("phone", ""),
            "url": d["place_url"],
            "x": float(d["x"]),
            "y": float(d["y"]),
            "distance": int(d["distance"]) if d.get("distance") else None,
            "source": "kakao_local",
        }
        for d in body["documents"]
    ]
    return places, body["meta"]["is_end"]


def kakao_all(
    query: str, category_code: str = "", x=None, y=None, radius=None
) -> list[dict]:
    """카카오가 노출 가능한 만큼(최대 45개) 전부 받는다.
    페이지당 15개 × 최대 3페이지. is_end면 조기 종료."""
    places, seen = [], set()
    for page in (1, 2, 3):
        batch, is_end = kakao_keyword(
            query, page=page, category_code=category_code, x=x, y=y, radius=radius
        )
        for p in batch:
            if p["id"] not in seen:
                seen.add(p["id"])
                places.append(p)
        if is_end:
            break
    return places


def verify_names(names: list[str], region_hint: str = "") -> list[dict]:
    """상호명 문자열 목록을 카카오로 역조회해 실존·주소·좌표를 확보한다.
    이름당 첫 결과만 채택. 검증 못 한 이름은 조용히 버려진다."""
    out, seen = [], set()
    for n in names:
        ps, _ = kakao_keyword(f"{region_hint} {n}".strip())
        if not ps:
            ps, _ = kakao_keyword(n)
        for p in ps[:1]:
            if p["id"] not in seen:
                seen.add(p["id"])
                out.append(p)
    return out


# ---------------------------------------------------------------- RRF

def rrf_merge(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """여러 검색 결과를 Reciprocal Rank Fusion으로 병합.

    RRF(d) = Σ 1/(k + rank). k=60은 업계 표준값(Cormack et al. 2009).
    여러 쿼리·소스에서 반복 등장 = 관점 간 합의이므로 상위로 올라간다.
    단순 등장 횟수(마케팅 물량)와 달리, 서로 다른 업종어·소스로 검색했을 때도
    걸린다는 뜻이라 실제 관련성 신호에 가깝다.
    """
    scores, store = {}, {}
    for places in result_lists:
        for rank, p in enumerate(places, 1):
            scores[p["id"]] = scores.get(p["id"], 0.0) + 1.0 / (k + rank)
            store.setdefault(p["id"], p)

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    out = []
    for pid, s in ranked:
        p = dict(store[pid])
        p["rrf"] = round(s, 5)
        p["hit_lists"] = sum(1 for lst in result_lists if any(q["id"] == pid for q in lst))
        out.append(p)
    return out


# ---------------------------------------------------------------- 다이닝코드 수확

def diningcode_names(query: str, display: int = 30) -> dict:
    """네이버 웹문서 검색으로 다이닝코드 스니펫을 긁어 매장명을 뽑는다.

    다이닝코드 리스트 페이지('X 맛집 Top45')는 스니펫에 상위 매장이
    '매장명 (카테고리, ★평점)' 형태로 나열된다. 페이지 자체는 봇 차단이
    있지만 네이버 검색 스니펫으로는 우회해서 읽을 수 있다.

    쿼리에 'Top'을 넣어야 리스트 페이지가 걸린다. 개별 매장 페이지가 아니라
    리스트 페이지를 노려야 한 번에 여러 매장이 나온다.

    카카오 '맛집' 태그가 없어 kakao_all이 놓치는 로컬 매장을 보강하는 용도.
    포괄어 쿼리("X 맛집")에서 특히 값어치가 크다 — 업종을 특정하지 못하는
    쿼리는 fanout_queries가 확장을 포기하므로, 그 빈자리를 다이닝코드가 채운다.

    반환: {매장명: {"category":..., "rating":..., "src": 스니펫제목}}
    """
    resp = requests.get(
        "https://openapi.naver.com/v1/search/webkr",
        headers={
            "X-Naver-Client-Id": os.environ["NAVER_CLIENT_ID"],
            "X-Naver-Client-Secret": os.environ["NAVER_CLIENT_SECRET"],
        },
        params={"query": query, "display": display},
        timeout=5,
    )
    resp.raise_for_status()

    def clean(t):
        return re.sub(r"<[^>]+>", "", t or "").strip()

    out = {}
    for item in resp.json()["items"]:
        if "diningcode.com" not in item.get("link", ""):
            continue
        title = clean(item["title"])
        desc = clean(item["description"])

        # 패턴 1: "매장명 (카테고리, ★4.3)"  ← Top 리스트 페이지
        for m in re.finditer(r"([^,()]+?)\s*\(([^,()]+?),\s*★([\d.]+)\)", desc):
            name = re.sub(r"^.*?(맛집|랭킹|Top\d*|등)\s+", "", m.group(1).strip()).strip()
            if 1 < len(name) <= 30:
                out.setdefault(name, {
                    "category": m.group(2).strip(),
                    "rating": float(m.group(3)),
                    "src": title[:40],
                })

        # 패턴 2: "가게A 마산 ; 가게B 마산 ; ..."  ← 세미콜론 나열형
        if "★" not in desc and ";" in desc:
            for chunk in desc.split(";"):
                name = re.sub(r"\s+(마산|창원|서울|부산)\s*$", "", chunk.strip())
                if 1 < len(name) <= 30 and not name.startswith('"'):
                    out.setdefault(name, {"category": "", "rating": None, "src": title[:40]})

    return out


def in_same_region(place: dict, base: list[dict]) -> bool:
    """place가 base(1차 카카오 결과)와 같은 구(區)에 있는지.

    다이닝코드 수확에는 다른 지역이 섞인다 — '가포' 검색에 청주 흥덕구
    '가포산로' 리스트가 딸려오는 식. 도로명 접두로 걸러보면 지역마다
    도로명 구성이 달라('가포'는 가포*로 통일, '신림역'은 신림로·남부순환로·
    관악로로 분산) 일관되게 동작하지 않는다. 구 단위만 보는 이유다.
    같은 구 안의 먼 매장은 RRF 하위로 밀리므로 최종 선별에서 걸러진다.
    """
    parts = place["address"].split()
    if len(parts) < 3:
        return False
    base_gu = {b["address"].split()[2] for b in base if len(b["address"].split()) > 2}
    return parts[2] in base_gu


# ---------------------------------------------------------------- 쿼리 확장

_FANOUT_PROMPT = """\
사용자가 찾는 것: "{query}"

카카오맵은 같은 가게를 사람들이 부르는 이름과 다르게 분류한다.
(예: 화덕피자집이 '피자'가 아니라 '이탈리안'이나 '파스타'로 등록된다)

이 요청에 **구체적인 업종·메뉴어**가 있으면, 카카오맵이 쓸 법한 다른 분류명으로
검색어 3~4개를 만들어라. 지역명은 그대로 두고 업종어만 바꾼다.
    "합성동 피자"   → ["합성동 이탈리안", "합성동 파스타", "합성동 화덕피자"]
    "신림역 고기집" → ["신림역 삼겹살", "신림역 갈비", "신림역 돼지고기"]
    "신림역 횟집"   → ["신림역 일식", "신림역 해산물"]

요청이 '맛집', '먹을 데', '식당', '추천' 같은 포괄어뿐이고 구체적 업종어가
없으면 반드시 빈 배열 []을 반환하라. 그런 경우 업종을 추측하면 그 동네에
실제로 없는 업종을 검색하게 되어 결과가 더 나빠진다.

결과는 문자열 배열 JSON으로만 답하라.
"""


def fanout_queries(query: str) -> list[str]:
    """업종어 확장 쿼리를 만든다. 원 쿼리는 항상 포함.
    포괄어라 확장할 수 없다고 LLM이 판단하면 원 쿼리만 반환한다."""
    try:
        raw = llm(_FANOUT_PROMPT.format(query=query), json_mode=True)
        qs = json.loads(raw)
        qs = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
    except (json.JSONDecodeError, TypeError):
        qs = []

    out, seen = [], set()
    for q in [query] + qs[:4]:
        key = re.sub(r"\s+", "", q)
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


# ---------------------------------------------------------------- 크롤링

async def scrape_urls(urls: list[str], max_chars: int = 0) -> list[dict]:
    """URL 병렬 크롤링 → [{'url', 'text'}].

    실패해도 예외를 던지지 않고 FAIL_PREFIX로 시작하는 문자열을 담는다.
    max_chars가 0이면 본문 전체를 담는다.
    """

    @track("scrape")
    async def fetch_one(client: httpx.AsyncClient, url: str) -> str:
        m = re.search(r"blog\.naver\.com/([^/?]+)/(\d+)", url)
        target = (
            f"https://blog.naver.com/PostView.naver?blogId={m.group(1)}&logNo={m.group(2)}"
            if m
            else url
        )

        async with SCRAPE_SEM:
            try:
                resp = await client.get(target)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                return f"{FAIL_PREFIX}열기 실패: {type(e).__name__}"

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = ""
        for selector in _BODY_SELECTORS:
            node = soup.select_one(selector)
            if node:
                candidate = re.sub(
                    r"\s+", " ", node.get_text(separator=" ", strip=True)
                ).strip()
                if len(candidate) > 150:
                    text = candidate
                    break
        if not text:
            text = re.sub(
                r"\s+", " ", (soup.body or soup).get_text(separator=" ", strip=True)
            ).strip()

        if len(text) < 150:
            return f"{FAIL_PREFIX}본문 추출 실패"
        return text[:max_chars] if max_chars else text

    async with httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT}, timeout=10.0, follow_redirects=True
    ) as client:
        texts = await asyncio.gather(*(fetch_one(client, u) for u in urls))

    return [{"url": u, "text": t} for u, t in zip(urls, texts)]


# ---------------------------------------------------------------- 네이버 검색

@track("naver")
async def naver_search(
    query: str, kind: str = "blog", display: int = 10, sort: str = "sim"
) -> tuple[list[dict], str]:
    """네이버 검색 API 비동기 호출.

    blog / news / webkr 모두 응답 구조가 같아 하나로 처리한다.
    kind: 'blog'(블로그) / 'news'(뉴스) / 'webkr'(웹문서)
    sort: 'sim'(정확도) / 'date'(최신). webkr은 sort를 무시한다.

    반환: (결과 리스트, 상태). 상태는 'ok' / 'empty' / '429'
    """
    async with NAVER_SEM:
        for attempt in range(3):
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"https://openapi.naver.com/v1/search/{kind}",
                    headers={
                        "X-Naver-Client-Id": os.environ["NAVER_CLIENT_ID"],
                        "X-Naver-Client-Secret": os.environ["NAVER_CLIENT_SECRET"],
                    },
                    params={"query": query, "display": display, "sort": sort},
                )
            if resp.status_code == 429:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            break
        else:
            return [], "429"

    items = resp.json().get("items", [])
    if not items:
        return [], "empty"

    return [
        {
            "title": strip_tags(i.get("title", "")),
            "description": strip_tags(i.get("description", "")),
            "link": i.get("link", ""),
            "date": i.get("postdate", "") or i.get("pubDate", ""),
        }
        for i in items
    ], "ok"
