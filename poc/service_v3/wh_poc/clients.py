"""Layer 1 — 외부 클라이언트.

각 API를 감싸는 얇은 래퍼. 툴로 노출하지 않는다.
"""

import asyncio
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


# ---------------------------------------------------------------- 카카오

@track("kakao")
def kakao_keyword(
    query: str, page: int = 1, category_code: str = ""
) -> tuple[list[dict], bool]:
    """카카오 로컬 키워드 검색 1페이지. (결과, is_end) 반환.

    카카오는 상호명·카테고리·태그 문자열 매칭이라 서술형 검색어에 약하다.
    페이지당 15개, 전체 노출 가능 문서는 45개가 상한.
    """
    params = {"query": query, "size": 15, "page": page}
    if category_code:
        params["category_group_code"] = category_code

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
            "source": "kakao_local",
        }
        for d in body["documents"]
    ]
    return places, body["meta"]["is_end"]


# ---------------------------------------------------------------- 네이버

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

    # 날짜 필드명이 종류별로 다르다: 블로그는 postdate, 뉴스는 pubDate, 웹문서는 없음
    return [
        {
            "title": strip_tags(i.get("title", "")),
            "description": strip_tags(i.get("description", "")),
            "link": i.get("link", ""),
            "date": i.get("postdate", "") or i.get("pubDate", ""),
        }
        for i in items
    ], "ok"


# ---------------------------------------------------------------- 크롤링

async def scrape_urls(urls: list[str], max_chars: int = 0) -> list[dict]:
    """URL 병렬 크롤링 → [{'url', 'text'}].

    실패해도 예외를 던지지 않고 FAIL_PREFIX로 시작하는 문자열을 담는다.
    실패를 별도 타입(None)이 아닌 같은 타입의 특수값으로 두면
    후속 처리에서 None 체크를 반복할 필요가 없다.
    max_chars가 0이면 본문 전체를 담는다.
    """

    @track("scrape")
    async def fetch_one(client: httpx.AsyncClient, url: str) -> str:
        # 네이버 블로그는 iframe 껍데기라 PostView로 우회해야 본문이 나온다
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

        # 셀렉터를 순회하되, 잡히긴 했는데 내용이 짧으면(광고 div 등) 다음으로 넘어간다
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


# ---------------------------------------------------------------- LLM

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
