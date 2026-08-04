"""검색·취득 게이트웨이. 유스케이스가 흐름을 지휘하고, 여기서 I/O를 병렬로 실행한다.

- NaverPlaces.search : 쿼리 여러 개를 병렬로 검색하고 URL 기준 중복 제거
- NaverPlaces.fetch  : 상위 후보 URL들의 본문을 병렬로 취득

검색 커버리지를 '코드'가 보장하므로, 모델의 tool-loop(멀티턴 검색) 능력에
의존하지 않는다 — 약한 모델에서도 동일한 폭으로 검색된다.
"""
import asyncio
import html
import re
from itertools import chain

import httpx
from bs4 import BeautifulSoup

from ..config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, USER_AGENT
from ..domain.models import SearchQuery

# 본문 컨테이너 후보 (우선순위대로)
_BODY_SELECTORS = [
    ".se-main-container",        # 네이버 블로그 신에디터
    "#postViewArea",             # 네이버 블로그 구에디터
    "#dic_area",                 # 네이버 뉴스 신형
    "#articleBodyContents",      # 네이버 뉴스 구형
    "article",                   # 표준 언론사
    "#article-view-content-div", # 다수 지역 언론
    ".article-body", ".news-body", ".article_body",
    "#content", ".content",
]

_DROP_TAGS = ["script", "style", "nav", "header", "footer", "aside",
              "form", "button", "iframe", "noscript"]


class NaverPlaces:
    """검색+본문취득을 묶은 게이트웨이. composition root에서 유스케이스에 주입된다.

    log은 진행 상황을 한 줄로 남기는 콜러블(없으면 조용히 동작). 검색·취득이
    '몇 쿼리 → 몇 후보 → 몇 본문'인지 파이프라인 로그에 보이게 한다.
    """

    def __init__(self, log=None) -> None:
        self._log = log or (lambda _line: None)

    async def search(self, queries: list[SearchQuery]) -> list[dict]:
        """쿼리들을 병렬로 검색해 URL 중복을 제거한 후보 목록을 돌려준다."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            batches = await asyncio.gather(*(_search_one(client, q) for q in queries))
        hits = _dedupe_by_url(chain.from_iterable(batches))
        self._log(f"🔎 병렬 검색 {len(queries)}쿼리 → 후보 {len(hits)}건(중복 제거)")
        return hits

    async def fetch(self, urls: list[str]) -> list[dict]:
        """상위 후보 URL들의 본문을 병렬로 취득한다. 각 항목은 {url, text}."""
        bodies = await fetch_pages(urls)
        self._log(f"📄 본문 병렬 취득 {len(bodies)}건")
        return bodies


async def _search_one(client: httpx.AsyncClient, q: SearchQuery) -> list[dict]:
    """검색어 하나를 비동기로 던진다(search의 워커). 실패하면 빈 목록."""
    try:
        response = await client.get(
            f"https://openapi.naver.com/v1/search/{q.kind}.json",
            headers={
                "X-Naver-Client-Id": NAVER_CLIENT_ID,
                "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
            },
            params={"query": q.query, "display": 6, "sort": "sim"},
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return []
    return [
        {
            "title": _strip_tags(item.get("title", "")),
            "desc": _strip_tags(item.get("description", "")),
            "url": item.get("link", ""),
            "kind": q.kind,
        }
        for item in response.json().get("items", [])
    ]


def _dedupe_by_url(hits) -> list[dict]:
    """URL 기준 중복 제거(먼저 나온 것 유지). URL 없는 항목은 버린다."""
    seen: set[str] = set()
    out: list[dict] = []
    for hit in hits:
        url = hit.get("url", "")
        if url and url not in seen:
            seen.add(url)
            out.append(hit)
    return out


async def fetch_pages(urls: list[str]) -> list:
    """여러 웹 페이지의 본문을 동시에(병렬로) 가져온다. 각 항목은 {url, text}."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=10.0, follow_redirects=True
    ) as client:
        texts = await asyncio.gather(*(_afetch_one(client, u) for u in urls))
    return [{"url": u, "text": t} for u, t in zip(urls, texts)]


async def _afetch_one(client: httpx.AsyncClient, url: str) -> str:
    """URL 하나를 비동기로 취득 (fetch_pages의 워커)."""
    try:
        response = await client.get(_resolve_naver_url(url))
        response.raise_for_status()
    except httpx.HTTPError as e:
        return _open_failed(e)
    return _page_text(response.text)


# ── 이하 저수준 파싱 세부 ──────────────────────────────

def _page_text(raw_html: str) -> str:
    """응답 HTML → 본문 텍스트. fetch_page/fetch_pages 공통 후처리."""
    text = _extract_body(BeautifulSoup(raw_html, "html.parser"))
    if len(text) < 150:
        return "[본문 추출 실패] 이 URL은 근거로 쓰지 마세요."
    return text[:1000]   # 컨텍스트 폭발 방지 (여러 본문이 coordinator 문맥에 쌓임)


def _open_failed(error: httpx.HTTPError) -> str:
    return f"[열기 실패: {type(error).__name__}] 이 URL은 건너뛰세요."

def _strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def _resolve_naver_url(url: str) -> str:
    """네이버 블로그 iframe → PostView 실주소로 변환."""
    m = re.search(r"blog\.naver\.com/([^/?]+)/(\d+)", url)
    if m:
        return f"https://blog.naver.com/PostView.naver?blogId={m.group(1)}&logNo={m.group(2)}"
    return url


def _extract_body(soup: BeautifulSoup) -> str:
    """본문 컨테이너만 골라 텍스트를 뽑는다. 메뉴/헤더/광고 제거."""
    for tag in soup(_DROP_TAGS):
        tag.decompose()

    for selector in _BODY_SELECTORS:
        node = soup.select_one(selector)
        if node:
            text = _collapse(node.get_text(separator=" ", strip=True))
            if len(text) > 150:
                return text

    return _collapse((soup.body or soup).get_text(separator=" ", strip=True))


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
