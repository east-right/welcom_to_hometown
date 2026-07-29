"""검색·취득 함수툴. 코디네이터가 호출한다.

- naver_search : 발견 (URL + 스니펫)
- fetch_page   : 취득 1건 (본문 텍스트만 추출)
- fetch_pages  : 취득 N건을 동시(병렬)로 — 순차보다 빠르다

함수 docstring은 ADK가 LLM에게 넘기는 툴 설명으로 쓰이므로 유지한다.
"""
import asyncio
import html
import re

import httpx
from bs4 import BeautifulSoup

from ..config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, USER_AGENT

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


def naver_search(query: str, kind: str = "blog") -> list:
    """네이버에서 정보를 검색해 제목/요약/URL 목록을 돌려준다.

    Args:
        query: 검색어. 지역명 포함. 예 '창원 국화축제 2026'
        kind: 'blog'(후기), 'news'(공지·개최일정), 'local'(장소·주소) 중 하나
    """
    try:
        response = httpx.get(
            f"https://openapi.naver.com/v1/search/{kind}.json",
            headers={
                "X-Naver-Client-Id": NAVER_CLIENT_ID,
                "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
            },
            params={"query": query, "display": 8, "sort": "sim"},
            timeout=5.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        return [{"error": f"검색 실패({type(e).__name__}). 다른 검색어로 다시 시도하세요."}]

    return [
        {
            "title": _strip_tags(item.get("title", "")),
            "desc": _strip_tags(item.get("description", "")),
            "url": item.get("link", ""),
        }
        for item in response.json().get("items", [])
    ]


def fetch_page(url: str) -> str:
    """웹 페이지의 본문 텍스트를 가져온다.

    naver_search로 찾은 URL 중, 운영시간·기간·휴무일 같은 상세 정보가
    필요한 2~3개만 골라 이 툴로 본문을 읽어라.

    Args:
        url: 읽을 페이지 전체 URL
    """
    try:
        response = httpx.get(
            _resolve_naver_url(url),
            headers={"User-Agent": USER_AGENT},
            timeout=10.0,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        return _open_failed(e)
    return _page_text(response.text)


async def fetch_pages(urls: list[str]) -> list:
    """여러 웹 페이지의 본문을 동시에(병렬로) 가져온다.

    본문을 확인할 URL 2~3개를 한 번에 넘겨라. 하나씩 순차로 읽는 것보다 빠르다.
    각 항목은 {url, text} 형태로 돌아온다.

    Args:
        urls: 읽을 페이지 전체 URL 목록 (2~3개 권장)
    """
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
    return text[:2000]


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
