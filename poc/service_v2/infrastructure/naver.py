"""네이버 blog 후기 검색 + 본문 취득.

카카오가 못 주는 '살'(영업시간·기간·후기·광고여부)의 소스.
collect() 하나가 한 장소의 검색→본문까지 async로 처리하므로, 여러 장소를
asyncio.gather로 동시에 요약 파이프라인에 태울 수 있다.
"""
import asyncio
import html
import re

import httpx
from bs4 import BeautifulSoup

from ..config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, USER_AGENT  # noqa: F401 (호환)

_SEARCH_URL = "https://openapi.naver.com/v1/search/blog.json"

_BODY_SELECTORS = [
    ".se-main-container", "#postViewArea", "#dic_area", "#articleBodyContents",
    "article", "#article-view-content-div",
    ".article-body", ".news-body", ".article_body", "#content", ".content",
]
_DROP_TAGS = ["script", "style", "nav", "header", "footer", "aside",
              "form", "button", "iframe", "noscript"]


class NaverBlog:
    def __init__(self, client_id: str, client_secret: str) -> None:
        self._api_headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        }

    async def collect(self, query: str, n: int = 3) -> list[dict]:
        """검색 → 상위 n개 후기 본문 병렬 취득까지. [{title, url, text}] 목록."""
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            hits = await self._search(client, query, n)
            bodies = await asyncio.gather(*(_afetch_one(client, h["url"]) for h in hits))
        return [{**hit, "text": body} for hit, body in zip(hits, bodies)]

    async def _search(self, client: httpx.AsyncClient, query: str, n: int) -> list[dict]:
        try:
            response = await client.get(
                _SEARCH_URL, headers=self._api_headers,
                params={"query": query, "display": n, "sort": "sim"},
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        return [
            {"title": _strip_tags(item.get("title", "")), "url": item.get("link", "")}
            for item in response.json().get("items", [])
        ]


async def _afetch_one(client: httpx.AsyncClient, url: str) -> str:
    try:
        response = await client.get(_resolve_naver_url(url), headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
    except httpx.HTTPError as error:
        return f"[열기 실패: {type(error).__name__}]"
    return _page_text(response.text)


# ── 저수준 파싱 세부 ──────────────────────────────────

def _page_text(raw_html: str) -> str:
    text = _extract_body(BeautifulSoup(raw_html, "html.parser"))
    return text[:1200] if len(text) >= 150 else "[본문 추출 실패]"


def _strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def _resolve_naver_url(url: str) -> str:
    """네이버 블로그 iframe → PostView 실주소로 변환."""
    m = re.search(r"blog\.naver\.com/([^/?]+)/(\d+)", url)
    if m:
        return f"https://blog.naver.com/PostView.naver?blogId={m.group(1)}&logNo={m.group(2)}"
    return url


def _extract_body(soup: BeautifulSoup) -> str:
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
