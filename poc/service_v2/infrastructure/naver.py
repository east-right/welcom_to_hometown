"""네이버 검색(blog/news) + 본문 취득.

두 쓰임:
  - 맛집/카페 경로: collect()로 blog 후기 검색+본문(enricher가 장소마다 병렬).
  - 축제/장소 경로: search(news|blog) + read()를 explorer 에이전트가 도구로 직접 호출.
"""
import asyncio
import html
import re

import httpx
from bs4 import BeautifulSoup

from ..config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, USER_AGENT  # noqa: F401

_SEARCH = "https://openapi.naver.com/v1/search/{kind}.json"

_BODY_SELECTORS = [
    ".se-main-container", "#postViewArea", "#dic_area", "#articleBodyContents",
    "article", "#article-view-content-div",
    ".article-body", ".news-body", ".article_body", "#content", ".content",
]
_DROP_TAGS = ["script", "style", "nav", "header", "footer", "aside",
              "form", "button", "iframe", "noscript"]


class NaverSearch:
    def __init__(self, client_id: str, client_secret: str) -> None:
        self._api = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        }

    async def search(self, query: str, kind: str = "blog", n: int = 5) -> list[dict]:
        """kind(blog|news) 검색 → [{title, desc, url}]. 실패하면 빈 목록."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.get(
                    _SEARCH.format(kind=kind), headers=self._api,
                    params={"query": query, "display": n, "sort": "sim"},
                )
                r.raise_for_status()
            except httpx.HTTPError:
                return []
        return [
            {"title": _strip(i.get("title", "")),
             "desc": _strip(i.get("description", "")),
             "url": i.get("link", "")}
            for i in r.json().get("items", [])
        ]

    async def read(self, urls: list[str]) -> list[dict]:
        """URL들의 본문을 병렬로 읽어 [{url, text}]. text가 '['로 시작하면 실패."""
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=10.0, follow_redirects=True
        ) as client:
            texts = await asyncio.gather(*(_afetch_one(client, u) for u in urls))
        return [{"url": u, "text": t} for u, t in zip(urls, texts)]

    async def collect(self, query: str, n: int = 3) -> list[dict]:
        """맛집/카페 enricher용: blog 검색 + 본문까지 한 번에 [{title, url, text}]."""
        hits = await self.search(query, "blog", n)
        bodies = await self.read([h["url"] for h in hits])
        return [{"title": h["title"], "url": h["url"], "text": b["text"]}
                for h, b in zip(hits, bodies)]


async def _afetch_one(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(_resolve_naver_url(url))
        r.raise_for_status()
    except httpx.HTTPError as e:
        return f"[열기 실패: {type(e).__name__}]"
    return _page_text(r.text)


# ── 저수준 파싱 세부 ──────────────────────────────────

def _page_text(raw_html: str) -> str:
    text = _extract_body(BeautifulSoup(raw_html, "html.parser"))
    return text[:1200] if len(text) >= 150 else "[본문 추출 실패]"


def _strip(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def _resolve_naver_url(url: str) -> str:
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
