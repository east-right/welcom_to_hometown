"""네이버 blog 후기 검색 + 본문 병렬 취득.

카카오가 못 주는 '살'(영업시간·기간·후기·광고여부)의 소스.
검색으로 후기 URL을 찾고, 본문을 병렬로 긁어 요약 에이전트에 넘긴다.
"""
import asyncio
import html
import re

import httpx
from bs4 import BeautifulSoup

from ..config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, USER_AGENT

# 본문 컨테이너 후보 (우선순위대로)
_BODY_SELECTORS = [
    ".se-main-container", "#postViewArea", "#dic_area", "#articleBodyContents",
    "article", "#article-view-content-div",
    ".article-body", ".news-body", ".article_body", "#content", ".content",
]
_DROP_TAGS = ["script", "style", "nav", "header", "footer", "aside",
              "form", "button", "iframe", "noscript"]


class NaverBlog:
    def __init__(self, client_id: str, client_secret: str) -> None:
        self._headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        }

    def search(self, query: str, n: int = 3) -> list[dict]:
        """blog 후기를 검색해 [{title, url}] 목록을 돌려준다. 실패하면 빈 목록."""
        try:
            response = httpx.get(
                "https://openapi.naver.com/v1/search/blog.json",
                headers=self._headers,
                params={"query": query, "display": n, "sort": "sim"},
                timeout=5.0,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        return [
            {"title": _strip_tags(item.get("title", "")), "url": item.get("link", "")}
            for item in response.json().get("items", [])
        ]

    async def fetch(self, urls: list[str]) -> list[str]:
        """여러 URL의 본문을 동시에(병렬로) 긁어 텍스트 목록으로 돌려준다."""
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=10.0, follow_redirects=True
        ) as client:
            return await asyncio.gather(*(_afetch_one(client, u) for u in urls))


async def _afetch_one(client: httpx.AsyncClient, url: str) -> str:
    try:
        response = await client.get(_resolve_naver_url(url))
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
