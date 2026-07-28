"""검색·취득 함수툴. 서브 에이전트가 호출한다.

- naver_search : 발견 (URL + 스니펫)
- fetch_page   : 취득 (본문 텍스트만 추출)
"""
import html
import re

import httpx
from bs4 import BeautifulSoup

from config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, USER_AGENT


def _strip(t: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", t or "")).strip()


def naver_search(query: str, kind: str = "blog") -> list:
    """네이버에서 정보를 검색해 제목/요약/URL 목록을 돌려준다.

    Args:
        query: 검색어. 지역명 포함. 예 '창원 국화축제 2026'
        kind: 'blog'(후기), 'news'(공지·개최일정), 'local'(장소·주소) 중 하나
    """
    try:
        r = httpx.get(
            f"https://openapi.naver.com/v1/search/{kind}.json",
            headers={
                "X-Naver-Client-Id": NAVER_CLIENT_ID,
                "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
            },
            params={"query": query, "display": 8, "sort": "sim"},
            timeout=5.0,
        )
        r.raise_for_status()
    except httpx.HTTPError as e:
        return [{"error": f"검색 실패({type(e).__name__}). 다른 검색어로 다시 시도하세요."}]

    return [
        {
            "title": _strip(it.get("title", "")),
            "desc": _strip(it.get("description", "")),
            "url": it.get("link", ""),
        }
        for it in r.json().get("items", [])
    ]


def _naver_real_url(url: str) -> str:
    """네이버 블로그 iframe → PostView 실주소로 변환."""
    m = re.search(r"blog\.naver\.com/([^/?]+)/(\d+)", url)
    if m:
        return f"https://blog.naver.com/PostView.naver?blogId={m.group(1)}&logNo={m.group(2)}"
    return url


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


def _extract_body(soup: BeautifulSoup) -> str:
    """본문 컨테이너만 골라 텍스트를 뽑는다. 메뉴/헤더/광고 제거."""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                     "form", "button", "iframe", "noscript"]):
        tag.decompose()

    for sel in _BODY_SELECTORS:
        node = soup.select_one(sel)
        if node:
            text = node.get_text(separator=" ", strip=True)
            if len(text) > 150:
                return text

    body = soup.body or soup
    return body.get_text(separator=" ", strip=True)


def fetch_page(url: str) -> str:
    """웹 페이지의 본문 텍스트를 가져온다.

    naver_search로 찾은 URL 중, 운영시간·기간·휴무일 같은 상세 정보가
    필요한 2~3개만 골라 이 툴로 본문을 읽어라.

    Args:
        url: 읽을 페이지 전체 URL
    """
    try:
        r = httpx.get(
            _naver_real_url(url),
            headers={"User-Agent": USER_AGENT},
            timeout=10.0,
            follow_redirects=True,
        )
        r.raise_for_status()
    except httpx.HTTPError as e:
        return f"[열기 실패: {type(e).__name__}] 이 URL은 건너뛰세요."

    text = _extract_body(BeautifulSoup(r.text, "html.parser"))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 150:
        return "[본문 추출 실패] 이 URL은 근거로 쓰지 마세요."
    return text[:2000]
