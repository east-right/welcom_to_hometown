"""Layer 1 — 외부 클라이언트.

각 API를 감싸는 얇은 래퍼. 툴로 노출하지 않는다.
  llm_async     유틸 LLM 비동기 호출
  embed_texts   텍스트 임베딩 (리랭크용)
  kakao_keyword / kakao_all   카카오 로컬 검색
  verify_names  매장명을 카카오로 역조회해 실존·좌표 확보 (웹수확 검증)
  in_same_region  구(區) 단위 지역 일치 판정
  naver_search  네이버 블로그/뉴스/웹문서 검색
  scrape_urls   본문 병렬 크롤 (+ 협찬 판정)
"""

import asyncio
import itertools
import os
import re
from urllib.parse import urlparse

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

# LLM 백엔드 스왑 — 기본은 호스티드 Gemini. WH_LLM_BACKEND=local이면 유틸 LLM과
# 에이전트 LLM만 vLLM(OpenAI 호환)으로 라우팅한다. 임베딩은 항상 Gemini(아래 _client).
# env를 안 주면 현행(gemini)과 100% 동일하게 동작한다.
LLM_BACKEND = os.getenv("WH_LLM_BACKEND", "gemini").lower()
LOCAL_MODEL = os.getenv("WH_LOCAL_MODEL", "")           # 예: openai/Qwen/Qwen3.8-27B-FP8
LOCAL_API_BASE = os.getenv("WH_LOCAL_API_BASE", "")     # 예: https://<pod>-8000.proxy.runpod.net/v1
LOCAL_API_KEY = os.getenv("WH_LOCAL_API_KEY", "EMPTY")  # vLLM은 보통 키 검사를 안 한다

# 임베딩은 백엔드와 무관하게 Gemini를 쓰므로 GOOGLE_API_KEY는 항상 필요하다.
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


@track("llm")
async def llm_async(
    prompt: str, *, json_mode: bool = False, temperature: float = 0.0
) -> str:
    """유틸 LLM 비동기 호출. 병렬 처리용. 백엔드는 WH_LLM_BACKEND로 스왑한다."""
    async with LLM_SEM:
        if LLM_BACKEND == "local":
            return await _llm_local(prompt, json_mode=json_mode, temperature=temperature)
        return await _llm_gemini(prompt, json_mode=json_mode, temperature=temperature)


async def _llm_gemini(prompt: str, *, json_mode: bool, temperature: float) -> str:
    config = types.GenerateContentConfig(temperature=temperature)
    if json_mode:
        config.response_mime_type = "application/json"
    resp = await _client.aio.models.generate_content(
        model=UTIL_MODEL, contents=prompt, config=config
    )
    return resp.text


async def _llm_local(prompt: str, *, json_mode: bool, temperature: float) -> str:
    """vLLM OpenAI 호환 엔드포인트 호출. 에이전트의 LiteLlm과 같은 litellm을 쓴다
    (새 의존성 없음). json_mode는 OpenAI식 response_format으로 매핑한다.

    Qwen3 같은 추론 모델은 기본으로 답 앞에 <think> 사고과정을 붙인다. 유틸 LLM은
    HyDE 문서·요약처럼 '결과 텍스트'만 필요하므로 chat_template_kwargs로 thinking을 꺼
    Gemini(비추론)와 같은 형태의 출력을 받는다. thinking 미지원 모델은 이 인자를 무시한다."""
    import litellm

    kwargs = {
        "model": LOCAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "api_base": LOCAL_API_BASE or None,
        "api_key": LOCAL_API_KEY or None,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = await litellm.acompletion(**kwargs)
    return resp.choices[0].message.content


# ---------------------------------------------------------------- 임베딩

EMBED_MODEL = os.getenv("WH_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768  # 3072 기본에서 축소 — 벡터가 가벼워지고 코사인도 빨라진다


@track("embed")
async def embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트 목록을 임베딩 벡터로. 100개씩 배치로 병렬 호출한다.

    RAG처럼 저장해두는 게 아니라, 한 요청 안에서 리랭크에 쓰고 버리는 일회성 벡터다.
    """
    if not texts:
        return []

    async def one_batch(batch: list[str]) -> list[list[float]]:
        async with LLM_SEM:
            resp = await _client.aio.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
            )
        return [e.values for e in resp.embeddings]

    chunks = [texts[i : i + 100] for i in range(0, len(texts), 100)]
    batched = await asyncio.gather(*[one_batch(c) for c in chunks])
    return [v for b in batched for v in b]


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


# ---------------------------------------------------------------- 수확 보강 (지역 일치 판정)

def in_same_region(place: dict, base: list[dict]) -> bool:
    """place가 base(카카오 결과)와 같은 구(區)에 있는지.

    웹수확에는 다른 지역이 섞인다 — '가포' 검색에 청주 흥덕구 '가포산로' 리스트가
    딸려오는 식. 도로명 접두로 걸러보면 지역마다 도로명 구성이 달라 일관되지 않아
    구 단위만 본다.
    """
    parts = place["address"].split()
    if len(parts) < 3:
        return False
    base_gu = {b["address"].split()[2] for b in base if len(b["address"].split()) > 2}
    return parts[2] in base_gu


# ---------------------------------------------------------------- 크롤링

# 협찬/체험단 판정용. 내돈내산 후기는 자기 사진을 올려 이미지가 네이버(pstatic)에 재호스팅되지만,
# 협찬 배지는 대행사 서버(레뷰·리뷰노트 등)에서 들고 와 첫 사진 URL이 비네이버다.
_STICKER_HOSTS = ("storep-phinf", "simg.pstatic", "ssl.pstatic", "dthumb")  # 이모티콘·지도카드
_SPONSOR_WORDS = ("협찬", "소정의", "원고료", "제공받", "대가를", "업체로부터", "지원받", "고료를")


def _is_sponsored(soup) -> bool:
    """네이버 블로그 글이 협찬/체험단인지 추정한다.

    스티커·지도카드를 건너뛴 '첫 사진'의 도메인이 네이버(pstatic/naver)가 아니면 협찬으로 본다
    (배지를 대행사 서버에서 불러온 경우). 본문 앞부분의 고지 문구도 함께 본다.
    """
    cont = soup
    for s in _BODY_SELECTORS:
        node = soup.select_one(s)
        if node:
            cont = node
            break

    for img in cont.find_all("img"):
        u = img.get("data-lazy-src") or img.get("src") or img.get("data-src") or ""
        if u.startswith("//"):
            u = "https:" + u
        dom = urlparse(u).netloc
        if not dom or any(h in dom for h in _STICKER_HOSTS):
            continue  # 이모티콘·카드는 사진이 아니므로 건너뛴다
        if "pstatic" not in dom and "naver" not in dom:
            return True  # 첫 사진이 비네이버 → 협찬 배지
        break  # 첫 사진이 네이버 → 사진 신호로는 정상

    return any(w in cont.get_text(" ", strip=True)[:600] for w in _SPONSOR_WORDS)


async def scrape_urls(urls: list[str], max_chars: int = 0) -> list[dict]:
    """URL 병렬 크롤링 → [{'url', 'text', 'sponsored'}].

    실패해도 예외를 던지지 않고 text에 FAIL_PREFIX로 시작하는 문자열을 담는다.
    sponsored는 네이버 블로그 협찬 추정 플래그(블로그가 아니면 의미 없음).
    max_chars가 0이면 본문 전체를 담는다.
    """

    @track("scrape")
    async def fetch_one(client: httpx.AsyncClient, url: str) -> dict:
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
                return {"text": f"{FAIL_PREFIX}열기 실패: {type(e).__name__}", "sponsored": False}

        soup = BeautifulSoup(resp.text, "html.parser")
        sponsored = _is_sponsored(soup)  # 태그 정리 전에 이미지·고지 판정
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
            return {"text": f"{FAIL_PREFIX}본문 추출 실패", "sponsored": sponsored}
        return {"text": text[:max_chars] if max_chars else text, "sponsored": sponsored}

    async with httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT}, timeout=10.0, follow_redirects=True
    ) as client:
        results = await asyncio.gather(*(fetch_one(client, u) for u in urls))

    return [{"url": u, **d} for u, d in zip(urls, results)]


# ---------------------------------------------------------------- 네이버 검색

# 검색 키 — 2번째 키(NAVER_CLIENT_ID_2)가 있으면 라운드로빈으로 부하를 나눈다.
# 429는 앱 키당 걸리므로 키를 번갈아 쓰면 각 키가 자기 한도 안에 머물러 재시도가 준다.
_NAVER_KEYS = [(os.environ["NAVER_CLIENT_ID"], os.environ["NAVER_CLIENT_SECRET"])]
if os.getenv("NAVER_CLIENT_ID_2") and os.getenv("NAVER_CLIENT_SECRET_2"):
    _NAVER_KEYS.append(
        (os.environ["NAVER_CLIENT_ID_2"], os.environ["NAVER_CLIENT_SECRET_2"])
    )
_naver_cycle = itertools.cycle(_NAVER_KEYS)


@track("naver")
async def naver_search(
    query: str, kind: str = "blog", display: int = 10, sort: str = "sim"
) -> tuple[list[dict], str]:
    """네이버 검색 API 비동기 호출.

    blog / news / webkr 모두 응답 구조가 같아 하나로 처리한다.
    kind: 'blog'(블로그) / 'news'(뉴스) / 'webkr'(웹문서)
    sort: 'sim'(정확도) / 'date'(최신). webkr은 sort를 무시한다.

    키가 여러 개면 호출·재시도마다 라운드로빈해 429를 분산한다.
    반환: (결과 리스트, 상태). 상태는 'ok' / 'empty' / '429'
    """
    async with NAVER_SEM:
        for attempt in range(3):
            cid, csec = next(_naver_cycle)  # 호출·재시도마다 키 교대
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"https://openapi.naver.com/v1/search/{kind}",
                    headers={
                        "X-Naver-Client-Id": cid,
                        "X-Naver-Client-Secret": csec,
                    },
                    params={"query": query, "display": display, "sort": sort},
                )
            if resp.status_code == 429:
                # 429는 키(앱)당 걸리므로, 아직 안 써본 키가 남았으면 sleep 없이 즉시
                # 다른 키로 재시도한다. 모든 키가 429일 때만(=진짜 과부하) 백오프한다.
                if attempt + 1 < len(_NAVER_KEYS):
                    continue
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            if resp.status_code == 401 and len(_NAVER_KEYS) > 1:
                # 스코프 없는/잘못된 키 — 크래시 말고 다음 키로 교대 재시도
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
