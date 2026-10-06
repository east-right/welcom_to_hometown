"""실행 로그를 두 갈래로 남긴다.

- FlowLog  : 파이프라인 데이터 흐름(쿼리증강 → 카카오 → 네이버 URL/fetch). 콘솔+파일.
- AgentLog : 각 에이전트의 '변동 입력 프롬프트 + JSON 출력'. 파일만(프롬프트가 길다).

같은 실행 이름(stem)으로 <stem>.flow.log / <stem>.agents.log 로 짝지어 저장돼
서로 매칭하기 쉽다. review 레코드는 매장명 태그가 붙어 flow의 그 매장 블록과 대응된다.
"""
import sys
from datetime import datetime
from pathlib import Path


class FileSink:
    """한 파일에 append하는 얇은 싱크. echo=True면 콘솔에도(인코딩 안전)."""

    def __init__(self, path: str, echo: bool = False) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._echo = echo
        if echo:
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except (AttributeError, ValueError):
                pass

    def write(self, text: str = "") -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
        if self._echo:
            try:
                print(text)
            except UnicodeEncodeError:
                enc = sys.stdout.encoding or "utf-8"
                print(text.encode(enc, "replace").decode(enc))


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


class FlowLog:
    """실행 흐름 트레이스: 질문 → 각 툴 호출(입력/출력) → 결과 카드.

    툴의 '입력(무슨 인자)'과 '출력(무슨 결과)'을 다 남기되, 크롤링 원문(본문)은 넣지 않는다.
    """

    def __init__(self, sink: FileSink) -> None:
        self._sink = sink

    def question(self, text: str) -> None:
        bar = "═" * 58
        self._sink.write(f"\n{bar}\n[질문] {text}\n{bar}")

    def queries(self, queries) -> None:
        parts = " · ".join(f"{q.query}({q.category or '-'})" for q in queries)
        self._sink.write(f"\n{_now()} [1. 쿼리분해] {len(queries)}개: {parts}")

    def places(self, places) -> None:
        names = " · ".join(p.name for p in places) or "(없음)"
        self._sink.write(f"{_now()} [2. 카카오 수집] {len(places)}곳: {names}")

    def enrich_header(self, n: int) -> None:
        self._sink.write(f"{_now()} [3. 리뷰수집 {n}곳 전부 병렬] (매장 → blog URL, ✓취득/✗실패)")

    def enrich_one(self, name: str, urls) -> None:
        """urls: [(url, ok)] — 한 매장의 blog URL과 취득 성패(원문 제외)."""
        marks = " · ".join((f"✓ {u}" if ok else f"✗ {u}") for u, ok in urls) or "(없음)"
        self._sink.write(f"    {name}  →  {marks}")

    def cards(self, feed) -> None:
        names = " · ".join(c.name for c in feed.cards) or "(없음)"
        self._sink.write(f"\n{_now()} [4. 결과] 카드 {len(feed.cards)}개: {names}\n")


class AgentLog:
    """에이전트 입출력(변동 프롬프트 + JSON 결과)만 읽기 좋은 섹션으로 남긴다."""

    def __init__(self, sink: FileSink) -> None:
        self._sink = sink

    def record(self, agent: str, model: str, prompt: str, output: str,
               elapsed: float, tag: str = "") -> None:
        head = f"[{agent}: {tag}]" if tag else f"[{agent}]"
        block = (
            f"\n===== {_now()} {head} {model}  {elapsed:.1f}s =====\n"
            f"--- INPUT ---\n{prompt}\n"
            f"--- OUTPUT ---\n{output}"
        )
        self._sink.write(block)   # 한 번에 써서 병렬 실행에도 블록이 안 섞인다
