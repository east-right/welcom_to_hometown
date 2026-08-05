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
    """파이프라인 데이터 흐름. 사람이 흐름을 눈으로 따라가게 한 줄씩."""

    def __init__(self, sink: FileSink) -> None:
        self._sink = sink

    def plan(self, queries) -> None:
        parts = " · ".join(f"{q.query}({q.category or '-'})" for q in queries)
        self._sink.write(f"{_now()} [쿼리증강] {len(queries)}개: {parts}")

    def kakao(self, places) -> None:
        names = " · ".join(p.name for p in places) or "(없음)"
        self._sink.write(f"{_now()} [카카오] {len(places)}곳: {names}")

    def reviews(self, place_name: str, items) -> None:
        """items: [(url, ok, length)] — 한 매장의 후기 URL과 fetch 성패."""
        lines = [f"  {place_name}"]
        for url, ok, length in items:
            lines.append(f"    ✓ {url}  ({length}자)" if ok else f"    ✗ {url}  [fetch 실패]")
        self._sink.write("\n".join(lines))

    def reviews_header(self) -> None:
        self._sink.write(f"{_now()} [네이버 후기 URL + fetch]")

    def note(self, line: str) -> None:
        self._sink.write(f"{_now()} {line}")


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
