"""턴 단위 오케스트레이션 트레이서.

ADK의 before/after 콜백을 걸어 한 턴 안에서 일어나는 LLM 호출과 툴 실행을
시간순으로 기록한다. core.py의 track/timed_tool이 '툴 내부 어디가 느린가'를 본다면,
이쪽은 '턴 전체에서 무엇이 언제 일어났는가'를 본다.

기록되는 것:
    - LLM 호출마다: 소요시간, 입출력 토큰, 모델이 뭘 결정했는지(툴 호출 or 텍스트)
    - 툴 호출마다: 소요시간, 인자, 반환 요약
    - 프롬프트 전문 (capture_prompt=True일 때만. 길어서 기본은 끔)

Langfuse 같은 외부 도구 없이 노트북에서 바로 볼 수 있게 만든 것이라,
영속화나 세션 간 비교가 필요해지면 그때 OTel 연동을 얹으면 된다.
"""

import json
import time
from typing import Any


def _brief(value: Any, limit: int = 80) -> str:
    """긴 값을 한 줄로 줄인다. 파일 내용처럼 큰 건 길이만 표시."""
    if isinstance(value, dict):
        if "content" in value:
            other = {k: v for k, v in value.items() if k != "content"}
            body = f"<{len(str(value['content']))}자>"
            return f"{other} content={body}" if other else body
        s = json.dumps(value, ensure_ascii=False)
    else:
        s = str(value)
    s = s.replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "…"


class Tracer:
    """한 Chat 인스턴스에 붙어 턴별 타임라인을 쌓는다."""

    def __init__(self, capture_prompt: bool = False):
        self.capture_prompt = capture_prompt
        self.turns: list[dict] = []
        self._cur: dict | None = None
        self._llm_t0: float = 0.0
        self._tool_t0: dict[str, float] = {}

    # ------------------------------------------------------------ 턴 경계

    def start_turn(self, query: str) -> None:
        self._cur = {
            "turn": len(self.turns) + 1,
            "query": query,
            "steps": [],
            "started": time.perf_counter(),
            "elapsed": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
        }

    def end_turn(self, result: dict | None = None) -> None:
        if self._cur is None:
            return
        self._cur["elapsed"] = time.perf_counter() - self._cur["started"]
        self._cur["result_kind"] = (result or {}).get("kind", "?")
        self._cur["places"] = len((result or {}).get("places", []))
        self.turns.append(self._cur)
        self._cur = None

    # ------------------------------------------------------------ 콜백

    # ADK는 콜백을 키워드 인자로 호출한다. 이름이 정확히 맞아야 한다.
    #   model: callback_context, llm_request / callback_context, llm_response
    #   tool : tool, args, tool_context / tool, args, tool_context, tool_response

    def before_model(self, callback_context, llm_request):
        self._llm_t0 = time.perf_counter()
        if self._cur is None or not self.capture_prompt:
            return None
        # contents는 대화 전체 이력이라 매우 길다. 마지막 것만 남긴다.
        try:
            last = llm_request.contents[-1] if llm_request.contents else None
            text = " ".join(
                p.text for p in (last.parts or []) if getattr(p, "text", None)
            )
        except Exception:
            text = ""
        self._cur.setdefault("prompts", []).append(text[:2000])
        return None

    def after_model(self, callback_context, llm_response):
        if self._cur is None:
            return None
        elapsed = time.perf_counter() - self._llm_t0

        tin = tout = 0
        um = getattr(llm_response, "usage_metadata", None)
        if um:
            tin = getattr(um, "prompt_token_count", 0) or 0
            tout = getattr(um, "candidates_token_count", 0) or 0
        self._cur["tokens_in"] += tin
        self._cur["tokens_out"] += tout

        # 모델이 무엇을 결정했는지: 툴 호출인가 텍스트인가
        decision = "(빈 응답)"
        content = getattr(llm_response, "content", None)
        if content and content.parts:
            calls = [p.function_call.name for p in content.parts if p.function_call]
            texts = [p.text for p in content.parts if getattr(p, "text", None)]
            if calls:
                decision = "→ " + ", ".join(calls)
            elif texts:
                decision = _brief("".join(texts), 60)

        self._cur["steps"].append(
            {
                "kind": "llm",
                "sec": elapsed,
                "tok_in": tin,
                "tok_out": tout,
                "detail": decision,
            }
        )
        return None

    def before_tool(self, tool, args, tool_context):
        self._tool_t0[tool.name] = time.perf_counter()
        return None

    def after_tool(self, tool, args, tool_context, tool_response):
        if self._cur is None:
            return None
        t0 = self._tool_t0.pop(tool.name, None)
        self._cur["steps"].append(
            {
                "kind": "tool",
                "name": tool.name,
                "sec": (time.perf_counter() - t0) if t0 else 0.0,
                "args": _brief(args, 60),
                "detail": _brief(tool_response),
            }
        )
        return None

    # ------------------------------------------------------------ 출력

    def report(self, turn: int | None = None) -> None:
        """타임라인 출력. turn을 주면 그 턴만, 없으면 전체."""
        targets = self.turns if turn is None else [
            t for t in self.turns if t["turn"] == turn
        ]
        for t in targets:
            print(
                f"\n[턴 {t['turn']}] \"{t['query'][:40]}\""
                f"    총 {t['elapsed']:.1f}초  "
                f"| {t['result_kind']} places={t['places']} "
                f"| 토큰 in {t['tokens_in']:,} out {t['tokens_out']:,}"
            )
            n = len(t["steps"])
            for i, s in enumerate(t["steps"]):
                edge = "└" if i == n - 1 else "├"
                if s["kind"] == "llm":
                    print(
                        f"  {edge} LLM   {s['sec']:6.1f}초  "
                        f"in {s['tok_in']:>6,} out {s['tok_out']:>5,}  {s['detail']}"
                    )
                else:
                    print(
                        f"  {edge} TOOL  {s['sec']:6.1f}초  "
                        f"{s['name']}({s['args']})"
                    )
                    print(f"  {'│' if i < n - 1 else ' '}          → {s['detail']}")

    def totals(self) -> None:
        """턴 전체 합계. LLM 대기와 툴 실행의 비중을 본다."""
        if not self.turns:
            print("기록된 턴이 없습니다.")
            return
        wall = sum(t["elapsed"] for t in self.turns)
        llm = sum(s["sec"] for t in self.turns for s in t["steps"] if s["kind"] == "llm")
        tool = sum(s["sec"] for t in self.turns for s in t["steps"] if s["kind"] == "tool")
        tin = sum(t["tokens_in"] for t in self.turns)
        tout = sum(t["tokens_out"] for t in self.turns)

        print(f"\n{'=' * 56}")
        print(f"턴 {len(self.turns)}개 | 총 {wall:.1f}초")
        print(f"  LLM 대기   {llm:6.1f}초  ({llm / wall * 100:4.1f}%)")
        print(f"  툴 실행    {tool:6.1f}초  ({tool / wall * 100:4.1f}%)")
        print(f"  그 외      {wall - llm - tool:6.1f}초")
        print(f"  토큰       in {tin:,} / out {tout:,}")

        # 툴별 누적
        agg: dict[str, list[float]] = {}
        for t in self.turns:
            for s in t["steps"]:
                if s["kind"] == "tool":
                    agg.setdefault(s["name"], []).append(s["sec"])
        if agg:
            print("  툴별 누적:")
            for name, secs in sorted(agg.items(), key=lambda x: -sum(x[1])):
                print(f"    {name:18} {sum(secs):6.1f}초 ({len(secs)}회)")

    def to_dict(self) -> list[dict]:
        """저장·비교용 원본 데이터."""
        return self.turns
