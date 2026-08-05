"""runs.jsonl 집계. 모델별로 지연·토큰·출력 유효성을 비교표로 뽑는다.

사용:
    uv run python poc/service_v2/eval/aggregate.py [logs/runs.jsonl]

각 레코드(에이전트 1회 실행)를 (model, agent)로 묶어 계산한다:
  - n       : 실행 횟수
  - ok%     : 출력이 그 에이전트에 유효했나 (약한 모델 판별의 핵심)
              모두 구조화 출력이므로 JSON 파싱 + 필수 키로 판정
              slot_extractor=slots/ready, query_expander=queries, selector=cards
  - avg_s/p50_s : 지연(초, 네트워크+추론 벽시계)
  - avg_tok : 평균 total 토큰 (제공자가 usage를 줄 때만)

모델을 바꿔가며 돌린 뒤 이 표를 보면 어느 모델·사이즈가
어느 에이전트에서 무너지는지 한눈에 보인다.
"""
import argparse
import json
import statistics
from collections import defaultdict

_AGENT_ORDER = {"slot_extractor": 0, "review_agent": 1, "selector": 2}


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def is_ok(agent: str, output: str) -> bool:
    """이 에이전트가 쓸모 있는 출력을 냈는가 (모델 실패 판별)."""
    output = output or ""
    try:
        data = json.loads(output)         # 모든 에이전트가 구조화 출력: 유효 JSON인가
    except (json.JSONDecodeError, TypeError):
        return False
    if agent == "slot_extractor":
        return isinstance(data, dict) and "slots" in data and "ready" in data
    if agent == "review_agent":
        return isinstance(data, dict)          # 후기가 비어도 유효 JSON이면 성공
    if agent == "selector":
        return isinstance(data, dict) and "cards" in data
    return bool(output)


def _tok(rec: dict) -> int:
    t = rec.get("tokens") or {}
    return t.get("total", 0) if isinstance(t, dict) else 0


def summarize(rows: list[dict]) -> dict:
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r.get("model", "?"), r.get("agent", "?"))].append(r)

    stats = {}
    for (model, agent), recs in groups.items():
        elapsed = [r.get("elapsed_sec", 0) or 0 for r in recs]
        oks = [is_ok(agent, r.get("output", "")) for r in recs]
        toks = [_tok(r) for r in recs]
        stats[(model, agent)] = {
            "n": len(recs),
            "ok": 100 * sum(oks) / len(recs),
            "avg_s": statistics.mean(elapsed),
            "p50_s": statistics.median(elapsed),
            "avg_tok": statistics.mean(toks) if toks else 0,
        }
    return stats


def render(stats: dict) -> str:
    models = sorted({m for (m, _) in stats})
    lines = []
    for model in models:
        lines.append(f"\nmodel: {model}")
        lines.append(f"  {'agent':<16}{'n':>4}{'ok%':>7}{'avg_s':>9}{'p50_s':>8}{'avg_tok':>10}")
        agents = sorted(
            (a for (m, a) in stats if m == model),
            key=lambda a: _AGENT_ORDER.get(a, 99),
        )
        for agent in agents:
            s = stats[(model, agent)]
            lines.append(
                f"  {agent:<16}{s['n']:>4}{s['ok']:>6.0f}%"
                f"{s['avg_s']:>9.2f}{s['p50_s']:>8.2f}{s['avg_tok']:>10.0f}"
            )
    return "\n".join(lines) if lines else "(집계할 레코드가 없습니다)"


def main() -> None:
    ap = argparse.ArgumentParser(description="runs.jsonl 모델별 집계")
    ap.add_argument("path", nargs="?", default="logs/runs.jsonl", help="JSONL 경로")
    args = ap.parse_args()

    rows = load(args.path)
    print(f"# {args.path} — 레코드 {len(rows)}건")
    print(render(summarize(rows)))


if __name__ == "__main__":
    main()
