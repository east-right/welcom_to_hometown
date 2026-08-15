"""정답셋 기반 평가.

    python eval.py                              기본 경로(../../data/test.json)
    python eval.py path/to/test.json            경로 지정
    python eval.py test.json -n 3               앞 3개만
    python eval.py test.json -o result.json     결과 저장 경로 지정

## 왜 단계별로 재는가

정답을 못 찾았을 때 원인이 둘 중 하나다.
  - 수집 실패: collect_places가 애초에 그 매장을 안 가져옴 → 검색어 문제
  - 선별 실패: 후보에는 있었는데 최종 8개에 못 듦 → 랭킹 문제
이 둘은 고칠 곳이 완전히 다르므로 지표를 분리해서 잰다.

    Recall@수집   정답이 collect_places 결과 안에 있나
    Recall@최종   정답이 최종 추천 안에 있나
    MRR           최종 추천에서 첫 정답이 몇 번째인가

Precision은 쓰지 않는다. 정답이 1개인데 시스템이 5~8개를 내므로
낮게 나오는 게 당연하고, 나머지가 나쁜 답이라는 뜻이 아니다.
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from wh_poc import Chat, load_json, set_verbose

DEFAULT_DATA = "../../data/test.json"


# ---------------------------------------------------------------- 매칭

_PREFIX = re.compile(r"^(24시|24h|원조|진짜|리얼|정통|옛날)")
_SUFFIX = re.compile(r"(본점|직영점|\d+호점|[가-힣]{1,6}점)$")


def _norm(s: str) -> str:
    """공백·기호·대소문자 차이를 흡수한다."""
    return re.sub(r"[\s\-_()·.,'\"]", "", str(s)).lower()


def _core(s: str) -> str:
    """상호명에서 수식어를 벗겨 핵심만 남긴다.

    구어체 상호명과 카카오 등록명은 앞뒤 수식어가 자주 어긋난다.
    '24시서울밥집'과 '서울밥집 신림점'은 같은 곳이지만 양방향 포함이 성립하지 않아,
    접두(24시·원조)와 접미(본점·OO점)를 벗겨야 매칭된다.
    """
    t = _norm(s)
    prev = None
    while prev != t:                 # 접두·접미가 겹칠 수 있어 반복
        prev = t
        t = _PREFIX.sub("", t)
        t = _SUFFIX.sub("", t)
    return t or _norm(s)             # 전부 벗겨지면 원본을 쓴다


def _expand(answer: str) -> list[str]:
    """'등대집or 가포등대집' 같은 별칭 표기를 분리한다."""
    parts = re.split(r"\s*\bor\b\s*|\s*/\s*", answer, flags=re.I)
    return [p.strip() for p in parts if p.strip()]


def _hit(answer: str, candidates: list[str]) -> int:
    """정답이 후보 목록 몇 번째에 있나. 없으면 -1."""
    variants = []
    for v in _expand(answer):
        variants += [_norm(v), _core(v)]
    variants = [v for v in variants if len(v) >= 2]

    for i, c in enumerate(candidates):
        forms = {_norm(c), _core(c)}
        for v in variants:
            for f in forms:
                if v in f or f in v:
                    return i
    return -1


# ---------------------------------------------------------------- 평가

async def eval_one(case: dict, idx: int, total: int) -> dict:
    q = case["question"]
    answers = case["answer"]

    print(f"\n[{idx}/{total}] {q}")
    t0 = time.perf_counter()

    chat = Chat(show_tools=False, show_status=False)
    try:
        out = await chat.send(q)
    except Exception as e:
        print(f"   실패: {type(e).__name__}: {e}")
        return {
            "question": q, "answer": answers, "error": f"{type(e).__name__}: {e}",
            "elapsed": time.perf_counter() - t0,
        }

    elapsed = time.perf_counter() - t0
    result = out["result"] or {}
    kind = result.get("kind", "?")
    final_names = [p.get("name", "") for p in result.get("places", [])]

    # 수집 단계 결과 — 툴이 남긴 파일에서 읽는다
    pool_names: list[str] = []
    if pf := out["files"].get("file"):
        try:
            pool_names = [p["name"] for p in load_json(pf)]
        except Exception:
            pass

    # 정답별 히트 위치
    detail = []
    for a in answers:
        pool_rank = _hit(a, pool_names)
        final_rank = _hit(a, final_names)
        detail.append({
            "answer": a,
            "in_pool": pool_rank >= 0,
            "pool_rank": pool_rank + 1 if pool_rank >= 0 else None,
            "in_final": final_rank >= 0,
            "final_rank": final_rank + 1 if final_rank >= 0 else None,
        })

    n_pool = sum(1 for d in detail if d["in_pool"])
    n_final = sum(1 for d in detail if d["in_final"])
    ranks = [d["final_rank"] for d in detail if d["final_rank"]]
    rr = 1.0 / min(ranks) if ranks else 0.0

    # 콘솔 요약
    mark = "O" if n_final else ("~" if n_pool else "X")
    print(f"   [{mark}] 수집 {n_pool}/{len(answers)} | 최종 {n_final}/{len(answers)} "
          f"| RR {rr:.2f} | {kind} | {elapsed:.1f}초")
    for d in detail:
        if d["in_final"]:
            s = f"최종 {d['final_rank']}위"
        elif d["in_pool"]:
            s = f"수집 {d['pool_rank']}위 → 최종 탈락"
        else:
            s = "미수집"
        print(f"       {d['answer'][:24]:26} {s}")
    if final_names:
        print(f"       추천: {', '.join(final_names)}")

    tracer_turn = chat.tracer.turns[-1] if (chat.tracer and chat.tracer.turns) else {}

    return {
        "question": q,
        "answer": answers,
        "kind": kind,
        "predicted": final_names,
        "pool": pool_names,
        "detail": detail,
        "n_answer": len(answers),
        "n_pool_hit": n_pool,
        "n_final_hit": n_final,
        "rr": rr,
        "elapsed": elapsed,
        "tool_calls": [t["name"] for t in out["tool_calls"]],
        "tokens_in": tracer_turn.get("tokens_in", 0),
        "tokens_out": tracer_turn.get("tokens_out", 0),
        "files": out["files"],
        "message": result.get("message", ""),
    }


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("\n평가 가능한 결과가 없습니다.")
        return {}

    n_ans = sum(r["n_answer"] for r in ok)
    n_pool = sum(r["n_pool_hit"] for r in ok)
    n_final = sum(r["n_final_hit"] for r in ok)

    # 케이스 단위: 정답을 하나라도 맞혔나
    case_pool = sum(1 for r in ok if r["n_pool_hit"])
    case_final = sum(1 for r in ok if r["n_final_hit"])

    mrr = sum(r["rr"] for r in ok) / len(ok)
    avg_sec = sum(r["elapsed"] for r in ok) / len(ok)
    tok_in = sum(r["tokens_in"] for r in ok)
    tok_out = sum(r["tokens_out"] for r in ok)

    print(f"\n{'=' * 64}")
    print(f"케이스 {len(ok)}개 (오류 {len(rows) - len(ok)}개) | 정답 총 {n_ans}개\n")
    print(f"  정답 단위")
    print(f"    Recall@수집   {n_pool}/{n_ans}  ({n_pool / n_ans * 100:5.1f}%)")
    print(f"    Recall@최종   {n_final}/{n_ans}  ({n_final / n_ans * 100:5.1f}%)")
    print(f"\n  케이스 단위 (하나라도 맞힘)")
    print(f"    Recall@수집   {case_pool}/{len(ok)}  ({case_pool / len(ok) * 100:5.1f}%)")
    print(f"    Recall@최종   {case_final}/{len(ok)}  ({case_final / len(ok) * 100:5.1f}%)")
    print(f"\n  MRR           {mrr:.3f}")
    print(f"  평균 소요      {avg_sec:.1f}초")
    print(f"  토큰          in {tok_in:,} / out {tok_out:,}")

    # 수집은 됐는데 최종에서 떨어진 것 = 랭킹 문제
    lost = n_pool - n_final
    if lost:
        print(f"\n  수집됐지만 최종 탈락: {lost}개  ← 랭킹 개선 여지")
        for r in ok:
            for d in r["detail"]:
                if d["in_pool"] and not d["in_final"]:
                    print(f"    {d['answer'][:20]:22} (수집 {d['pool_rank']}위) "
                          f"| {r['question'][:28]}")

    missed = n_ans - n_pool
    if missed:
        print(f"\n  아예 수집 안 됨: {missed}개  ← 검색어 개선 여지")
        for r in ok:
            for d in r["detail"]:
                if not d["in_pool"]:
                    print(f"    {d['answer'][:20]:22} | {r['question'][:28]}")

    # kind 분포 — question이 많으면 되묻기가 과한 것
    kinds = {}
    for r in ok:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"\n  응답 종류: {kinds}")

    return {
        "n_case": len(ok), "n_answer": n_ans,
        "recall_pool": n_pool / n_ans, "recall_final": n_final / n_ans,
        "case_recall_pool": case_pool / len(ok), "case_recall_final": case_final / len(ok),
        "mrr": mrr, "avg_sec": avg_sec,
        "tokens_in": tok_in, "tokens_out": tok_out,
        "kinds": kinds,
    }


async def main():
    ap = argparse.ArgumentParser(description="정답셋 평가")
    ap.add_argument("data", nargs="?", default=DEFAULT_DATA, help="테스트 JSON 경로")
    ap.add_argument("-n", type=int, default=0, help="앞 N개만 평가 (0=전체)")
    ap.add_argument("-o", "--out", default="", help="결과 저장 경로")
    ap.add_argument("-v", "--verbose", action="store_true", help="툴 내부 로그")
    args = ap.parse_args()

    set_verbose(args.verbose)

    cases = load_json(args.data)
    if args.n:
        cases = cases[: args.n]

    rows = []
    for i, c in enumerate(cases, 1):
        rows.append(await eval_one(c, i, len(cases)))

    stats = summarize(rows)

    out_path = Path(args.out or f"eval_{int(time.time())}.json")
    out_path.write_text(
        json.dumps({"summary": stats, "cases": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n기록 저장: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
