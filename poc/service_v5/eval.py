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
from collections import defaultdict
from pathlib import Path

from wh_poc import Chat, load_json, set_verbose
from wh_poc.clients import pop_llm_latency

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


# 한글과 알파벳은 둘 다 \w라 그 사이에 \b가 성립하지 않는다.
# '등대집or 가포등대집'을 분리하려면 한글 뒤에 붙은 or를 따로 잡아야 한다.
_ALIAS_SEP = re.compile(
    r"\s*(?:\bor\b|(?<=[가-힣])or(?=[\s가-힣]))\s*|\s*/\s*", re.I
)


def _expand(answer: str) -> list[str]:
    """'등대집or 가포등대집' 같은 별칭 표기를 분리한다."""
    return [p.strip() for p in _ALIAS_SEP.split(answer) if p.strip()]


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
        pop_llm_latency()  # 다음 케이스로 지연 기록이 새지 않게 비운다
        print(f"   실패: {type(e).__name__}: {e}")
        return {
            "question": q, "answer": answers, "error": f"{type(e).__name__}: {e}",
            "elapsed": time.perf_counter() - t0,
        }

    elapsed = time.perf_counter() - t0
    result = out["result"] or {}
    kind = result.get("kind", "?")
    final_names = [p.get("name", "") for p in result.get("places", [])]

    # 파이프라인 단계별 후보 — 각 툴이 남긴 파일에서 읽는다.
    # 수집(collect_places) → 1차필터·리랭크(collect_info의 info_rerank) → 최종.
    pool_names: list[str] = []
    if pf := out["files"].get("file"):
        try:
            pool_names = [p["name"] for p in load_json(pf)]
        except Exception:
            pass

    filter_names, rerank_names = [], []
    if rf := out["files"].get("rerank_file"):
        try:
            rr_data = load_json(rf)
            filter_names = [x["name"] for x in rr_data["ranking"]]                 # 1차 필터 통과
            rerank_names = [x["name"] for x in rr_data["ranking"] if x["kept"]]    # 리랭크 채택
        except Exception:
            pass

    # 정답별 단계 통과 위치
    detail = []
    for a in answers:
        pool_rank = _hit(a, pool_names)
        final_rank = _hit(a, final_names)
        detail.append({
            "answer": a,
            "in_pool": pool_rank >= 0,
            "pool_rank": pool_rank + 1 if pool_rank >= 0 else None,
            "in_filter": _hit(a, filter_names) >= 0,
            "in_rerank": _hit(a, rerank_names) >= 0,
            "in_final": final_rank >= 0,
            "final_rank": final_rank + 1 if final_rank >= 0 else None,
        })

    n_pool = sum(1 for d in detail if d["in_pool"])
    n_filter = sum(1 for d in detail if d["in_filter"])
    n_rerank = sum(1 for d in detail if d["in_rerank"])
    n_final = sum(1 for d in detail if d["in_final"])
    ranks = [d["final_rank"] for d in detail if d["final_rank"]]
    rr = 1.0 / min(ranks) if ranks else 0.0

    # 콘솔 요약
    mark = "O" if n_final else ("~" if n_pool else "X")
    na = len(answers)
    print(f"   [{mark}] 수집 {n_pool}/{na} | 필터 {n_filter}/{na} | 리랭크 {n_rerank}/{na} "
          f"| 최종 {n_final}/{na} | RR {rr:.2f} | {kind} | {elapsed:.1f}초")
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

    # 시간 분해(백엔드 의존성) — tracing.totals와 같은 기준.
    steps = tracer_turn.get("steps", [])
    bd = defaultdict(float)
    for s in steps:
        for k, v in s.get("breakdown", {}).items():
            bd[k] += v
    timing = {
        "agent_llm_sec": round(sum(s["sec"] for s in steps if s["kind"] == "llm"), 2),
        "util_llm_sec": round(bd.get("llm", 0.0), 2),   # 툴내부 로컬 LLM(HyDE·요약·포매터)
        "tool_sec": round(sum(s["sec"] for s in steps if s["kind"] == "tool"), 2),
        "embed_sec": round(bd.get("embed", 0.0), 2),
        "io_sec": round(bd.get("naver", 0.0) + bd.get("scrape", 0.0) + bd.get("kakao", 0.0), 2),
    }

    # 스텝 타임라인 — 콜별 걸린 시간(LLM 대기/툴 실행)을 그대로 남긴다.
    step_timeline = [
        {
            "kind": s["kind"],
            "name": s.get("name", "llm"),
            "sec": round(s["sec"], 2),
            "tok_in": s.get("tok_in", 0),
            "tok_out": s.get("tok_out", 0),
            "args": s.get("args", ""),        # 툴에 던진 인자(검색쿼리 등)
            "detail": s.get("detail", ""),
        }
        for s in steps
    ]

    # 로컬 서빙 지연(스트리밍 계측). 마지막 호출 = 최종 JSON 포매터(가장 무거운 생성).
    lat = pop_llm_latency()
    latency = {}
    if lat:
        n = len(lat)
        tot_out = sum(r["out_tok"] for r in lat)
        tpot_mean = (sum(r["tpot"] * r["out_tok"] for r in lat) / tot_out) if tot_out else 0.0
        latency = {
            "n_calls": n,
            "out_tok_total": tot_out,
            "ttft_mean": round(sum(r["ttft"] for r in lat) / n, 3),
            "ttft_max": round(max(r["ttft"] for r in lat), 3),
            "tpot_mean": round(tpot_mean, 4),                       # 초/토큰
            "decode_tps": round(1 / tpot_mean, 1) if tpot_mean else 0.0,
            "formatter": {k: round(lat[-1][k], 3) for k in ("ttft", "e2el", "tpot", "tps", "out_tok")},
            # 콜별 상세 — 각 로컬 LLM 호출의 TTFT·E2EL·TPOT·출력토큰 전량.
            "calls": [{k: round(r[k], 3) for k in ("ttft", "e2el", "tpot", "tps", "out_tok")} for r in lat],
        }
        f = latency["formatter"]
        print(f"       ⏱ TTFT {latency['ttft_mean']:.2f}s(max {latency['ttft_max']:.2f}) "
              f"| TPOT {latency['tpot_mean']*1000:.0f}ms → {latency['decode_tps']:.1f} tok/s "
              f"| 호출 {n} out {tot_out}tok | 포매터 E2EL {f['e2el']:.1f}s({f['out_tok']}tok)")

    return {
        "question": q,
        "answer": answers,
        "kind": kind,
        "predicted": final_names,
        "pool": pool_names,
        "detail": detail,
        "n_answer": len(answers),
        "n_pool_hit": n_pool,
        "n_filter_hit": n_filter,
        "n_rerank_hit": n_rerank,
        "n_final_hit": n_final,
        "rr": rr,
        "elapsed": elapsed,
        "timing": timing,
        "latency": latency,
        "steps": step_timeline,
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
    n_filter = sum(r.get("n_filter_hit", 0) for r in ok)
    n_rerank = sum(r.get("n_rerank_hit", 0) for r in ok)
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
    print(f"  정답 단위 funnel (단계별 recall, 괄호=직전 단계 대비 손실)")
    print(f"    수집     {n_pool}/{n_ans}  ({n_pool / n_ans * 100:5.1f}%)")
    print(f"    1차필터  {n_filter}/{n_ans}  ({n_filter / n_ans * 100:5.1f}%)   (-{n_pool - n_filter})")
    print(f"    리랭크   {n_rerank}/{n_ans}  ({n_rerank / n_ans * 100:5.1f}%)   (-{n_filter - n_rerank})")
    print(f"    최종     {n_final}/{n_ans}  ({n_final / n_ans * 100:5.1f}%)   (-{n_rerank - n_final})")
    print(f"\n  케이스 단위 (하나라도 맞힘)")
    print(f"    Recall@수집   {case_pool}/{len(ok)}  ({case_pool / len(ok) * 100:5.1f}%)")
    print(f"    Recall@최종   {case_final}/{len(ok)}  ({case_final / len(ok) * 100:5.1f}%)")
    print(f"\n  MRR           {mrr:.3f}")
    print(f"  평균 소요      {avg_sec:.1f}초")
    print(f"  토큰          in {tok_in:,} / out {tok_out:,}")

    # 로컬 서빙 지연 집계(스트리밍 계측이 있는 케이스만).
    lats = [r["latency"] for r in ok if r.get("latency")]
    lat_stats = {}
    if lats:
        tot_out = sum(la["out_tok_total"] for la in lats)
        tpot = sum(la["tpot_mean"] * la["out_tok_total"] for la in lats) / tot_out if tot_out else 0.0
        ttft_mean = sum(la["ttft_mean"] for la in lats) / len(lats)
        ttft_max = max(la["ttft_max"] for la in lats)
        fmt_e2el = sum(la["formatter"]["e2el"] for la in lats) / len(lats)
        lat_stats = {
            "ttft_mean": round(ttft_mean, 3), "ttft_max": round(ttft_max, 3),
            "tpot_mean": round(tpot, 4), "decode_tps": round(1 / tpot, 1) if tpot else 0.0,
            "out_tok_total": tot_out, "formatter_e2el_mean": round(fmt_e2el, 2),
        }
        print(f"\n  로컬 서빙 지연 (vLLM 스트리밍 계측)")
        print(f"    TTFT       평균 {ttft_mean:.2f}초 / 최대 {ttft_max:.2f}초")
        print(f"    TPOT       {tpot * 1000:.0f}ms/토큰  →  decode {1 / tpot if tpot else 0:.1f} tok/s")
        print(f"    포매터 E2EL 평균 {fmt_e2el:.1f}초 (최종 JSON 생성)")

    # 수집은 됐는데 최종에서 떨어진 것 = 어느 단계에서 샜나
    lost = n_pool - n_final
    if lost:
        print(f"\n  수집됐지만 최종 탈락: {lost}개  ← 단계별 누수")
        for r in ok:
            for d in r["detail"]:
                if d["in_pool"] and not d["in_final"]:
                    if not d.get("in_filter"):
                        stage = "1차필터 탈락"
                    elif not d.get("in_rerank"):
                        stage = "리랭크 탈락"
                    else:
                        stage = "최종LLM 탈락"
                    print(f"    {d['answer'][:18]:20} {stage:12} | {r['question'][:26]}")

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
        "recall_pool": n_pool / n_ans,
        "recall_filter": n_filter / n_ans,
        "recall_rerank": n_rerank / n_ans,
        "recall_final": n_final / n_ans,
        "case_recall_pool": case_pool / len(ok), "case_recall_final": case_final / len(ok),
        "mrr": mrr, "avg_sec": avg_sec,
        "tokens_in": tok_in, "tokens_out": tok_out,
        "latency": lat_stats,
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

    out_path = Path(args.out or f"eval_{int(time.time())}.json")

    # 케이스마다 결과를 파일에 덮어써 부분 저장한다. 중간에 죽어도(로컬은 오래
    # 걸려 끊길 수 있다) 완료분이 남는다. summary는 마지막에 채운다.
    def _dump(stats: dict) -> None:
        out_path.write_text(
            json.dumps({"summary": stats, "cases": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    rows = []
    for i, c in enumerate(cases, 1):
        rows.append(await eval_one(c, i, len(cases)))
        _dump({})                 # 부분 저장
        print(f"   [저장 {i}/{len(cases)}] {out_path.name}", flush=True)

    _dump(summarize(rows))
    print(f"\n기록 저장: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
