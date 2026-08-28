"""리랭크 방법별 recall@K 스윕 (최종 LLM 선정 이전, 리랭크 순위 기준).

과거 원자료(experiments/{C,G,I}.json)와 그 tool_file 덤프(rerank_file=HyDE순위,
url_file=스니펫)에서 재구성한다. '쿼리-문서 임베딩'은 저장된 스니펫에 생쿼리를 다시
임베딩해 오프라인 재계산.
"""
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

Ks = [10, 15, 20, 25]

# ── 매칭 (eval.py와 동일 규칙) ──────────────────────────
_PREFIX = re.compile(r"^(24시|24h|원조|진짜|리얼|정통|옛날)")
_SUFFIX = re.compile(r"(본점|직영점|\d+호점|[가-힣]{1,6}점)$")


def _norm(s):
    return re.sub(r"[\s\-_()·.,'\"]", "", str(s)).lower()


def _core(s):
    t = _norm(s)
    prev = None
    while prev != t:
        prev = t
        t = _PREFIX.sub("", t)
        t = _SUFFIX.sub("", t)
    return t or _norm(s)


_ALIAS_SEP = re.compile(r"\s*(?:\bor\b|(?<=[가-힣])or(?=[\s가-힣]))\s*|\s*/\s*", re.I)


def _expand(answer):
    return [p.strip() for p in _ALIAS_SEP.split(answer) if p.strip()]


def _rank_of(answer, names):
    """정답이 순위 리스트 몇 번째(1-base)에 있나. 없으면 None."""
    variants = []
    for v in _expand(answer):
        variants += [_norm(v), _core(v)]
    variants = [v for v in variants if len(v) >= 2]
    for i, c in enumerate(names):
        forms = {_norm(c), _core(c)}
        for v in variants:
            for f in forms:
                if v in f or f in v:
                    return i + 1
    return None


def recall_at_k(cases_ranks, total_answers, n_cases):
    """cases_ranks: [[정답별 최소순위 or None], ...]  → K별 (answer%, case%)."""
    out = {}
    for K in Ks:
        a_hit = sum(1 for cr in cases_ranks for r in cr if r is not None and r <= K)
        c_hit = sum(1 for cr in cases_ranks if any(r is not None and r <= K for r in cr))
        out[K] = (a_hit / total_answers * 100, c_hit / n_cases * 100)
    return out


def cos(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else -1.0


def load(run):
    return json.load(open(f"experiments/{run}.json", encoding="utf-8"))


def p(path):
    return path.replace("\\", "/")


def method_from_ranking(run, use_pool=False):
    """rerank_file의 순위(HyDE/judge) 또는 pool 순서(baseline)로 K스윕."""
    d = load(run)
    cases_ranks = []
    tot = 0
    n_cases = 0
    for c in d["cases"]:
        rf = c.get("files", {}).get("rerank_file")
        if not rf:
            continue
        n_cases += 1
        answers = c["answer"]
        tot += len(answers)
        rr = json.load(open(p(rf), encoding="utf-8"))
        ranked = [x["name"] for x in sorted(
            rr["ranking"], key=lambda x: x["rank"] if x.get("rank") is not None else 10**9)]
        if use_pool:
            # 리랭크 없음 baseline = 같은 후보군을 '수집 순서'로. url_file 키가
            # collect_info가 places를 순회한 순서(=수집 순서)이자 동일 후보군.
            uf = c.get("files", {}).get("url_file")
            urls = json.load(open(p(uf), encoding="utf-8"))
            cand = set(ranked)
            names = [n for n in urls.keys() if n in cand]
        else:
            names = ranked
        cases_ranks.append([_rank_of(a, names) for a in answers])
    return recall_at_k(cases_ranks, tot, n_cases), tot, d["summary"], n_cases


async def method_plain_embed(run):
    """저장된 스니펫에 생쿼리를 임베딩해 쿼리-문서 코사인 순위로 K스윕."""
    from wh_poc.clients import embed_texts
    d = load(run)
    cases_ranks = []
    tot = 0
    n_cases = 0
    for c in d["cases"]:
        rf = c.get("files", {}).get("rerank_file")
        uf = c.get("files", {}).get("url_file")
        if not rf or not uf:
            continue
        n_cases += 1
        answers = c["answer"]
        tot += len(answers)
        query = c["question"]
        rr = json.load(open(p(rf), encoding="utf-8"))
        cand = [x["name"] for x in rr["ranking"]]
        urls = json.load(open(p(uf), encoding="utf-8"))
        snip_texts, owner = [], []
        for name in cand:
            for b in urls.get(name, []):
                snip_texts.append(f"{b['title']} {b['description']}")
                owner.append(name)
        if not snip_texts:
            cases_ranks.append([None] * len(answers))
            continue
        vecs = await embed_texts([query] + snip_texts)
        qv, svs = vecs[0], vecs[1:]
        score = {n: -1.0 for n in cand}
        for name, v in zip(owner, svs):
            score[name] = max(score[name], cos(qv, v))
        names = sorted(cand, key=lambda n: -score[n])
        cases_ranks.append([_rank_of(a, names) for a in answers])
    return recall_at_k(cases_ranks, tot, n_cases), tot, n_cases


def fmt(res):
    return "  ".join(f"@{K} {res[K][0]:5.1f}%/{res[K][1]:5.1f}%" for K in Ks)


async def main():
    print("형식: @K  정답단위%/케이스단위%   (I 런 후보군 공통, judge만 G 런)\n")

    base, totI, si, ncI = method_from_ranking("I_hyde_local", use_pool=True)
    print(f"① 가장 기본 (수집순서, 리랭크 없음)  [케이스 {ncI}개 / 정답 {totI}개]")
    print("   ", fmt(base))

    try:
        plain, _, _ = await method_plain_embed("I_hyde_local")
        print("② 쿼리-문서 임베딩 (HyDE 아님, 생쿼리 재계산)")
        print("   ", fmt(plain))
    except Exception as e:
        print("② 쿼리-문서 임베딩 — 임베딩 quota로 보류:", type(e).__name__)

    hyde, _, sI, _ = method_from_ranking("I_hyde_local", use_pool=False)
    print("③ HyDE 임베딩 + 필터 (I)")
    print("   ", fmt(hyde))

    judge, totG, sG, ncG = method_from_ranking("G_llm_judge", use_pool=False)
    print(f"④ LLM-as-judge (G)  [케이스 {ncG}개 / 정답 {totG}개]")
    print("   ", fmt(judge))

    print("\n── 토큰·시간 (전체 파이프라인, 각 런 summary) ──")
    print(f"  I(HyDE)  tok_in {sI['tokens_in']:,} / out {sI['tokens_out']:,} | 평균 {sI['avg_sec']:.1f}s/케이스")
    print(f"  G(judge) tok_in {sG['tokens_in']:,} / out {sG['tokens_out']:,} | 평균 {sG['avg_sec']:.1f}s/케이스")
    # C(HyDE 스니펫) 참고
    dC = load("C_hyde_snippet")["summary"]
    print(f"  C(HyDE스니펫) tok_in {dC['tokens_in']:,} / out {dC['tokens_out']:,} | 평균 {dC['avg_sec']:.1f}s/케이스")


asyncio.run(main())
