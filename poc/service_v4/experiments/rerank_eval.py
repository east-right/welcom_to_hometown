"""리랭크 방법 3종 head-to-head eval (동일 후보군·동일 스니펫, 방법만 교체).

  A. 본문 HyDE   : HyDE문서 ↔ 크롤 본문 임베딩 코사인
  B. 스니펫 HyDE : HyDE문서 ↔ 스니펫(제목+요약) 임베딩 코사인
  C. 스니펫 judge: 스니펫을 통째로 LLM에 주고 순위 배열 반환

공정성: 후보군·스니펫은 과거 I 런 산출물(collect_places + info_urls)에서 고정으로
가져온다. HyDE 문서는 케이스당 1회 생성해 A·B가 공유. 본문만 그 스니펫 URL을 새로 크롤.

지표: recall@10/15/20/25 (정답단위/케이스단위), 방법별 리랭크 토큰, 방법별 리랭크 시간.
  python experiments/rerank_eval.py
"""
import asyncio
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from google.genai import types

from wh_poc import prompts
from wh_poc.clients import _client, UTIL_MODEL, FAIL_PREFIX, embed_texts, scrape_urls

Ks = [10, 15, 20, 25]
RERANK_KEEP = 15
BODY_CLIP = 2000          # 본문 임베딩 입력 클립(임베딩 토큰 상한 대비)
SNIP_PER = 2              # judge 블록/본문 크롤에 쓸 매장당 스니펫 수
SRC = "experiments/I_hyde_local.json"

JUDGE_RERANK = """\
사용자 요청: "{user_query}"

아래 후보 매장 목록에서 사용자 요청에 가장 잘 맞는 곳을 최대 {k}곳 골라
**적합한 순서대로** 번호만 배열로 출력하라. 각 항목은 '[번호] 상호명 | 카테고리'와
그 매장의 블로그 후기 제목 몇 개다.

판단 기준:
- 요청의 업종·조건(지역·상황·동반자·분위기 등)에 실제로 부합하는가.
- 후기 제목에서 실제 방문 후기가 붙은 진짜 로컬 가게로 보이는가.
- 막연한 '맛집' 요청이면 그 지역에서 후기가 실히 쌓인 로컬 맛집을 우선한다.
- 프랜차이즈·체인·요청과 무관한 업종은 뒤로 미룬다.

번호 배열 JSON만 출력하라. 예: [12, 3, 45, 7]

후보:
{candidates}
"""

# ── 매칭 (eval.py 규칙) ─────────────────────────────────
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


def _rank_of(answer, names):
    variants = []
    for v in [p.strip() for p in _ALIAS_SEP.split(answer) if p.strip()]:
        variants += [_norm(v), _core(v)]
    variants = [v for v in variants if len(v) >= 2]
    for i, c in enumerate(names):
        forms = {_norm(c), _core(c)}
        for v in variants:
            for f in forms:
                if v in f or f in v:
                    return i + 1
    return None


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else -1.0


def approx_tokens(texts):
    """임베딩 입력 토큰 근사(한글 ~2.5자/토큰)."""
    return int(sum(len(t) for t in texts) / 2.5)


async def gen_capture(prompt, json_mode=False):
    cfg = types.GenerateContentConfig(temperature=0.0)
    if json_mode:
        cfg.response_mime_type = "application/json"
    resp = await _client.aio.models.generate_content(
        model=UTIL_MODEL, contents=prompt, config=cfg
    )
    u = resp.usage_metadata
    return resp.text, (u.prompt_token_count or 0) + (u.candidates_token_count or 0)


async def embed_retry(texts, tries=5):
    for k in range(tries):
        try:
            return await embed_texts(texts)
        except Exception as e:
            if "429" in str(e) and k < tries - 1:
                await asyncio.sleep(20 * (k + 1))
                continue
            raise


def order_by_score(cand, score):
    return sorted(cand, key=lambda n: -score.get(n, -1.0))


def p(path):
    return path.replace("\\", "/")


async def run():
    data = json.load(open(SRC, encoding="utf-8"))
    cases = [c for c in data["cases"]
             if c.get("files", {}).get("rerank_file") and c.get("files", {}).get("url_file")]
    if len(sys.argv) > 1:
        cases = cases[: int(sys.argv[1])]

    # 방법별 누적
    ranks = {m: [] for m in "ABC"}          # [[정답별 최소순위]]
    tok = {m: 0 for m in "ABC"}
    sec = {m: 0.0 for m in "ABC"}
    tot_ans = 0

    for idx, c in enumerate(cases, 1):
        q, answers = c["question"], c["answer"]
        tot_ans += len(answers)
        rr = json.load(open(p(c["files"]["rerank_file"]), encoding="utf-8"))
        cand = [x["name"] for x in rr["ranking"]]
        urls = json.load(open(p(c["files"]["url_file"]), encoding="utf-8"))
        places = json.load(open(p(c["files"]["file"]), encoding="utf-8"))
        cat = {pl["name"]: pl.get("category", "") for pl in places}

        # HyDE 문서 1회 생성 (A·B 공유)
        hypo, hgtok = await gen_capture(prompts.HYDE_REVIEW.format(user_query=q))
        hypo = hypo.strip() or q

        # ── B. 스니펫 HyDE ──
        t0 = time.perf_counter()
        s_texts, s_owner = [], []
        for n in cand:
            for b in urls.get(n, [])[:SNIP_PER]:
                s_texts.append(f"{b['title']} {b['description']}")
                s_owner.append(n)
        vecs = await embed_retry([hypo] + s_texts)
        hv, svs = vecs[0], vecs[1:]
        scB = {n: -1.0 for n in cand}
        for n, v in zip(s_owner, svs):
            scB[n] = max(scB[n], cos(hv, v))
        namesB = order_by_score(cand, scB)
        sec["B"] += time.perf_counter() - t0
        tok["B"] += hgtok + approx_tokens([hypo] + s_texts)

        # ── A. 본문 HyDE ── (스니펫 URL을 새로 크롤)
        t0 = time.perf_counter()
        links, l_owner = [], []
        for n in cand:
            for b in urls.get(n, [])[:SNIP_PER]:
                links.append(b["link"])
                l_owner.append(n)
        scraped = await scrape_urls(links, max_chars=BODY_CLIP)
        b_texts, b_owner = [], []
        for n, s in zip(l_owner, scraped):
            txt = s["text"]
            if txt.startswith(FAIL_PREFIX):
                continue
            b_texts.append(txt[:BODY_CLIP])
            b_owner.append(n)
        scA = {n: -1.0 for n in cand}
        if b_texts:
            vecs = await embed_retry([hypo] + b_texts)
            hv, bvs = vecs[0], vecs[1:]
            for n, v in zip(b_owner, bvs):
                scA[n] = max(scA[n], cos(hv, v))
        namesA = order_by_score(cand, scA)
        sec["A"] += time.perf_counter() - t0
        tok["A"] += hgtok + approx_tokens([hypo] + b_texts)

        # ── C. 스니펫 judge ──
        t0 = time.perf_counter()
        blocks = []
        for i, n in enumerate(cand):
            titles = "; ".join(b["title"] for b in urls.get(n, [])[:SNIP_PER]) or "(후기 없음)"
            blocks.append(f"[{i}] {n} | {cat.get(n, '')}\n     {titles}")
        raw, jtok = await gen_capture(
            JUDGE_RERANK.format(user_query=q, k=RERANK_KEEP, candidates="\n".join(blocks)),
            json_mode=True,
        )
        try:
            order = [i for i in json.loads(raw) if isinstance(i, int) and 0 <= i < len(cand)]
        except (json.JSONDecodeError, TypeError):
            order = []
        seen, namesC = set(), []
        for i in order:
            if i not in seen:
                seen.add(i)
                namesC.append(cand[i])
        for n in cand:                      # 남은 후보는 원래(수집) 순서로 채움
            if n not in namesC:
                namesC.append(n)
        sec["C"] += time.perf_counter() - t0
        tok["C"] += jtok

        for m, names in (("A", namesA), ("B", namesB), ("C", namesC)):
            ranks[m].append([_rank_of(a, names) for a in answers])

        print(f"[{idx}/{len(cases)}] {q[:24]}  본문크롤 {len(b_texts)}/{len(links)}  후보 {len(cand)}")

    return cases, ranks, tok, sec, tot_ans


def recall_table(cases_ranks, tot, ncase):
    out = {}
    for K in Ks:
        a = sum(1 for cr in cases_ranks for r in cr if r is not None and r <= K)
        cc = sum(1 for cr in cases_ranks if any(r is not None and r <= K for r in cr))
        out[K] = (a / tot * 100, cc / ncase * 100)
    return out


async def main():
    cases, ranks, tok, sec, tot = await run()
    nc = len(cases)
    names = {"A": "본문 HyDE", "B": "스니펫 HyDE", "C": "스니펫 judge"}
    print(f"\n{'='*70}\n리랭크 방법 비교  [케이스 {nc} / 정답 {tot}]  (동일 후보군, I 런 고정)\n")
    print(f"{'방법':<14}{'@10':>14}{'@15':>14}{'@20':>14}{'@25':>14}")
    tables = {}
    for m in "ABC":
        t = recall_table(ranks[m], tot, nc)
        tables[m] = t
        row = "".join(f"{t[K][0]:5.1f}/{t[K][1]:5.1f}".rjust(14) for K in Ks)
        print(f"{names[m]:<14}{row}")
    print("\n(정답단위% / 케이스단위%)\n")
    print(f"{'방법':<14}{'리랭크토큰':>14}{'리랭크시간(합)':>16}{'평균/케이스':>14}")
    for m in "ABC":
        print(f"{names[m]:<14}{tok[m]:>14,}{sec[m]:>14.1f}s{sec[m]/nc:>13.2f}s")

    json.dump(
        {"n_case": nc, "n_answer": tot,
         "recall": {m: {str(K): tables[m][K] for K in Ks} for m in "ABC"},
         "tokens": tok, "sec_total": sec, "sec_per_case": {m: sec[m] / nc for m in "ABC"}},
        open("experiments/rerank_eval_out.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=2,
    )
    print("\n저장: experiments/rerank_eval_out.json")


asyncio.run(main())
