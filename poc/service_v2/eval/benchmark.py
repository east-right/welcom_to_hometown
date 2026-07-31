"""테스트셋 기반 평가. 답변 품질(recall/precision) + 서빙 지표(TTFT/E2EL/ITL).

두 가지를 서로 다른 방식으로 잰다:
  - 답변 품질: 전체 플로우(understand→recommend)를 돌려 추천 카드 이름을
    정답 맛집 리스트와 대조. recall = 정답 중 맞힌 비율, precision = 추천 중 정답 비율.
  - 서빙 지표: vLLM OpenAI 엔드포인트에 '직접 스트리밍' 호출로 토큰 타이밍 측정.
    (ADK 파이프라인은 토큰 단위 시각을 안 내주므로 raw 스트리밍이 정확)
      TTFT = 첫 토큰까지 시간, E2EL = 전체 생성 시간, ITL = 토큰 간 평균 간격

사용:
  uv run python -m poc.service_v2.eval.benchmark quality        # recall/precision
  uv run python -m poc.service_v2.eval.benchmark serving        # TTFT/E2EL/ITL
  uv run python -m poc.service_v2.eval.benchmark all            # 둘 다
  # 테스트셋 경로 지정: ... all --testset data/test.json
"""
import argparse
import asyncio
import json
import re
import statistics
import sys
import time

import httpx

from ..config import MODEL, MODEL_API_BASE, MODEL_API_KEY
from ..domain.models import Slots
from ..main import build_recommender

_DEFAULT_TESTSET = "data/test.json"


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def load_testset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 답변 품질: 이름 매칭 ────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def _variants(answer: str) -> list[str]:
    """'등대집or 가포등대집', 'A/B', 'A, B' 같은 표기를 개별 후보로 쪼갠다."""
    parts = re.split(r"\s*(?:or|/|,|또는)\s*", answer, flags=re.IGNORECASE)
    return [v for p in parts if len(v := _norm(p)) >= 2]


def _same(a: str, b: str) -> bool:
    return a in b or b in a


def score(answers: list[str], recommended: list[str]) -> dict:
    """recall/precision 계산. 부분일치(공백 제거 후 포함관계)로 관대하게 매칭."""
    rec_norm = [_norm(r) for r in recommended]

    hit_answers = sum(
        any(_same(v, rn) for v in _variants(a) for rn in rec_norm)
        for a in answers
    )
    hit_cards = sum(
        any(_same(v, rn) for a in answers for v in _variants(a))
        for rn in rec_norm
    )
    recall = hit_answers / len(answers) if answers else 0.0
    precision = hit_cards / len(recommended) if recommended else 0.0
    return {"recall": recall, "precision": precision,
            "matched": hit_answers, "n_answer": len(answers), "n_reco": len(recommended)}


async def _recommend_names(rec, question: str) -> tuple[list[str], str]:
    """질문 → 슬롯 추출 → 추천. 추천 카드 이름 목록을 돌려준다."""
    ext = await rec.understand(f"User: {question}")
    slots = ext.slots
    if not slots.purpose:                 # 최소 목적은 있어야 검색 의미가 있음
        slots = Slots(**{**slots.model_dump(), "purpose": question})
    feed = await rec.recommend(slots)
    return [c.name for c in feed.cards], slots.summary()


async def quality(testset: list[dict]) -> None:
    rec = build_recommender()
    recalls, precisions = [], []
    print(f"[답변 품질] 모델={MODEL}  문항={len(testset)}\n")
    for i, item in enumerate(testset, 1):
        q, answers = item["question"], item["answer"]
        try:
            names, slot_sum = await _recommend_names(rec, q)
        except Exception as e:
            print(f"{i}. ❌ {q}\n   실패: {type(e).__name__} — {str(e)[:120]}\n")
            recalls.append(0.0); precisions.append(0.0)
            continue
        s = score(answers, names)
        recalls.append(s["recall"]); precisions.append(s["precision"])
        print(f"{i}. {q}")
        print(f"   추천: {names}")
        print(f"   정답: {answers}")
        print(f"   recall={s['recall']:.2f} ({s['matched']}/{s['n_answer']})  "
              f"precision={s['precision']:.2f}\n")
    print("── 요약 ──")
    print(f"  평균 recall    : {statistics.mean(recalls):.3f}")
    print(f"  평균 precision : {statistics.mean(precisions):.3f}")


# ── 서빙 지표: 직접 스트리밍 ─────────────────────────────

def _served_model() -> str | None:
    """vLLM에 보낼 모델 id. hosted_vllm/<id> → <id>. 네이티브(문자열)면 None."""
    return MODEL.split("/", 1)[1] if "/" in MODEL else None


async def measure_stream(client: httpx.AsyncClient, model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True, "temperature": 0, "max_tokens": 200,
    }
    headers = {"Authorization": f"Bearer {MODEL_API_KEY}"} if MODEL_API_KEY else {}

    t0 = time.perf_counter()
    ttft = None
    last = t0
    n = 0
    async with client.stream("POST", f"{MODEL_API_BASE}/chat/completions",
                             json=payload, headers=headers) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            delta = json.loads(data)["choices"][0]["delta"].get("content")
            if delta:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                n += 1
                last = now
    e2el = last - t0
    itl = (last - (t0 + ttft)) / (n - 1) if (ttft and n > 1) else 0.0
    return {"ttft": ttft or 0.0, "e2el": e2el, "itl": itl, "tokens": n}


async def serving(testset: list[dict]) -> None:
    model = _served_model()
    if not model or not MODEL_API_BASE:
        print(f"[서빙 지표] WH_MODEL_API_BASE 없음(네이티브 모델 {MODEL}) → 측정 생략")
        return
    print(f"[서빙 지표] 모델={model}  프롬프트={len(testset)}개 (직접 스트리밍)\n")
    ttfts, e2els, itls = [], [], []
    async with httpx.AsyncClient(timeout=180.0) as client:
        for i, item in enumerate(testset, 1):
            try:
                m = await measure_stream(client, model, item["question"])
            except httpx.HTTPError as e:
                print(f"{i}. ❌ 스트리밍 실패: {type(e).__name__}")
                continue
            ttfts.append(m["ttft"]); e2els.append(m["e2el"]); itls.append(m["itl"])
            print(f"{i}. TTFT={m['ttft']*1000:6.0f}ms  E2EL={m['e2el']:5.2f}s  "
                  f"ITL={m['itl']*1000:5.1f}ms  ({m['tokens']}토큰)")
    if ttfts:
        print("\n── 요약 (mean / p50) ──")
        print(f"  TTFT : {statistics.mean(ttfts)*1000:6.0f}ms / {statistics.median(ttfts)*1000:6.0f}ms")
        print(f"  E2EL : {statistics.mean(e2els):6.2f}s / {statistics.median(e2els):6.2f}s")
        print(f"  ITL  : {statistics.mean(itls)*1000:6.1f}ms / {statistics.median(itls)*1000:6.1f}ms")


def main() -> None:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="테스트셋 평가 (recall/precision + TTFT/E2EL/ITL)")
    ap.add_argument("cmd", choices=["quality", "serving", "all"])
    ap.add_argument("--testset", default=_DEFAULT_TESTSET)
    args = ap.parse_args()

    testset = load_testset(args.testset)
    if args.cmd in ("quality", "all"):
        asyncio.run(quality(testset))
    if args.cmd == "all":
        print()
    if args.cmd in ("serving", "all"):
        asyncio.run(serving(testset))


if __name__ == "__main__":
    main()
