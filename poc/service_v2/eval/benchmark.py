"""테스트셋 실행. recall + 추천 결과 + 실행별 로그(flow/agents) + 서버측 시간지표.

precision(정확도)은 추천 10개 중 매칭이라 원래 낮게 나와 무의미해 빼고, recall(정답 중
몇 개를 담았나)만 잰다.
  - report : test.json 전체를 파이프라인에 태워 recall·추천 결과 저장 + flow/agents 로그.
  - serving: vLLM OpenAI 엔드포인트에 '직접 스트리밍' 호출로 토큰 타이밍(vLLM일 때만).
      TTFT = 첫 토큰까지 시간, E2EL = 전체 생성 시간, ITL = 토큰 간 평균 간격

사용:
  uv run python -m poc.service_v2.eval.benchmark report
  uv run python -m poc.service_v2.eval.benchmark serving
  uv run python -m poc.service_v2.eval.benchmark all
  # 테스트셋 경로 지정: ... report --testset data/test.json
"""
import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime

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


# ── recall: 정답 매장 중 몇 개를 추천에 담았나 (precision은 안 잰다) ──

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def _variants(answer: str) -> list[str]:
    """'등대집or 가포등대집', 'A/B', 'A, B' 같은 표기를 개별 후보로 쪼갠다."""
    parts = re.split(r"\s*(?:or|/|,|또는)\s*", answer, flags=re.IGNORECASE)
    return [v for p in parts if len(v := _norm(p)) >= 2]


def recall_of(answers: list[str], recommended: list[str]) -> dict:
    """정답 중 추천에 담긴 비율. 공백 제거 후 포함관계로 관대하게 매칭."""
    rec_norm = [_norm(r) for r in recommended]
    hit = sum(
        any(v in rn or rn in v for v in _variants(a) for rn in rec_norm)
        for a in answers
    )
    return {"recall": hit / len(answers) if answers else 0.0,
            "matched": hit, "n_answer": len(answers)}


async def _recommend_names(rec, question: str) -> tuple[list[str], str]:
    """질문 → 슬롯 추출 → 추천. 추천 카드 이름 목록을 돌려준다."""
    ext = await rec.understand(f"User: {question}")
    slots = ext.slots
    if not slots.purpose:                 # 최소 목적은 있어야 검색 의미가 있음
        slots = Slots(**{**slots.model_dump(), "purpose": question})
    feed = await rec.recommend(slots)
    return [c.name for c in feed.cards], slots.summary()


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


def _new_records(path: str, before_size: int) -> list[dict]:
    """이번 실행이 runs.jsonl에 새로 남긴 레코드만 읽는다."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        f.seek(before_size)
        return [json.loads(line) for line in f if line.strip()]


async def report(testset: list[dict]) -> None:
    """test.json 전체를 파이프라인에 태워 recall + 추천 결과 + 서버측 시간지표를 저장.

    precision은 빼고 recall만 잰다. 추천 결과 JSON과 flow/agents 로그를 같은
    이름(report_<model>_<ts>)으로 짝지어 실행별로 분리 저장한다.
    """
    runs = "logs/runs.jsonl"
    before = os.path.getsize(runs) if os.path.exists(runs) else 0

    stem = f"report_{MODEL.replace('/', '_')}_{datetime.now():%Y%m%d_%H%M%S}"
    rec = build_recommender(log_name=stem)   # 이 실행 전용 flow/agents 로그
    per_q = []
    print(f"[리포트] 모델={MODEL}  문항={len(testset)}\n")
    for i, item in enumerate(testset, 1):
        q, answers = item["question"], item.get("answer", [])
        try:
            names, _ = await _recommend_names(rec, q)
            s = recall_of(answers, names)
        except Exception as e:
            names, s = [], {"recall": 0.0, "matched": 0, "n_answer": len(answers)}
            print(f"{i}. ❌ {q} — {type(e).__name__}")
        else:
            print(f"{i}. {q}  recall={s['recall']:.2f} ({s['matched']}/{s['n_answer']})")
            print(f"   정답: {answers}")
            print(f"   추천: {names}")
        per_q.append({"question": q, "answer": answers, "recommended": names, **s})

    # 이번 실행이 남긴 서버측 지표(model_sec/ttft/itl = 통신시간 제외)
    steps = _new_records(runs, before)
    model_secs = [r["model_sec"] for r in steps if r.get("model_sec") is not None]
    ttfts = [r["ttft"] for r in steps if r.get("ttft") is not None]
    itls = [r["itl"] for r in steps if r.get("itl") is not None]

    summary = {
        "model": MODEL,
        "questions": len(testset),
        "recall": round(statistics.mean(x["recall"] for x in per_q), 3),
        "steps": len(steps),
        "model_sec_mean": round(statistics.mean(model_secs), 3) if model_secs else None,
        "ttft_mean": round(statistics.mean(ttfts), 3) if ttfts else None,
        "itl_mean": round(statistics.mean(itls), 4) if itls else None,
    }
    print("\n── 리포트 요약 (시간지표는 서버측=통신 제외) ──")
    for key, value in summary.items():
        print(f"  {key:16}: {value}")
    if summary["model_sec_mean"] is None:
        print("  ⚠️ 서버측 지표 비어있음 — vLLM /metrics 이름이 버전과 다를 수 있음 (metrics.py 후보 확인)")

    out = f"logs/{stem}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_question": per_q}, f, ensure_ascii=False, indent=2)
    print(f"\n  → 리포트: {out}")
    print(f"  → flow  : logs/{stem}.flow.log")
    print(f"  → agents: logs/{stem}.agents.log")


def main() -> None:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="테스트셋 실행(추천 결과·로그) + 서버측 시간지표")
    ap.add_argument("cmd", choices=["report", "serving", "all"])
    ap.add_argument("--testset", default=_DEFAULT_TESTSET)
    args = ap.parse_args()

    testset = load_testset(args.testset)
    if args.cmd in ("report", "all"):
        asyncio.run(report(testset))
    if args.cmd == "all":
        print()
    if args.cmd in ("serving", "all"):
        asyncio.run(serving(testset))


if __name__ == "__main__":
    main()
