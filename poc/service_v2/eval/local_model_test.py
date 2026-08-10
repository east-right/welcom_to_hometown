"""로컬/원격 vLLM 모델 연결·추천 수동 테스트.

.env의 WH_MODEL / WH_MODEL_API_BASE / WH_MODEL_API_KEY를 그대로 써서
지금 붙어있는 모델로 단발 추천(search_agent)을 돌려본다.

사용:
  # 1) (vLLM일 때) 엔드포인트 살아있나 + 어떤 모델이 떠 있나
  uv run python -m poc.service_v2.eval.local_model_test check

  # 2) 질문 하나 → search_agent 단발 추천(카드)
  uv run python -m poc.service_v2.eval.local_model_test ask "다음주에 창원 귀산 오션뷰 카페"
"""
import argparse
import asyncio
import sys

import httpx

from ..config import MODEL, MODEL_API_BASE, MODEL_API_KEY
from ..main import build_search


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def check() -> None:
    """엔드포인트 /models 호출 → 살아있나 + 모델 id 확인(vLLM일 때만 의미)."""
    print(f"[모델] {MODEL}")
    if not MODEL_API_BASE:
        print("  (WH_MODEL_API_BASE 없음 → ADK 네이티브 모델. 엔드포인트 체크 생략)")
        return
    print(f"[엔드포인트] {MODEL_API_BASE}/models")
    headers = {"Authorization": f"Bearer {MODEL_API_KEY}"} if MODEL_API_KEY else {}
    try:
        r = httpx.get(f"{MODEL_API_BASE}/models", headers=headers, timeout=15.0)
        print(f"  HTTP {r.status_code}")
        if r.status_code == 200:
            ids = [m.get("id") for m in r.json().get("data", [])]
            print(f"  ✅ 서빙 중 모델: {ids}")
        else:
            print(f"  ⚠️ 응답: {r.text[:200]}")
    except httpx.HTTPError as e:
        print(f"  ❌ 연결 실패: {type(e).__name__} — pod Logs / 포트 노출 확인")


async def _ask(question: str) -> None:
    feed = await build_search().find(question)
    print(f"  카드 {len(feed.cards)}개:")
    for c in feed.cards:
        meta = c.period or " / ".join(x for x in (c.hours, c.closed) if x)
        print(f"   - [{c.domain}] {c.name} | {meta} | {c.source}")


def ask(question: str) -> None:
    print(f"[추천] \"{question}\"")
    try:
        asyncio.run(_ask(question))
    except Exception as e:
        print(f"  ❌ 실패: {type(e).__name__} — {str(e)[:250]}")


def main() -> None:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="모델 연결·단발 추천 수동 테스트")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="엔드포인트/모델 확인(vLLM)")
    p_ask = sub.add_parser("ask", help="질문 하나 → 추천 카드")
    p_ask.add_argument("question")

    args = ap.parse_args()
    if args.cmd == "check":
        check()
    elif args.cmd == "ask":
        ask(args.question)


if __name__ == "__main__":
    main()
