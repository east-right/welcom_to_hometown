"""로컬/원격 vLLM 모델 연결·플로우 수동 테스트.

.env의 WH_MODEL / WH_MODEL_API_BASE / WH_MODEL_API_KEY를 그대로 써서
지금 붙어있는 모델로 각 단계를 하나씩 돌려본다.

사용:
  # 1) 엔드포인트 살아있나 + 어떤 모델이 떠 있나
  uv run python -m poc.service_v2.eval.local_model_test check

  # 2) 슬롯 추출만 (툴 없음, 빠름)
  uv run python -m poc.service_v2.eval.local_model_test slots "창원에 부모님이랑 주말에 맛집"

  # 3) 전체 추천 (coordinator 툴콜링 + 카드)
  uv run python -m poc.service_v2.eval.local_model_test recommend \
        --region 창원 --date 주말 --companion 부모님 --purpose 맛집

  # 4) 위 3개를 순서대로
  uv run python -m poc.service_v2.eval.local_model_test all "창원에 부모님이랑 주말에 맛집"
"""
import argparse
import asyncio
import sys

import httpx

from ..config import MODEL, MODEL_API_BASE, MODEL_API_KEY
from ..domain.models import Slots
from ..main import build_recommender


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def check() -> None:
    """엔드포인트 /models 호출 → 살아있나 + 모델 id 확인."""
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
            _warn_if_model_mismatch(ids)
        else:
            print(f"  ⚠️ 응답: {r.text[:200]}")
    except httpx.HTTPError as e:
        print(f"  ❌ 연결 실패: {type(e).__name__} — pod Logs / 포트 노출 확인")


def _warn_if_model_mismatch(served_ids: list) -> None:
    want = MODEL.split("/", 1)[1] if "/" in MODEL else MODEL  # hosted_vllm/<id> → <id>
    if served_ids and want not in served_ids:
        print(f"  ⚠️ WH_MODEL의 id('{want}')가 서빙 목록에 없음 → 404 날 수 있음")


async def _understand(query: str):
    rec = build_recommender()
    ext = await rec.understand(f"User: {query}")
    print(f"  ready   : {ext.ready}")
    print(f"  slots   : {ext.slots.model_dump()}")
    if ext.question:
        print(f"  question: {ext.question}")
    return ext


def slots(query: str) -> None:
    print(f"[슬롯 추출] \"{query}\"")
    asyncio.run(_understand(query))


def recommend(s: Slots) -> None:
    print(f"[전체 추천] {s.summary()}")
    rec = build_recommender()
    try:
        feed = asyncio.run(rec.recommend(s))
    except Exception as e:  # 컨텍스트 초과·툴콜링 실패 등을 깔끔히 보여줌
        print(f"  ❌ 실패: {type(e).__name__} — {str(e)[:250]}")
        return
    print(f"  카드 {len(feed.cards)}개:")
    for c in feed.cards:
        meta = c.period or " / ".join(x for x in (c.hours, c.closed) if x)
        print(f"   - [{c.domain}] {c.name} | {meta} | {c.source}")


def main() -> None:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="vLLM 모델 연결·플로우 수동 테스트")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="엔드포인트/모델 확인")

    p_slots = sub.add_parser("slots", help="슬롯 추출만")
    p_slots.add_argument("query", help="사용자 발화 (한 문장에 지역/일자/동반/목적 다 넣으면 좋음)")

    p_rec = sub.add_parser("recommend", help="전체 추천")
    p_rec.add_argument("--region", default="창원")
    p_rec.add_argument("--date", default="주말")
    p_rec.add_argument("--companion", default="부모님")
    p_rec.add_argument("--purpose", default="맛집")

    p_all = sub.add_parser("all", help="check → slots → recommend")
    p_all.add_argument("query", nargs="?", default="창원에 부모님이랑 주말에 맛집")

    args = ap.parse_args()

    if args.cmd == "check":
        check()
    elif args.cmd == "slots":
        slots(args.query)
    elif args.cmd == "recommend":
        recommend(Slots(region=args.region, date=args.date,
                        companion=args.companion, purpose=args.purpose))
    elif args.cmd == "all":
        check()
        print()
        ext = asyncio.run(_understand(args.query))
        print()
        if ext.slots.is_full():
            recommend(ext.slots)
        else:
            print("  (슬롯이 덜 차서 recommend 생략 — 한 문장에 지역/일자/동반/목적 다 넣어봐)")


if __name__ == "__main__":
    main()
