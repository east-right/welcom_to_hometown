"""CLI 대화 진입점.

실행:  python main.py

흐름:
  1. 사용자 입력 → 슬롯 추출
  2. 부족하면 되물음 (슬롯 필링)
  3. 지역·일자·동반자·목적이 다 차면 → 검색 → 카드 출력
  4. 이어서 대화 계속 (추가 요청 / 새 검색)
"""
import asyncio

from config import check_keys
from engine import extract_slots, render_cards, search_and_format
from schema import Slots

BANNER = """
╔══════════════════════════════════════════════╗
║   Welcome Come to Hometown                   ║
║   어디로, 언제, 누구랑 갈지 편하게 말해주세요   ║
║   (종료: quit / 처음부터: reset)              ║
╚══════════════════════════════════════════════╝
"""


async def conversation() -> None:
    check_keys()
    print(BANNER)

    history: list[str] = []   # "User: ..." / "Assistant: ..." 누적
    slots = Slots()

    while True:
        user = input("\n🙋 나: ").strip()
        if not user:
            continue
        if user.lower() in ("quit", "exit", "종료"):
            print("👋 안녕히 가세요!")
            break
        if user.lower() in ("reset", "처음", "다시"):
            history, slots = [], Slots()
            print("🔄 처음부터 시작할게요.")
            continue

        history.append(f"User: {user}")

        # ── 슬롯 추출 (지금까지 대화 전체를 넘김) ──
        convo = "\n".join(history)
        try:
            ex = await extract_slots(convo)
        except Exception as e:
            print(f"  (슬롯 파악 실패: {e}) 다시 말씀해 주세요.")
            continue

        slots = ex.slots
        print(f"  📋 파악: {slots.summary()}")

        # ── 아직 정보가 부족하면 되물음 ──
        if not ex.ready:
            q = ex.question or "조금만 더 알려주실 수 있을까요?"
            print(f"🤖 봇: {q}")
            history.append(f"Assistant: {q}")
            continue

        # ── 정보 충분 → 검색 실행 ──
        print("🤖 봇: 조건이 다 모였어요. 찾아볼게요!\n")
        try:
            feed = await search_and_format(slots, verbose=True)
        except Exception as e:
            print(f"  (검색 실패: {e})")
            continue

        print("\n" + "─" * 50)
        print(render_cards(feed))
        print("─" * 50)

        result_note = f"[{len(feed.cards)}개 추천 완료]"
        history.append(f"Assistant: {result_note}")
        print("\n  다른 조건으로 더 찾거나(예: '카페도 알려줘'), "
              "reset으로 새로 시작할 수 있어요.")


if __name__ == "__main__":
    try:
        asyncio.run(conversation())
    except (KeyboardInterrupt, EOFError):
        print("\n👋 종료합니다.")
