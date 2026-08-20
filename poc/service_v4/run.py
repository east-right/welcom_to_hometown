"""CLI 진입점.

    python run.py                          대화형 모드
    python run.py "신림역 백반집 추천해줘"      단발 질의
    python run.py -v "성수동 카공카페"         툴 내부 로그까지
    python run.py --trace "가평 캠핑장"        끝나고 타임라인 출력
"""

import argparse
import asyncio

from wh_poc import Chat, set_verbose


async def interactive(chat: Chat) -> None:
    print("대화를 시작합니다. 종료하려면 빈 줄이나 'quit'을 입력하세요.\n")
    while True:
        try:
            text = input("👤 ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in {"quit", "exit", "q"}:
            break
        await chat.say(text)
    if chat.history:
        print("\n--- 대화 요약 ---")
        chat.summary()


async def main() -> None:
    ap = argparse.ArgumentParser(description="장소 추천 에이전트")
    ap.add_argument("query", nargs="?", help="질의. 없으면 대화형 모드로 시작한다")
    ap.add_argument("-v", "--verbose", action="store_true", help="툴 내부 로그 출력")
    ap.add_argument("-t", "--tools", action="store_true", help="툴 호출 인자까지 출력")
    ap.add_argument("--trace", action="store_true", help="끝나고 턴별 타임라인 출력")
    args = ap.parse_args()

    set_verbose(args.verbose)
    chat = Chat(show_tools=args.tools)

    if args.query:
        await chat.say(args.query)
    else:
        await interactive(chat)

    if args.trace and chat.history:
        chat.trace()
        chat.totals()


if __name__ == "__main__":
    asyncio.run(main())
