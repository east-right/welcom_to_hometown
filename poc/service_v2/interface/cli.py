"""대화형 CLI. 질문을 받아 search_agent가 만든 추천 카드를 보여준다."""
from ..config import check_keys
from . import view

_QUIT = {"quit", "exit", "종료"}


async def run(search) -> None:
    check_keys()
    print(view.BANNER)

    while True:
        user = input("\n🙋 나: ").strip()
        if not user:
            continue
        if user.lower() in _QUIT:
            print("👋 안녕히 가세요!")
            break

        print("🤖 찾아볼게요...\n")
        try:
            feed = await search.find(user)
        except Exception as e:
            print(f"  (처리 실패: {e}) 다시 말씀해 주세요.")
            continue
        print(view.feed(feed))
