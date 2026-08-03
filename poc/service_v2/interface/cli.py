"""대화형 CLI. 입력 루프와 한 턴의 흐름 제어만 담당한다.

추천 로직은 Recommender에, 출력 형태는 view에 위임한다.
"""
from ..config import check_keys
from ..domain.models import Slots
from . import view

_QUIT = {"quit", "exit", "종료"}
_RESET = {"reset", "처음", "다시"}


async def run(recommender) -> None:
    """설정을 확인하고 대화 루프를 돈다."""
    check_keys()
    print(view.BANNER)

    chat = Chat(recommender)
    while True:
        user = input("\n🙋 나: ").strip()
        if not user:
            continue
        if user.lower() in _QUIT:
            print("👋 안녕히 가세요!")
            break
        if user.lower() in _RESET:
            chat.reset()
            print("🔄 처음부터 시작할게요.")
            continue

        try:
            await chat.turn(user)
        except Exception as e:
            print(f"  (처리 실패: {e}) 다시 말씀해 주세요.")


class Chat:
    """멀티턴 대화 상태(히스토리·슬롯)와 한 턴 처리."""

    def __init__(self, recommender) -> None:
        self._recommender = recommender
        self._history: list[str] = []
        self._slots = Slots()

    async def turn(self, user: str) -> None:
        self._history.append(f"User: {user}")

        extraction = await self._recommender.understand("\n".join(self._history))
        self._slots = extraction.slots
        print(view.slot_summary(self._slots))

        if extraction.ready:
            await self._present_recommendations()
        else:
            self._ask(extraction.question)

    def reset(self) -> None:
        self._history = []
        self._slots = Slots()

    def _ask(self, question: str) -> None:
        text = question or "조금만 더 알려주실 수 있을까요?"
        print(f"🤖 봇: {text}")
        self._history.append(f"Assistant: {text}")

    async def _present_recommendations(self) -> None:
        print("🤖 봇: 조건이 다 모였어요. 찾아볼게요!\n")
        feed = await self._recommender.recommend(self._slots)
        print(view.feed(feed))
        print("\n  다른 조건으로 더 찾거나(예: '카페도 알려줘'), "
              "reset으로 새로 시작할 수 있어요.")
        self._history.append(f"Assistant: [{len(feed.cards)}개 추천 완료]")
