"""사용자에게 보여줄 문자열 구성. 출력 형태는 여기서만 다룬다."""
from ..domain.models import Card, Feed, Slots

BANNER = """
╔══════════════════════════════════════════════╗
║   Welcome Come to Hometown                   ║
║   어디로, 언제, 누구랑 갈지 편하게 말해주세요   ║
║   (종료: quit / 처음부터: reset)              ║
╚══════════════════════════════════════════════╝
"""

DIVIDER = "─" * 50


def slot_summary(slots: Slots) -> str:
    return f"  📋 파악: {slots.summary()}"


def feed(feed: Feed) -> str:
    """추천 피드를 구분선으로 감싼 카드 목록 문자열로 만든다."""
    if not feed.cards:
        body = "  조건에 맞는 결과를 찾지 못했어요."
    else:
        body = "\n".join(_card(i, c) for i, c in enumerate(feed.cards, 1))
    return f"\n{DIVIDER}\n{body}\n{DIVIDER}"


def _card(index: int, card: Card) -> str:
    return "\n".join([_card_head(index, card), f"     {card.reason}", f"     {card.source}"])


def _card_head(index: int, card: Card) -> str:
    head = f"  {index}. [{card.domain}] {card.name}"
    for icon, value in (("📅", card.period), ("🕒", card.hours), ("🚫", card.closed)):
        if value:
            head += f"  {icon} {value}"
    return head
