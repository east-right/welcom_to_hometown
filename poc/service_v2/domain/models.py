"""도메인 모델. 수집할 정보(슬롯)와 결과(카드).

이 계층은 외부 라이브러리 연동·I/O 없이 데이터와 그 규칙만 담는다.
"""
from pydantic import BaseModel, Field


class Slots(BaseModel):
    """멀티턴에서 채워야 하는 기본 정보. 빈 값이면 아직 안 받은 것."""

    region: str = Field(default="", description="지역. 예 '창원', '창원 근교'")
    date: str = Field(default="", description="방문 일자/기간. 예 '7월 25~27일', '이번 주말'")
    companion: str = Field(default="", description="동반자. 예 '부모님', '친구', '연인', '혼자'")
    purpose: str = Field(default="", description="목적/관심사. 예 '축제 구경', '맛집', '카페', '나들이'")

    def missing(self) -> list[str]:
        """아직 안 채워진 슬롯 이름 목록."""
        return [name for name, value in self._fields() if not value]

    def is_full(self) -> bool:
        return not self.missing()

    def summary(self) -> str:
        return " / ".join(
            f"{label}={value or '?'}"
            for label, value in (
                ("지역", self.region),
                ("일자", self.date),
                ("동반", self.companion),
                ("목적", self.purpose),
            )
        )

    def _fields(self) -> tuple[tuple[str, str], ...]:
        return (
            ("region", self.region),
            ("date", self.date),
            ("companion", self.companion),
            ("purpose", self.purpose),
        )


class SlotExtraction(BaseModel):
    """사용자 발화에서 슬롯을 추출하고, 부족하면 되물을 질문을 담는다."""

    slots: Slots = Field(description="현재까지 파악된 슬롯. 못 채운 건 빈 문자열")
    ready: bool = Field(description="검색을 시작해도 될 만큼 정보가 충분한가")
    question: str = Field(
        default="",
        description="아직 부족하면 사용자에게 물어볼 질문 한 개. 충분하면 빈 문자열",
    )


class Card(BaseModel):
    """추천 카드. 도메인별로 채우는 필드가 다르다."""

    domain: str = Field(description="restaurant/cafe/festival/spot 중 하나")
    name: str = Field(description="장소/행사 이름")
    reason: str = Field(description="추천 이유 한 줄")
    period: str = Field(default="", description="행사 기간. festival만. 확인 못하면 빈 문자열")
    hours: str = Field(default="", description="운영시간. 매장/장소만. 확인 못하면 빈 문자열")
    closed: str = Field(default="", description="휴무일. 매장/장소만. 확인 못하면 빈 문자열")
    source: str = Field(description="근거 URL")


class Feed(BaseModel):
    cards: list[Card] = Field(description="추천 카드 목록 (최대 10개, 우선순위 순)")
