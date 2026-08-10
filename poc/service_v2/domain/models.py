"""도메인 모델. 데이터와 그 규칙만 담는다. 외부 라이브러리·I/O는 모른다."""
from typing import Literal

from pydantic import BaseModel, Field

Domain = Literal["restaurant", "cafe", "festival", "spot"]


class KakaoQuery(BaseModel):
    """카카오 지역검색용 '짧은' 키워드 쿼리. 긴 서술형은 결과가 0이라 쪼개 넣는다."""

    query: str = Field(description="짧은 키워드(2~4단어). 지역명 포함. 예 '창원 귀산 오션뷰 카페'")
    category: str = Field("", description="'' | FD6(음식점) | CE7(카페) | AT4(관광명소)")


class QueryPlan(BaseModel):
    """서술형 질문을 카카오가 먹는 짧은 쿼리 여러 개로 분해한 결과."""

    queries: list[KakaoQuery] = Field(description="서로 다른 짧은 검색어 3~6개")


class Review(BaseModel):
    """한 장소의 블로그 리뷰 요약(살). 질문과 연관되게 서브모델이 3줄로 뽑는다."""

    reason: str = ""     # 질문 연관 3줄 요약(추천 이유)
    hours: str = ""      # 운영시간
    closed: str = ""     # 휴무일
    period: str = ""     # 축제·행사 기간
    ad: bool = False     # 광고/협찬 여부
    source: str = ""     # 근거 URL


class Place(BaseModel):
    """카카오 로컬이 준 '검증된 장소' 엔티티(상호·주소·좌표). 환각 없음."""

    id: str = ""
    name: str = ""
    category: str = ""
    address: str = ""
    phone: str = ""
    url: str = ""
    x: float | None = None   # 경도
    y: float | None = None   # 위도

    @classmethod
    def from_kakao(cls, doc: dict) -> "Place":
        return cls(
            id=doc.get("id", ""),
            name=doc.get("place_name", ""),
            category=doc.get("category_name", ""),
            address=doc.get("road_address_name") or doc.get("address_name", ""),
            phone=doc.get("phone", ""),
            url=doc.get("place_url", ""),
            x=float(doc["x"]) if doc.get("x") else None,
            y=float(doc["y"]) if doc.get("y") else None,
        )


class Card(BaseModel):
    """최종 추천 카드. 도메인별로 채우는 시간 필드가 다르다."""

    domain: Domain
    name: str
    reason: str
    period: str = ""     # festival만
    hours: str = ""      # 매장/명소
    closed: str = ""     # 매장/명소
    source: str


class Feed(BaseModel):
    cards: list[Card] = Field(default_factory=list)
