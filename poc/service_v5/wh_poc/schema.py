"""에이전트 응답 스키마.

모든 응답이 같은 껍데기를 쓴다. kind로 종류를 구분하고, 프론트엔드는 kind를 보고
카드를 띄울지 말풍선만 띄울지 결정한다.
"""

from typing import List

from pydantic import BaseModel, Field


class PlaceCard(BaseModel):
    name: str = Field(description="장소명")
    score: int = Field(default=0, description="요청 적합도 점수 0~100. 높을수록 잘 맞음")
    address: str = Field(description="주소. 파일에 있는 값을 그대로 쓴다")
    category: str = Field(default="", description="카테고리")
    reason: str = Field(description="이 요청에 왜 맞는지 2~3문장")
    highlights: List[str] = Field(description="핵심 강점 2~3개. 짧은 구문으로")
    caution: str = Field(default="", description="주의사항. 없으면 빈 문자열")
    period: str = Field(default="", description="행사 기간. 해당 없으면 빈 문자열")
    url: str = Field(default="", description="카카오맵 상세 페이지 URL")


class FeedOutput(BaseModel):
    kind: str = Field(
        description=(
            "응답 종류. "
            "'recommendation'=추천 결과가 있음(places를 채운다) / "
            "'question'=정보가 부족해 사용자에게 되물음 / "
            "'chat'=인사·안내·일반 대화, 또는 이미 보여준 결과에 대한 후속 답변"
        )
    )
    message: str = Field(
        description=(
            "사용자에게 말풍선으로 보여줄 문장. "
            "kind가 'recommendation'이면 두 문장 이내의 짧은 요약만 쓴다. "
            "각 장소의 설명은 places에 이미 들어가므로 여기에 목록을 나열하지 마라. "
            "'question'이면 물어볼 내용, 'chat'이면 답변 전문을 쓴다."
        )
    )
    places: List[PlaceCard] = Field(
        default=[],
        description="추천 장소 5~8곳. kind가 'recommendation'일 때만 채우고 그 외에는 빈 배열.",
    )
