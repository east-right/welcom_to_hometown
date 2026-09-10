"""에이전트 응답 스키마.

모든 응답이 같은 껍데기를 쓴다. kind로 종류를 구분하고, 프론트엔드는 kind를 보고
카드를 띄울지 말풍선만 띄울지 결정한다.
"""

from typing import List

from pydantic import BaseModel, Field


class PlaceCard(BaseModel):
    name: str = Field(description="장소명")
    score: int = Field(default=0, description="요청 적합도 점수 0~100. 높을수록 잘 맞음")
    reason: str = Field(description="이 요청에 왜 맞는지 2~3문장")
    # 아래는 에이전트가 비워도 되는 부가필드 — Chat.send가 근거 파일에서 채운다.
    address: str = Field(default="", description="주소. 비워두면 코드가 채운다")
    category: str = Field(default="", description="카테고리")
    highlights: List[str] = Field(default=[], description="핵심 강점 2~3개. 비워두면 코드가 채운다")
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
        description=(
            "kind가 'recommendation'일 때, 근거 파일(summary_file)의 리랭크된 후보를 "
            "'전부' 담고 각자 score를 매긴다(일부만 고르지 마라). 최종 상위 선별은 코드가 한다. "
            "그 외 kind에서는 빈 배열."
        ),
    )
