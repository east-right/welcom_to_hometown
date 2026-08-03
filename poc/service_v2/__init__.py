"""지역 추천 대화 서비스 (계층 분리 리팩토링본).

계층 의존 방향은 안쪽(도메인)으로만 향한다.

    interface → application → (agents · infrastructure) → domain

- domain         : 순수 데이터 모델. 외부 의존 없음.
- infrastructure : 외부 연동(ADK 실행, 네이버 검색, 로깅) 등 '어떻게'의 세부.
- agents         : LLM 에이전트 정의(지시문 + 툴 배선).
- application    : 유스케이스. '무엇을' 하는지만 한 단계 추상화로 읽힌다.
- interface      : CLI 입출력.
"""
