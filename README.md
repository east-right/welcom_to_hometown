# Welcome to HomeTown

멀티턴 대화로 지역·일자·동반자·목적을 모은 뒤,
도메인별 검색 에이전트가 네이버 검색 + 본문 취득으로 추천 카드를 만든다.
Google ADK로 구현한 멀티에이전트 시스템이다.

---

## 실행

### 1. 의존성 설치

```bash
uv sync
```

> uv가 없으면: `pip install uv` 또는
> `curl -LsSf https://astral.sh/uv/install.sh | sh` (mac/linux)

### 2. API 키 설정

`.env` 파일 생성 후 아래 키 등록:

```
GOOGLE_API_KEY=제미나이_키
NAVER_CLIENT_ID=네이버_클라이언트_ID
NAVER_CLIENT_SECRET=네이버_시크릿
WH_MODEL=gemini-3.6-flash
```

- 제미나이 키: https://aistudio.google.com/apikey
- 네이버 검색 키: https://developers.naver.com/apps/#/register
  (앱 등록 시 **"검색" API** 추가 필수)

### 3. 실행

```bash
cd service
uv run python main.py
```

종료: `quit` / 처음부터: `reset`

---

## 사용 예시

정보를 한 번에 말해도 되고, 부족하면 봇이 되묻는다.

```
🙋 나: 부모님 모시고 축제 가고 싶어
  📋 파악: 지역=? / 일자=? / 동반=부모님 / 목적=축제
🤖 봇: 어느 지역으로 가실 예정인가요?

🙋 나: 창원 근처, 이번 주말
  📋 파악: 지역=창원 근처 / 일자=이번 주말 / 동반=부모님 / 목적=축제
🤖 봇: 조건이 다 모였어요. 찾아볼게요!
  ...추천 카드 6개...
```

---

## 구조 (MSA식 역할 분리)

```
service/
├── main.py        CLI 대화 진입점 (슬롯 필링 루프)
├── config.py      환경 설정
├── schema.py      슬롯·카드 데이터 모델
├── tools.py       naver_search(발견) + fetch_page(취득)
├── agents.py      슬롯추출기 / 검색서브4 / 코디네이터 / 포매터
└── engine.py      에이전트 실행 헬퍼 + 오케스트레이션
```

---

## 흐름

```
입력 → slot_extractor (슬롯 추출)
        ├ 부족 → 되물음 → (반복)
        └ 충분 → coordinator (위임)
                    ├→ restaurant_agent  ┐
                    ├→ cafe_agent        │ naver_search(발견)
                    ├→ festival_agent    │ + fetch_page(취득)
                    └→ spot_agent        ┘
                    └→ formatter (JSON) → 추천 카드
```

---

## 슬롯

| 슬롯 | 예시 |
|------|------|
| region (지역) | 창원, 창원 근교 |
| date (일자) | 7월 25~27일, 이번 주말 |
| companion (동반자) | 부모님, 친구, 혼자 |
| purpose (목적) | 축제, 맛집, 카페, 나들이 |

4개가 다 차면 검색이 시작된다.

---

## 핵심 설계

- **발견/취득 분리**: `naver_search`로 URL을 찾고, `fetch_page`가 본문만 추출 (메뉴·광고를 걷어내고 본문 컨테이너만 파싱)
- **도메인별 카드**: 축제는 개최 기간, 매장은 운영시간, 휴무일, 본문에서 확인 안 된 정보는 생성하지 않는다.
- **이원화**: 검색 툴을 쓰는 에이전트는 JSON 강제(`output_schema`)를 못 쓰므로, 검색 담당과 JSON 출력 agent를 분리