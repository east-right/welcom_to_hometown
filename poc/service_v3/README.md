# wh_poc — 장소 추천 에이전트

카카오 로컬로 후보를 잡고, 네이버 블로그·웹문서·뉴스로 근거를 모아 추천하는 ADK 에이전트.

## 구조

```
wh_poc/
├── core.py      Layer 0  공용 유틸 · 계측 · 파일 I/O
├── clients.py   Layer 1  카카오 · 네이버 · 크롤링 · LLM
├── prompts.py            툴 내부 프롬프트 템플릿
├── tools.py     Layer 2  에이전트에 노출되는 툴 5개
├── schema.py             응답 스키마 (kind / message / places)
├── agent.py     Layer 3  LlmAgent + 멀티턴 Chat
└── tracing.py            턴 단위 오케스트레이션 트레이서

run.py      CLI 진입점
tests.py    레이어별 독립 테스트
```

## 설치

```bash
pip install -r requirements.txt
cp .env.example .env    # 키 채우기
```

필요한 키: `GOOGLE_API_KEY`, `KAKAO_REST_API_KEY`, `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`

## 실행

```bash
python run.py                         # 대화형
python run.py "신림역 백반집 추천해줘"    # 단발
python run.py -v -t "성수동 카공카페"    # 툴 내부 로그까지

python tests.py                       # 전체 테스트
python tests.py clients               # Layer 1만
python tests.py multiturn             # 멀티턴 시나리오
python tests.py probe                 # 되묻기 남발 점검
python tests.py trace                 # 오케스트레이션 타임라인
```

노트북에서:

```python
from wh_poc import Chat, set_verbose

set_verbose(False)
chat = Chat()
await chat.say("신림역 혼밥하기 좋은 백반집")
await chat.say("그중에 웨이팅 없는 곳은?")
chat.summary()
```

## 관측

두 층으로 나뉜다. 목적이 다르다.

**툴 내부 (`core.py`의 `track` / `timed_tool`)** — 툴 하나 안에서 어느 API가 느린가.

```
[collect_info] 총 28.1초
   scrape      1307.2초  (186회, 평균 7.03초)
   naver        286.5초  (80회, 평균 3.58초)
   llm           66.0초  (38회, 평균 1.74초)
```

누적 시간은 벽시계가 아니라 **일감의 총량**이다. 누적 ≫ 총합이면 병렬이 잘 먹고 있는 것,
누적 ≈ 총합이면 직렬화된 것이라 세마포어를 올려도 소용없다.

**턴 전체 (`tracing.py`)** — 한 턴에서 LLM과 툴이 언제 무엇을 했는가.

```python
chat = Chat()                 # trace=True가 기본
await chat.say("신림동 백반")
chat.trace()                  # 타임라인
chat.totals()                 # LLM 대기 vs 툴 실행 비중
```

```
[턴 1] "신림동 백반"    총 32.1초  | recommendation places=5 | 토큰 in 21,020 out 1,385
  ├ LLM      1.4초  in  2,180 out    45  → collect_places
  ├ TOOL     0.3초  collect_places({"query": "신림동 백반"})
  │          → {"status": "too_broad", "count": 30, ...}
  ├ LLM      1.1초  in  2,410 out    60  → collect_info
  ├ TOOL    28.2초  collect_info({"region_prefix": "신림"})
  │          → {"count": 25, "no_info": 5, ...}
  └ LLM      2.2초  in 13,900 out 1,240  신림역 근처 5곳을 찾았어요
```

ADK의 `before/after_model_callback`, `before/after_tool_callback`을 쓴다.
프롬프트 전문이 필요하면 `Chat(capture_prompt=True)`.

외부 도구(Langfuse 등)는 세션 간 비교나 팀 공유가 필요해질 때 얹으면 된다.
OTel 기반이라 이 트레이서와 공존할 수 있다.

## 설계 원칙

**툴 경계 = 에이전트가 판단해야 하는 지점.**
코드가 관측 가능한 조건(결과 개수, `is_end`, API 응답 상태)은 함수 안으로 내리고,
의미론적 판단만 에이전트에게 남긴다. API 제공자(카카오/네이버) 기준으로 자르지 않는다.

**툴은 신호를 주고, 판단은 에이전트가 한다.**
`too_broad`는 "후보가 상한을 넘었다"는 사실만 알려준다. 그게 문제인지(되물을지)는
발화를 이해해야 알 수 있다. `count=30`은 "신림역 백반집"이든 "서울 카페"든 똑같이 찍힌다.

**모든 툴 결과는 파일로 떨구고 경로만 반환.**
원본이 컨텍스트에 올라가지 않아 할루시네이션 여지가 줄고, 에이전트는 `read_file`로
필요할 때만 읽는다. 크롤링한 본문은 요약이 나오면 용도가 끝나므로 저장하지 않는다.

**한 곳에서만 쓰는 로직은 함수로 빼지 않는다.**
재사용되는 것만 Layer 0/1에 둔다.

## 에이전트에게 남긴 판단

1. **자연어 → 카카오가 먹는 검색어로 압축** — "조용히 작업할 카페" → "카공카페"
2. **재시도 시 어휘 교체 + `region_prefix`** — "백반집"→"밥집", "관천"→"신림"
3. **`collect_info` / `collect_context` 선택** — 카페는 방문 경험, 축제는 일정 사실
4. **`too_broad`일 때 되물을지 진행할지** — 파이프라인에 되돌아가는 경로를 만든다

## 체크포인트를 collect_places 뒤에만 둔 이유

| 툴 | 소요 | 되돌리기 비용 |
|---|---|---|
| `collect_places` | 0.3초 | 거의 없음 |
| `collect_context` | 8초 | 중간 |
| `collect_info` | 28초 | 큼 |

0.3초짜리 뒤에서 물으면 재실행해도 0.3초다. 반대로 `collect_info` 뒤에서 물으면
이미 28초를 태운 뒤라 절약되는 게 없다. 그래서 후기가 부실할 때는 되묻지 않고
`message`로 한계만 밝히고 진행한다.

## 삽질로 알아낸 것들

| 발견 | 대응 |
|---|---|
| 429는 실패의 주범이 아니었다 | 세마포어를 조정해도 실패 수가 그대로였다. 진짜 원인은 검색어 매칭 |
| 상호명 단독 검색은 동음이의어에 취약 | "도화불백"→마포 도화동 글. 지역명을 앞에 붙여야 정확도가 오른다 |
| 도로명은 실제 리뷰 표현과 다르다 | "관천로"→"관천"이 아니라 "신림". 지역 상식이라 코드가 못 한다 → `region_prefix` |
| `sim`/`date`는 매장마다 유리한 쪽이 다르다 | 사전 판단 불가. 둘 다 던지고 `link`로 dedup |
| 네이버 블로그는 iframe 껍데기 | `PostView.naver`로 우회해야 본문이 나온다 |
| `[봉천역] 털보네…`가 실패로 오탐 | 대괄호 판정 대신 `__SCRAPE_FAIL__` 마커 사용 |
| "결과가 적다"와 "더 이상 없다"는 다르다 | `is_end`로 `exhausted` 판정. 축제 쿼리의 무의미한 retry 제거 |
| `asyncio.run()`은 주피터·서버 양쪽에서 터진다 | 툴 전부 `async def` 통일 |
| `temp:` state는 다음 턴에 사라진다 | 파일 경로는 세션 스코프, 재시도 카운터만 `temp:` |
| `output_schema`가 있어도 모델이 항상 따르진 않는다 | 파싱 실패 시 `kind='chat'`으로 감싸는 폴백 |

## 남은 과제

| 항목 | 메모 |
|---|---|
| `scrape` 평균 7초 | 186건을 SEM=20으로 밀면 건당이 느려진다. 낮추는 쪽이 빠를 수 있다 |
| `naver` 평균 3.5초 | 검색 API치고 비정상. 429 재시도가 평균을 끌어올린 것으로 보임 |
| 재실행 낭비 | 좁혀서 재검색하면 겹치는 매장의 후기를 다시 긁는다. `id` 대조로 스킵 가능 |
| 컨텍스트 누적 | 턴이 쌓이면 이전 `read_file` 결과(10KB)가 계속 남는다 |
| 후기 미확보율 | 흔한 상호명("옛집", "행복한식당")은 여전히 취약 |
| `category_code` | 에이전트가 임의 지정. 잘못 넣으면 결과가 급감 |
| 되묻기 횟수 제한 | instruction으로만 걸어둠. state 카운터가 확실하다 |
| 세마포어 전역 | 프로세스 단위 공유. 서버화 시 요청 간 간섭 검토 |
