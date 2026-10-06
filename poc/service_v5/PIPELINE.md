# wh_poc v5 — 파이프라인

지역 장소를 찾아 추천하는 ADK 에이전트. 카카오로 후보를 잡고, 웹수확으로 카카오가
놓친 로컬 노포를 보강한 뒤, 블로그 스니펫으로 요청에 맞는 상위 15곳을 리랭크해 요약한다.

---

## 레이어

```
interface   run.py / eval.py / tests.py        진입점·평가·테스트
agent       agent.py    LlmAgent + 멀티턴 Chat + INSTRUCTION
tools       tools.py    에이전트에 노출되는 툴 5개 (+ 헬퍼)
prompts     prompts.py  툴 내부 프롬프트
clients     clients.py  카카오·네이버·크롤·임베딩·LLM 래퍼
core        core.py     세마포어·계측·파일 I/O·로그
schema.py   응답 스키마 (kind / message / places)
tracing.py  턴 단위 오케스트레이션 트레이서
```

의존 방향은 안쪽으로: `interface → agent → tools → (clients · prompts) → core`.

---

## 전체 흐름

```
사용자 질문
   │
   ▼
[agent]  자연어 → 검색어 압축 ('지역 + 짧은 명사', 막연한 맛집은 '식사 맛집')
   │     + region_prefix 판단 + collect_info/context 선택
   ▼
[collect_places]  후보 수집  (status: ok / too_broad)
   ├ 1) 카카오 로컬 전량 (≤45)
   ├ 2) 웹수확 — 네이버 웹검색 스니펫 → LLM 상호명 추출 → 카카오 역조회 → 같은 구만
   ├ 3) 프랜차이즈 제거 (음식점 체인)
   └ 4) 업종 1차 필터 — base가 단일업종이면 다른 업종 수확 제거 (막연한 맛집은 필터 안 함)
          → 풀 ~60개, 자르지 않고 파일로 저장
   │
   ▼   (too_broad이고 좁힐 기준 없으면 → 에이전트가 되물음)
   │
[collect_info]  블로그 후기 리랭크·요약   ← 카페·식당류
   ├ 1) 매장별 블로그 검색 → 스니펫(제목+요약) 최대 5개
   ├ 2) HyDE 리랭크 — '기준 문서' 생성 → 임베딩 → 스니펫과 max 유사도 → 상위 15
   ├ 3) 상위 15만 본문 크롤 (매장당 3편) — 협찬/체험단 리뷰 제외
   └ 4) 매장별 통합 요약 (강점·약점·평가) → summary_file
   │
   ▼   (축제·행사·관광이면 collect_info 대신 collect_context: 웹문서·뉴스 사실 정리)
   │
[agent]  read_file로 근거 확인 → 요청에 맞는 5~8곳 선정 → kind='recommendation'
   ▼
응답 (message + places[] 카드)
```

모든 툴 결과는 파일(`tool_file/`)로 떨구고 경로만 반환한다. 원본이 컨텍스트에 안 올라가
할루시네이션 여지가 줄고, 에이전트는 `read_file`로 필요할 때만 읽는다.

---

## 툴 5개

| 툴 | 하는 일 | 반환 |
|---|---|---|
| `get_today` | 오늘 날짜·요일·시각 (상대 시점·행사용) | date/weekday/time/year/month |
| `collect_places` | 후보 수집 (카카오 + 웹수확 + 필터) | status, count, file |
| `collect_info` | 블로그 후기 리랭크·요약 (상설 매장) | count, url_file, rerank_file, summary_file |
| `collect_context` | 일정·운영정보 정리 (축제·관광) | count, source_file, context_file |
| `read_file` | 툴 산출 파일 읽기 (답변 전 필수) | content |

에이전트가 판단하는 것은 넷: ① 검색어 압축 ② region_prefix ③ info/context 선택
④ too_broad일 때 되물을지. 나머지(수집·수확·필터·리랭크·동시성)는 전부 툴 안에 있다.

---

## 핵심 설계

### 1. 수집 = 카카오 + 웹수확
카카오 로컬은 '맛집' 태그 없는 로컬 노포(순대집·국숫집)를 놓친다. 그래서 네이버 웹검색
스니펫에서 상호명을 뽑아(LLM) 카카오로 역조회해 보강한다. `"{query} Top"`으로 검색해
랭킹 리스트 페이지를 노려 수확 밀도를 높인다.
- 카카오 45 + 웹수확 ≈ Recall@수집 96% (카카오만으론 75%).

### 2. 프랜차이즈 필터 (`_is_franchise`)
카카오는 프랜차이즈 브랜드를 카테고리 끝에 붙인다(`음식점>카페>커피전문점>스타벅스`).
상호명이 카테고리 마지막 조각으로 **시작**하고 카테고리 깊이가 3+이면 프랜차이즈로 보고
제거. 일반 매장(서울'갈비', '손'칼국수)은 안 걸린다.

### 3. 업종 1차 필터 (`_dominant_cats`)
base(카카오)의 2단계 업종 분포에서 한 업종이 ≥50%면(구체 쿼리) 그 업종군만 수확에서
남기고 나머지는 노이즈로 제거(카페 검색에 딸려온 횟집 등). 여러 업종에 퍼져 있으면
(막연한 '맛집') 필터하지 않는다 — 우동=일식처럼 소수 업종 정답이 잘리지 않게.

### 4. HyDE 리랭크 (스니펫 임베딩)
요청으로 '기준 문서'를 생성해 임베딩하고, 각 매장 블로그 스니펫과 max 코사인 유사도로
상위 15를 뽑는다(질문 문장이 아니라 문서를 쓰는 게 HyDE — 후기끼리 같은 텍스트 공간에
있어 유사도가 잘 맞는다). 문서는 조건부:
- **구체 쿼리**(부모님·바다뷰 등): 그 조건이 드러나는 담백한 방문 후기.
- **막연한 '맛집'**: 특정 후기 대신 '진짜 로컬 맛집 신호'(노포·3대·비프랜차이즈·웨이팅·
  장인·방송·현지인 단골)를 서술 → 로컬 노포 블로그와 임베딩이 더 잘 맞는다.
- 본문이 아니라 **스니펫**으로 리랭크한다(본문≈스니펫이라 비싼 크롤이 랭킹엔 무의미).

### 5. 협찬 필터 (`_is_sponsored`)
크롤한 블로그의 첫 사진(스티커·지도카드 제외)이 네이버(pstatic)가 아니면 협찬 배지
(레뷰·리뷰노트 등 대행사 서버)로 보고 제외. 본문 앞 고지 문구도 함께 본다. 리랭크 통과
상위 15의 요약 입력에서 마케팅 카피를 걷어낸다.

### 6. 되묻기 체크포인트는 collect_places 뒤에만
collect_places는 빠르고 collect_info는 느리다(크롤·요약). 느린 것 뒤에서 되물으면 이미
비용을 태운 뒤라 절약이 없다. 그래서 후기가 부실할 땐 되묻지 않고 message로 한계만 밝힌다.

---

## 조절 노브 (`tools.py` 상수)

| 상수 | 값 | 의미 |
|---|---|---|
| `MAX_COUNT` | 45 | 카카오 상한 (too_broad 판정) |
| `RERANK_SNIPPETS` | 5 | 매장당 확보할 블로그 스니펫 수 |
| `RERANK_KEEP` | 15 | 리랭크로 남길 매장 수 (크롤·요약 비용이 비례) |
| `INFO_PER_PLACE` | 3 | 남긴 매장당 요약에 쓸 본문 수 |
| `INFO_BODY_CHARS` | 2000 | 요약에 넣는 본문 최대 길이 |

동시성: `NAVER_SEM=10 / SCRAPE_SEM=20 / LLM_SEM=10` (`core.py`, 실측값).
모델: 에이전트·유틸 LLM `gemini-3.1-flash-lite`, 임베딩 `gemini-embedding-001`(768차원).

---

## LLM 백엔드 스왑 (로컬 모델)

에이전트·유틸 LLM은 env 하나로 호스티드 Gemini ↔ 로컬 vLLM(OpenAI 호환)을 갈아끼운다.
**임베딩은 항상 Gemini**라 `GOOGLE_API_KEY`는 어느 백엔드에서도 필요하다. 분기 지점은
딱 둘 — `clients.llm_async`(유틸 LLM)와 `agent.agent_model`(에이전트 LLM). 나머지
파이프라인은 프로바이더 무관이다.

| env | 기본 | 의미 |
|---|---|---|
| `WH_LLM_BACKEND` | `gemini` | `local`이면 아래 3개로 vLLM 호출 |
| `WH_LOCAL_MODEL` | — | LiteLLM 식별자. 예: `openai/Qwen2.5-7B-Instruct` |
| `WH_LOCAL_API_BASE` | — | vLLM OpenAI 호환 엔드포인트. 예: `https://<pod>-8000.proxy.runpod.net/v1` |
| `WH_LOCAL_API_KEY` | `EMPTY` | vLLM은 보통 키 검사를 안 한다 |

env를 안 주면 현행(gemini)과 100% 동일. 로컬 경로는 `litellm`을 쓴다(`requirements.txt`).

> ⚠️ 검증 필요: 로컬 모델에서 `output_schema`(구조화 출력) + 툴 함수콜링이 Gemini만큼
> 되는지는 백엔드/모델 의존이라 `eval.py`로 재측정해야 확정된다. structured output이
> 약하면 `Chat.send`의 JSON 파싱 폴백이 받아준다.

---

## 측정 (`eval.py`)

정답셋을 에이전트에 넣어(LLM이 실제로 툴을 호출) 단계별 recall을 잰다. 단계마다 고칠
곳이 다르므로 funnel로 분리한다.

```
수집 → 1차필터 → 리랭크(15) → 최종(5~8)
```
- `Recall@수집` 정답이 collect_places 풀에 있나 (검색어/소스 문제)
- `Recall@리랭크` 정답이 리랭크 top-15에 있나 (리랭커 문제)
- `Recall@최종` 정답이 최종 추천에 있나 (최종 LLM 선정 문제)
- `MRR` 최종에서 첫 정답의 순위

각 정답이 어느 단계에서 샜는지(수집실패/리랭크탈락/최종LLM탈락)까지 출력한다.
설계 결정의 근거·실험 이력은 `experiments/results.md`에 단계별(A~I)로 기록돼 있다.

---

## 실행

```bash
python run.py "신림역 백반집 추천해줘"     # 단발
python run.py -v "가포 맛집"               # 툴 내부 로그까지
python eval.py test.json                   # 정답셋 funnel 평가
uvicorn api:app --reload                    # 웹 + FastAPI

# 로컬 모델로 스왑
WH_LLM_BACKEND=local WH_LOCAL_MODEL=openai/Qwen2.5-7B-Instruct \
  WH_LOCAL_API_BASE=https://<pod>-8000.proxy.runpod.net/v1 python eval.py test.json
```

필요한 키: `GOOGLE_API_KEY`, `KAKAO_REST_API_KEY`, `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`
