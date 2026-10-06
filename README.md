# Welcome to HomeTown

자연어 질의("신림역 고기집 추천해줘", "창원 가포 맛집")를 받아 **지역 맛집을 찾아
추천 카드로 돌려주는 agentic 파이프라인**. 카카오로 후보를 모으고, 웹수확으로 카카오가
놓친 로컬 노포를 보강한 뒤, 블로그 후기를 멀티HyDE로 리랭크·요약하고, LLM Judge가
요청 적합도로 점수를 매겨 상위 매장을 고른다. Google ADK로 구현했다.

현행 코드는 **`poc/service_v5`** 에 있다. 파이프라인 상세는
[`poc/service_v5/PIPELINE.md`](poc/service_v5/PIPELINE.md),
로컬 LLM(vLLM) 벤치는 [`poc/service_v5/vllm_logs/BENCH_SUMMARY.md`](poc/service_v5/vllm_logs/BENCH_SUMMARY.md).

---

## 실행

### 1. 의존성 설치

```bash
uv sync
```

> uv가 없으면: `pip install uv` 또는
> `curl -LsSf https://astral.sh/uv/install.sh | sh` (mac/linux)

### 2. API 키 설정

리포 루트에 `.env` 생성 후 등록:

```
GOOGLE_API_KEY=제미나이_키
KAKAO_REST_API_KEY=카카오_REST_키
NAVER_CLIENT_ID=네이버_클라이언트_ID
NAVER_CLIENT_SECRET=네이버_시크릿
```

- 제미나이 키(LLM + 임베딩): https://aistudio.google.com/apikey
- 카카오 로컬 키: https://developers.kakao.com (REST API 키)
- 네이버 검색 키: https://developers.naver.com/apps/#/register
  (앱 등록 시 **"검색" API** 추가 필수)

### 3. 실행

```bash
cd poc/service_v5
uv run python run.py                        # 대화형 모드
uv run python run.py "신림역 백반집 추천해줘"   # 단발 질의
uv run python run.py -v "창원 가포 맛집"        # 툴 내부 로그까지
uv run python eval.py test.json             # 정답셋 단계별 recall 평가
uv run uvicorn api:app --reload             # 웹 + FastAPI
```

---

## 사용 예시

```
👤 신림역 근처 고기집 추천해줘
🤖 신림역 주변 고기집을 정리해 드립니다.
   ...추천 카드 (매장명 · 점수 · 선정 이유 · 주소 · 강점/주의)...
```

조건이 모호해 후보가 너무 넓으면(`too_broad`) 봇이 되묻는다.

---

## 구조 (MSA식 레이어 분리)

```
poc/service_v5/
├── run.py        CLI 진입점 (대화형 / 단발)
├── eval.py       정답셋 단계별 recall 평가
├── api.py        FastAPI + 웹 UI
└── wh_poc/
    ├── agent.py    LlmAgent + 멀티턴 Chat + INSTRUCTION
    ├── tools.py    에이전트에 노출되는 툴 5개 (+ 수집·필터·리랭크 헬퍼)
    ├── prompts.py  툴 내부 프롬프트 (HyDE·요약·Judge 스코어링)
    ├── clients.py  카카오·네이버·크롤·임베딩·LLM 래퍼
    ├── core.py     세마포어·계측·파일 I/O·로그
    ├── schema.py   응답 스키마 (kind / message / places)
    └── tracing.py  턴 단위 오케스트레이션 트레이서
```

의존 방향은 안쪽으로: `interface → agent → tools → (clients · prompts) → core`.

---

## 흐름

```
사용자 질문
   │
   ▼
[agent]  자연어 → 검색어 압축 + region_prefix 판단 + info/context 선택
   ▼
[collect_places]  카카오 로컬(≤45) + 네이버 웹수확(노포 보강) + 프랜차이즈·업종 필터
   │              → 풀 ~60개 파일로 저장  (너무 넓으면 too_broad → 되물음)
   ▼
[collect_info]  블로그 스니펫 → 멀티HyDE 리랭크 top15 → 본문 크롤·요약(협찬 제외)
   │            → LLM Judge가 전수 0~100점 스코어링 → 점수순 정렬
   ▼
[agent]  read_file로 근거 확인 → 코드가 점수 상위 FINAL_K곳으로 추천 카드 구성
   ▼
응답 (message + places[] 카드)
```

모든 툴 결과는 `tool_file/`에 떨구고 경로만 반환한다 — 원본이 컨텍스트에 안 올라가
할루시네이션 여지가 줄고, 에이전트는 `read_file`로 필요할 때만 읽는다.

---

## 툴 5개

| 툴 | 하는 일 |
|---|---|
| `get_today` | 오늘 날짜·요일·시각 (상대 시점·행사용) |
| `collect_places` | 후보 수집 (카카오 + 웹수확 + 프랜차이즈·업종 필터) |
| `collect_info` | 블로그 후기 멀티HyDE 리랭크·요약 + Judge 전수 스코어링 (상설 매장) |
| `collect_context` | 일정·운영정보 정리 (축제·관광) |
| `read_file` | 툴 산출 파일 읽기 (답변 전 필수) |

---

## 핵심 설계

- **발견/취득 분리**: 카카오 로컬로 후보를 잡고, 네이버 웹검색 스니펫에서 상호명을
  뽑아(LLM) 카카오 역조회로 노포를 보강 → Recall@수집 ≈ 96% (카카오만으론 ~75%).
- **멀티HyDE 리랭크**: 요청으로 서로 다른 3관점(전통노포 / 인기웨이팅 / 미디어검증)의
  '기준 문서'를 생성·임베딩해 매장 후기와 **max 유사도**로 상위 15를 뽑는다. 질문 문장이
  아니라 가상 후기를 쓰는 게 HyDE — 후기끼리 같은 텍스트 공간이라 유사도가 잘 맞는다.
- **LLM Judge 전수 스코어링**: 리랭크 top15 요약을 질의와 함께 읽혀 0~100점을 **전부**
  매기고(bi-encoder → cross-encoder 2단), 점수순으로 최종 추천을 코드가 구성한다.
  (LLM이 상위 몇 개만 고르고 자르는 lazy-cut 방지 → 결정적·재현 가능)
- **협찬 필터**: 크롤한 블로그의 대표 이미지가 네이버(pstatic)가 아니면 대행사 협찬으로
  보고 요약 입력에서 제외 → 마케팅 카피 차단.
- **백엔드 스왑**: 에이전트·유틸 LLM은 env 하나로 Gemini ↔ 로컬 vLLM을 갈아끼운다.
  임베딩은 항상 Gemini. (`WH_LLM_BACKEND` / `WH_AGENT_BACKEND`, 상세는 PIPELINE.md)

---

## 조절 노브 (`wh_poc/tools.py`)

| 상수 | 기본 | 의미 |
|---|---|---|
| `MAX_COUNT` | 45 | 카카오 상한 (too_broad 판정) |
| `RERANK_SNIPPETS` | 10 | 매장당 확보할 블로그 스니펫 수 |
| `RERANK_KEEP` | 15 | 리랭크로 남길 매장 수 (`WH_RERANK_KEEP`) |
| `INFO_PER_PLACE` | 3 | 남긴 매장당 요약에 쓸 본문 수 |
| `FINAL_K` | 8 | 최종 추천 매장 수 (`WH_FINAL_K`) |

모델: 에이전트·유틸 작업은 경량 호스티드 LLM으로 돌리고(env로 로컬 vLLM 스왑 가능),
리랭크 임베딩은 Gemini를 쓴다. 구체 모델·백엔드 설정은 `PIPELINE.md` 참고.
