# service_v5 로컬 모델 이식 — 트러블슈팅 기록

service_v4(Gemini 전용) → service_v5로 복사한 뒤, 에이전트·유틸 LLM을 **자체 호스팅
오픈웨이트 모델**(RunPod 서버리스 vLLM)로 붙이면서 겪은 문제와 해결을 순서대로 기록한다.

- **모델**: Qwen3.8-27B-FP8 (밀집 하이브리드 Mamba/Gated-DeltaNet + 풀 어텐션, 추론 모델)
- **서빙**: RunPod Serverless, vLLM 0.27, OpenAI 호환 엔드포인트(`/v2/<id>/openai/v1`)
- **어댑터**: `WH_LLM_BACKEND=local` 스위치로 LLM만 vLLM으로 라우팅, 임베딩은 항상 Gemini
- **연결**: ADK `LiteLlm`(에이전트) + `litellm`(유틸 LLM), 둘 다 OpenAI 호환 경로 사용

핵심 교훈: **문제는 대부분 모델·프롬프트가 아니라 "배선(wiring)과 서빙 설정"에 있었다.**

---

## 배경 지식 (증상 해석용)

- **서버리스 콜드스타트**: 워커가 요청 없으면 scale-to-zero. 첫 호출은 모델 로딩(27GB)으로
  수십 초~1분 무응답 행이 정상. `/openai/v1/*`가 code=000으로 오래 걸리면 대개 콜드스타트.
  `/health`·네이티브 `/run`은 즉답이므로 진단에 활용.
- **네이티브 vs OpenAI 경로**: RunPod은 `/run`+`/status`(네이티브, job 큐)와
  `/openai/v1/*`(OpenAI 호환) 두 경로가 있다. v5(litellm)는 후자를 쓴다. 네이티브가 되고
  OpenAI가 안 되면 vLLM OpenAI 서버 쪽 문제.

---

## 문제 1 — vLLM 부팅 실패 (startup crash, exit code 1)

**증상**
```
ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (211).
... EngineCore failed to start → vLLM serve exited during startup with code 1
```
`/openai/v1/*` 전체 무응답. 워커가 크래시 루프.

**원인**
Qwen3.8은 하이브리드 Mamba 모델이라 **디코딩 시퀀스 1개당 Mamba 캐시 블록 1개**가 필요.
이 GPU 메모리 예산으론 211블록까지 확보되는데 워커 기본값 `max_num_seqs=256`이 초과 →
CUDA 그래프 캡처 불가 → 엔진 초기화 거부 → 서버 사망. **GPU가 약해서가 아니라 설정
미스매치.** (A6000 48GB면 용량 자체는 충분)

**해결**
- vLLM args `--max-num-seqs 200` (≤211). 또는 `gpu_memory_utilization` 상향(near-OOM이라
  비권장).

**무해 노이즈 (원인 아님)**
- `Qwen3VLVideoProcessor / min_frames` ERROR — transformers docstring 린트. VL 모델 아님.
- `FP8 not native support` — A6000은 Ampere라 FP8 텐서코어 없음. Marlin 에뮬레이션으로
  동작(밀집이라 지원). 메모리는 절약되나 속도 이득은 없음.

---

## 문제 2 — 함수 호출 시 500 / 무응답

**증상**
plain chat은 200. 그런데 `tools`를 포함한 요청만 500(초기) → 파서 교체 후엔 무응답 행.
에이전트 실행 시 `litellm.InternalServerError: 500`.

**원인 (2단계)**
1. 처음엔 vLLM이 **함수 호출 활성화 플래그 없이** 떠서 tools 요청을 못 받음.
2. 플래그 켠 뒤엔 **툴콜 파서 형식 불일치**. Qwen3.8은 툴콜을
   `<tool_call><function=get_weather><parameter=city>Seoul</parameter>...` **XML 형식**으로
   출력하는데, `--tool-call-parser hermes`는 **JSON 형식**을 기대 → 파싱 실패 →
   `tool_calls`가 빈 채로 XML이 `content`에 raw로 남음 → ADK가 툴콜 없다고 보고 깨짐.

**해결 (RunPod 워커 설정)**
RunPod 대시보드에서 토글/값으로 노출된다. 실제로 건드린 것:
| 설정 | 값 | 비고 |
|---|---|---|
| `enable_auto_tool_choice` | **false → true** | 켜야 tools 요청 처리 |
| `enable_reasoning` | **false → true** | reasoning 파서 활성화 |
| `tool_call_parser` | `qwen3_xml` | **hermes 아님!** Qwen3 XML 형식 |
| `reasoning_parser` | `qwen3` | 추론 텍스트를 content에서 분리 |

- 대응 vLLM args: `--enable-auto-tool-choice --tool-call-parser qwen3_xml
  --enable-reasoning --reasoning-parser qwen3`
- `enable_auto_tool_choice`(true)만 켜고 `tool_call_parser` 안 주면 부팅 크래시(짝 필수).
- 검증: `finish_reason: "tool_calls"` + `tool_calls:[{function:{name,arguments}}]`로 파싱되면 OK.

---

## 문제 3 — 응답에 `<think>` 사고과정 오염

**증상**
유틸 LLM(HyDE 문서 생성) 응답 앞에 `We need answer... </think>` 추론 텍스트가 딸려 나옴.
HyDE 임베딩이 사고과정으로 오염, 에이전트 최종 JSON 파싱 위험.

**원인**
Qwen3.8은 추론 모델이라 기본으로 `<think>` 블록을 생성. (json_mode 호출은 JSON 문법
제약이 억제해줘서 깨끗했지만, plain 호출은 오염됨.)

**해결 (v5 코드)**
`clients._llm_local`와 `agent.agent_model` 둘 다:
```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```
Qwen3의 thinking 모드를 꺼서 Gemini(비추론)와 같은 형태의 출력. thinking 미지원 모델은
이 인자를 무시하므로 무해.

---

## 문제 4 — 툴을 아예 안 부르고 장소를 환각 (핵심)

**증상**
에이전트가 `collect_places`를 안 부르고 자기 지식으로 바로 답. eval 전 케이스 `tools=0`,
`predicted`가 `귀산식당2`, `귀산식당3` 같은 **환각**. full eval recall 0%.
thinking ON/OFF 무관, 오히려 ON이 더 나쁨(전부 chat).

**원인 (ADK 소스로 확인)**
에이전트가 `output_schema=FeedOutput` + `tools`를 동시에 씀.
`adk/utils/output_schema_utils.py`의 `can_use_output_schema_with_tools()`:
```python
if isinstance(model, LiteLlm):
    return True          # LiteLlm이면 무조건 True
```
→ ADK가 LiteLlm을 "네이티브 지원"으로 오인해 안전 우회(set_model_response 툴 주입)를
**안 쓰고**, `response_format`(스키마)를 tools와 함께 vLLM에 그대로 넘김.
→ vLLM이 FeedOutput 스키마로 **guided decoding(제약 디코딩)**을 걸어 출력을
`{kind,message,places}`로 강제 → 모델은 한 방에 채우는 게 유효해져 **툴 부를 이유 소멸** →
places를 환각으로 채움.

실측 확인: 스키마 ON → 툴콜 0 / 스키마 OFF → 툴콜 시도함(진단으로 확정).

**해결 — 워커–포매터 2단계 분리 (approach A)**
전문용어: **툴 사용과 구조화 출력의 분리(decoupling tool-use from structured/constrained
decoding)**, **worker–formatter 패턴**, 출력을 스키마에 맞추는 **LLM-as-formatter /
후처리 정형화(post-hoc structured output)**.

v5 코드 변경:
- `agent.build_agent`: `LLM_BACKEND != "local"`일 때만 `output_schema`를 건다. 로컬은
  **tools만**으로 돌려 툴콜을 살린다.
- `agent._format_feed`: 스키마 없이 돌린 워커의 자연어 답 + 근거(`summary_file`)를
  `prompts.FORMAT_FEED`로 FeedOutput JSON으로 정형화(json_mode).
- `agent.Chat.send`: 로컬이면 `_format_feed`, gemini면 기존 `json.loads`.

**결과**: 로컬 Qwen3.8이 `collect_places→collect_info(HyDE 리랭크)→포매터` 전 파이프라인을
돌려 **실제 장소·주소로** 정확히 추천(정답 청하횟집 적중, 환각 없음).

---

## 남은 이슈 (정확도 아님)

- **속도**: `collect_info`가 로컬 LLM 요약 콜 ~16개(HyDE 1 + 요약 최대 15)를 때려 느림.
  단발 쿼리에 수백 초. 서버리스 콜드스타트 + FP8 에뮬레이션 + 추론 모델이 겹침.
  → 최적화 여지: 요약 병렬도/개수, persistent pod, 네이티브 FP8 GPU(L40S 등).
- **종료 시 exit 139(세그폴트)**: 결과 산출 **후** 인터프리터 정리 중 litellm/grpc의
  Windows 크래시. 결과엔 무해.
- **Qwen3.8 함수콜링이 다소 느슨**: 없는 툴(`ask_user`) 호출한 적 있음.

---

## 최종 설정 요약

**RunPod 워커 설정 (대시보드에서 건드린 것)**
| 설정 | 값 |
|---|---|
| `max_num_seqs` | `256 → 200` (≤211) |
| `enable_auto_tool_choice` | `false → true` |
| `enable_reasoning` | `false → true` |
| `tool_call_parser` | `qwen3_xml` |
| `reasoning_parser` | `qwen3` |

대응 vLLM args:
```
--max-num-seqs 200
--enable-auto-tool-choice --tool-call-parser qwen3_xml
--enable-reasoning --reasoning-parser qwen3
```

**.env**
```
WH_LLM_BACKEND=local
WH_LOCAL_MODEL=openai/qwen/qwen3.8-27b-fp8     # /openai/v1/models 이름과 대소문자 일치
WH_LOCAL_API_BASE=https://api.runpod.ai/v2/<endpoint_id>/openai/v1
WH_LOCAL_API_KEY=<runpod_key>
```

**v5 코드 변경 파일**
- `wh_poc/clients.py` — `llm_async` 백엔드 분기, `_llm_local`(litellm + enable_thinking off)
- `wh_poc/agent.py` — `agent_model`(LiteLlm), `build_agent`(로컬 output_schema 제외),
  `_format_feed`(포매터), `Chat.send`(로컬 분기)
- `wh_poc/prompts.py` — `FORMAT_FEED`
- `requirements.txt` — `litellm`

**4개 장벽 요약**
| # | 문제 | 층위 | 해결 |
|---|---|---|---|
| 1 | vLLM 부팅 크래시 | 서빙 설정 | `--max-num-seqs 200` |
| 2 | 함수콜링 500/무응답 | 서빙 설정 | `--tool-call-parser qwen3_xml` (+auto-tool-choice) |
| 3 | `<think>` 오염 | 디코딩 제어 | `enable_thinking=false` |
| 4 | 툴 우회 + 환각 | 배선(ADK) | output_schema 제외 + 워커–포매터 분리 |

---

## 속도 분석 (Qwen3.8-27B-FP8 / RunPod pod A6000 실측)

정확도는 잡혔고(청하횟집·백반집 등 실장소 정확 추천), 남은 건 속도다. 단발 추천 1건이
**~170~220초**. `--trace`에 구성별 분해(`로컬 LLM / 외부 I/O / 임베딩`)와 collect_info
단계별 벽시계 타이머를 추가해 병목을 실측했다.

### 시간이 어디로 가나 (신림역 백반집, MTP 켠 pod)
| 구간 | 벽시계 | 성격 |
|---|---|---|
| `collect_info` | ~64초 | **최대 병목.** 내부 LLM 일감 합 ~494초가 7.7배 병렬로 압축 |
| 최종답 생성 | ~32~40초 | 909토큰 생성(decode) |
| 에이전트 판단 7콜 | ~64초 | 출력 44~49토큰인데도 느림 = **전부 prefill** |
| 외부 I/O(naver/scrape) | (병렬로 collect_info에 흡수) | 백엔드 무관 |

### 핵심 진단 — 판단은 prefill-bound(확정), 요약은 미검증, 최종답은 decode
- **에이전트 판단 = prefill-bound (Gemini 대조로 확정)**: 판단 콜은 입력 ~3.5k / 출력
  44토큰. 같은 입력·출력을 **Gemini 0.9초 vs pod 3~9초**로 처리 — 차이가 곧 순수 추론
  (prefill) 속도다. pod prefill이 ~500 tok/s(FP8 Marlin 에뮬)라 8k만 해도 초 단위, Gemini는 ~0.1초.
- **요약 콜 = LLM-bound이나 prefill/decode 비율은 미검증**: 후기 ~6천 자 입력 / JSON
  ~200토큰 출력. 처음엔 "MTP 켜도 요약 일감 493→494초 불변"을 근거로 prefill-bound라 봤으나,
  요약은 `json_mode`(guided decoding)라 **vLLM이 speculative decoding을 안 걸었을 수 있어**
  그 무효는 증거가 못 된다(spec draft 누적이 요약 출력량보다 적었던 정황). decode(200토큰
  @22tok/s ≈ 9초)도 무시 못 한다. 정확한 판정은 per-request **TTFT vs TPOT**
  (`vllm:time_to_first_token_seconds` / `time_per_output_token_seconds`)로 갈라야 하는데 pod가 꺼져 미측정.
- **generation은 단일 요청 ~22 tok/s**가 바닥(FP8 에뮬). 최종답 40초 ≈ 910토큰 / 22 → **decode-bound 확정**.

> ⚠️ **측정 주의**: `--trace`의 카테고리 "합"(naver/scrape/llm 누적, `@track`)은 **세마포어
> 대기까지 포함**해, 콜 수 > 세마포어면 크게 부풀려진다(naver 합 400초+가 wall ~12초에
> 압축된 것처럼 보임 — 불가능한 병렬도). **병렬도·병목 판단은 "합"이 아니라 벽시계(wall)로.**
> 위 표의 "7.7배 병렬"·"요약 494초"도 이 편향이 섞인 대략치다.

### 켜본 것 — 효과 실측
- **prefix caching**: 하이브리드 Mamba에서도 **작동함**(auto-disable 아님, `/metrics`
  hit 확인). 그러나 히트율 16~27%로 약함. 매 턴 새로 붙는 툴 결과 delta는 캐시 불가 +
  캐시 트래픽 대부분이 재사용 없는 요약 콜이라 희석 → **판단 지연 거의 못 줄임.**
- **MTP(speculative, `mtp`)**: 수용률 **83%**(accepted 2913/draft 3508), mean acceptance
  length ~1.77 → 순수 decode **~1.8배**. **decode-bound인 최종답 생성엔 유효**. 요약엔
  일감이 불변이었으나 이는 json_mode로 spec decode가 안 걸린 탓일 수 있어 무효라 단정 못 함
  (위 요약 진단 참고). 총 시간 개선은 decode 비중만큼이라 제한적.
- 결론: **두 서빙 레버(prefix/MTP) 모두 우리 최대 병목(요약 prefill)을 못 잡는다.**
  근본은 FP8 Marlin 에뮬(A6000엔 네이티브 FP8 없음)의 낮은 prefill/decode 처리량.

### 그래서 실질 레버 (측정으로 확정된 우선순위)
1. **일감 자체 줄이기(코드, 백엔드 무관)** — `RERANK_KEEP` 15→10(요약 콜 수↓),
   `INFO_PER_PLACE` 3→2·`INFO_BODY_CHARS`↓(콜당 prefill 입력↓). 최대 일감(요약 494초)을
   직접 깎는 유일한 방법.
2. **read_file 과호출 억제(프롬프트)** — Qwen3.8이 summary/rerank/collect_places 파일을
   **3번** 읽어 최종 prefill을 15.5k로 부풀렸다. INSTRUCTION에 "summary_file 하나만" 명시하면
   판단 턴 3→1 + 최종답 prefill 축소.
3. **네이티브 FP8 GPU(L40S/H100)** — prefill·decode 처리량 자체를 올리는 본질적 해법.

계측 코드: `core.timed_tool`이 카테고리별(llm/naver/scrape/embed) 누적을 집계 →
`tracing`이 `--trace` 요약과 툴 스텝에 분해 표시. `collect_info`엔 단계별 벽시계
타이머(스니펫수집/리랭크/크롤/요약) 추가(로그, `-v`).

### 파이프라인 자체 최적화 (백엔드 무관 — Gemini 기준 실측/적용)

서빙(pod)과 별개로, LLM이 아닌 외부 I/O·파싱 병목을 코드로 잡았다(**Gemini 단발 53→28초**).
이건 pod에서도 그대로 이득이다(외부 I/O·크롤은 백엔드 무관).

- **collect_places kakao 병렬화**: 웹수확 이름 검증이 동기 kakao **131콜 직렬(14.6초)**이었다.
  문자열 유사도(`_norm_name`)로 base 중복을 먼저 걸러 역조회 콜을 줄이고, `to_thread`+`KAKAO_SEM`
  으로 병렬화 → **17.9→6.3초**.
- **naver 스니펫수집 멀티키**: 429는 앱 키당 걸리므로 `NAVER_CLIENT_ID_2`를 라운드로빈 +
  `NAVER_SEM` 키수 연동. 429 시 안 써본 키 있으면 sleep 없이 즉시 교대 → **12.4→7.0초**.
- **크롤 파싱 lxml**: 느림의 진짜 범인이 다운로드가 아니라 **BeautifulSoup**였다(실측). BS4는
  트리를 파이썬 객체로 재구성해 GIL을 잡아 `to_thread`로 빼도 직렬화, 2MB 네이버 페이지에서
  콜당 4~5초. **raw lxml**(lxml.html, C 레벨·GIL 해제)로 → **파싱 12.6배**(30개 12.8→1.0초),
  본문·협찬 판정은 BS4와 0건 불일치(동치 검증). 크롤 wall 8→2~6초.
- **read_file 1회 제한**: 에이전트가 summary_file 하나만 읽게 INSTRUCTION 명시 → 판단 라운드트립·
  최종 prefill 축소.

정리: **LLM 추론 속도(pod)와 무관한 병목(kakao 직렬·naver 429·BS4 파싱)이 컸고, 코드로 제거하니
Gemini 기준 절반 가까이 단축.** pod의 남은 격차는 순수 LLM 추론 속도(FP8 에뮬)라 GPU 교체 몫.
