# 로컬 LLM(vLLM) 벤치 정리 — service_v5 장소추천 에이전트

/ 테스트셋 `test.json` 12질의, 단일질문 상세는 "창원 귀산 바다가 보이는 커피 맛집".
/ 하드웨어 L40S 48GB(네이티브 FP8). 임베딩은 전 구성 공통 Gemini(gemini-embedding-001).

## 0. 결론 한 줄
- **파인튜닝 없이 tool-call 안정성은 gemma-4-31B > Qwen3-27B.** gemma는 전 구성에서 툴 루프 0건, Qwen은 `get_today` 68회 루프(단일 케이스 564초) 등 폭주 발생.
- 속도는 **spec 세팅(num_speculative_tokens)을 맞추면 gemma ≈ Qwen.** 초기 "gemma 우위"는 num 불일치(Qwen 2 vs gemma 4) 탓이었음.

## 1. 구성(누가 어디서 도나)
- **agent LLM**(메인 에이전트: 툴선택+최종 워커텍스트) — 스위치 `WH_AGENT_BACKEND`
- **util LLM**(HyDE·상호명추출·매장요약 15개·최종 포매터 JSON) — 스위치 `WH_LLM_BACKEND`
- **embedding**(리랭크) — 항상 Gemini

| 구성 | agent | util(요약·포매터) |
|---|---|---|
| all-Gemini(베이스) | Gemini | Gemini |
| main-only | 로컬 | Gemini |
| all-local | 로컬 | 로컬 |

## 2. Speculative decoding (서버측)
- **Qwen**: `method: mtp` + 모델 내장 MTP 헤드(별도 모델 없음). num 2 → 4 실험.
- **gemma**: `method: mtp` + **assistant 동반 모델**(`gemma-4-31B-it-assistant`)이 드래프터, num=4. (별도 소형 LM이 드래프트, 큰 모델이 검증 — 무손실.)
- **n-gram**(prompt-lookup, 모델 없음)도 gemma에서 실험.

### 드래프터 종류별 효과 (all-local 1질문)
| 구성 | decode(집계) | 포매터 E2EL | 총시간 |
|---|---|---|---|
| Qwen spec2 (MTP헤드) | 28.8 tok/s | 34.2s(1592tok) | 101.3s |
| Qwen spec4 (MTP헤드) | 31.9 tok/s | 27.1s(1708tok) | 95.0s |
| gemma nospec | 14.6 tok/s | 83.8s(1505tok) | 180.7s |
| gemma n-gram | 11.7 tok/s | 34.0s(1293tok) | 130.2s |
| **gemma MTP-assistant(4)** | **37.5 tok/s** | **22.0s(1521tok)** | **80.4s** |

- **n-gram**: 입력 복붙형(포매터)만 가속(2.1배), **자유생성(요약)은 역효과**(10.4→7.2 tps). VRAM 0.
- **draft/assistant(gemma), 내장MTP(Qwen)**: 자유생성·복붙형 **둘 다** 가속.

### 서버 로그 수용률 (num=4)
- gemma assistant: mean acceptance ~5, 자유생성 73~90% / 복붙형 96~98%.
- Qwen spec4: mean acceptance 4.70~4.94, per-position 위치4까지 0.88~0.98, avg 92~99%.
- → **num=4에서 Qwen ≈ gemma** (draft 품질 대등). 서버 피크 generation: gemma 72~75, Qwen 69.6 tok/s.
- **num_speculative_tokens 값 관련(gemma assistant)**: gemma-4 assistant 공개 예제는 대부분 **num=1**(assistant가 1토큰 예측 모듈이라 그게 native). num>1이면 vLLM이 그 모듈을 **autoregressive로 반복**해 여러 토큰을 뽑는다. 우리는 **num=4**로 돌렸고(로그의 per-position 값이 4개 = 4 확인), **위치 4까지 수용률 0.6~0.97 / mean acceptance ~4.6~4.9**로 잘 버텨 → **예제 기본(1)보다 올린 게 실이득**(패스당 확정 토큰↑). 단 assistant 종류/훈련 깊이에 따라 과하게 올리면 뒤 위치 수용률이 떨어질 수 있으니 `Mean acceptance length`로 확인하며 조정.

## 3. 전체 12케이스 (main-only, 요약·포매터=Gemini)
| 구성 | 최종recall | 케이스recall | MRR | 평균 | 중앙 | max | 툴루프 |
|---|---|---|---|---|---|---|---|
| all-Gemini(구코드*) | 20.8% | — | — | 38s | 39 | 70 | 0 |
| Qwen (MTP2, 가드前**) | 45.8% | 66.7% | 0.408 | 104.7s | 53 | **564** | **2** |
| gemma nospec | 37.5% | 58.3% | 0.274 | 57.3s | 63 | 72 | 0 |
| gemma n-gram | 29.2% | 41.7% | 0.163 | 52.6s | 56 | 64 | 0 |
| **gemma MTP-assistant** | 33.3% | 58.3% | 0.35 | **37.2s** | 39 | 50 | 0 |

\* eval_gemini는 워커/포매터·가드 도입 前 구코드 → recall 직접비교 주의.
\** Qwen 12케이스는 루프 가드 도입 前 run(그래서 case7이 564초까지 폭주). recall 차이는 네이버 수집 런투런 편차 영향(spec은 무손실이라 품질 무관).

## 4. tool-call 안정성 (파인튜닝 없음) — 핵심
- **gemma-4-31B**: 전 구성(nospec/n-gram/draft, 12케이스×3) **툴 루프 0건**, 스텝 5~9로 일정. `get_today`·`collect_context` 헛호출 없음(카페 질의에 날짜툴 안 부름).
- **Qwen3-27B**: `get_today` **68회 연속 호출**(단일 케이스 145스텝/564초), `collect_context` 중복 호출 등 **파이프라인 반복 폭주 성향**.
- 대응(모델 공통): **멱등 툴 가드**(collect_places/collect_info/read_file/get_today 재호출 시 재작업 없이 캐시+중단신호) + **`max_llm_calls=30` 하드 상한**. Qwen은 이 가드가 있어야 안전, gemma는 가드 없이도 애초에 안 돎.
- → **"오픈모델을 에이전트로 쓸 때 tool-call 준수/루프 억제"가 실사용 안정성의 핵심 변수. 현시점 gemma4가 우위.**

## 5. vLLM 운영 학습
- **KV 캐시는 병목 아님**: 피크 all-local 62~70%, main-only 5~18%. 늘 여유. **KV를 줄여도 decode는 안 빨라짐**(decode=메모리 대역폭 지배). 남는 VRAM의 올바른 사용처 = **draft/assistant 모델**.
- **prefix caching는 워크로드가 상한**: 에이전트 이력(재전송) 78%까지, 고유본문 요약은 14%. decode-bound엔 지렛대 아님.
- **FP8**: L40S=네이티브 FP8. A6000(Ampere)=weight-only(Marlin, dequant) → decode ~10~15%차(대역폭), prefill/대batch 차이는 더 큼. 48GB면 31B FP8 적재 가능.
- **num_speculative_tokens**: 2→4가 실이득(Qwen 28.8→31.9, 위치4까지 수용률 높음). 값 키울 땐 로그 `Mean acceptance length`가 실제로 오르는지로 판단.
- **기동 gotcha**: 재기동 시 `--max-model-len`이 기본(16384)으로 복귀 자주 → 매번 확인(32768 필요). `--tool-call-parser`/`--reasoning-parser`는 계열별(gemma4 ≠ qwen3). 로컬 리즈닝모델은 `chat_template_kwargs {enable_thinking:false}`.
- 구조화출력(JSON grammar)+spec 동시 사용 시 `grammar_matcher ... accept new token 198` 경고 뜨나 무해.

## 6. 아키텍처 판단
- **속도·단순함 우선(Gemini 허용)** → main-only. gemma+MTP-assistant = 37s(Gemini 베이스 38s와 동급인데 에이전트는 로컬).
- **완전 로컬 독립** → all-local. spec 필수(gemma 181→80s). 더 줄이려면 **요약·추출을 소형 모델로(tiered)** — 단 소형모델 요약 품질=recall 리스크, 검증 필요.
- 공통: **워커/포매터 분리**(느린 로컬은 선택+이유만, 무거운 JSON은 빠른 모델이 근거에서 채움) + **루프 가드/상한**.

## 7. 산출 파일 (현재 보관본, 같은 폴더)
- `eval_gemma_local.json` — gemma 로컬 구성 run
- `eval_qwen_full.json` — Qwen 로컬 구성 run
- `eval_gemma_max50.json` — gemma 후보 확장(max50) run
- (정직 기록) 위 표를 만든 콜별 상세(`eval_*_1q*.json`)·vLLM /metrics 시계열(`eval_*_vllm.jsonl`)·초기 통합비교 파일은 이전 정리 때 삭제되어 **현재 미보관**. 해당 수치는 본 문서의 표로만 남아 있음.

## 8. 이번 라운드 정정 사항(정직 기록)
- "gemma는 MTP 없음" → **틀림.** gemma-4는 assistant 동반 모델로 MTP 지원.
- "Qwen 내장 MTP 헤드는 얕아 위치3,4 수용률 낮음" → **틀림.** spec4 로그에서 위치4까지 0.88~0.98.
- "gemma가 Qwen보다 빠름" → **부분 정정.** num 불일치(2 vs 4) 영향이 컸고, num=4로 맞추면 대등.
