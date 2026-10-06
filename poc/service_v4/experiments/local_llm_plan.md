# 계획: "추론 LLM 불요"를 로컬 LLM으로 재검증

**상태: 미실행 (계획 문서).** 현재 리랭크 비교([rerank_methods.md](rerank_methods.md))의
"스니펫 HyDE가 추론 LLM 없이 judge에 근접" 이라는 결론은 **호스티드 gemini-3.1-flash-lite**
기준이다. HyDE를 최종 노선으로 택한 핵심 근거가 "로컬 배포 친화(추론 LLM 불요)"인 만큼,
그 주장은 **실제 로컬 LLM에서 한 번 더** 재봐야 확정된다.

## 왜 다시 재야 하나 — 지금 숫자의 한계

- recall·비용이 전부 **호스티드 Gemini** 위에서 측정됐다. 로컬 LLM은 (a) 추론 품질이 낮고
  (b) GPU·지연 비용을 직접 지불한다 — 두 축 다 judge에 불리하게 작동할 수 있다.
- 정확히 하면 HyDE도 LLM을 쓴다: **HyDE 문서 생성 1콜(짧음, out ~100토큰)**. judge는
  **전 후보(28~59곳) 스니펫을 한 프롬프트에 넣어 읽고 순위 추론**(긴 입력 + 추론). 즉
  - HyDE = "짧은 생성 1콜 + 임베딩(모델 무관 코사인)"
  - judge = "긴 문맥 추론 1콜"
  이 비대칭이 **약한/느린 로컬 LLM에서 벌어질 것**이라는 게 검증 가설이다.

## 가설

- **H1 (품질)**: 로컬 LLM에서 judge의 recall@K가 하락(추론 품질↓) → 운영점 @15에서 이미
  대등하던 스니펫 HyDE가 judge를 **역전**할 수 있다.
- **H2 (지연/비용)**: judge의 케이스당 지연이 급증(긴 입력 + 디코딩)한다. HyDE의 LLM
  부담은 짧은 생성 1콜이라 거의 안 변한다 → 지연 격차가 벌어진다.
- **H3 (강건성)**: 모델을 바꿔도 HyDE는 흔들림이 작다(임베딩 고정 + 짧은 생성). judge는
  모델 능력에 recall이 민감하다.

## 측정 항목 (방법별, 로컬 LLM 기준)

| 항목 | 본문/스니펫 HyDE | judge |
|---|---|---|
| recall@10/15/20/25 | 재측정 | 재측정 |
| 케이스당 리랭크 지연(s) | HyDE 문서 생성콜 시간 | 추론콜 시간 |
| 서빙 지표 | (생성콜) TTFT/E2EL/tok·s | (추론콜) TTFT/E2EL/tok·s |
| LLM 입력/출력 토큰 | 생성 1콜(작음) | 후보 통째(큼) |
| GPU/인프라 | 임베딩만(추론 GPU 불요) | 추론 GPU 상주 필요 |

임베딩(gemini-embedding-001)은 그대로 두고 **LLM만 로컬로** 스왑해 변수 격리. (선택: 로컬
임베딩 모델까지 바꾸는 2차 실험은 별도.)

## 셋업 (기존 인프라 재사용)

- v2에 이미 **로컬 LLM 배선**이 있다: `service_v2/infrastructure/model.py`의 `build_model`이
  `WH_MODEL=openai/<hf-id>` + `WH_MODEL_API_BASE`(vLLM OpenAI 호환) + `WH_MODEL_API_KEY`로
  LiteLlm 객체를 만든다. v2 `eval/benchmark.py serving`은 TTFT/E2EL/ITL도 잰다.
- v4는 `google.genai`를 직접 부르므로(`wh_poc/clients.py`), 이 재검증을 위해
  **judge/HyDE-생성 LLM 콜만 로컬 엔드포인트로 라우팅**하는 얇은 어댑터가 필요하다.
  가장 작은 변경: `experiments/rerank_eval.py`의 `gen_capture`를 OpenAI 호환 클라이언트로
  갈아끼우고(`WH_MODEL_API_BASE`), 나머지(임베딩·후보군·매칭)는 그대로 둔다.
- 후보 모델 예: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct 등 vLLM(RunPod) 상주.

## 절차

1. vLLM로 로컬 LLM 서빙(OpenAI 호환) → `WH_MODEL_API_BASE`/`WH_MODEL_API_KEY` 설정.
2. `rerank_eval.py`의 LLM 콜(gen_capture)만 로컬 엔드포인트로 스왑. 후보군은 동일 I 런 고정.
3. 동일 11케이스로 재실행 → recall@10/15/20/25 + 케이스당 지연 + 서빙 지표 수집.
4. 호스티드(현재) 결과와 나란히 표로 비교.

## 성공 판정 (HyDE 선택을 확정하는 조건)

- **판정 A**: 로컬에서 judge@15 ≤ 스니펫 HyDE@15 → "품질에서도 HyDE 우위" 확정(H1).
- **판정 B**: judge 케이스당 지연 ≫ HyDE, 또는 추론 GPU 상주 비용이 비현실적 → "비용/인프라
  에서 HyDE 우위" 확정(H2).
- 위 둘 중 하나만 성립해도 **"추론 LLM 불요 = HyDE의 로컬 친화"** 주장이 데이터로 선다.
- 반대로 judge가 로컬에서도 품질을 유지하고 지연이 감당 가능하면 → judge 재고 여지(정직하게
  그 결과도 기록).

## 산출물 (실행 시)

- `experiments/rerank_local_llm.json` (recall·지연·서빙 지표)
- `experiments/rerank_local_llm.md` (호스티드 vs 로컬 비교표 + 판정)
- 확정 시 [metrics.md](metrics.md) §리랭크 방법별 recall@K와 §두 카드에 로컬 결과 반영.
