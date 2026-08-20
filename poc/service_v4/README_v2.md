# wh_poc v2 — 변경 요약

기존 v1(멀티턴) 대비 `collect_places`만 교체. 나머지 툴(collect_info, collect_context,
read_file, get_today)과 Chat/Tracer는 그대로.

## 무엇이 바뀌었나

**collect_places가 3단계 수집 → RRF 병합 → 정제 구조로 바뀜:**

```
1. 카카오 원 쿼리 45개 전량       kakao_all()
2. 업종어 확장 (LLM)              fanout_queries()
     "합성동 피자" → [이탈리안, 파스타, 화덕피자]
     "가포 맛집"   → [] (포괄어는 확장 포기, 3번이 보강)
3. 다이닝코드 Top 리스트 수확     diningcode_names() + verify_names()
     "{query} Top" 스니펫에서 매장명 추출 → 카카오 역조회로 실존 검증
4. RRF 병합                       rrf_merge(), k=60
5. 정제 (카카오/다이닝코드 쿼터)   _refine(), 기본 15개로 축소
```

## 사라진 것

- `MIN_COUNT`, `MAX_ATTEMPTS`, `status='retry'` — 어휘 재시도를 fan-out이 대신함
- 블로그 유래 폴백(`prompts.EXTRACT_NAMES` 경로) — 다이닝코드 경로로 대체

## 남은 것 (그대로 재사용)

- `collect_info`, `collect_context`, `read_file`, `get_today`
- `Chat`, `Tracer`, `INSTRUCTION`(일부 문구만 수정)
- eval.py, test.json, tests.py

## 실측 근거 (6~9케이스, 수동 확인)

4회 반복 평가에서 계속 미수집이던 매장들이 이 구조로 확보됨:

| 매장 | v1 | v2 |
|---|---|---|
| 우동한그릇 | 4회 미수집 | 확보 (다이닝코드) |
| 개성순대 | 4회 미수집 | 확보 (다이닝코드) |
| 뜨라또리아 다젠나 | 4회 미수집 | 1위 (업종 fan-out) |
| 모티키친 | 4회 미수집 | 확보 (업종 fan-out) |

## 다음 할 일

- [ ] eval.py로 14케이스 정식 재측정 (Recall@수집 / Recall@최종 / MRR)
- [ ] REFINE_TARGET(15) 값 자체를 10/15/20으로 스윕해서 비용 대비 손실 확인
- [ ] `in_same_region`이 여전히 놓치는 케이스(귀산옥 등) 개별 확인은 보류 —
      개별 진단보다 전체 재측정이 우선
