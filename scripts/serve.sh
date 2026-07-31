#!/usr/bin/env bash
# RunPod pod에서 vLLM을 모델별 올바른 파서와 함께 띄운다.
# pod 하나 켜두고, 모델 바꿀 때 Ctrl+C 후 ./serve.sh <key> 로 재실행 (pod 재시작 불필요).
#
# 사용:
#   chmod +x serve.sh
#   ./serve.sh 8b        # 또는 14b / 32b / 35b / 72b / gemma
#
# 로컬(.env)은 URL·키 그대로, WH_MODEL만 아래 M 값에 맞춰 hosted_vllm/<M> 로.
# 파서값은 각 모델 카드로 한 번씩 확정해서 여기 박아두면 그 뒤론 안 건드림.
set -euo pipefail

EXTRA=""; TOOL=""; REASON=""
case "${1:-}" in
  4b)    M="Qwen/Qwen3-4B";             TOOL=hermes;      REASON=deepseek_r1 ;;
  8b)    M="Qwen/Qwen3-8B";             TOOL=hermes;      REASON=deepseek_r1 ;;
  14b)   M="Qwen/Qwen3-14B";            TOOL=hermes;      REASON=deepseek_r1 ;;
  32b)   M="Qwen/Qwen3-32B";            TOOL=hermes;      REASON=deepseek_r1 ;;
  35b)   M="Qwen/Qwen3.6-35B-A3B";      TOOL=qwen3_coder; REASON=deepseek_r1; EXTRA="--max-num-seqs 32" ;;
  72b)   M="Qwen/Qwen2.5-72B-Instruct"; TOOL=hermes;      REASON="";          EXTRA="--tensor-parallel-size 2" ;;
  gemma) M="google/gemma-2-9b-it";      TOOL="";          REASON="" ;;   # 툴콜링 미지원(비교군)
  *) echo "usage: ./serve.sh <8b|14b|32b|35b|72b|gemma>"; exit 1 ;;
esac

A="--model $M --host 0.0.0.0 --port 8000 --max-model-len 32768 $EXTRA"
[ -n "$TOOL" ]   && A="$A --enable-auto-tool-choice --tool-call-parser $TOOL"
[ -n "$REASON" ] && A="$A --reasoning-parser $REASON"

echo ">> vllm serve $A"
exec vllm serve $A
