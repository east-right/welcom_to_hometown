"""vLLM 서버 metric 실시간 샘플러.

실행 중인 vLLM(로컬 백엔드)의 /metrics(Prometheus)를 주기적으로 폴링해
KV 캐시 사용률·prefix 캐시 히트율·동시요청·토큰 처리량·서버측 TTFT/TPOT를
타임스탬프별 JSONL로 남긴다. eval과 나란히 띄워두면 어느 구간에서 KV가 차고
prefix가 얼마나 맞았는지 벽시계로 맞춰볼 수 있다.

    python vllm_metrics.py -o metrics.jsonl -i 0.5
    (eval을 다른 셸에서 돌리는 동안 켜두고, 끝나면 Ctrl-C 또는 kill)

누적 카운터는 값 그대로 + 직전 샘플 대비 델타(구간 히트율/처리량)를 함께 적는다.
"""
import argparse
import json
import time

import httpx
from dotenv import dotenv_values

# 폴링할 metric — gauge는 스냅샷, counter는 누적+델타로 본다.
GAUGES = ["vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting"]
COUNTERS = [
    "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
    "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count",
    "vllm:request_time_per_output_token_seconds_sum",
    "vllm:request_time_per_output_token_seconds_count",
]


def parse(text: str, names: list[str]) -> dict:
    """Prometheus 텍스트에서 이름별 값을 합산해 뽑는다(라벨 시리즈가 여럿이면 합)."""
    want = set(names)
    out = {n: 0.0 for n in names}
    for ln in text.splitlines():
        if not ln or ln.startswith("#"):
            continue
        base = ln.split("{", 1)[0].split(" ", 1)[0]
        if base in want:
            try:
                out[base] += float(ln.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    return out


def _rate(cur: dict, prev: dict, hits: str, qs: str) -> float | None:
    dq = cur[qs] - prev[qs]
    return round((cur[hits] - prev[hits]) / dq, 4) if dq > 0 else None


def _avg(cur: dict, prev: dict, s: str, c: str) -> float | None:
    dc = cur[c] - prev[c]
    return round((cur[s] - prev[s]) / dc, 4) if dc > 0 else None


def main():
    ap = argparse.ArgumentParser(description="vLLM /metrics 샘플러")
    ap.add_argument("-o", "--out", default="vllm_metrics.jsonl")
    ap.add_argument("-i", "--interval", type=float, default=0.5)
    ap.add_argument("-d", "--duration", type=float, default=0, help="초. 0=무한(kill까지)")
    args = ap.parse_args()

    env = dotenv_values("C:/VSCode/wh_poc/.env")
    base = env["WH_LOCAL_API_BASE"].rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    url = f"{root}/metrics"
    key = env.get("WH_LOCAL_API_KEY", "EMPTY")

    print(f"샘플링 {url}  every {args.interval}s  → {args.out}", flush=True)
    t0 = time.time()
    prev = None
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        while True:
            now = time.time()
            try:
                r = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=5)
                g = parse(r.text, GAUGES)
                c = parse(r.text, COUNTERS)
            except Exception:
                time.sleep(args.interval)      # 일시 실패는 건너뛴다
                continue

            rec = {
                "t": round(now - t0, 2),
                "kv_cache_perc": round(g["vllm:kv_cache_usage_perc"] * 100, 2),
                "running": int(g["vllm:num_requests_running"]),
                "waiting": int(g["vllm:num_requests_waiting"]),
                "prefix_hits": int(c["vllm:prefix_cache_hits_total"]),
                "prefix_queries": int(c["vllm:prefix_cache_queries_total"]),
                "prompt_tok": int(c["vllm:prompt_tokens_total"]),
                "prompt_tok_cached": int(c["vllm:prompt_tokens_cached_total"]),
                "gen_tok": int(c["vllm:generation_tokens_total"]),
            }
            if prev:
                rec["prefix_hit_rate_interval"] = _rate(
                    c, prev, "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")
                rec["ttft_avg_interval"] = _avg(
                    c, prev, "vllm:time_to_first_token_seconds_sum",
                    "vllm:time_to_first_token_seconds_count")
                rec["tpot_avg_interval"] = _avg(
                    c, prev, "vllm:request_time_per_output_token_seconds_sum",
                    "vllm:request_time_per_output_token_seconds_count")
            # 누적 prefix 히트율
            q = c["vllm:prefix_cache_queries_total"]
            rec["prefix_hit_rate_cum"] = round(c["vllm:prefix_cache_hits_total"] / q, 4) if q else None

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            prev = c
            n += 1
            if args.duration and (now - t0) >= args.duration:
                break
            time.sleep(args.interval)
    print(f"샘플 {n}개 기록 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
