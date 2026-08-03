"""vLLM /metrics(Prometheus) 스냅샷. 서버가 잰 값이라 클라↔서버 '통신 시간'이 안 들어간다.

리포트용 순수 지표(모델시간·TTFT·ITL)를 여기서 얻는다. 실행 전후로 snapshot을 찍어
그 사이(=한 스텝) 델타를 계산한다. 단일 클라이언트·순차 실행이라 델타 귀속이 깨끗하다.
"""
import httpx

# 관심 히스토그램 (vLLM 버전 따라 이름이 다를 수 있어 후보를 둔다)
_INFER = ["vllm:request_inference_time_seconds"]          # 순수 추론시간(prefill+decode)
_TTFT = ["vllm:time_to_first_token_seconds"]              # 첫 토큰까지
_ITL = ["vllm:time_per_output_token_seconds",             # 토큰당 시간(=ITL)
        "vllm:inter_token_latency_seconds"]


class VllmMetrics:
    def __init__(self, api_base: str, api_key: str = "") -> None:
        root = api_base.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        self._url = f"{root}/metrics"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def snapshot(self) -> dict:
        """현재 누적 히스토그램의 {sum, count}."""
        try:
            text = httpx.get(self._url, headers=self._headers, timeout=10.0).text
        except httpx.HTTPError:
            return {"sum": {}, "count": {}}
        return _parse(text)

    def step(self, before: dict, after: dict) -> dict:
        """스텝 전후 스냅샷 → 서버측 지표(통신시간 제외)."""
        return {
            "model_sec": _sum_delta(before, after, _INFER),   # 이 스텝 총 추론시간
            "ttft": _mean_delta(before, after, _TTFT),        # 평균 TTFT
            "itl": _mean_delta(before, after, _ITL),          # 평균 ITL
        }


def _parse(text: str) -> dict:
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        head = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            value = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        if head.endswith("_sum"):
            sums[head[:-4]] = sums.get(head[:-4], 0.0) + value
        elif head.endswith("_count"):
            counts[head[:-6]] = counts.get(head[:-6], 0.0) + value
    return {"sum": sums, "count": counts}


def _sum_delta(before: dict, after: dict, names: list[str]) -> float | None:
    for n in names:
        if n in after["sum"]:
            return after["sum"].get(n, 0.0) - before["sum"].get(n, 0.0)
    return None


def _mean_delta(before: dict, after: dict, names: list[str]) -> float | None:
    for n in names:
        if n in after["sum"]:
            ds = after["sum"].get(n, 0.0) - before["sum"].get(n, 0.0)
            dc = after["count"].get(n, 0.0) - before["count"].get(n, 0.0)
            if dc > 0:
                return ds / dc
    return None
