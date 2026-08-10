"""모델 비교용 구조화 로그. 실행 1회(에이전트 1회 호출) = JSONL 한 줄.

사람용 로그(logger·tracer)와 별개로, 모델별 결과를 기계가 집계·비교할 수 있게
logs/runs.jsonl 에 append 한다. 모델을 바꿔가며 돌린 뒤 이 파일을 model 필드로
묶으면 지연·토큰·출력·실패를 나란히 비교할 수 있다.

한 줄 예:
  {"ts": "...", "model": "openai/meta-llama/Llama-3.1-8B-Instruct",
   "agent": "search_agent", "elapsed_sec": 2.3,
   "tokens": {"prompt": 812, "output": 64, "total": 876},
   "chars": 210, "output": "{...}"}
"""
import json
from datetime import datetime
from pathlib import Path


class RunRecorder:
    def __init__(self, path: str = "logs/runs.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(exist_ok=True)

    def record(self, **fields) -> None:
        row = {"ts": datetime.now().isoformat(timespec="seconds"), **fields}
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
