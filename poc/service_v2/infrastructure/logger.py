"""실행 기록을 콘솔과 파일에 동시에 남긴다."""
from datetime import datetime
from pathlib import Path


class RunLogger:
    """한 줄 로그를 타임스탬프와 함께 콘솔·파일에 기록한다."""

    def __init__(self, log_dir: str = "logs") -> None:
        directory = Path(log_dir)
        directory.mkdir(exist_ok=True)
        self._file = directory / f"run_{datetime.now():%Y%m%d}.log"

    def write(self, line: str) -> None:
        stamped = f"{datetime.now():%H:%M:%S} {line}"
        print(stamped)
        with self._file.open("a", encoding="utf-8") as f:
            f.write(stamped + "\n")
