"""실행 기록을 콘솔과 파일에 동시에 남긴다."""
import sys
from datetime import datetime
from pathlib import Path


class RunLogger:
    """한 줄 로그를 타임스탬프와 함께 콘솔·파일에 기록한다."""

    def __init__(self, log_dir: str = "logs") -> None:
        directory = Path(log_dir)
        directory.mkdir(exist_ok=True)
        self._file = directory / f"run_{datetime.now():%Y%m%d}.log"
        # Windows 콘솔(cp949) 등에서 이모지·대시가 깨지지 않도록 UTF-8로 전환(가능하면).
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    def write(self, line: str) -> None:
        stamped = f"{datetime.now():%H:%M:%S} {line}"
        self._console(stamped)
        with self._file.open("a", encoding="utf-8") as f:
            f.write(stamped + "\n")

    @staticmethod
    def _console(text: str) -> None:
        """콘솔 인코딩이 특정 문자를 못 써도 절대 예외로 죽지 않는다."""
        try:
            print(text)
        except UnicodeEncodeError:
            enc = sys.stdout.encoding or "utf-8"
            print(text.encode(enc, "replace").decode(enc))
