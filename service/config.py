"""환경 설정 로딩. 모든 설정은 여기를 통해서만 읽는다."""
import os

from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("WH_MODEL", "gemini-3.6-flash")

NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def check_keys() -> None:
    """필수 키 확인. 없으면 바로 알려준다."""
    missing = []
    if not NAVER_CLIENT_ID:
        missing.append("NAVER_CLIENT_ID")
    if not NAVER_CLIENT_SECRET:
        missing.append("NAVER_CLIENT_SECRET")
    if missing:
        raise RuntimeError(
            f".env에 다음 키가 없습니다: {', '.join(missing)}\n"
            "developers.naver.com에서 발급 후 .env에 넣어주세요."
        )
