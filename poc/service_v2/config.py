"""환경 설정 로딩. 모든 설정은 여기를 통해서만 읽는다."""
import os

from dotenv import load_dotenv

load_dotenv(override=True)

MODEL = os.environ.get("WH_MODEL", "gemini-3.6-flash")
# 원격 오픈모델(RunPod의 vLLM/TGI 등 OpenAI 호환 엔드포인트)로 붙을 때만 채운다.
MODEL_API_BASE = os.environ.get("WH_MODEL_API_BASE", "")
MODEL_API_KEY = os.environ.get("WH_MODEL_API_KEY", "")

NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def check_keys() -> None:
    """필수 키 확인. 없으면 바로 알려준다."""
    missing = [
        name
        for name, value in (
            ("NAVER_CLIENT_ID", NAVER_CLIENT_ID),
            ("NAVER_CLIENT_SECRET", NAVER_CLIENT_SECRET),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            f".env에 다음 키가 없습니다: {', '.join(missing)}\n"
            "developers.naver.com에서 발급 후 .env에 넣어주세요."
        )
