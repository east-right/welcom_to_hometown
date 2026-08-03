"""웹 어댑터. CLI(cli.py)와 나란히 같은 Recommender를 구동하되, HTTP/JSON으로 노출한다.

- GET  /              → 프론트(index.html)
- POST /api/chat      → 한 턴 처리(슬롯 되물음 or 검색 완료 시 피드)
- POST /api/reset     → 세션 초기화

대화 상태(히스토리·슬롯)는 세션ID별 인메모리로 보관한다(데모용).
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import check_keys
from ..domain.models import Feed, Slots
from ..main import build_recommender

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Welcome to Hometown")
_recommender = build_recommender()
_sessions: dict[str, "_WebChat"] = {}


# ── 요청/응답 스키마 ────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ResetRequest(BaseModel):
    session_id: str


class ChatResponse(BaseModel):
    slots: Slots
    ready: bool
    reply: str
    feed: list | None = None   # ready=true면 카드 목록, 아니면 None


# ── 세션: 한 대화의 히스토리·슬롯 + 한 턴 처리 ──────────
class _WebChat:
    """cli.Chat의 턴 흐름을 미러링하되, 출력 대신 결과를 반환한다."""

    def __init__(self, recommender) -> None:
        self._recommender = recommender
        self._history: list[str] = []
        self._slots = Slots()

    async def turn(self, user: str) -> ChatResponse:
        self._history.append(f"User: {user}")

        extraction = await self._recommender.understand("\n".join(self._history))
        self._slots = extraction.slots

        if not extraction.ready:
            return self._ask(extraction.question)
        return await self._recommend()

    def _ask(self, question: str) -> ChatResponse:
        reply = question or "조금만 더 알려주실 수 있을까요?"
        self._history.append(f"Assistant: {reply}")
        return ChatResponse(slots=self._slots, ready=False, reply=reply, feed=None)

    async def _recommend(self) -> ChatResponse:
        feed: Feed = await self._recommender.recommend(self._slots)
        self._history.append(f"Assistant: [{len(feed.cards)}개 추천 완료]")
        return ChatResponse(
            slots=self._slots,
            ready=True,
            reply="조건에 맞는 곳을 찾았어요! 피드에서 확인해보세요.",
            feed=[card.model_dump() for card in feed.cards],
        )


def _session(session_id: str) -> _WebChat:
    if session_id not in _sessions:
        _sessions[session_id] = _WebChat(_recommender)
    return _sessions[session_id]


# ── 라우트 ─────────────────────────────────────────
@app.on_event("startup")
def _check_config() -> None:
    check_keys()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    return await _session(req.session_id).turn(req.message)


@app.post("/api/reset")
async def reset(req: ResetRequest) -> dict:
    _sessions.pop(req.session_id, None)
    return {"ok": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
