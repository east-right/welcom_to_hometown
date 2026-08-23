"""FastAPI 웹 어댑터 — v4 Chat 엔진을 HTTP/JSON으로 노출한다.

  GET  /            → 프론트(web/index.html)
  POST /api/chat    → 질문 → 추천 카드 (session_id로 멀티턴 대화 유지)
  POST /api/reset   → 세션 대화 초기화

  실행: uvicorn api:app --reload    (poc/service_v4 디렉터리에서)
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wh_poc import Chat, set_verbose

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Welcome to Hometown")
set_verbose(False)

# v4는 멀티턴이라 session_id별로 Chat(대화 상태)을 유지한다. CLI(run.py)와 같은 엔진.
_sessions: dict[str, Chat] = {}


def _session(sid: str) -> Chat:
    if sid not in _sessions:
        _sessions[sid] = Chat(show_status=False, show_tools=False)
    return _sessions[sid]


def _to_card(p: dict) -> dict:
    """v4 PlaceCard → 프론트 카드. 카테고리로 도메인(배지)을 추정한다."""
    cat = p.get("category", "")
    if p.get("period"):
        domain = "festival"
    elif "카페" in cat:
        domain = "cafe"
    elif "관광" in cat or "명소" in cat:
        domain = "spot"
    else:
        domain = "restaurant"
    return {
        "domain": domain,
        "name": p.get("name", ""),
        "reason": p.get("reason", ""),
        "address": p.get("address", ""),
        "highlights": p.get("highlights", []),
        "period": p.get("period", ""),
        "caution": p.get("caution", ""),
        "source": p.get("url", ""),
    }


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ResetRequest(BaseModel):
    session_id: str


class ChatResponse(BaseModel):
    reply: str
    kind: str = "chat"
    feed: list | None = None


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    out = await _session(req.session_id).send(req.message)
    result = out["result"] or {}
    cards = [_to_card(p) for p in result.get("places", [])]
    return ChatResponse(
        reply=result.get("message", ""),
        kind=result.get("kind", "chat"),
        feed=cards or None,
    )


@app.post("/api/reset")
async def reset(req: ResetRequest) -> dict:
    _sessions.pop(req.session_id, None)
    return {"ok": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
