"""웹 어댑터. CLI와 같은 search_agent 엔진을 HTTP/JSON으로 노출한다.

- GET  /              → 프론트(index.html)
- POST /api/chat      → 질문 → 추천 카드
- POST /api/reset     → (상태 없음) OK만 반환
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import check_keys
from ..main import build_search

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Welcome to Hometown")
_search = build_search()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ResetRequest(BaseModel):
    session_id: str


class ChatResponse(BaseModel):
    reply: str
    feed: list | None = None


@app.on_event("startup")
def _check_config() -> None:
    check_keys()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    feed = await _search.find(req.message)
    cards = [card.model_dump() for card in feed.cards]
    reply = "조건에 맞는 곳을 찾았어요!" if cards else "조건에 맞는 결과를 못 찾았어요. 다르게 말해줄래요?"
    return ChatResponse(reply=reply, feed=cards or None)


@app.post("/api/reset")
async def reset(req: ResetRequest) -> dict:
    return {"ok": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
