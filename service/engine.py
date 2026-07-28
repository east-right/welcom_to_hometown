import time
import json
from datetime import datetime
from pathlib import Path

from google.adk.runners import InMemoryRunner
from google.genai import types

import agents
from schema import Feed, SlotExtraction, Slots

APP = "wh"
USER = "u1"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"run_{datetime.now():%Y%m%d}.log"


def _log(line: str) -> None:
    """콘솔 + 파일 동시 기록."""
    stamped = f"{datetime.now():%H:%M:%S} {line}"
    print(stamped)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(stamped + "\n")


async def _run_once(agent, text: str, verbose: bool = True) -> str:
    """에이전트 1회 실행 + 상세 로그.

    남기는 것:
      - 어떤 에이전트에 무슨 입력이 들어갔나
      - 각 툴을 무슨 인자로 부르고 뭘 받았나
      - 최종 출력
      - 전체 소요 시간 + 툴별 시간
    """
    runner = InMemoryRunner(agent=agent, app_name=APP)
    s = await runner.session_service.create_session(app_name=APP, user_id=USER)

    t_start = time.time()
    _log(f"┌─ [{agent.name}] 시작")
    _log(f"│  입력: {text[:120]}")

    final = ""
    pending_tool: dict = {}   # tool_name → 호출 시각

    async for ev in runner.run_async(
        user_id=USER, session_id=s.id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    ):
        author = ev.author
        for p in (ev.content.parts if ev.content else []) or []:
            # 툴/서브 호출
            if getattr(p, "function_call", None):
                fc = p.function_call
                pending_tool[fc.name] = time.time()
                args = json.dumps(dict(fc.args), ensure_ascii=False)
                _log(f"│  → [{author}] 호출 {fc.name}({args[:100]})")
            # 툴/서브 응답
            elif getattr(p, "function_response", None):
                fr = p.function_response
                dt = time.time() - pending_tool.get(fr.name, time.time())
                resp = str(fr.response)
                _log(f"│  ← [{author}] {fr.name} 응답 ({dt:.1f}초, {len(resp)}자)")
                _log(f"│      {resp[:150]}")

        if ev.is_final_response() and ev.content:
            final = "".join(x.text or "" for x in ev.content.parts)

    total = time.time() - t_start
    _log(f"│  출력: {final[:150]}")
    _log(f"└─ [{agent.name}] 완료 ({total:.1f}초)\n")
    return final


async def extract_slots(history: str) -> SlotExtraction:
    out = await _run_once(agents.slot_extractor, history)
    return SlotExtraction.model_validate_json(out)


async def search_and_format(slots: Slots, verbose: bool = True) -> Feed:
    request = (
        f"조건: 지역={slots.region}, 일자={slots.date}, "
        f"동반자={slots.companion}, 목적={slots.purpose}. "
        f"이 조건에 맞는 곳을 찾아줘."
    )
    _log("=" * 55)
    _log(f"[검색 요청] {slots.summary()}")

    t0 = time.time()
    text = await _run_once(agents.coordinator, request)
    _log(f"[코디네이터 전체] {time.time()-t0:.1f}초")

    formatted = await _run_once(agents.formatter, text)
    return Feed.model_validate_json(formatted)


def render_cards(feed: Feed) -> str:
    if not feed.cards:
        return "  조건에 맞는 결과를 찾지 못했어요."
    lines = []
    for i, c in enumerate(feed.cards, 1):
        head = f"  {i}. [{c.domain}] {c.name}"
        if c.period: head += f"  📅 {c.period}"
        if c.hours:  head += f"  🕒 {c.hours}"
        if c.closed: head += f"  🚫 {c.closed}"
        lines.append(head)
        lines.append(f"     {c.reason}")
        lines.append(f"     {c.source}")
    return "\n".join(lines)