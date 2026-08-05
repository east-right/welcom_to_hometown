"""LLM 출력에서 JSON만 안전하게 뽑는 유틸.

thinking 모델이 <think> 블록이나 잡텍스트를 앞뒤로 붙여도, 파싱 가능한
마지막 JSON 객체만 골라낸다(모델은 보통 최종 답을 끝에 둔다).
"""
import json
import re

_DECODER = json.JSONDecoder()


def json_only(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    best, i = None, text.find("{")
    while i != -1:
        try:
            _, end = _DECODER.raw_decode(text[i:])
            best, i = text[i:i + end], text.find("{", i + end)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
    return best if best is not None else text
