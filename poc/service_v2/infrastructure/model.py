"""모델 팩토리. 이름 하나로 Gemini(ADK 네이티브 문자열)와 원격 오픈모델(LiteLLM 객체)을 다룬다.

규칙 (WH_MODEL 값 기준):
  - 슬래시 없음  → ADK 네이티브 문자열 그대로.        예) "gemini-3.6-flash"
  - 슬래시 있음  → LiteLLM 식별자로 보고 LiteLlm 객체로 감싼다.
                   예) "openai/meta-llama/Llama-3.1-8B-Instruct"

HuggingFace 모델을 RunPod GPU에서 vLLM/TGI로 띄우면 보통 **OpenAI 호환 엔드포인트**가
나온다. 그 URL을 WH_MODEL_API_BASE로, 키를 WH_MODEL_API_KEY로 주면 그쪽으로 호출한다.

  WH_MODEL=openai/meta-llama/Llama-3.1-8B-Instruct
  WH_MODEL_API_BASE=https://<pod-id>-8000.proxy.runpod.net/v1
  WH_MODEL_API_KEY=EMPTY

모델 스왑은 이 env들만 바꾸면 된다.
"""


def build_model(name: str, api_base: str = "", api_key: str = ""):
    """모델 이름(+엔드포인트) → LlmAgent에 넘길 모델(str 또는 LiteLlm)."""
    if "/" not in name:
        return name  # Gemini 등 ADK 네이티브
    # 원격 오픈모델을 쓸 때만 임포트 (litellm 미설치 환경에서도 Gemini 경로는 동작)
    from google.adk.models.lite_llm import LiteLlm

    kwargs = {}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    return LiteLlm(model=name, **kwargs)


def model_label(model) -> str:
    """로그·기록용 모델 식별 문자열. str이면 그대로, LiteLlm이면 내부 model명."""
    if isinstance(model, str):
        return model
    return getattr(model, "model", type(model).__name__)
