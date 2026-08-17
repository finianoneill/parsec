import json

import httpx
import pytest

from parsec.gateway.openai_adapter import OpenAIAdapter
from parsec.models.gateway import ModelRequest


def make_transport(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "judge-model-x",
                "choices": [
                    {"message": {"role": "assistant", "content": '{"synthesis_score": 4}'},
                     "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            },
        )

    return httpx.MockTransport(handler)


async def test_request_translation_and_response_mapping():
    captured: dict = {}
    adapter = OpenAIAdapter(api_key="sk-test", transport=make_transport(captured))
    request = ModelRequest(
        model="judge-model-x",
        max_tokens=500,
        system=[{"type": "text", "text": "You are grading."}],
        messages=[{"role": "user", "content": "Question:\nQ\n\nReport:\nA"}],
    )
    resp = await adapter.complete(request)

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    body = captured["body"]
    assert body["model"] == "judge-model-x"
    assert body["max_completion_tokens"] == 500
    assert body["messages"][0] == {"role": "system", "content": "You are grading."}
    assert body["messages"][1]["role"] == "user"

    assert resp.text == '{"synthesis_score": 4}'
    assert resp.usage.input_tokens == 50
    assert resp.usage.output_tokens == 10


async def test_tools_rejected():
    adapter = OpenAIAdapter(api_key="sk-test", transport=make_transport({}))
    request = ModelRequest(
        model="m", max_tokens=10, tools=[{"name": "t"}],
        messages=[{"role": "user", "content": "x"}],
    )
    with pytest.raises(NotImplementedError):
        await adapter.complete(request)


def test_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAIAdapter()
