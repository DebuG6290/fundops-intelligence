from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm.sarvam_provider import LLMConfigurationError, SarvamProvider


class FakeSarvamClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.chat = SimpleNamespace(completions=self.complete)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


def _response(content="{}", tool_calls=None):
    return SimpleNamespace(
        id="req-1",
        model="sarvam-105b",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls or []))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
    )


def test_live_provider_requires_environment_key(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError):
        SarvamProvider()


def test_sarvam_provider_uses_stable_v1_adapter_and_json_mode():
    client = FakeSarvamClient([_response('{"ok": true}')])
    provider = SarvamProvider(client=client, model="sarvam-105b")
    response = provider.create_response("system", {"exception_id": "EXC-1"}, [])

    assert response.output_text == '{"ok": true}'
    assert client.calls[0]["model"] == "sarvam-105b"
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert provider.last_call["request_id"] == "req-1"
    assert provider.last_call["input_tokens"] == 12


def test_sarvam_function_tools_map_to_chat_completion_shape():
    client = FakeSarvamClient([_response()])
    provider = SarvamProvider(client=client)
    provider.create_response("system", {}, [{
        "type": "function", "name": "lookup", "description": "Lookup record",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }])
    tool = client.calls[0]["tools"][0]
    assert tool["function"]["name"] == "lookup"
    assert tool["function"]["parameters"]["type"] == "object"


def test_provider_redacts_api_key_from_errors(monkeypatch):
    secret = "sk_test_secret"
    monkeypatch.setenv("SARVAM_API_KEY", secret)

    class BrokenClient:
        chat = SimpleNamespace(completions=lambda **kwargs: (_ for _ in ()).throw(ValueError(secret)))

    with pytest.raises(RuntimeError) as error:
        SarvamProvider(client=BrokenClient()).create_response("system", {}, [])
    assert secret not in str(error.value)
