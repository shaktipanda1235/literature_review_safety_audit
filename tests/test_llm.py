import pytest

from app import llm
from app.config import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER


@pytest.mark.parametrize("role", ["grader", "auditor"])
def test_get_llm_uses_environment_model_and_zero_temperature(monkeypatch, role):
    calls = []

    def fake_init_chat_model(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(llm, "init_chat_model", fake_init_chat_model)
    monkeypatch.setenv("LLM_PROVIDER", "example_provider")
    monkeypatch.setenv("LLM_MODEL", "example-model-v2")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.8")

    model = llm.get_llm(role)

    assert model is not None
    assert calls == [
        {
            "model": "example-model-v2",
            "model_provider": "example_provider",
            "temperature": 0.0,
        }
    ]


def test_get_llm_uses_configured_defaults_and_default_temperature(monkeypatch):
    calls = []
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    monkeypatch.setattr(llm, "init_chat_model", lambda **kwargs: calls.append(kwargs))

    llm.get_llm()

    assert calls == [
        {
            "model": DEFAULT_LLM_MODEL,
            "model_provider": DEFAULT_LLM_PROVIDER,
            "temperature": 0.2,
        }
    ]


def test_prompt_loader_reads_packaged_prompt_and_rejects_paths():
    prompt = llm.load_prompt("safety_review")

    assert "untrusted evidence" in prompt
    assert "document IDs" in prompt
    with pytest.raises(ValueError, match="plain filename"):
        llm.load_prompt("../secrets")


def test_fake_llm_returns_deterministic_response():
    fake = llm.create_fake_llm(["controlled response"])

    assert fake.invoke("test prompt").content == "controlled response"