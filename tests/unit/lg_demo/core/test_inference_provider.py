import pytest

from lg_demo.core.model_provider import ChatOllamaProvider, HfCloudProvider


@pytest.fixture(autouse=True)
def reset_model_registry():
    import lg_demo.core.model_provider as module

    module.ModelProvider._ModelProvider__MODEL_REGISTRY.clear()


def test_chat_ollama_provider_builds_model_with_given_params(monkeypatch):
    captured = {}

    class FakeChatOllama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import lg_demo.core.model_provider as module

    monkeypatch.setattr(module, "ChatOllama", FakeChatOllama)

    provider = ChatOllamaProvider(model_name="llama3", temperature=0.25, num_gpu=2)

    assert provider.get_model().__class__ is FakeChatOllama
    assert captured == {"model": "llama3", "temperature": 0.25, "num_gpu": 2}


def test_hf_cloud_provider_builds_openai_client_with_hf_router(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import lg_demo.core.model_provider as module

    monkeypatch.setattr(module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("HF_CLOUD_MODEL", "hf/model")

    provider = HfCloudProvider()

    assert provider.get_model().__class__ is FakeChatOpenAI
    assert captured["model"] == "hf/model"
    assert captured["base_url"] == "https://router.huggingface.co/v1"
    assert captured["api_key"] == "test-token"


def test_hf_cloud_provider_raises_when_hf_token_missing(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_CLOUD_MODEL", "hf/model")
    provider = HfCloudProvider()

    with pytest.raises(KeyError):
        provider.get_model()
