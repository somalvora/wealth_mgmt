import pytest
from hypothesis import given
from hypothesis import strategies as st
from langchain_ollama import ChatOllama, OllamaEmbeddings

from household_agent.llm import (
    DEFAULT_CHAT_MODEL,
    ENV_CHAT_MODEL,
    ENV_EMBED_MODEL,
    HashEmbeddings,
    ModelSpec,
    chat_model_spec,
    get_chat_model,
    get_embeddings,
)


def test_spec_keeps_colons_in_model_name():
    assert ModelSpec.parse("ollama:qwen3:8b") == ModelSpec("ollama", "qwen3:8b")
    assert str(ModelSpec.parse("hash")) == "hash"


def test_default_and_env_override(monkeypatch):
    monkeypatch.delenv(ENV_CHAT_MODEL, raising=False)
    assert str(chat_model_spec()) == DEFAULT_CHAT_MODEL
    monkeypatch.setenv(ENV_CHAT_MODEL, "ollama:qwen3:8b")
    model = get_chat_model()
    assert isinstance(model, ChatOllama) and model.model == "qwen3:8b" and model.temperature == 0


def test_embeddings_select_by_env(monkeypatch):
    monkeypatch.setenv(ENV_EMBED_MODEL, "hash")
    assert isinstance(get_embeddings(), HashEmbeddings)
    monkeypatch.setenv(ENV_EMBED_MODEL, "ollama:nomic-embed-text")
    assert isinstance(get_embeddings(), OllamaEmbeddings)


@pytest.mark.parametrize("spec", ["openai:gpt-x", "ollama", ":x", "ollama:"])
def test_unknown_or_incomplete_spec_rejected(spec):
    with pytest.raises(ValueError):
        get_chat_model(spec)


def test_hash_embeddings_similarity():
    e = HashEmbeddings()
    q = e.embed_query("property tax increase")
    near, far = e.embed_documents(["Secured property tax bill", "grocery receipt"])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
    assert dot(q, near) > dot(q, far)


@given(st.text(max_size=200))
def test_hash_embeddings_are_deterministic_and_normalized(text):
    e = HashEmbeddings(dim=64)
    v = e.embed_query(text)
    assert v == e.embed_query(text) and len(v) == 64
    norm = sum(x * x for x in v)
    assert norm == 0 or abs(norm - 1) < 1e-9
