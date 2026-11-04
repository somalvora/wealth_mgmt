import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings

from household_agent import llm
from household_agent.llm import (
    DEFAULT_CHAT_MODEL,
    ENV_CACHE,
    ENV_CHAT_MODEL,
    ENV_EMBED_MODEL,
    ENV_GROQ_KEY,
    ENV_RPM,
    HashEmbeddings,
    ModelSpec,
    SQLiteChatCache,
    chat_model_spec,
    get_chat_model,
    get_embeddings,
)

FAKE_KEY = "gsk_test_not_a_real_key"


@pytest.fixture
def groq_env(monkeypatch, tmp_path):
    """Fake key, cache in tmp, fresh shared limiter; no network is touched by construction."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENV_GROQ_KEY, FAKE_KEY)
    monkeypatch.setenv(ENV_CACHE, str(tmp_path / "cache.sqlite"))
    monkeypatch.delenv(ENV_RPM, raising=False)
    monkeypatch.setattr(llm, "_LIMITERS", {})
    return tmp_path


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


# --- Groq provider ----------------------------------------------------------------------

def test_groq_spec_builds_a_rate_limited_cached_client(groq_env):
    model = get_chat_model("groq:qwen/qwen3-32b")
    assert isinstance(model, ChatGroq) and model.model_name == "qwen/qwen3-32b"
    assert model.temperature < 1e-6  # ChatGroq sends 1e-8 for 0; the API rejects exactly 0
    assert model.groq_api_key.get_secret_value() == FAKE_KEY and model.max_retries == llm.GROQ_MAX_RETRIES
    assert model.reasoning_format == "hidden"  # Qwen3 <think> blocks would break JSON replies
    assert isinstance(model.cache, SQLiteChatCache) and model.cache.path == groq_env / "cache.sqlite"
    assert model.rate_limiter.requests_per_second == pytest.approx(20 / 60)


def test_all_groq_models_share_one_rate_limiter(groq_env):
    router, specialist = get_chat_model("groq:qwen/qwen3-32b"), get_chat_model("groq:llama-3.1-8b-instant")
    assert router.rate_limiter is specialist.rate_limiter
    assert specialist.reasoning_format is None


def test_key_and_limits_can_come_from_dotenv_without_touching_environ(groq_env, monkeypatch):
    monkeypatch.delenv(ENV_GROQ_KEY)
    (groq_env / ".env").write_text(f"{ENV_GROQ_KEY}=gsk_from_file\n{ENV_RPM}=6\n")
    model = get_chat_model("groq:llama-3.1-8b-instant")
    assert model.groq_api_key.get_secret_value() == "gsk_from_file"
    assert model.rate_limiter.requests_per_second == pytest.approx(0.1)
    assert ENV_GROQ_KEY not in os.environ


def test_cache_off_and_env_spec(groq_env, monkeypatch):
    monkeypatch.setenv(ENV_CACHE, "off")
    monkeypatch.setenv(ENV_CHAT_MODEL, "groq:qwen/qwen3-32b")
    assert get_chat_model().cache is None


def test_cached_answer_is_replayed_without_calling_the_model(tmp_path):
    cache = SQLiteChatCache(tmp_path / "c.sqlite")
    first = GenericFakeChatModel(messages=iter([AIMessage("routed: ledger")]), cache=cache)
    assert first.invoke("route this").content == "routed: ledger"
    # a fresh model with no answers left can only succeed by reading the cache, even after reopening the file
    again = GenericFakeChatModel(messages=iter([]), cache=SQLiteChatCache(tmp_path / "c.sqlite"))
    assert again.invoke("route this").content == "routed: ledger"


def test_cached_tool_calls_survive_the_round_trip(tmp_path):
    cache = SQLiteChatCache(tmp_path / "c.sqlite")
    msg = AIMessage("", tool_calls=[{"name": "query_ledger", "args": {"month": "2026-10"}, "id": "c1"}])
    cache.update("p", "m", [ChatGeneration(message=msg, generation_info={"finish_reason": "tool_calls"})])
    [gen] = cache.lookup("p", "m")
    assert gen.message.tool_calls == msg.tool_calls and gen.generation_info == {"finish_reason": "tool_calls"}


def test_api_key_never_reaches_the_cache(groq_env):
    model = get_chat_model("groq:qwen/qwen3-32b")
    assert FAKE_KEY not in model._get_llm_string()
    model.cache.update("p", model._get_llm_string(), [ChatGeneration(message=AIMessage("ok"))])
    assert FAKE_KEY.encode() not in model.cache.path.read_bytes()


# --- Groq negative paths ----------------------------------------------------------------

def test_groq_without_a_key_fails_before_any_call(groq_env, monkeypatch):
    monkeypatch.delenv(ENV_GROQ_KEY)
    with pytest.raises(ValueError, match=ENV_GROQ_KEY):
        get_chat_model("groq:qwen/qwen3-32b")


@pytest.mark.parametrize("rpm", ["0", "-5"])
def test_non_positive_rate_rejected(groq_env, monkeypatch, rpm):
    monkeypatch.setenv(ENV_RPM, rpm)
    with pytest.raises(ValueError, match=ENV_RPM):
        get_chat_model("groq:qwen/qwen3-32b")


def test_groq_needs_a_model_name_and_has_no_embeddings(groq_env):
    with pytest.raises(ValueError):
        get_chat_model("groq:")
    with pytest.raises(ValueError):
        get_embeddings("groq:nomic-embed-text")


def test_cache_misses_and_ignores_non_chat_generations(tmp_path):
    cache = SQLiteChatCache(tmp_path / "c.sqlite")
    cache.update("p", "m", [Generation(text="plain")])
    assert cache.lookup("p", "m") is None
    cache.update("p", "m", [ChatGeneration(message=AIMessage("a"))])
    assert cache.lookup("p", "other model") is None and cache.lookup("other prompt", "m") is None
    cache.clear()
    assert cache.lookup("p", "m") is None


@settings(max_examples=40, deadline=None)
@given(st.text(max_size=80), st.text(max_size=80), st.text(max_size=200))
def test_cache_round_trips_any_prompt_and_answer(prompt, llm_string, answer):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cache = SQLiteChatCache(f"{tmp}/c.sqlite")
        cache.update(prompt, llm_string, [ChatGeneration(message=AIMessage(answer))])
        [gen] = cache.lookup(prompt, llm_string)
        assert gen.message.content == answer
