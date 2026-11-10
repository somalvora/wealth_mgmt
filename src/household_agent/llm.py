"""Model selection (design §3.5). This is the only module that names a model.

Models are chosen by spec strings ``"<provider>:<name>"`` taken from the environment:

- ``HH_CHAT_MODEL``  default ``ollama:qwen2.5:7b`` (``ollama:qwen3:8b`` for comparison);
  hosted APIs for synthetic data only: ``gemini:<model>`` (e.g. ``gemini:gemini-2.5-flash``,
  Google AI Studio) or ``groq:<model>`` (e.g. ``groq:qwen/qwen3-32b``)
- ``HH_EMBED_MODEL`` default ``ollama:nomic-embed-text``; ``hash`` = offline deterministic;
  ``gemini:gemini-embedding-001`` hosted

Hosted settings (environment or a ``.env`` file in the working directory, never committed):

- ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) for ``gemini:``; ``GROQ_API_KEY`` for ``groq:``
- ``HH_REQUESTS_PER_MINUTE`` default 10; one limiter per provider is shared by every model
  in the process, so the router and the specialists together stay under the free-tier limit
- ``HH_LLM_CACHE`` default ``.cache/llm.sqlite``; ``off`` disables. Answers are cached by
  prompt and model settings, so re-running an eval makes no new calls (delete to re-sample)

Constructing an Ollama, Gemini or Groq client does not contact the server, so this module imports
and tests without either. Unit tests inject LangChain fake chat models instead.
"""

import hashlib
import json
import math
import os
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values
from langchain_core.caches import BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import message_to_dict, messages_from_dict
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.rate_limiters import InMemoryRateLimiter

ENV_CHAT_MODEL = "HH_CHAT_MODEL"
ENV_EMBED_MODEL = "HH_EMBED_MODEL"
DEFAULT_CHAT_MODEL = "ollama:qwen2.5:7b"
DEFAULT_EMBED_MODEL = "ollama:nomic-embed-text"
ENV_GROQ_KEY = "GROQ_API_KEY"
ENV_GEMINI_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
ENV_RPM = "HH_REQUESTS_PER_MINUTE"
ENV_CACHE = "HH_LLM_CACHE"
DEFAULT_RPM = 10
DEFAULT_CACHE = ".cache/llm.sqlite"
HOSTED_MAX_RETRIES = 6  # 429s are retried with backoff
HOSTED_TIMEOUT_S = 90  # one request that gets no answer fails (and is retried) instead of hanging the run


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    name: str

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        provider, _, name = spec.strip().partition(":")
        if not provider:
            raise ValueError(f"invalid model spec {spec!r}")
        return cls(provider.lower(), name)

    def __str__(self) -> str:
        return f"{self.provider}:{self.name}" if self.name else self.provider


def chat_model_spec() -> ModelSpec:
    return ModelSpec.parse(os.environ.get(ENV_CHAT_MODEL, DEFAULT_CHAT_MODEL))


def embed_model_spec() -> ModelSpec:
    return ModelSpec.parse(os.environ.get(ENV_EMBED_MODEL, DEFAULT_EMBED_MODEL))


def get_chat_model(spec: str | None = None, *, temperature: float = 0.0) -> BaseChatModel:
    s = ModelSpec.parse(spec) if spec else chat_model_spec()
    if s.provider == "ollama" and s.name:
        from langchain_ollama import ChatOllama

        return ChatOllama(model=s.name, temperature=temperature)
    if s.provider == "groq" and s.name:
        from langchain_groq import ChatGroq

        key = setting(ENV_GROQ_KEY)
        if not key:
            raise ValueError(f"{s} needs {ENV_GROQ_KEY} in the environment or .env")
        return ChatGroq(
            model=s.name,
            temperature=temperature,
            api_key=key,
            max_retries=HOSTED_MAX_RETRIES,
            timeout=HOSTED_TIMEOUT_S,
            rate_limiter=_shared_limiter("groq"),
            cache=_cache(),
            # Qwen3 otherwise prefixes answers with <think>…</think>, which breaks JSON parsing
            reasoning_format="hidden" if "qwen3" in s.name.lower() else None,
        )
    if s.provider == "gemini" and s.name:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=s.name,
            temperature=temperature,
            google_api_key=_gemini_key(s),
            max_retries=HOSTED_MAX_RETRIES,
            timeout=HOSTED_TIMEOUT_S,
            rate_limiter=_shared_limiter("gemini"),
            cache=_cache(),
        )
    raise ValueError(f"unsupported chat model {s}")


def setting(name: str, default: str | None = None) -> str | None:
    """Environment first, then ``.env`` in the working directory; ``.env`` never mutates os.environ."""
    if name in os.environ:
        return os.environ[name]
    return dotenv_values(".env").get(name) if Path(".env").is_file() else default


def _gemini_key(s: ModelSpec) -> str:
    key = next((k for k in map(setting, ENV_GEMINI_KEYS) if k), None)
    if not key:
        raise ValueError(f"{s} needs {ENV_GEMINI_KEYS[0]} in the environment or .env")
    return key


_LIMITERS: dict[str, InMemoryRateLimiter] = {}


def _shared_limiter(provider: str) -> InMemoryRateLimiter:
    if provider not in _LIMITERS:
        rpm = float(setting(ENV_RPM) or DEFAULT_RPM)
        if rpm <= 0:
            raise ValueError(f"{ENV_RPM} must be positive, got {rpm}")
        _LIMITERS[provider] = InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.1)
    return _LIMITERS[provider]


def _cache() -> BaseCache | None:
    path = setting(ENV_CACHE) or DEFAULT_CACHE
    return None if path.lower() == "off" else SQLiteChatCache(path)


def get_embeddings(spec: str | None = None) -> Embeddings:
    s = ModelSpec.parse(spec) if spec else embed_model_spec()
    if s.provider == "ollama" and s.name:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=s.name)
    if s.provider == "hash":
        return HashEmbeddings()
    if s.provider == "gemini" and s.name:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=s.name, google_api_key=_gemini_key(s))
    raise ValueError(f"unsupported embedding model {s}")


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings for tests and offline demos.

    Each lowercase token is hashed into one of ``dim`` buckets and the vector is
    L2-normalized, so texts sharing words have higher cosine similarity. Not semantic.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class SQLiteChatCache(BaseCache):
    """Disk cache of chat answers keyed by (prompt, model settings incl. bound tools).

    Messages are stored as plain dicts (``message_to_dict``), not pickles or LangChain's
    generic object loader, so reading the cache can only produce message objects.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._run("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, body TEXT NOT NULL)")

    def _run(self, sql: str, args: tuple = ()) -> list[tuple]:
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                return conn.execute(sql, args).fetchall()
        finally:
            conn.close()

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        return hashlib.sha256(json.dumps([prompt, llm_string]).encode()).hexdigest()

    def lookup(self, prompt: str, llm_string: str) -> list[Generation] | None:
        rows = self._run("SELECT body FROM answers WHERE key = ?", (self._key(prompt, llm_string),))
        if not rows:
            return None
        stored = json.loads(rows[0][0])
        messages = messages_from_dict([g["message"] for g in stored])
        return [ChatGeneration(message=m, generation_info=g["info"]) for m, g in zip(messages, stored, strict=True)]

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        if not all(isinstance(g, ChatGeneration) for g in return_val):
            return  # only chat answers are cached
        body = json.dumps([{"message": message_to_dict(g.message), "info": g.generation_info} for g in return_val])
        self._run("INSERT OR REPLACE INTO answers (key, body) VALUES (?, ?)", (self._key(prompt, llm_string), body))

    def clear(self, **kwargs) -> None:
        self._run("DELETE FROM answers")
