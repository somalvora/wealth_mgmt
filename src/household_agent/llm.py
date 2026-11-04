"""Model selection (design §3.5). This is the only module that names a model.

Models are chosen by spec strings ``"<provider>:<name>"`` taken from the environment:

- ``HH_CHAT_MODEL``  default ``ollama:qwen2.5:7b`` (``ollama:qwen3:8b`` for comparison)
- ``HH_EMBED_MODEL`` default ``ollama:nomic-embed-text``; ``hash`` = offline deterministic

Constructing an Ollama client does not contact the server, so this module imports and
tests without Ollama installed. Unit tests inject LangChain fake chat models instead.
"""

import hashlib
import math
import os
import re
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

ENV_CHAT_MODEL = "HH_CHAT_MODEL"
ENV_EMBED_MODEL = "HH_EMBED_MODEL"
DEFAULT_CHAT_MODEL = "ollama:qwen2.5:7b"
DEFAULT_EMBED_MODEL = "ollama:nomic-embed-text"


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
    raise ValueError(f"unsupported chat model {s}")


def get_embeddings(spec: str | None = None) -> Embeddings:
    s = ModelSpec.parse(spec) if spec else embed_model_spec()
    if s.provider == "ollama" and s.name:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=s.name)
    if s.provider == "hash":
        return HashEmbeddings()
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
