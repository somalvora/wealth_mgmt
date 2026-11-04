"""Verified retrieval over ChromaDB (design §8): ``retrieve_verified`` and ``validate_claims``.

Access control happens *before* similarity search. The service asks the ``Authorizer``
which accounts the actor may ``retrieve`` for the stated purpose and sends that list to
Chroma as a mandatory ``where`` predicate, so an unauthorized chunk is never a candidate
and nothing is redacted after the fact. Public Tier-1 sources (for example an IRS
publication) use the pseudo-account ``PUBLIC``. Each household has its own collections.

Pipeline: authorize → metadata filter (household, accounts, status, jurisdiction, tax year)
→ up to 20 candidates → MMR (λ = 0.65) → up to 5 passages → per-passage verification
(content hash, authority tier, freshness, tax year) → audit.

Embeddings always come from a LangChain ``Embeddings`` object (``llm.get_embeddings()``);
Chroma's own embedding function is never used. Retrieved text is untrusted data: chunks
that look like instructions are flagged at indexing time and audited as data-quality
findings, never followed.
"""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

from langchain_core.embeddings import Embeddings

from household_agent.audit import AuditLog
from household_agent.authz import Actor, AuthorizationError, Authorizer, Decision
from household_agent.models import Action, Role

RETRIEVAL_VERSION = "retrieval_1.0"
COLLECTIONS = ("financial_history", "policies_compliance")
PUBLIC = "PUBLIC"
CANDIDATE_K = 20
FINAL_K = 5
MMR_LAMBDA = 0.65
CHUNK_WORDS = 500  # design: 500 tokens / 100 overlap; words approximate tokens here
CHUNK_OVERLAP = 100
STATEMENT_ROWS_PER_CHUNK = 25
CANNOT_VERIFY = ("The system cannot verify the rule for this tax year. "
                 "This item needs an updated official source or CPA review.")

_HEADING = re.compile(r"^(?:#+\s+|Section\s+\d+\b)", re.IGNORECASE)
_INJECTION = re.compile(
    r"ignore (?:all |any )?(?:previous|prior|above) instructions|disregard (?:the )?(?:system|previous)|"
    r"you are now|system prompt|send (?:this|the) (?:statement|file|data) to",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


class RetrievalRequestError(ValueError):
    """A retrieval request is missing a mandatory predicate or is malformed."""


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    household_id: str
    collection: Literal["financial_history", "policies_compliance"]
    account_id: str  # access scope; ``PUBLIC`` for public Tier-1 sources
    source_name: str
    doc_type: str
    authority_tier: int
    publisher: str
    text: str
    jurisdiction: str = ""
    effective_from: date | None = None
    effective_to: date | None = None
    effective_tax_year: int | None = None
    valid_until: date | None = None
    statement_period: str = ""  # YYYY-MM for statements
    document_status: str = "active"
    source_version: str = "1"

    @property
    def source_hash(self) -> str:
        return "sha256:" + hashlib.sha256(self.text.encode()).hexdigest()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    section: str


def _window(words: list[str], size: int, overlap: int) -> list[str]:
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


def chunk_document(doc: SourceDocument, *, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Split on headings first, then by a sliding word window within long sections."""
    if doc.doc_type == "bank_statement":
        return chunk_statement(doc)
    sections: list[tuple[str, list[str]]] = []
    for line in doc.text.splitlines():
        if _HEADING.match(line.strip()) or not sections:
            sections.append((line.strip().lstrip("#").strip() if _HEADING.match(line.strip()) else "", []))
        sections[-1][1].append(line)
    chunks = []
    for title, lines in sections:
        for part in _window(" ".join(lines).split(), size, overlap):
            chunks.append(Chunk(f"{doc.doc_id}_{len(chunks):03d}", part, title))
    return chunks


def chunk_statement(doc: SourceDocument, rows_per_chunk: int = STATEMENT_ROWS_PER_CHUNK) -> list[Chunk]:
    """Row groups with the header (``#`` lines and column header) repeated in every chunk."""
    lines = [ln for ln in doc.text.splitlines() if ln.strip()]
    header = [ln for ln in lines if ln.startswith("#") or ln.startswith("date,")]
    rows = [ln for ln in lines if ln not in header]
    return [
        Chunk(f"{doc.doc_id}_{i // rows_per_chunk:03d}", "\n".join(header + rows[i:i + rows_per_chunk]),
              f"rows {i + 1}-{min(i + rows_per_chunk, len(rows))}")
        for i in range(0, len(rows), rows_per_chunk)
    ]


def _mmr(query: Sequence[float], candidates: Sequence[Sequence[float]], k: int, lam: float) -> list[int]:
    dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
    relevance = [dot(query, c) for c in candidates]
    chosen: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(chosen) < k:
        def score(i: int) -> float:
            redundancy = max((dot(candidates[i], candidates[j]) for j in chosen), default=0.0)
            return lam * relevance[i] - (1 - lam) * redundancy
        best = max(remaining, key=lambda i: (score(i), -i))
        chosen.append(best)
        remaining.remove(best)
    return chosen


@dataclass
class RetrievalService:
    client: Any  # chromadb client; tests use PersistentClient(tmp_path)
    embeddings: Embeddings
    authorizer: Authorizer
    audit: AuditLog

    def _collection(self, household_id: str, name: str):
        if name not in COLLECTIONS:
            raise RetrievalRequestError(f"unknown collection {name!r}")
        return self.client.get_or_create_collection(
            f"{household_id}__{name}", embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )

    # --- indexing --------------------------------------------------------------------

    def _authorize_index(self, actor: Actor, doc: SourceDocument, at: datetime) -> Decision:
        if actor.household_id != doc.household_id:
            return Decision(False, "actor is not in the document's household")
        if doc.account_id == PUBLIC:
            ok = actor.role in (Role.HOUSEHOLD_ADMIN, Role.ADULT_MEMBER) and doc.authority_tier == 1
            return Decision(ok, "public Tier-1 source" if ok else "only adults may index public Tier-1 sources")
        return self.authorizer.authorize(actor, doc.account_id, Action.INDEX, "document_indexing", at)

    def index(self, actor: Actor, doc: SourceDocument, *, correlation_id: str, at: datetime) -> list[str]:
        decision = self._authorize_index(actor, doc, at)
        chunks = chunk_document(doc) if decision else []
        flagged = [c.chunk_id for c in chunks if _INJECTION.search(c.text)]
        self.audit.append(
            correlation_id=correlation_id, actor_id=actor.member_id, household_id=actor.household_id,
            action=Action.INDEX.value, purpose="document_indexing", resource=f"doc:{doc.doc_id}",
            decision="allow" if decision else "deny", reason=decision.reason,
            policy_version=decision.policy_version, consent_version=decision.consent_version, at=at,
            details={"chunks": len(chunks), "source_hash": doc.source_hash, "instruction_like_chunks": flagged},
        )
        if not decision:
            raise AuthorizationError(decision)
        if not chunks:
            return []
        vectors = self.embeddings.embed_documents([c.text for c in chunks])
        self._collection(doc.household_id, doc.collection).upsert(
            ids=[c.chunk_id for c in chunks], embeddings=vectors, documents=[c.text for c in chunks],
            metadatas=[self._metadata(doc, c, c.chunk_id in flagged) for c in chunks],
        )
        return [c.chunk_id for c in chunks]

    @staticmethod
    def _metadata(doc: SourceDocument, chunk: Chunk, injection_suspect: bool) -> dict[str, Any]:
        iso = lambda d: d.isoformat() if d else ""  # noqa: E731  (Chroma metadata cannot be None)
        return {
            "doc_id": doc.doc_id, "household_id": doc.household_id, "account_id": doc.account_id,
            "source_name": doc.source_name, "doc_type": doc.doc_type, "authority_tier": doc.authority_tier,
            "publisher": doc.publisher, "jurisdiction": doc.jurisdiction, "effective_from": iso(doc.effective_from),
            "effective_to": iso(doc.effective_to), "effective_tax_year": doc.effective_tax_year or 0,
            "valid_until": iso(doc.valid_until), "statement_period": doc.statement_period,
            "document_status": doc.document_status, "source_version": doc.source_version,
            "source_hash": doc.source_hash, "section": chunk.section,
            "chunk_hash": "sha256:" + hashlib.sha256(chunk.text.encode()).hexdigest(),
            "injection_suspect": injection_suspect,
        }

    # --- retrieval -------------------------------------------------------------------

    def allowed_scopes(self, actor: Actor, purpose: str, at: datetime) -> list[str]:
        if actor.role is Role.DEPENDENT:
            return []
        accounts = [a.account_id for a in self.authorizer.accounts(actor.household_id)
                    if self.authorizer.authorize(actor, a.account_id, Action.RETRIEVE, purpose, at)]
        return [*accounts, PUBLIC]

    def retrieve_verified(
        self,
        actor: Actor,
        purpose: str,
        query: str,
        *,
        correlation_id: str,
        at: datetime,
        collections: Sequence[str] = COLLECTIONS,
        jurisdictions: Sequence[str] = (),
        tax_years: Sequence[int] = (),
        statement_periods: Sequence[str] = (),
        candidate_k: int = CANDIDATE_K,
        final_k: int = FINAL_K,
        mmr_lambda: float = MMR_LAMBDA,
    ) -> dict[str, Any]:
        if not purpose or not query.strip():
            raise RetrievalRequestError("retrieval requires a purpose and a non-empty query")
        scopes = self.allowed_scopes(actor, purpose, at)
        qvec = self.embeddings.embed_query(query)
        candidates: list[dict[str, Any]] = []
        for name in collections if scopes else ():  # dependents: nothing to search
            col = self._collection(actor.household_id, name)
            where = self._where(actor.household_id, scopes, name, jurisdictions, tax_years, statement_periods)
            r = col.query(query_embeddings=[qvec], n_results=candidate_k, where=where,
                          include=["documents", "metadatas", "embeddings", "distances"])
            for cid, text, meta, emb in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["embeddings"][0],
                                            strict=True):
                candidates.append({"chunk_id": cid, "collection": name, "text": text, "metadata": meta,
                                   "_embedding": [float(x) for x in emb]})
        candidates.sort(key=lambda c: c["chunk_id"])  # deterministic MMR tie-breaks
        picked = _mmr(qvec, [c["_embedding"] for c in candidates], final_k, mmr_lambda)
        passages = [self._verify({k: v for k, v in candidates[i].items() if k != "_embedding"}, at, tax_years)
                    for i in picked]
        self.audit.append(
            correlation_id=correlation_id, actor_id=actor.member_id, household_id=actor.household_id,
            acting_agent="policy_compliance", action=Action.RETRIEVE.value, purpose=purpose, decision="info",
            reason=RETRIEVAL_VERSION, at=at,
            details={"query": query, "scopes": scopes, "collections": list(collections),
                     "candidates": [c["chunk_id"] for c in candidates],
                     "selected": [p["chunk_id"] for p in passages],
                     "outcomes": {p["chunk_id"]: p["verification"] for p in passages},
                     "instruction_like": [p["chunk_id"] for p in passages if p["metadata"]["injection_suspect"]]},
        )
        return {"passages": passages, "candidates": len(candidates), "retrieval_version": RETRIEVAL_VERSION}

    @staticmethod
    def _where(household_id: str, scopes: list[str], collection: str, jurisdictions: Sequence[str],
               tax_years: Sequence[int], statement_periods: Sequence[str]) -> dict[str, Any]:
        clauses: list[dict[str, Any]] = [
            {"household_id": household_id},
            {"account_id": {"$in": scopes}},
            {"document_status": "active"},
        ]
        if collection == "policies_compliance":
            if jurisdictions:
                clauses.append({"jurisdiction": {"$in": list(jurisdictions)}})
            if tax_years:
                clauses.append({"$or": [{"effective_tax_year": {"$in": list(tax_years)}}, {"effective_tax_year": 0}]})
        if collection == "financial_history" and statement_periods:
            clauses.append({"statement_period": {"$in": list(statement_periods)}})
        return {"$and": clauses}

    @staticmethod
    def _verify(p: dict[str, Any], at: datetime, tax_years: Sequence[int]) -> dict[str, Any]:
        m = p["metadata"]
        if "sha256:" + hashlib.sha256(p["text"].encode()).hexdigest() != m["chunk_hash"]:
            outcome = "hash_mismatch"
        elif m["valid_until"] and date.fromisoformat(m["valid_until"]) < at.date():
            outcome = "expired"
        elif m["effective_to"] and date.fromisoformat(m["effective_to"]) < at.date() and not tax_years:
            outcome = "expired"
        elif tax_years and m["effective_tax_year"] and m["effective_tax_year"] not in tax_years:
            outcome = "wrong_period"
        else:
            outcome = "verified"
        return {**p, "authority_tier": m["authority_tier"], "verification": outcome}


# --- claim validation ------------------------------------------------------------------

ClaimKind = Literal["tax", "fact", "calculated"]


@dataclass(frozen=True)
class Claim:
    text: str
    kind: ClaimKind
    evidence_ids: tuple[str, ...] = ()
    calc_run_ids: tuple[str, ...] = ()
    tax_year: int | None = None


def _numbers(text: str) -> set[str]:
    """Numbers in canonical form: no separators, no trailing decimal zeros ("3,200.00" → "3200")."""
    out = set()
    for n in _NUMBER.findall(text):
        n = n.replace(",", "")
        out.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


_STOPWORDS = frozenset("""about after also been before does from have into more only over that than their them then
there these they this under were what when where which while with would your""".split())
MIN_WORD_OVERLAP = 0.5  # share of a claim's content words that its cited passages must contain


def _content_words(text: str) -> set[str]:
    """Lower-cased words of 4+ letters, without stopwords or a plural "s"."""
    words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", text)} - _STOPWORDS
    return {w[:-1] if w.endswith("s") and not w.endswith("ss") else w for w in words}


def validate_claims(
    claims: Sequence[Claim],
    passages: Sequence[dict[str, Any]],
    calc_outputs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Per-claim verdict: ``supported``, ``remove`` or ``rephrase_as_question`` (design §8 step 8).

    ``passages`` are the results of ``retrieve_verified`` in this job; ``calc_outputs`` maps
    calculation ``run_id`` to its stored output. A claim may only cite those. Every number
    in a claim must appear in its cited evidence, which blocks invented amounts, and a claim
    citing passages must share at least half of its content words with them, which blocks a
    true-sounding sentence pinned on an unrelated source.
    """
    by_id = {p["chunk_id"]: p for p in passages}
    calc_outputs = calc_outputs or {}
    verdicts = []
    for claim in claims:
        cited = [by_id[i] for i in claim.evidence_ids if i in by_id]
        usable = [p for p in cited if p["verification"] == "verified"]
        evidence_text = " ".join(p["text"] for p in usable) + " " + " ".join(
            str(calc_outputs[r]) for r in claim.calc_run_ids if r in calc_outputs)
        unknown = [i for i in claim.evidence_ids if i not in by_id] + [r for r in claim.calc_run_ids if r not in calc_outputs]

        def verdict(v: str, reason: str, message: str | None = None) -> dict[str, Any]:
            return {"claim": claim.text, "verdict": v, "reason": reason, "message": message,
                    "evidence_ids": [p["chunk_id"] for p in usable], "calc_run_ids": list(claim.calc_run_ids)}

        if unknown:
            verdicts.append(verdict("remove", f"cites evidence not retrieved in this job: {unknown}"))
        elif claim.kind == "calculated" and not claim.calc_run_ids:
            verdicts.append(verdict("remove", "calculated claim without a calculation run"))
        elif claim.kind == "tax" and not any(
            p["authority_tier"] == 1 and p["metadata"]["effective_tax_year"] == claim.tax_year for p in usable
        ):
            verdicts.append(verdict("rephrase_as_question", "no verified Tier-1 source for the tax year", CANNOT_VERIFY))
        elif claim.kind != "calculated" and not usable:
            verdicts.append(verdict("rephrase_as_question", "no verified supporting passage"))
        elif missing := sorted(_numbers(claim.text) - _numbers(evidence_text)):
            verdicts.append(verdict("remove", f"numbers not found in cited evidence: {missing}"))
        elif usable and (words := _content_words(claim.text)) and len(
                words & _content_words(" ".join(p["text"] for p in usable))) < MIN_WORD_OVERLAP * len(words):
            verdicts.append(verdict("rephrase_as_question", "cited passages do not say what the claim says"))
        else:
            verdicts.append(verdict("supported", "supported by cited evidence"))
    return verdicts
