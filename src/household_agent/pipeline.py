"""Ingestion pipeline: parse -> categorize -> reconcile -> authorize + persist."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from household_agent.authz import Actor
from household_agent.categorize import Categorizer
from household_agent.ingest import parse_statement
from household_agent.ledger import LedgerStore
from household_agent.models import ReconciliationStatus
from household_agent.reconcile import ReconciliationIssue, reconcile


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    source_name: str
    status: ReconciliationStatus
    document_confidence: float
    unverified: bool
    issues: tuple[ReconciliationIssue, ...]


def ingest_statement(
    store: LedgerStore,
    categorizer: Categorizer,
    actor: Actor,
    content: bytes,
    source_name: str,
    *,
    purpose: str,
    correlation_id: str,
    at: datetime | None = None,
    allow_pdf: bool = True,
) -> IngestResult:
    parsed = parse_statement(content, source_name, allow_pdf=allow_pdf)
    categorized = [categorizer.categorize(r.description) for r in parsed.rows]
    recon = reconcile(
        parsed,
        [c.mapped for c in categorized],
        store.historical_row_hashes(parsed.header.account_id),
    )
    document_id = store.ingest_document(actor, purpose, correlation_id, parsed, categorized, recon, at)
    return IngestResult(document_id, source_name, recon.status, recon.document_confidence, recon.unverified, recon.issues)


def ingest_file(store: LedgerStore, categorizer: Categorizer, actor: Actor, path: Path, **kwargs) -> IngestResult:
    return ingest_statement(store, categorizer, actor, path.read_bytes(), path.name, **kwargs)
