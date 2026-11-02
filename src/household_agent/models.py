"""Typed domain records shared by every component (design §6)."""

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class OwnerScope(StrEnum):
    PRIVATE = "private"
    JOINT = "joint"


class Role(StrEnum):
    HOUSEHOLD_ADMIN = "household_admin"
    ADULT_MEMBER = "adult_member"
    DELEGATE = "delegate"  # advisor/CPA: read-only, time-bounded
    DEPENDENT = "dependent"


class Action(StrEnum):
    UPLOAD = "upload"
    WRITE = "write"
    READ = "read"
    RETRIEVE = "retrieve"
    INDEX = "index"
    REPORT = "report"
    REPORT_AGGREGATE = "report_aggregate"
    EXPORT = "export"
    APPROVE = "approve"
    RESUME = "resume"


class ReconciliationStatus(StrEnum):
    RECONCILED = "reconciled"
    QUALIFIED = "qualified"
    UNRECONCILED = "unreconciled"


class Account(BaseModel):
    model_config = ConfigDict(frozen=True)

    account_id: str
    household_id: str
    kind: str  # checking, credit_card, brokerage, mortgage, ...
    owner_member_ids: frozenset[str]
    owner_scope: OwnerScope
    currency: str = "USD"


class Transaction(BaseModel):
    """Canonical transaction record. Rows are never updated; corrections add a new version."""

    model_config = ConfigDict(frozen=True)

    transaction_id: str
    version: int = 1
    household_id: str
    account_id: str
    owner_scope: OwnerScope
    posted_at: date
    amount_minor: int
    currency: str
    merchant_raw: str
    merchant_normalized: str
    category: str
    category_confidence: float = Field(ge=0, le=1)
    document_confidence: float = Field(ge=0, le=1)
    source_document_id: str
    source_locator: str
    parser_version: str
    reconciliation_status: ReconciliationStatus
    row_hash: str
    ingested_at: datetime
    correction_reason: str | None = None
