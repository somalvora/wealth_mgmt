"""Statement reconciliation and document confidence (design §3.3, §6).

Material problems -> ``unreconciled`` (blocked from calculations).
Bounded, explainable exceptions -> ``qualified`` (provisional views only).
Everything passes -> ``reconciled``.
"""

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from household_agent.ingest import ParsedRow, ParsedStatement
from household_agent.models import ReconciliationStatus

CONFIDENCE_THRESHOLD = 0.85
TOLERANCE_MINOR = 0  # same-currency statements must balance to the cent
REVERSAL_WINDOW_DAYS = 3
SUPPORTED_CURRENCIES = frozenset({"USD"})

# Weights for document_confidence; they sum to 1.
CONFIDENCE_WEIGHTS = {
    "row_parse_rate": Decimal("0.35"),
    "balance_continuity": Decimal("0.25"),
    "totals_match": Decimal("0.10"),
    "duplicate_free": Decimal("0.10"),
    "mapped_merchant_rate": Decimal("0.20"),
}

Severity = Literal["material", "bounded", "info"]


@dataclass(frozen=True)
class ReconciliationIssue:
    code: str
    severity: Severity
    message: str
    amount_minor: int | None = None
    line_numbers: tuple[int, ...] = ()


@dataclass(frozen=True)
class ReconciliationResult:
    status: ReconciliationStatus
    issues: tuple[ReconciliationIssue, ...]
    document_confidence: float
    unverified: bool
    confidence_components: dict[str, float] = field(default_factory=dict)
    row_hashes: tuple[str, ...] = ()  # aligned with ParsedStatement.rows


def row_hashes(parsed: ParsedStatement) -> tuple[str, ...]:
    """Stable per-row hash. The occurrence index distinguishes genuine repeated rows inside one
    statement while still matching the same rows when an overlapping statement is re-imported."""
    seen: Counter[tuple] = Counter()
    out = []
    for r in parsed.rows:
        key = (parsed.header.account_id, r.posted_at.isoformat(), r.amount_minor, " ".join(r.description.upper().split()))
        seen[key] += 1
        out.append(hashlib.sha256(repr((*key, seen[key])).encode()).hexdigest())
    return tuple(out)


def reconcile(
    parsed: ParsedStatement,
    mapped: Sequence[bool],
    historical_hashes: set[str],
    supported_currencies: frozenset[str] = SUPPORTED_CURRENCIES,
) -> ReconciliationResult:
    if len(mapped) != len(parsed.rows):
        raise ValueError("mapped flags must align with parsed rows")
    h, rows = parsed.header, parsed.rows
    issues: list[ReconciliationIssue] = []

    if parsed.row_errors:
        issues.append(ReconciliationIssue(
            "row_parse_error", "material", f"{len(parsed.row_errors)} row(s) could not be parsed",
            line_numbers=tuple(e.line_number for e in parsed.row_errors),
        ))
    if h.currency not in supported_currencies:
        issues.append(ReconciliationIssue("unsupported_currency", "material", f"currency {h.currency} not supported"))
    if h.period_end < h.period_start:
        issues.append(ReconciliationIssue("invalid_period", "material", "period end precedes start"))

    outside = [r for r in rows if not h.period_start <= r.posted_at <= h.period_end]
    if outside:
        issues.append(ReconciliationIssue(
            "outside_period", "material", f"{len(outside)} row(s) fall outside the statement period",
            line_numbers=tuple(r.line_number for r in outside),
        ))

    net = sum(r.amount_minor for r in rows)
    gap = h.opening_balance_minor + net - h.closing_balance_minor
    balance_ok = abs(gap) <= TOLERANCE_MINOR
    if not balance_ok:
        issues.append(ReconciliationIssue(
            "balance_mismatch", "material", "opening balance + transactions does not equal closing balance",
            amount_minor=gap,
        ))

    credits = sum(r.amount_minor for r in rows if r.amount_minor > 0)
    debits = -sum(r.amount_minor for r in rows if r.amount_minor < 0)
    totals_ok = credits == h.total_credits_minor and debits == h.total_debits_minor
    if not totals_ok:
        issues.append(ReconciliationIssue(
            "totals_mismatch", "material",
            f"credits {credits} vs stated {h.total_credits_minor}; debits {debits} vs stated {h.total_debits_minor}",
        ))

    hashes = row_hashes(parsed)
    already = [r.line_number for r, rh in zip(rows, hashes, strict=True) if rh in historical_hashes]
    if already:
        issues.append(ReconciliationIssue(
            "duplicate_of_ingested", "material",
            f"{len(already)} row(s) already ingested from another statement", line_numbers=tuple(already),
        ))

    issues.extend(_possible_duplicates(rows))
    issues.extend(_reversal_pairs(rows))

    components = {
        "row_parse_rate": _ratio(len(rows), len(rows) + len(parsed.row_errors)),
        "balance_continuity": Decimal(int(balance_ok)),
        "totals_match": Decimal(int(totals_ok)),
        "duplicate_free": Decimal(int(not any(i.code in {"duplicate_of_ingested", "possible_duplicate"} for i in issues))),
        "mapped_merchant_rate": _ratio(sum(mapped), len(mapped)),
    }
    score = sum(CONFIDENCE_WEIGHTS[k] * v for k, v in components.items()).quantize(Decimal("0.01"))

    if any(i.severity == "material" for i in issues):
        status = ReconciliationStatus.UNRECONCILED
    elif any(i.severity == "bounded" for i in issues):
        status = ReconciliationStatus.QUALIFIED
    else:
        status = ReconciliationStatus.RECONCILED

    return ReconciliationResult(
        status=status,
        issues=tuple(issues),
        document_confidence=float(score),
        unverified=score < Decimal(str(CONFIDENCE_THRESHOLD)),
        confidence_components={k: float(v) for k, v in components.items()},
        row_hashes=hashes,
    )


def _ratio(numerator: int, denominator: int) -> Decimal:
    return Decimal(1) if denominator == 0 else Decimal(numerator) / Decimal(denominator)


def _possible_duplicates(rows: Sequence[ParsedRow]) -> list[ReconciliationIssue]:
    groups: dict[tuple, list[ParsedRow]] = {}
    for r in rows:
        groups.setdefault((r.posted_at, r.amount_minor, " ".join(r.description.upper().split())), []).append(r)
    return [
        ReconciliationIssue(
            "possible_duplicate", "bounded",
            f"{len(g)} identical rows on {g[0].posted_at}; confirm they are separate charges",
            amount_minor=g[0].amount_minor * (len(g) - 1),
            line_numbers=tuple(r.line_number for r in g),
        )
        for g in groups.values() if len(g) > 1
    ]


def _reversal_pairs(rows: Sequence[ParsedRow]) -> list[ReconciliationIssue]:
    issues, used = [], set()
    for i, a in enumerate(rows):
        for j in range(i + 1, len(rows)):
            b = rows[j]
            if (
                i not in used and j not in used
                and a.amount_minor == -b.amount_minor != 0
                and abs((a.posted_at - b.posted_at).days) <= REVERSAL_WINDOW_DAYS
            ):
                used.update((i, j))
                issues.append(ReconciliationIssue(
                    "reversal_pair", "info", "charge and matching reversal",
                    amount_minor=abs(a.amount_minor), line_numbers=(a.line_number, b.line_number),
                ))
    return issues
