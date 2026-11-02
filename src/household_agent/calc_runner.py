"""The ``run_calculation`` tool (design §3.2, §3.4).

Agents name a function from ``REGISTRY`` and pass ledger transactions; they cannot supply
code. The runner enforces reconciliation eligibility, then records reproducible evidence:
library version, input hash, output, and output hash.
"""

import dataclasses
import hashlib
import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import household_finance as hf
from household_agent.ledger import LedgerStore
from household_agent.models import ReconciliationStatus, Transaction
from household_agent.reconcile import CONFIDENCE_THRESHOLD


class IneligibleInputError(ValueError):
    """An input transaction may not be used for this calculation (design §3.3)."""


def _variance_between(txns: Sequence[Transaction], params: dict) -> Any:
    if len(txns) != 2:
        raise hf.CalculationError(f"variance_between_transactions needs exactly 2 transactions, got {len(txns)}")
    prior, current = sorted(txns, key=lambda t: t.posted_at)
    return hf.variance(abs(prior.amount_minor), abs(current.amount_minor))


def _drift(txns: Sequence[Transaction], params: dict) -> Any:
    split = date.fromisoformat(params["current_period_start"])
    return hf.recurring_drift([t for t in txns if t.posted_at < split], [t for t in txns if t.posted_at >= split])


REGISTRY: dict[str, Callable[[Sequence[Transaction], dict], Any]] = {
    "cash_flow": lambda t, p: hf.cash_flow(t, date.fromisoformat(p["start"]), date.fromisoformat(p["end"])),
    "category_totals": lambda t, p: hf.category_totals(t, date.fromisoformat(p["start"]), date.fromisoformat(p["end"])),
    "variance_between_transactions": _variance_between,
    "recurring_drift": _drift,
}


@dataclass(frozen=True)
class CalculationRun:
    run_id: str
    fn: str
    library_version: str
    input_ids: list[str]
    params: dict
    input_hash: str
    output: Any
    output_hash: str
    provisional: bool


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (Decimal, date)):
        return str(value)
    return value


def _hash(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class CalculationRunner:
    def __init__(self, store: LedgerStore):
        self.store = store

    def run(
        self,
        fn: str,
        transactions: Sequence[Transaction],
        *,
        actor_id: str,
        household_id: str,
        correlation_id: str,
        params: dict | None = None,
        allow_provisional: bool = False,
        acting_agent: str | None = None,
    ) -> CalculationRun:
        params = params or {}
        if fn not in REGISTRY:
            raise KeyError(f"unknown calculation {fn!r}; allowed: {sorted(REGISTRY)}")

        provisional = False
        for t in transactions:
            if t.household_id != household_id:
                raise IneligibleInputError(f"{t.transaction_id} belongs to another household")
            if t.reconciliation_status is ReconciliationStatus.UNRECONCILED:
                raise IneligibleInputError(f"{t.transaction_id} is unreconciled")
            if t.reconciliation_status is ReconciliationStatus.QUALIFIED or t.document_confidence < CONFIDENCE_THRESHOLD:
                if not allow_provisional:
                    raise IneligibleInputError(f"{t.transaction_id} is provisional (qualified or unverified)")
                provisional = True

        inputs = sorted((t.transaction_id, t.version, t.amount_minor, t.row_hash) for t in transactions)
        input_hash = _hash({"fn": fn, "library_version": hf.__version__, "params": params, "inputs": inputs})
        output = _jsonable(REGISTRY[fn](transactions, params))
        run = CalculationRun(
            run_id=f"calc_run_{uuid.uuid4().hex[:12]}",
            fn=fn,
            library_version=hf.__version__,
            input_ids=[i[0] for i in inputs],
            params=params,
            input_hash=input_hash,
            output=output,
            output_hash=_hash(output),
            provisional=provisional,
        )
        now = datetime.now(UTC)
        self.store.record_calculation_run(
            {**dataclasses.asdict(run), "correlation_id": correlation_id, "created_at": now.isoformat()}
        )
        self.store.audit.append(
            correlation_id=correlation_id, actor_id=actor_id, household_id=household_id,
            action="calculate", decision="info", resource=f"calc:{run.run_id}", acting_agent=acting_agent,
            details={"fn": fn, "input_hash": input_hash, "output_hash": run.output_hash, "provisional": provisional},
            at=now,
        )
        return run
