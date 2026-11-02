"""Phase 1 vertical-slice demo: ``uv run python -m household_agent.demo``.

Runs parse -> authorize -> reconcile -> calculate on the synthetic household, then shows
the guardrails: an access denial, an unreconciled-data block, and audit-chain verification.
"""

import sqlite3
import uuid
from datetime import date

from household_finance import format_minor
from household_agent.audit import AuditLog
from household_agent.authz import AuthorizationError, Authorizer
from household_agent.calc_runner import CalculationRunner
from household_agent.categorize import Categorizer
from household_agent.ledger import LedgerStore
from household_agent.pipeline import ingest_statement
from household_agent.planning import plan_allocation
from household_agent.synthetic import demo_planning_inputs
from household_agent.synthetic import (
    ACCOUNTS, ALEX, DEMO_AS_OF, DEMO_OCTOBERS, HOUSEHOLD, JORDAN, RULES, demo_statements,
)


def main() -> None:
    conn = sqlite3.connect(":memory:")
    store = LedgerStore(conn, Authorizer(ACCOUNTS), AuditLog(conn))
    categorizer = Categorizer(RULES)
    corr = f"corr_demo_{uuid.uuid4().hex[:8]}"

    print("1. Ingest synthetic statements (owner uploads)")
    owners = {"jordan_card": JORDAN}
    for name, content in demo_statements().items():
        actor = next((a for key, a in owners.items() if key in name), ALEX)
        r = ingest_statement(store, categorizer, actor, content.encode(), name,
                             purpose="statement_ingestion", correlation_id=corr, at=DEMO_AS_OF)
        flag = " [unverified]" if r.unverified else ""
        print(f"   {name:<38} {r.status.value:<13} confidence={r.document_confidence:.2f}{flag}")
        for issue in r.issues:
            print(f"      - {issue.severity}: {issue.code} ({issue.message})")

    print("\n2. Year-over-year property-tax comparison (Ledger Analyst path)")
    txns = store.query_transactions(
        ALEX, "historical_comparison", corr, ["acct_rental_checking"], *DEMO_OCTOBERS,
        merchant_normalized="San Mateo County Tax Collector", at=DEMO_AS_OF,
    )
    run = CalculationRunner(store).run(
        "variance_between_transactions", txns, actor_id=ALEX.member_id, household_id=HOUSEHOLD,
        correlation_id=corr, acting_agent="ledger_analyst",
    )
    out = run.output
    print(f"   inputs: {[format_minor(t.amount_minor) for t in txns]} from {[t.source_locator for t in txns]}")
    print(f"   Variance: {format_minor(out['dollar_variance_minor'])} ({out['pct_variance']}%)  run_id={run.run_id}")
    print(f"   input_hash={run.input_hash[:23]}...  library=household_finance {run.library_version}")

    print("\n3. Guardrail: Jordan tries to read Alex's private checking")
    try:
        store.query_transactions(JORDAN, "budgeting", corr, ["acct_alex_checking"], *DEMO_OCTOBERS, at=DEMO_AS_OF)
    except AuthorizationError as exc:
        print(f"   DENIED: {exc}")

    print("\n4. Guardrail: the corrupt September statement is excluded from calculations")
    sept = store.query_transactions(ALEX, "budgeting", corr, ["acct_alex_checking"],
                                    date(2026, 9, 1), date(2026, 9, 30), include_provisional=True, at=DEMO_AS_OF)
    print(f"   September transactions available to calculations: {len(sept)}")

    print("\n5. Tree-of-Thought capital allocation (design §5.8; synthetic limits, scenario only)")
    inputs = demo_planning_inputs()
    plan = plan_allocation(inputs)
    print(f"   surplus={format_minor(inputs.surplus_minor)} above required reserve {format_minor(inputs.required_reserve_minor)}")
    print(f"   beam search b=3 k=2 depth=3: {plan.evaluations} nodes evaluated, "
          f"{sum(n.status in ('beam_cut', 'soft_pruned', 'hard_pruned') for n in plan.trace)} pruned")
    for leaf in plan.ranked_leaves[:3]:
        print(f"   {leaf.node_id:50} score={leaf.score.composite} liquidity={leaf.score.liquidity_safety}")
    best = plan.best
    print(f"   chosen: {best.node_id} -> " + ", ".join(f"{b}={format_minor(a)}" for b, a in best.allocation.items()))
    print(f"   status={plan.status} (interrupt for household approval; no money moves)")

    print(f"\n6. Audit chain verified: {store.audit.verify()} events, correlation_id={corr}")


if __name__ == "__main__":
    main()
