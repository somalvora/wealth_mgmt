"""Synthetic, non-sensitive demo household (design §12 step 1).

All names, accounts, and amounts are fictional. ``render_statement`` derives the header
totals from the rows so generated statements reconcile unless deliberately corrupted.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from household_finance import to_minor
from household_agent.authz import Actor, ConsentGrant
from household_agent.categorize import CategoryRule
from household_agent.ingest import FORMAT_ID
from household_agent.models import Account, Action, OwnerScope, Role

HOUSEHOLD = "hh_demo"

ALEX = Actor("alex", HOUSEHOLD, Role.HOUSEHOLD_ADMIN)
JORDAN = Actor("jordan", HOUSEHOLD, Role.ADULT_MEMBER)
CASEY_CPA = Actor("casey_cpa", HOUSEHOLD, Role.DELEGATE)
SAM = Actor("sam", HOUSEHOLD, Role.DEPENDENT)

ACCOUNTS = [
    Account(account_id="acct_rental_checking", household_id=HOUSEHOLD, kind="checking",
            owner_member_ids=frozenset({"alex", "jordan"}), owner_scope=OwnerScope.JOINT),
    Account(account_id="acct_alex_checking", household_id=HOUSEHOLD, kind="checking",
            owner_member_ids=frozenset({"alex"}), owner_scope=OwnerScope.PRIVATE),
    Account(account_id="acct_jordan_card", household_id=HOUSEHOLD, kind="credit_card",
            owner_member_ids=frozenset({"jordan"}), owner_scope=OwnerScope.PRIVATE),
]

RULES = [
    CategoryRule("SAN MATEO TAX COLLECTOR", "San Mateo County Tax Collector", "property_tax"),
    CategoryRule("RENT DEPOSIT", "Unit B Tenant", "rental_income"),
    CategoryRule("MAPLE ST HOA", "Maple Street HOA", "hoa_dues"),
    CategoryRule("BAYSIDE PLUMBING", "Bayside Plumbing", "repairs"),
    CategoryRule("PAYROLL", "Employer Payroll", "salary"),
    CategoryRule("SUMMIT MORTGAGE", "Summit Mortgage", "mortgage"),
    CategoryRule("PG&E", "PG&E", "utilities"),
    CategoryRule("CITY WATER", "City Water", "utilities"),
    CategoryRule("SAFEWAY", "Safeway", "groceries"),
    CategoryRule("STREAMCO", "StreamCo", "entertainment"),
    CategoryRule("CARD PAYMENT", "Card Payment", "transfer"),
]


def cpa_grant(expires_at: datetime) -> ConsentGrant:
    """Both owners let the CPA read the joint rental account for tax preparation."""
    return ConsentGrant(
        grant_id="grant_cpa_rental", household_id=HOUSEHOLD, grantor_id="alex", grantee_id="casey_cpa",
        account_ids=frozenset({"acct_rental_checking"}), actions=frozenset({Action.READ}),
        purposes=frozenset({"tax_preparation"}), effective_at=datetime(2026, 1, 1, tzinfo=UTC),
        expires_at=expires_at, version="cv_1",
    )


@dataclass(frozen=True)
class Row:
    posted_at: str
    description: str
    amount: str


def render_statement(
    account_id: str,
    period_start: str,
    period_end: str,
    opening_balance: str,
    rows: list[Row],
    *,
    closing_override: str | None = None,
) -> str:
    amounts = [to_minor(r.amount) for r in rows]
    opening = to_minor(opening_balance)
    closing = opening + sum(amounts)
    credits = sum(a for a in amounts if a > 0)
    debits = -sum(a for a in amounts if a < 0)

    def fmt(minor: int) -> str:
        return f"{'-' if minor < 0 else ''}{abs(minor) // 100}.{abs(minor) % 100:02d}"

    header = {
        "format": FORMAT_ID, "household_id": HOUSEHOLD, "account_id": account_id, "currency": "USD",
        "period_start": period_start, "period_end": period_end, "opening_balance": opening_balance,
        "closing_balance": closing_override or fmt(closing), "total_credits": fmt(credits), "total_debits": fmt(debits),
    }
    lines = [f"# {k}: {v}" for k, v in header.items()] + ["date,description,amount"]
    lines += [f"{r.posted_at},{r.description},{r.amount}" for r in rows]
    return "\n".join(lines) + "\n"


def demo_statements() -> dict[str, str]:
    return {
        "2025_10_rental_checking.csv": render_statement(
            "acct_rental_checking", "2025-10-01", "2025-10-31", "15000.00",
            [
                Row("2025-10-01", "RENT DEPOSIT UNIT B", "3100.00"),
                Row("2025-10-05", "MAPLE ST HOA DUES", "-425.00"),
                Row("2025-10-12", "BAYSIDE PLUMBING INV 2231", "-380.00"),
                Row("2025-10-18", "SAN MATEO TAX COLLECTOR - SECURED PROPERTY TAX - UNIT B", "-2750.00"),
            ],
        ),
        "2026_10_rental_checking.csv": render_statement(
            "acct_rental_checking", "2026-10-01", "2026-10-31", "17240.00",
            [
                Row("2026-10-01", "RENT DEPOSIT UNIT B", "3100.00"),
                Row("2026-10-05", "MAPLE ST HOA DUES", "-450.00"),
                Row("2026-10-09", "CITY WATER UTILITY", "-118.40"),
                Row("2026-10-18", "SAN MATEO TAX COLLECTOR - SECURED PROPERTY TAX - UNIT B", "-3200.00"),
            ],
        ),
        "2026_10_alex_checking.csv": render_statement(
            "acct_alex_checking", "2026-10-01", "2026-10-31", "9200.00",
            [
                Row("2026-10-01", "SUMMIT MORTGAGE PMT", "-3450.00"),
                Row("2026-10-03", "PAYROLL ACME CORP", "4100.00"),
                Row("2026-10-11", "PG&E ENERGY BILL", "-212.37"),
                Row("2026-10-14", "SAFEWAY #1182", "-186.52"),
                Row("2026-10-17", "PAYROLL ACME CORP", "4100.00"),
                Row("2026-10-28", "SAFEWAY #1182", "-143.08"),
            ],
        ),
        "2026_10_jordan_card.csv": render_statement(
            "acct_jordan_card", "2026-10-01", "2026-10-31", "-820.00",
            [
                Row("2026-10-02", "STREAMCO MONTHLY", "-12.99"),
                Row("2026-10-07", "SAFEWAY #0457", "-96.10"),
                Row("2026-10-07", "SAFEWAY #0457", "-96.10"),
                Row("2026-10-15", "CARD PAYMENT THANK YOU", "820.00"),
                Row("2026-10-21", "NOVAPIXEL DIGITAL", "-42.99"),
            ],
        ),
        # Deliberately broken: stated closing balance does not match the rows.
        "2026_09_alex_checking_corrupt.csv": render_statement(
            "acct_alex_checking", "2026-09-01", "2026-09-30", "8000.00",
            [
                Row("2026-09-03", "PAYROLL ACME CORP", "4100.00"),
                Row("2026-09-10", "SUMMIT MORTGAGE PMT", "-3450.00"),
            ],
            closing_override="9999.99",
        ),
    }


def write_demo_statements(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in demo_statements().items():
        path = directory / name
        path.write_text(content)
        paths.append(path)
    return paths


DEMO_AS_OF = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
DEMO_OCTOBERS = (date(2025, 10, 1), date(2026, 10, 31))
