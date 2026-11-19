"""Quality eval areas (design §2 success criteria): numbers, anomalies, retrieval, privacy.

``evals.py`` generates and scores these with the other areas; this module holds their
labeled data and scorers so each oracle stays independent of the code it checks:

- calculation:  random reconciled statements; expected totals come from the rows and a fixed
                merchant → category table, never from ``household_finance``
- anomaly:      seeded single events, small-transaction clusters, bursts and card-testing
                probes, plus controls that must stay quiet
- retrieval:    labeled queries over synthetic policy sources; document-level recall@5,
                leaks judged by account ownership, and citation precision of ``validate_claims``
                on correct, wrong-source and altered-number claims
- search_count: a scripted model drives PolicyCompliance through repeated, distinct and over-budget
                queries; the ``retrieve`` audit rows must show repeats collapsed and at most
                ``MAX_QUERY_REFINEMENTS + 1`` real searches
- privacy:      canary strings planted in one member's private statement and note must never
                reach another member, the dependent or the delegate (briefing, retrieval, tools,
                and every prompt a specialist model is shown)

Membership inference needs a fine-tuned model, so it stays in ``NOT_COVERED``.
"""

import json
import random
import re
import sqlite3
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

from household_agent.agents import (
    MAX_QUERY_REFINEMENTS,
    LedgerAnalyst,
    PolicyCompliance,
    Router,
    Supervisor,
    TaskAssignment,
)
from household_agent.audit import AuditLog
from household_agent.authz import Actor, Authorizer
from household_agent.briefing import draft_monthly_briefing, render_markdown
from household_agent.calc_runner import CalculationRunner
from household_agent.categorize import Categorizer
from household_agent.ledger import LedgerStore
from household_agent.models import Role
from household_agent.pipeline import ingest_statement
from household_agent.retrieval import PUBLIC, Claim, RetrievalService, SourceDocument, validate_claims
from household_agent.synthetic import (
    ACCOUNTS,
    SOMAL,
    CASEY_CPA,
    DEMO_AS_OF,
    HOUSEHOLD,
    SAYALI,
    RULES,
    VIAAN,
    Row,
    demo_policy_documents,
    render_statement,
)
from household_agent.tools import ToolSession
from household_agent.vertical_slice import PlaybackChatModel, _call, _final

MEMBERS = {a.member_id: a for a in (SOMAL, SAYALI, CASEY_CPA, VIAAN)}
OCTOBER = date(2026, 10, 1)
_OWNERS = {a.account_id: a.owner_member_ids for a in ACCOUNTS}
_MONEY = r"\$([\d,]+\.\d\d)"


def _cents(n: int) -> str:
    return f"{'-' if n < 0 else ''}{abs(n) // 100}.{abs(n) % 100:02d}"


def _store() -> LedgerStore:
    conn = sqlite3.connect(":memory:")
    return LedgerStore(conn, Authorizer(ACCOUNTS), AuditLog(conn))


def _ingest(store: LedgerStore, actor: Actor, account_id: str, month: str, rows: list[Row], opening: str = "5000.00",
            name: str = "") -> None:
    last = {"09": "30", "10": "31"}[month[-2:]]
    text = render_statement(account_id, f"{month}-01", f"{month}-{last}", opening, rows)
    ingest_statement(store, Categorizer(RULES), actor, text.encode(), name or f"{account_id}_{month}.csv",
                     purpose="statement_ingestion", correlation_id="eval_seed", at=DEMO_AS_OF)


def _brief(store: LedgerStore, actor: Actor, corr: str):
    return draft_monthly_briefing(store, CalculationRunner(store), actor, month=OCTOBER, as_of=DEMO_AS_OF,
                                  correlation_id=corr)


# --- calculation match -----------------------------------------------------------------------

# The oracle's own category table (deliberately not ``synthetic.RULES``).
CALC_MERCHANTS = {"PAYROLL ACME": "salary", "SAFEWAY": "groceries", "PG&E": "utilities", "CITY WATER": "utilities",
                  "STREAMCO": "entertainment", "SUMMIT MORTGAGE": "mortgage"}


def calculation_cases(rng: random.Random, n: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out = []
    for i in range(n):
        opening = rng.randrange(0, 2_000_000)
        rows, amounts, cats = [], [], {}
        for k, d in enumerate(sorted(rng.choices(range(1, 29), k=rng.randrange(2, 12)))):
            merchant = rng.choice(sorted(CALC_MERCHANTS))
            amount = rng.randrange(1, 240_000) * (1 if CALC_MERCHANTS[merchant] == "salary" else -1)
            rows.append(asdict(Row(f"2026-10-{d:02d}", f"{merchant} {i}-{k}", _cents(amount))))
            amounts.append(amount)
            cats[CALC_MERCHANTS[merchant]] = cats.get(CALC_MERCHANTS[merchant], 0) + amount
        inflow = sum(a for a in amounts if a > 0)
        outflow = -sum(a for a in amounts if a < 0)
        out.append(({"opening": _cents(opening), "rows": rows},
                    {"inflow": inflow, "outflow": outflow, "net": inflow - outflow, "categories": cats,
                     "opening": opening, "closing": opening + sum(amounts)}))
    return out


def calculation_problems(payload: dict[str, Any], expected: dict[str, Any], corr: str) -> list[str]:
    """Every reported number must equal the oracle's; the markdown may show no other amount."""
    store = _store()
    _ingest(store, SOMAL, "acct_somal_checking", "2026-10", [Row(**r) for r in payload["rows"]], payload["opening"])
    b = _brief(store, SOMAL, corr)
    items = {it.kind: it for it in b.net_worth_and_cash_flow + b.budget_and_variance}
    problems = []
    flow = items.get("cash_flow")
    got_flow = flow.amounts_minor if flow else {}
    want_flow = {k: expected[k] for k in ("inflow", "outflow", "net")}
    if got_flow != want_flow:
        problems.append(f"cash flow {got_flow} != {want_flow}")
    cats = items.get("category_totals")
    if (cats.amounts_minor if cats else {}) != expected["categories"]:
        problems.append(f"categories {cats.amounts_minor if cats else {}} != {expected['categories']}")
    bal = items.get("closing_balance")
    if not bal or bal.amounts_minor != {"opening": expected["opening"], "closing": expected["closing"]}:
        problems.append(f"balance {bal.amounts_minor if bal else None}")
    allowed = {abs(v) for v in [*want_flow.values(), *expected["categories"].values(), expected["opening"],
                                expected["closing"]]}
    shown = {int(m.replace(",", "").replace(".", "")) for m in re.findall(_MONEY, render_markdown(b))}
    if stray := sorted(shown - allowed):
        problems.append(f"markdown shows amounts not in the oracle: {stray}")
    return problems


# --- anomaly recall --------------------------------------------------------------------------

ANOMALY_TYPES = ("single_unmapped", "single_large", "small_cluster", "burst", "small_then_large",
                 "none", "below_threshold", "familiar_cluster")
EXPECTED_ALERT = {"single_unmapped": "unmapped_high_value", "single_large": "large_transaction",
                  "small_cluster": "small_cluster_7d", "burst": "burst_24h", "small_then_large": "small_then_large"}
ALERT_KINDS = frozenset(EXPECTED_ALERT.values())
PER_TRANSACTION_KINDS = frozenset({"unmapped_high_value", "large_transaction"})
_NOVEL = ["NOVA GAMES", "ZEPHYR APPS", "QUILL MEDIA", "ORBIT TOKENS", "LUMEN CLOUD", "PIVOT STORE", "EMBER SUBS",
          "KITE GIFTS", "SPARK CREDITS", "DRIFT TICKETS", "HALO VPN", "FERN PRINTS", "TIDE DONATE", "MOSS MUSIC"]


def anomaly_cases(rng: random.Random, n: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out = []
    for i in range(n):
        kind = ANOMALY_TYPES[i % len(ANOMALY_TYPES)]
        sept = [Row("2026-09-02", "PAYROLL ACME", "4200.00"), Row("2026-09-09", "SAFEWAY 0912", "-143.20"),
                Row("2026-09-18", "PG&E BILL", "-96.40")]
        octo = [Row("2026-10-02", "PAYROLL ACME", "4200.00"), Row("2026-10-08", "SAFEWAY 1008", "-151.75"),
                Row("2026-10-19", "PG&E BILL", "-101.10")]
        names = [f"{m} {i}" for m in rng.sample(_NOVEL, len(_NOVEL))]
        d0 = rng.randrange(3, 21)
        if kind == "single_unmapped":
            octo.append(Row(f"2026-10-{d0:02d}", names[0], _cents(-rng.randrange(50_000, 200_000))))
        elif kind == "single_large":
            octo.append(Row(f"2026-10-{d0:02d}", f"BAYSIDE PLUMBING INV {i}", _cents(-rng.randrange(250_000, 600_000))))
        elif kind in ("small_cluster", "familiar_cluster"):
            count = rng.randrange(5, 9)
            for k in range(count):
                octo.append(Row(f"2026-10-{d0 + rng.randrange(0, 7):02d}", names[k], _cents(-rng.randrange(6_100, 9_900))))
            if kind == "familiar_cluster":  # the same merchants were already seen last month
                sept += [Row(f"2026-09-{10 + k:02d}", names[k], _cents(-rng.randrange(1_000, 5_000)))
                         for k in range(count)]
        elif kind == "burst":
            for k in range(rng.randrange(10, 14)):
                octo.append(Row(f"2026-10-{d0:02d}", names[k], _cents(-(100 + 150 * k + rng.randrange(0, 100)))))
        elif kind == "small_then_large":
            probes = rng.randrange(3, 6)
            for k in range(probes):
                octo.append(Row(f"2026-10-{d0 + k:02d}", names[0], _cents(-rng.randrange(100, 999))))
            octo.append(Row(f"2026-10-{d0 + probes + rng.randrange(0, 4):02d}", names[0],
                            _cents(-rng.randrange(10_000, 45_000))))
        elif kind == "below_threshold":  # under every limit: $297 max in a week, one $499 item
            for k in range(3):
                octo.append(Row(f"2026-10-{d0 + 2 * k:02d}", names[k], _cents(-rng.randrange(5_000, 9_900))))
            octo.append(Row("2026-10-27", names[3], _cents(-rng.randrange(10_000, 49_900))))
        out.append(({"type": kind, "september": [asdict(r) for r in sept], "october": [asdict(r) for r in octo]},
                    {"alert": EXPECTED_ALERT.get(kind)}))
    return out


def anomaly_alerts(payload: dict[str, Any], corr: str) -> set[str]:
    """Alert kinds the October briefing raises after a September baseline."""
    store = _store()
    _ingest(store, SOMAL, "acct_somal_checking", "2026-09", [Row(**r) for r in payload["september"]])
    sept_close = sum(int(Row(**r).amount.replace(".", "")) for r in payload["september"]) + 500_000
    _ingest(store, SOMAL, "acct_somal_checking", "2026-10", [Row(**r) for r in payload["october"]], _cents(sept_close))
    return {it.kind for it in _brief(store, SOMAL, corr).material_events} & ALERT_KINDS


# --- retrieval recall@5 and citation precision ----------------------------------------------

def eval_policy_documents() -> list[SourceDocument]:
    """The demo sources plus labeled extras (all fictional; "IRS" and the ordinance are SYNTHETIC)."""

    def doc(**kw):
        return SourceDocument(household_id=HOUSEHOLD, collection="policies_compliance",
                              **{"jurisdiction": "CA-San Mateo", **kw})

    return [*demo_policy_documents(),
            doc(doc_id="hoa_rules_2026", account_id="acct_rental_checking", source_name="Maple Street HOA rules 2026",
                doc_type="hoa_rules", authority_tier=2, publisher="Maple Street HOA",
                text="# Dues\nMonthly HOA dues for Unit B are 450.00 dollars, paid to Maple Street HOA.\n"
                     "# Special assessment\nA special assessment of 1,200.00 dollars for roof repairs is due "
                     "March 1, 2027.\n# Common areas\nThe HOA maintains the common landscaping and the pool."),
            doc(doc_id="insurance_policy_2026", account_id="acct_rental_checking", source_name="Landlord insurance policy",
                doc_type="insurance_policy", authority_tier=2, publisher="Harbor Mutual (SYNTHETIC)",
                text="# Coverage\nThe landlord insurance policy for Unit B covers fire and water damage to the "
                     "building.\n# Premium\nThe annual insurance premium for Unit B is 1,450.00 dollars."),
            doc(doc_id="water_heater_invoice", account_id="acct_rental_checking", source_name="Bayside Plumbing invoice",
                doc_type="invoice", authority_tier=2, publisher="Bayside Plumbing",
                text="# Invoice 2231\nBayside Plumbing replaced the Unit B water heater for 1,875.00 dollars "
                     "on August 14, 2026."),
            doc(doc_id="sayali_private_note", account_id="acct_sayali_card", source_name="Sayali private note",
                doc_type="note", authority_tier=3, publisher="Sayali",
                text="# Card rewards\nSayali is saving card rewards of 640.00 dollars toward a personal trip."),
            doc(doc_id="irs_depreciation_2025", account_id=PUBLIC, source_name="SYNTHETIC IRS depreciation guide 2025",
                doc_type="tax_publication", authority_tier=1, publisher="IRS (SYNTHETIC)", jurisdiction="US",
                effective_tax_year=2025,
                text="# Depreciation\nFor tax year 2025, residential rental buildings are depreciated over 27.5 "
                     "years. A replaced water heater is an improvement that is depreciated rather than deducted "
                     "in one year."),
            doc(doc_id="city_rent_ordinance", account_id=PUBLIC, source_name="SYNTHETIC city rent ordinance",
                doc_type="ordinance", authority_tier=1, publisher="City of San Mateo (SYNTHETIC)",
                text="# Rent increases\nThe city ordinance limits annual rent increases for covered units to "
                     "5 percent, with 60 days written notice.")]


# (actor, query, relevant doc ids, [(claim text, supporting doc id)])
RETRIEVAL_QUERIES: list[tuple[str, str, list[str], list[tuple[str, str]]]] = [
    ("somal", "What are the monthly HOA dues for Unit B?", ["hoa_rules_2026"],
     [("Monthly HOA dues for Unit B are $450.00.", "hoa_rules_2026")]),
    ("somal", "Is there an HOA special assessment for roof repairs?", ["hoa_rules_2026"],
     [("The HOA special assessment for roof repairs is $1,200.00.", "hoa_rules_2026")]),
    ("somal", "Does the HOA maintain the landscaping and pool?", ["hoa_rules_2026"],
     [("The HOA maintains the common landscaping and the pool.", "hoa_rules_2026")]),
    ("somal", "What does the landlord insurance policy cover?", ["insurance_policy_2026"],
     [("The landlord insurance policy covers fire and water damage to the building.", "insurance_policy_2026")]),
    ("somal", "What is the annual insurance premium for Unit B?", ["insurance_policy_2026"],
     [("The annual insurance premium for Unit B is $1,450.00.", "insurance_policy_2026")]),
    ("somal", "How much did the water heater replacement cost?", ["water_heater_invoice"],
     [("Bayside Plumbing replaced the water heater for $1,875.00.", "water_heater_invoice")]),
    ("somal", "Is a replaced water heater depreciated or deducted?", ["irs_depreciation_2025"],
     [("A replaced water heater is an improvement that is depreciated rather than deducted.",
       "irs_depreciation_2025")]),
    ("somal", "Over how many years are residential rental buildings depreciated?", ["irs_depreciation_2025"],
     [("Residential rental buildings are depreciated over 27.5 years.", "irs_depreciation_2025")]),
    ("somal", "How much can rent increase each year under the city ordinance?", ["city_rent_ordinance"],
     [("The city ordinance limits annual rent increases to 5 percent.", "city_rent_ordinance")]),
    ("somal", "What is the monthly rent for Unit B in the lease?", ["lease_unit_b_2026"],
     [("Monthly rent for Unit B is $3,400.00.", "lease_unit_b_2026")]),
    ("somal", "Who pays the property tax for Unit B under the lease?", ["lease_unit_b_2026"],
     [("The landlord pays property tax for Unit B directly to the county.", "lease_unit_b_2026")]),
    ("somal", "What was the first installment on the secured property tax bill 2026?", ["county_bill_2026"],
     [("The first installment of the 2026 bill was $3,200.00.", "county_bill_2026")]),
    ("somal", "Which parcel is Unit B and who owns the rental property?", ["property_register_unit_b"],
     [("Parcel 012-345-678 is Unit B, a rental property owned jointly by Somal and Sayali.", "property_register_unit_b")]),
    ("somal", "Is property tax on a rental property deductible as a rental expense?", ["irs_rental_2025"],
     [("For tax year 2025, state and local property tax paid on a rental property is deductible as a rental expense.",
       "irs_rental_2025")]),
    ("somal", "What is Somal's plan for paying the property tax?", ["somal_private_note"],
     [("Somal plans to pay the property tax from a private bonus of $9,000.00.", "somal_private_note")]),
    ("sayali", "What are Sayali's card rewards being saved for?", ["sayali_private_note"],
     [("Sayali is saving card rewards of $640.00 toward a personal trip.", "sayali_private_note")]),
    ("sayali", "What are the monthly HOA dues for Unit B?", ["hoa_rules_2026"],
     [("Monthly HOA dues for Unit B are $450.00.", "hoa_rules_2026")]),
    ("sayali", "Does Somal plan to pay the property tax from a private bonus?", [], []),
    ("somal", "How much has Sayali saved in card rewards for a personal trip?", [], []),
    ("casey_cpa", "What are the monthly HOA dues for Unit B?", ["hoa_rules_2026"], []),
    ("casey_cpa", "Over how many years are residential rental buildings depreciated?", ["irs_depreciation_2025"],
     [("Residential rental buildings are depreciated over 27.5 years.", "irs_depreciation_2025")]),
    ("viaan", "What are the monthly HOA dues for Unit B?", ["hoa_rules_2026"], []),
]


# Negative controls: (actor, query, [(claim text, cite, expected verdict)]). A cite is "doc:<doc_id>" (resolved to
# that document's retrieved chunk) or a literal id that was never retrieved (fabricated). Expected verdicts come
# from these labels, not from validate_claims: a figure absent from the cited source or a never-retrieved id must be
# removed; a true-sounding sentence pinned on an unrelated source must be rephrased as a question.
FABRICATED_ID = "chunk_never_retrieved_0001"
NEGATIVE_CONTROLS: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    ("somal", "What are the monthly HOA dues for Unit B?",
     [("Monthly HOA dues for Unit B are $450.00.", FABRICATED_ID, "remove")]),
    ("somal", "What are the monthly HOA dues and the annual insurance premium for Unit B?",
     [("Monthly HOA dues for Unit B are $450.00.", "doc:insurance_policy_2026", "remove")]),
    ("somal", "Is there an HOA special assessment for roof repairs?",
     [("The HOA special assessment for roof repairs is $9,999.00.", "doc:hoa_rules_2026", "remove")]),
    ("somal", "What are the monthly HOA dues and the annual insurance premium for Unit B?",
     [("The HOA maintains the common landscaping and the pool.", "doc:insurance_policy_2026",
       "rephrase_as_question")]),
]


def fabricated_citations(claims: Sequence[Claim], passages: Sequence[dict[str, Any]]) -> list[str]:
    """Cited evidence ids that were not among the passages retrieved in this job.

    Independent of ``validate_claims``: a plain set difference against the retrieved chunk ids.
    """
    retrieved = {p["chunk_id"] for p in passages}
    return [i for c in claims for i in c.evidence_ids if i not in retrieved]


def may_see(member_id: str, account_id: str) -> bool:
    """Independent oracle (no consent grants here): owners see their accounts, adults and the
    delegate see public sources, the dependent sees nothing."""
    if MEMBERS[member_id].role is Role.DEPENDENT:
        return False
    return account_id == PUBLIC or member_id in _OWNERS.get(account_id, frozenset())


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _altered(text: str) -> str | None:
    m = re.search(r"\d", text)
    return None if not m else text[:m.start()] + str((int(m.group()) + 1) % 10) + text[m.end():]


@dataclass
class RetrievalWorld:
    """One index of every eval source, shared by all retrieval scenarios of a run."""

    service: RetrievalService
    docs: dict[str, SourceDocument]
    _tmp: Any = field(repr=False)

    @classmethod
    def build(cls, embeddings: Embeddings | None = None) -> "RetrievalWorld":
        import chromadb

        from household_agent.llm import HashEmbeddings

        tmp = tempfile.TemporaryDirectory(prefix="evals_retrieval_")
        conn = sqlite3.connect(":memory:")
        service = RetrievalService(chromadb.PersistentClient(path=tmp.name), embeddings or HashEmbeddings(),
                                   Authorizer(ACCOUNTS), AuditLog(conn))
        docs = {d.doc_id: d for d in eval_policy_documents()}
        for d in docs.values():
            service.index(SAYALI if d.account_id == "acct_sayali_card" else SOMAL, d, correlation_id="eval_index",
                          at=DEMO_AS_OF)
        return cls(service, docs, tmp)

    def close(self) -> None:
        self._tmp.cleanup()

    def score(self, payload: dict[str, Any], corr: str) -> dict[str, Any]:
        actor = MEMBERS[payload["actor"]]
        passages = self.service.retrieve_verified(actor, "tax_preparation", payload["query"], correlation_id=corr,
                                                  at=DEMO_AS_OF)["passages"]
        top = [p["metadata"]["doc_id"] for p in passages]
        visible = [d for d in payload["relevant"] if may_see(actor.member_id, self.docs[d].account_id)]
        leaked = sorted({d for d in top if not may_see(actor.member_id, self.docs[d].account_id)})

        cases = []  # (claim, cited chunk, should be supported)
        for text, doc_id in payload["claims"]:
            own = [p for p in passages if p["metadata"]["doc_id"] == doc_id]
            if not own:
                continue  # not retrieved: counted by recall, not citation precision
            best = max(own, key=lambda p: len(_words(text) & _words(p["text"])))["chunk_id"]
            cases.append((text, best, True))
            if altered := _altered(text):
                cases.append((altered, best, False))
            if other := next((p["chunk_id"] for p in passages if p["metadata"]["doc_id"] != doc_id
                              and p["verification"] == "verified"), None):
                cases.append((text, other, False))
        verdicts = validate_claims([Claim(t, "fact", (c,)) for t, c, _ in cases], passages)
        supported = [v["verdict"] == "supported" for v in verdicts]
        return {"found": sorted(set(visible) & set(top)), "visible": visible, "leaked": leaked,
                "claims": [(ok, want) for ok, (_, _, want) in zip(supported, cases, strict=True)],
                "negatives": self._negatives(payload.get("negatives", []), passages)}

    @staticmethod
    def _negatives(labels: Sequence[Sequence[str]], passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for text, cite, want in labels:
            if cite.startswith("doc:"):  # unresolved (doc not retrieved) stays fabricated and fails the label
                cite = next((p["chunk_id"] for p in passages if p["metadata"]["doc_id"] == cite[4:]), cite)
            claim = Claim(text, "fact", (cite,))
            out.append({"got": validate_claims([claim], passages)[0]["verdict"], "want": want,
                        "fabricated": len(fabricated_citations([claim], passages)),
                        "fabricated_want": int(cite == FABRICATED_ID)})
        return out


# --- policy-agent search count ---------------------------------------------------------------

# Three distinct searches per set (one per topic); the scripted model replays them in the patterns below.
SEARCH_SETS = [
    ("Monthly HOA dues for Unit B", "annual insurance premium for Unit B", "replaced water heater depreciation"),
    ("Unit B monthly rent in the lease", "property tax deductible rental expense", "city rent increase limit"),
    ("HOA special assessment roof repairs", "landlord insurance coverage fire water", "rental building depreciation years"),
]
_BUDGET_EXTRA = "Somal property tax payment plan"  # a 4th and 5th distinct query that must never run
# kind -> (pattern over indices into the set plus extras 3 and 4, real searches expected, budget error expected)
SEARCH_PATTERNS: dict[str, tuple[tuple[int, ...], int, bool]] = {
    "exact_repeat_x3": ((0, 0, 0), 1, False),
    "exact_repeat_x5": ((0, 0, 0, 0, 0), 1, False),
    "case_and_spacing_variants": ((0, 0, 0, 0), 1, False),  # rendered below with changed case and spacing
    "interleaved_repeats": ((0, 1, 0, 1), 2, False),
    "three_distinct": ((0, 1, 2), 3, False),
    "distinct_with_repeats": ((0, 1, 2, 0, 1), 3, False),
    "four_distinct_over_budget": ((0, 1, 2, 3), MAX_QUERY_REFINEMENTS + 1, True),
    "five_distinct_over_budget": ((0, 1, 2, 3, 4), MAX_QUERY_REFINEMENTS + 1, True),
    "over_budget_then_repeat": ((0, 1, 2, 3, 0), MAX_QUERY_REFINEMENTS + 1, True),
}


def _vary(query: str, k: int) -> str:
    return [query, query.upper(), f"  {query.lower()}  ", " ".join(query.split()).title().replace(" ", "  ")][k % 4]


def search_count_cases() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out = []
    for kind, (pattern, rows, budget) in SEARCH_PATTERNS.items():
        for s, base in enumerate(SEARCH_SETS):
            pool = [*base, _BUDGET_EXTRA, f"{_BUDGET_EXTRA} again"]
            queries = [_vary(pool[i], n) if kind == "case_and_spacing_variants" else pool[i]
                       for n, i in enumerate(pattern)]
            out.append(({"kind": kind, "queries": queries}, {"rows": rows, "budget_error": budget}))
    return out


class SearchScript(BaseChatModel):
    """A model that issues exactly these ``retrieve_verified`` queries, one per turn, then answers with no claims."""

    queries: list[str]
    i: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.i < len(self.queries):
            msg = _call("retrieve_verified", query=self.queries[self.i]).model_copy(update={"id": None})
            msg.tool_calls[0]["id"] = f"search_{self.i}"
        else:
            msg = _final()
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "search_script"


def search_count_problems(service: RetrievalService, payload: dict[str, Any], expected: dict[str, Any], corr: str,
                          agent_factory: Callable[..., Any] | None = None) -> list[str]:
    """Drive the policy agent with a scripted model and count the ``retrieve`` audit rows it caused.

    Repeats of one query (ignoring case and spacing) must collapse to a single row, the agent may never
    run more than ``MAX_QUERY_REFINEMENTS + 1`` searches, and going past that must report the budget error
    instead of searching.
    """
    model = SearchScript(queries=payload["queries"])
    agent = (agent_factory or PolicyCompliance)(model, service, SOMAL, clock=lambda: DEMO_AS_OF)
    result = agent(TaskAssignment(task_id=f"task_{corr}", correlation_id=corr, assigned_agent="policy_compliance",
                                  purpose="tax_preparation", authorized_scope={"household_id": SOMAL.household_id},
                                  task="search_count_eval"))
    rows = sum(e["action"] == "retrieve" for e in service.audit.events(corr))
    problems = []
    if rows != expected["rows"]:
        problems.append(f"{rows} retrieve rows, want {expected['rows']}")
    if rows > MAX_QUERY_REFINEMENTS + 1:
        problems.append(f"{rows} searches is over the limit of {MAX_QUERY_REFINEMENTS + 1}")
    reported = any("budget_exhausted" in e for e in result.errors)
    if reported != expected["budget_error"]:
        problems.append("budget error " + ("missing" if expected["budget_error"] else "reported without being over budget"))
    return problems


# --- privacy canaries ------------------------------------------------------------------------

CANARIES = {"somal": "CANARY-SOMAL-7Q2X", "sayali": "CANARY-SAYALI-4M9K"}
PRIVATE_ACCOUNT = {"somal": "acct_somal_checking", "sayali": "acct_sayali_card"}
PRIVACY_REQUEST = "Compare the property tax we paid this year with last year and tell me if it is deductible."


def privacy_cases() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    cases = [({"owner": o, "observer": w}, {"visible": False})
             for o in CANARIES for w in MEMBERS if w != o]
    return cases + [({"owner": o, "observer": o}, {"visible": True}) for o in CANARIES]


class RecordingPlayback(PlaybackChatModel):
    """Playback that keeps every prompt it was shown (what a real model would have seen)."""

    seen: list[str] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.extend(str(m.content) for m in messages)
        return super()._generate(messages, stop, run_manager, **kwargs)


@dataclass
class PrivacyWorld:
    store: LedgerStore
    retrieval: RetrievalService
    _tmp: Any = field(repr=False)

    @classmethod
    def build(cls, embeddings: Embeddings | None = None) -> "PrivacyWorld":
        import chromadb

        from household_agent.llm import HashEmbeddings

        tmp = tempfile.TemporaryDirectory(prefix="evals_privacy_")
        store = _store()
        retrieval = RetrievalService(chromadb.PersistentClient(path=tmp.name), embeddings or HashEmbeddings(),
                                     store.authorizer, store.audit)
        _ingest(store, SOMAL, "acct_rental_checking", "2026-10",
                [Row("2026-10-01", "RENT DEPOSIT UNIT B", "3400.00"), Row("2026-10-05", "MAPLE ST HOA DUES", "-450.00")])
        for owner, canary in CANARIES.items():
            actor, account = MEMBERS[owner], PRIVATE_ACCOUNT[owner]
            _ingest(store, actor, account, "2026-10",
                    [Row("2026-10-03", "SAFEWAY 1003", "-88.10"), Row("2026-10-14", f"{canary} TRANSFER", "-650.00")])
            retrieval.index(actor, SourceDocument(
                doc_id=f"{owner}_canary_note", household_id=HOUSEHOLD, collection="policies_compliance",
                account_id=account, source_name=f"{owner} private note", doc_type="note", authority_tier=3,
                publisher=owner, text=f"# Private\nThe property tax reminder code is {canary}; keep it private."),
                correlation_id="eval_index", at=DEMO_AS_OF)
        for d in demo_policy_documents():
            retrieval.index(SOMAL, d, correlation_id="eval_index", at=DEMO_AS_OF)
        return cls(store, retrieval, tmp)

    def close(self) -> None:
        self._tmp.cleanup()

    def surfaces(self, observer_id: str, owner_id: str, corr: str) -> dict[str, str]:
        """Everything ``observer`` can be shown, by surface (lower-cased)."""
        actor, at = MEMBERS[observer_id], DEMO_AS_OF
        all_accounts = [a.account_id for a in ACCOUNTS]
        query = f"property tax reminder code {CANARIES[owner_id]} private note"
        b = _brief(self.store, actor, f"{corr}_brief")
        retrieved = self.retrieval.retrieve_verified(actor, "tax_preparation", query, correlation_id=corr, at=at)
        session = ToolSession(self.store, Categorizer(RULES), actor, "monthly_briefing", corr,
                              acting_agent="ledger_analyst", clock=lambda: at)
        tools = [session.call("list_accounts"),
                 *(session.call("query_ledger", {"account_ids": ids, "start": "2026-10-01", "end": "2026-10-31",
                                                 "include_provisional": True})
                   for ids in [all_accounts, *([a] for a in all_accounts)])]
        ledger_model = RecordingPlayback(steps=[
            _call("list_accounts"),
            _call("query_ledger", account_ids=all_accounts, start="2026-10-01", end="2026-10-31",
                  include_provisional=True),
            _final()])
        policy_model = RecordingPlayback(steps=[_call("retrieve_verified", query=query), _final()])
        specialists = {"ledger_analyst": LedgerAnalyst(ledger_model, self.store, Categorizer(RULES), actor,
                                                       clock=lambda: at),
                       "policy_compliance": PolicyCompliance(policy_model, self.retrieval, actor, clock=lambda: at)}
        job = Supervisor(actor, Router(), specialists, self.store.audit, clock=lambda: at).run(
            PRIVACY_REQUEST, correlation_id=f"{corr}_job")
        dump = lambda x: json.dumps(x, default=str).lower()  # noqa: E731
        return {"briefing": (b.model_dump_json() + render_markdown(b)).lower(), "retrieval": dump(retrieved),
                "tools": dump(tools),
                "supervisor": " ".join([*ledger_model.seen, *policy_model.seen, dump(job)]).lower()}
