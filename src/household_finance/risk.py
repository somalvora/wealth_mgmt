"""Cluster-risk rules (design §5.4, §5.5): patterns hidden below per-transaction limits.

``cluster_risk`` looks only at *new, unmapped* small debits: the merchant has no approved
category and did not appear in the baseline period. Three rules, all configurable:

- ``small_cluster_7d``: such debits total more than $300 within 7 days (any merchants, any
  of the accounts passed in, so one pattern spread over two cards is still one cluster);
- ``burst_24h``: 10 or more of them on one day (posting dates have no time, so a day stands
  in for 24 hours);
- ``small_then_large``: 3 or more tiny probes from one merchant followed within 7 days by a
  larger debit from the same merchant (card testing).

An alert is a review prompt, not a fraud determination. Indexes refer to ``current``.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from household_finance.calcs import LedgerEntry
from household_finance.money import format_minor

RISK_RULES_VERSION = "cluster_risk_1.0"


@dataclass(frozen=True)
class RiskAlert:
    rule: str
    start: date
    end: date
    count: int
    total_minor: int  # sum of the debits, as a positive number
    merchants: tuple[str, ...]
    entry_indexes: tuple[int, ...]
    explanation: str


def cluster_risk(
    current: Sequence[LedgerEntry],
    baseline: Iterable[LedgerEntry] = (),
    *,
    uncategorized: str = "uncategorized",
    small_max_minor: int = 10_000,
    cluster_sum_minor: int = 30_000,
    window_days: int = 7,
    burst_count: int = 10,
    probe_max_minor: int = 1_000,
    probe_count: int = 3,
) -> list[RiskAlert]:
    known = {e.merchant_normalized for e in baseline}
    new = sorted(
        (i for i, e in enumerate(current)
         if e.amount_minor < 0 and e.category == uncategorized and e.merchant_normalized not in known),
        key=lambda i: (current[i].posted_at, i),
    )
    small = [i for i in new if -current[i].amount_minor < small_max_minor]
    alerts: list[RiskAlert] = []

    def alert(rule: str, idx: Sequence[int], why: str) -> RiskAlert:
        entries = [current[i] for i in idx]
        total = -sum(e.amount_minor for e in entries)
        merchants = tuple(sorted({e.merchant_normalized for e in entries}))
        return RiskAlert(rule, min(e.posted_at for e in entries), max(e.posted_at for e in entries), len(idx),
                         total, merchants, tuple(sorted(idx)),
                         f"{len(idx)} new small charge(s) totaling {format_minor(total)} from {len(merchants)} "
                         f"merchant(s) {why}")

    # small_cluster_7d: greedy, non-overlapping windows that start at a charge
    i = 0
    while i < len(small):
        end = current[small[i]].posted_at + timedelta(days=window_days - 1)
        j = i
        while j + 1 < len(small) and current[small[j + 1]].posted_at <= end:
            j += 1
        window = small[i:j + 1]
        if -sum(current[k].amount_minor for k in window) > cluster_sum_minor:
            alerts.append(alert("small_cluster_7d", window,
                                f"within {window_days} days (limit {format_minor(cluster_sum_minor)})"))
            i = j + 1
        else:
            i += 1

    by_day: dict[date, list[int]] = defaultdict(list)
    for k in small:
        by_day[current[k].posted_at].append(k)
    for day in sorted(by_day):
        if len(by_day[day]) >= burst_count:
            alerts.append(alert("burst_24h", by_day[day], f"on {day} (limit {burst_count - 1} per day)"))

    by_merchant: dict[str, list[int]] = defaultdict(list)
    for k in new:
        by_merchant[current[k].merchant_normalized].append(k)
    for merchant in sorted(by_merchant):
        probes: list[int] = []
        for k in by_merchant[merchant]:
            amount = -current[k].amount_minor
            if amount < probe_max_minor:
                probes.append(k)
            elif (len(probes) >= probe_count
                  and current[k].posted_at - current[probes[-1]].posted_at <= timedelta(days=window_days)):
                alerts.append(alert("small_then_large", [*probes, k],
                                    f"after {len(probes)} probes under {format_minor(probe_max_minor)}, "
                                    f"then {format_minor(amount)}"))
                probes = []
    return alerts
