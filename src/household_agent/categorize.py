"""Merchant normalization and categorization from approved preference-memory rules (design §6).

Only user-approved rules assign a category. Anything unmatched stays ``uncategorized`` and
counts against document confidence; the agent may propose a category later, never apply one.
"""

from collections.abc import Iterable
from dataclasses import dataclass

UNCATEGORIZED = "uncategorized"
MAPPED_CONFIDENCE = 0.98


@dataclass(frozen=True)
class CategoryRule:
    pattern: str  # case-insensitive substring of the raw description
    merchant_normalized: str
    category: str


@dataclass(frozen=True)
class Categorized:
    merchant_normalized: str
    category: str
    confidence: float
    mapped: bool


class Categorizer:
    def __init__(self, rules: Iterable[CategoryRule]):
        self._rules = [(r.pattern.upper(), r) for r in rules]

    def categorize(self, description: str) -> Categorized:
        upper = description.upper()
        for pattern, rule in self._rules:
            if pattern in upper:
                return Categorized(rule.merchant_normalized, rule.category, MAPPED_CONFIDENCE, True)
        return Categorized(" ".join(description.split()).title(), UNCATEGORIZED, 0.0, False)
