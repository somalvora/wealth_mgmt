"""The ledger analyst is told which accounts it may read, so a small model need not guess ids."""
import json

from household_agent.synthetic import ALEX, JORDAN, SAM
from tests.test_tools import RENTAL, _session
from tests.test_vertical_slice import _check, _models
from household_agent.vertical_slice import run_vertical_slice


def _ids(rows):
    return [r["account_id"] for r in rows]


def test_readable_accounts_matches_list_accounts(seeded_store, categorizer):
    s = _session(seeded_store, categorizer, ALEX)
    assert _ids(s.readable_accounts()) == _ids(s.call("list_accounts")["accounts"]) == ["acct_alex_checking", RENTAL]


def test_readable_accounts_charges_no_tool_budget(seeded_store, categorizer):
    s = _session(seeded_store, categorizer, ALEX, max_calls=1)
    s.readable_accounts()
    assert s.calls == 0 and s.call("list_accounts")["ok"]


def test_a_dependent_is_told_about_no_accounts(seeded_store, categorizer):
    assert _session(seeded_store, categorizer, SAM).readable_accounts() == []


def test_jordan_is_not_told_about_alexs_checking(seeded_store, categorizer):
    assert "acct_alex_checking" not in _ids(_session(seeded_store, categorizer, JORDAN).readable_accounts())


def test_the_specialist_sees_readable_accounts_in_its_first_message(tmp_path):
    seen = []

    def spy(messages):
        seen.append(messages)
        from langchain_core.messages import AIMessage
        return AIMessage(content='{"claims": []}')

    run_vertical_slice(tmp_path, models=_models(ledger=lambda: _Spy(spy)))
    first = json.loads(seen[0][-1].content)
    assert "acct_alex_checking" in [a["account_id"] for a in first["inputs"]["readable_accounts"]]


class _Spy:
    def __init__(self, fn):
        self.fn = fn

    def bind_tools(self, *a, **k):
        return self

    def invoke(self, messages, *a, **k):
        return self.fn(messages)
