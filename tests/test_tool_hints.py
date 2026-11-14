"""Tool errors and descriptions steer a model that guessed an identifier back to valid ones."""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from household_agent.retrieval import COLLECTIONS, RetrievalRequestError
from household_agent.tools import _error
from household_agent.authz import AuthorizationError, Decision
from tests.test_tools import RENTAL, _session


def _denied(reason):
    return AuthorizationError(Decision(False, reason))


def test_unknown_account_error_points_to_list_accounts():
    out = _error(_denied("unknown account taxes"))
    assert out["error"]["type"] == "authorization_denied"
    assert "unknown account taxes" in out["error"]["message"] and "list_accounts" in out["error"]["message"]


def test_other_denials_get_no_hint():
    out = _error(_denied("no consent for purpose"))
    assert "list_accounts" not in out["error"]["message"]


def test_query_ledger_with_a_topic_as_account_gets_the_hint(seeded_store, categorizer):
    s = _session(seeded_store, categorizer)
    out = s.call("query_ledger", {"account_ids": ["taxes"], "start": "2026-10-01", "end": "2026-10-31"})
    assert not out["ok"] and "list_accounts" in out["error"]["message"]
    assert s.call("query_ledger", {"account_ids": [RENTAL], "start": "2026-10-01", "end": "2026-10-31"})["ok"]


def test_unknown_collection_lists_the_valid_ones():
    from household_agent.retrieval import RetrievalService

    svc = RetrievalService(client=None, embeddings=None, authorizer=None, audit=None)
    with pytest.raises(RetrievalRequestError) as exc:
        svc._collection("h1", "Internal Revenue Code")
    assert all(c in str(exc.value) for c in COLLECTIONS)


@settings(max_examples=30, deadline=None)
@given(name=st.text(min_size=1, max_size=30).filter(lambda n: n not in COLLECTIONS))
def test_property_any_unknown_collection_is_rejected_with_options(name):
    from household_agent.retrieval import RetrievalService

    svc = RetrievalService(client=None, embeddings=None, authorizer=None, audit=None)
    with pytest.raises(RetrievalRequestError, match="use one of"):
        svc._collection("h1", name)
