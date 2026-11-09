"""Policy JSON-repair retry and the review pause on the allocate path."""
import json
from typing import Any

import chromadb
import pytest
from chromadb.config import Settings
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from household_agent.agents import (
    MAX_DENIED_ATTEMPTS,
    PolicyCompliance,
    Router,
    Supervisor,
    TaskAssignment,
    _LoopState,
    repair_final,
)
from household_agent.allocation_advice import CHOOSE_PROMPT, PROPOSE_PROMPT
from household_agent.llm import HashEmbeddings
from household_agent.mcp_server import MEMBERS
from household_agent.retrieval import RetrievalService
from household_agent.synthetic import ALEX, DEMO_AS_OF, SAM, demo_policy_documents
from tests.test_allocation import ASK, FnModel, facts


class Scripted(BaseChatModel):
    steps: list[Any]
    i: int = 0
    seen: list[Any] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        step = self.steps[self.i]
        self.i += 1
        msg = (step(messages) if callable(step) else step).model_copy(update={"id": None})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "scripted"


GOOD = AIMessage(content=json.dumps({"claims": []}))
BAD = AIMessage(content="here are my findings: none")


def _assignment():
    return TaskAssignment(task_id="task_t", correlation_id="job_t", assigned_agent="policy_compliance",
                          purpose="budgeting", authorized_scope={"household_id": "hh_demo"}, task="t")


@pytest.fixture
def retrieval(tmp_path, seeded_store):
    client = chromadb.PersistentClient(str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False))
    svc = RetrievalService(client, HashEmbeddings(), seeded_store.authorizer, seeded_store.audit)
    for d in demo_policy_documents():
        svc.index(ALEX, d, correlation_id="idx", at=DEMO_AS_OF)
    return svc


def _policy(retrieval, steps):
    model = Scripted(steps=steps, seen=[])
    return model, PolicyCompliance(model, retrieval, ALEX, clock=lambda: DEMO_AS_OF)


# --- Policy JSON repair ----------------------------------------------------------------------

def test_an_unparseable_final_answer_is_repaired_with_one_retry(retrieval):
    model, agent = _policy(retrieval, [BAD, GOOD])
    r = agent(_assignment())
    assert r.status == "completed" and r.errors == [] and r.outputs == {"final_answer_repaired": True}
    assert model.i == 2
    assert "not valid JSON" in model.seen[1][-1].content  # the retry says what was wrong


def test_a_valid_answer_is_never_retried(retrieval):
    model, agent = _policy(retrieval, [GOOD])
    r = agent(_assignment())
    assert model.i == 1 and r.outputs == {}


def test_a_failed_repair_keeps_needs_review_and_retries_only_once(retrieval):
    model, agent = _policy(retrieval, [BAD, BAD, GOOD])
    r = agent(_assignment())
    assert model.i == 2 and r.status == "needs_review" and r.outputs == {}
    assert "after one repair retry" in r.errors[0]


def test_a_model_error_during_repair_keeps_the_original_parse_error(retrieval):
    def boom(_messages):
        raise RuntimeError("model down")

    _, agent = _policy(retrieval, [BAD, boom])
    r = agent(_assignment())
    assert r.status == "needs_review" and "unparseable" in r.errors[0]


def test_repair_does_not_apply_to_a_stopped_loop():
    state = _LoopState(messages=[AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])],
                       calls=0, errors=[], observations=[])
    assert repair_final(Scripted(steps=[], seen=[]), state) is None


def test_the_repair_prompt_never_echoes_the_offending_text():
    secret = "SSN 123-45-6789 appears here"
    model = Scripted(steps=[GOOD], seen=[])
    state = _LoopState(messages=[AIMessage(content=json.dumps({"claims": secret}))],
                       calls=0, errors=[], observations=[])
    claims, err = repair_final(model, state)
    assert err is None and claims == []
    assert secret not in model.seen[0][-1].content


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(junk=st.text(max_size=80).filter(lambda s: not s.strip().startswith(("{", "`"))))
def test_property_repair_makes_at_most_one_extra_model_call(retrieval, junk):
    model, agent = _policy(retrieval, [AIMessage(content=junk or "x"), AIMessage(content=junk or "x"), GOOD])
    r = agent(_assignment())
    assert model.i == 2 and r.status == "needs_review"


# --- review pause on the allocate path -------------------------------------------------------

def _bad_pick():
    def fn(system, payload):
        if system == PROPOSE_PROMPT:
            return json.dumps({"candidates": []})
        assert system == CHOOSE_PROMPT
        return json.dumps({"chosen": "all_to_highest_apr_debt", "claims": []})
    return FnModel(fn)


def _sup(store, model=None, actor=ALEX, **f):
    return Supervisor(actor, Router(), {}, store.audit, clock=lambda: DEMO_AS_OF, ledger=store,
                      allocation_facts=facts(**f), allocation_model=model, members=MEMBERS,
                      authorizer=store.authorizer)


def _flagged(store):
    sup = _sup(store, _bad_pick(), min_cash_buffer_months="5.2")
    saver = InMemorySaver()
    out = sup.run(ASK, correlation_id="j_ar", checkpointer=saver)
    return sup, saver, out


def _reviews(store, cid):
    return [(e["actor_id"], e["decision"]) for e in store.audit.events(cid) if e["action"] == "allocation_review"]


def test_a_flagged_allocation_pauses_for_review_with_a_checkpointer(seeded_store):
    _, _, out = _flagged(seeded_store)
    [intr] = out["__interrupt__"]
    assert out["status"] == "awaiting_review" and "briefing" not in out
    v = intr.value
    assert v["kind"] == "allocation_review" and v["correlation_id"] == "j_ar"
    assert v["recommended"] == "all_to_emergency_fund" and v["flags"]


def test_an_authorized_accept_resumes_to_a_reviewed_briefing(seeded_store):
    sup, saver, _ = _flagged(seeded_store)
    done = sup.resume("j_ar", {"reviewer": "jordan", "accepted": True, "note": "ok"}, checkpointer=saver)
    a = done["briefing"]["allocation"]
    assert done["status"] == "reviewed" and a["review"]["accepted"] is True
    assert a["recommended"] == "all_to_emergency_fund" and done["briefing"]["statements"]
    assert _reviews(seeded_store, "j_ar") == [("jordan", "allow")]
    assert seeded_store.audit.verify() > 0


def test_a_rejection_withholds_the_recommendation(seeded_store):
    sup, saver, _ = _flagged(seeded_store)
    done = sup.resume("j_ar", {"reviewer": "alex", "accepted": False, "note": "no"}, checkpointer=saver)
    a = done["briefing"]["allocation"]
    assert done["status"] == "review_rejected" and a["recommended"] is None
    assert not done["briefing"]["statements"]


@pytest.mark.parametrize("reviewer", ["sam", "casey_cpa", "mallory", ""])
def test_an_unauthorized_reviewer_is_refused_and_the_job_keeps_waiting(seeded_store, reviewer):
    sup, saver, _ = _flagged(seeded_store)
    out = sup.resume("j_ar", {"reviewer": reviewer, "accepted": True}, checkpointer=saver)
    assert out["status"] == "awaiting_review" and out["allocation_review_denied_attempts"] == 1 and out["__interrupt__"]
    assert _reviews(seeded_store, "j_ar") == [(reviewer or "unknown", "deny")]
    assert sup.resume("j_ar", {"reviewer": "alex", "accepted": True}, checkpointer=saver)["status"] == "reviewed"


def test_repeated_denied_attempts_block_the_job_and_recommend_nothing(seeded_store):
    sup, saver, _ = _flagged(seeded_store)
    out = None
    for _ in range(MAX_DENIED_ATTEMPTS):
        out = sup.resume("j_ar", {"reviewer": "mallory", "accepted": True}, checkpointer=saver)
    assert out["status"] == "needs_review" and "__interrupt__" not in out
    a = out["briefing"]["allocation"]
    assert a["review"]["blocked"] is True and a["recommended"] is None


def test_resume_adds_no_duplicate_run_or_audit_events(seeded_store):
    sup, saver, _ = _flagged(seeded_store)
    runs = seeded_store.conn.execute("SELECT COUNT(*) FROM calculation_runs").fetchone()[0]
    allocs = len([e for e in seeded_store.audit.events("j_ar") if e["action"] == "allocate"])
    sup.resume("j_ar", {"reviewer": "alex", "accepted": True}, checkpointer=saver)
    assert seeded_store.conn.execute("SELECT COUNT(*) FROM calculation_runs").fetchone()[0] == runs
    assert len([e for e in seeded_store.audit.events("j_ar") if e["action"] == "allocate"]) == allocs == 1


def test_a_clean_recommendation_does_not_pause(seeded_store):
    out = _sup(seeded_store).run(ASK, correlation_id="j_clean", checkpointer=InMemorySaver())
    assert out["status"] == "completed" and "__interrupt__" not in out


def test_missing_inputs_and_denied_do_not_pause(seeded_store):
    out = _sup(seeded_store).run("where should our extra money go?", correlation_id="j_m", checkpointer=InMemorySaver())
    assert out["allocation"]["status"] == "missing_inputs" and "__interrupt__" not in out
    out = _sup(seeded_store, actor=SAM).run(ASK, correlation_id="j_d", checkpointer=InMemorySaver())
    assert out["status"] == "refused" and "__interrupt__" not in out


def test_without_a_checkpointer_a_flagged_allocation_is_just_needs_review(seeded_store):
    out = _sup(seeded_store, _bad_pick(), min_cash_buffer_months="5.2").run(ASK, correlation_id="j_nc")
    assert out["status"] == "needs_review" and "__interrupt__" not in out


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(decision=st.fixed_dictionaries({"reviewer": st.sampled_from(["alex", "jordan"]), "accepted": st.booleans(),
                                       "note": st.text(max_size=20)}))
def test_property_an_authorized_decision_always_ends_the_pause(seeded_store, decision):
    cid = f"j_p_{abs(hash(json.dumps(decision, sort_keys=True)))}"
    sup = _sup(seeded_store, _bad_pick(), min_cash_buffer_months="5.2")
    saver = InMemorySaver()
    sup.run(ASK, correlation_id=cid, checkpointer=saver)
    done = sup.resume(cid, decision, checkpointer=saver)
    assert "__interrupt__" not in done
    assert done["status"] == ("reviewed" if decision["accepted"] else "review_rejected")
