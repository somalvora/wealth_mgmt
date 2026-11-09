import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langgraph.checkpoint.memory import InMemorySaver

from household_agent.agents import (
    LEDGER_TOOLS,
    POLICY_TOOLS,
    AgentResult,
    Router,
    Supervisor,
    rule_route,
)
from household_agent.audit import AuditLog
from household_agent.mcp_server import MEMBERS
from household_agent.reminders import (
    MAX_TITLE_LEN,
    ReminderDraft,
    ReminderError,
    build_ics,
    draft_reminder,
    export_ics,
    ics_filename,
    extract_due_date,
    transition,
)
from household_agent.synthetic import ALEX, DEMO_AS_OF

TODAY = DEMO_AS_OF.date()
DUE = date(TODAY.year + 1, 1, 15)
EVIDENCE = ["chunk_policy_1", "chunk_policy_2"]
STAMP = datetime(2026, 11, 9, 12, 0, 0, tzinfo=UTC)
REQUEST = f"Please add my insurance renewal on {DUE.isoformat()} to my calendar"
ALLOWED_PROPS = {"BEGIN", "END", "VERSION", "PRODID", "CALSCALE", "UID", "DTSTAMP", "DTSTART", "SUMMARY", "DESCRIPTION"}


def make(**kw):
    args = dict(title="Renew home insurance", due_date=DUE, evidence_ids=EVIDENCE[:1], job_evidence_ids=EVIDENCE,
                household_id="hh_demo", requested_by="alex", correlation_id="job_r", today=TODAY)
    return draft_reminder(**{**args, **kw})


def unfold(content: bytes) -> list[str]:
    text = content.decode()
    assert text.endswith("\r\n")
    return text.replace("\r\n ", "").split("\r\n")[:-1]


def approved(draft=None):
    return transition(draft or make(), "approved")


# --- happy path and determinism ------------------------------------------------------------

def test_draft_is_a_frozen_draft_with_the_given_fields():
    d = make()
    assert d.status == "draft" and d.due_date == DUE and d.source_evidence_ids == (EVIDENCE[0],)
    assert d.reminder_id.startswith("rem_")
    with pytest.raises(Exception):
        d.status = "created"


def test_same_input_gives_same_id_and_same_ics_bytes():
    a, b = make(), make(requested_by="jordan", correlation_id="other")
    assert a.reminder_id == b.reminder_id  # id is a hash of the content only
    assert build_ics(approved(a), STAMP) == build_ics(approved(b), STAMP)
    assert make(title="Different").reminder_id != a.reminder_id
    assert make(evidence_ids=EVIDENCE).reminder_id != a.reminder_id


def test_ics_has_the_expected_fields(tmp_path):
    path, content = export_ics(approved(), tmp_path / "out", STAMP)
    lines = unfold(content)
    d = make()
    assert path.read_bytes() == content and path.name == ics_filename(d)
    assert lines[0] == "BEGIN:VCALENDAR" and lines[-1] == "END:VCALENDAR"
    assert f"UID:{d.reminder_id}@reminder.household-agent.local" in lines
    assert f"DTSTART;VALUE=DATE:{DUE.strftime('%Y%m%d')}" in lines
    assert "DTSTAMP:20261109T120000Z" in lines and "SUMMARY:Renew home insurance" in lines
    assert "DESCRIPTION:Source evidence: chunk_policy_1" in lines


def test_ics_escapes_text_and_folds_long_lines():
    d = make(title="Pay; tax, now \\ ok", evidence_ids=EVIDENCE * 1)
    lines = unfold(build_ics(approved(d), STAMP))
    assert r"SUMMARY:Pay\; tax\, now \\ ok" in lines
    long = make(title="x" * MAX_TITLE_LEN, evidence_ids=[f"e{i}" * 40 for i in range(3)],
                job_evidence_ids=[f"e{i}" * 40 for i in range(3)])
    raw = build_ics(approved(long), STAMP)
    assert all(len(line) <= 75 for line in raw.split(b"\r\n")) and b"\r\n " in raw


def test_extract_due_date():
    assert extract_due_date("renews 2027-03-04!", TODAY) == date(2027, 3, 4)
    assert extract_due_date("due March 4, 2027", TODAY) == date(2027, 3, 4)
    assert extract_due_date("due January 2", TODAY) == date(TODAY.year + 1, 1, 2)  # next occurrence
    assert extract_due_date("no date here", TODAY) is None and extract_due_date("2027-02-31", TODAY) is None


# --- negative paths --------------------------------------------------------------------------

def test_past_due_date_is_rejected_but_today_is_allowed():
    with pytest.raises(ReminderError, match="past"):
        make(due_date=date(TODAY.year - 1, 1, 1))
    assert make(due_date=TODAY).due_date == TODAY


def test_invented_or_missing_evidence_is_rejected():
    with pytest.raises(ReminderError, match="not found"):
        make(evidence_ids=["chunk_invented"])
    with pytest.raises(ReminderError, match="at least one"):
        make(evidence_ids=[])


@pytest.mark.parametrize("title", ["Renew\nATTENDEE:mailto:evil@example.com", "a\rb", "a\x00b", "a b", "", "  ",
                                   "x" * (MAX_TITLE_LEN + 1)])
def test_bad_titles_are_rejected(title):
    with pytest.raises(ReminderError):
        make(title=title)


def test_invalid_transitions_and_exporting_a_non_approved_reminder(tmp_path):
    d = make()
    with pytest.raises(ReminderError):
        transition(d, "created")
    with pytest.raises(ReminderError):
        transition(transition(d, "rejected"), "approved")
    for bad in (d, transition(d, "rejected")):
        with pytest.raises(ReminderError):
            export_ics(bad, tmp_path / "out", STAMP)
    assert not (tmp_path / "out").exists()


# --- property: an arbitrary title never injects a property line ------------------------------

@settings(max_examples=200, deadline=None)
@given(title=st.text(max_size=150), evidence=st.text(min_size=1, max_size=60))
def test_arbitrary_text_never_adds_property_lines_and_lines_fit(title, evidence):
    try:
        d = make(title=title, evidence_ids=[evidence], job_evidence_ids=[evidence])
    except ReminderError:
        # Even a draft built around the validation (as if it had slipped through) must export safely.
        d = ReminderDraft.model_construct(reminder_id="rem_x", title=title, due_date=DUE, source_evidence_ids=(evidence,),
                                          household_id="h", requested_by="a", status="approved", correlation_id="c")
    raw = build_ics(approved(d) if d.status == "draft" else d, STAMP)
    assert raw.endswith(b"\r\n") and all(len(p) <= 75 for p in raw.split(b"\r\n"))
    assert raw.count(b"\r\n") == raw.count(b"\n") == raw.count(b"\r")
    names = [ln.split(":", 1)[0].split(";", 1)[0] for ln in unfold(raw)]
    assert set(names) <= ALLOWED_PROPS and names.count("SUMMARY") == 1 and names.count("DESCRIPTION") == 1
    assert len(names) == 12


# --- the Supervisor flow ---------------------------------------------------------------------

def _stub(evidence=("chunk_policy_1",)):
    calls = []

    def run(a):
        calls.append(a)
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent=a.assigned_agent,
                           status="completed", evidence_ids=list(evidence))
    run.calls = calls
    return run


def _sup(store, tmp_path, stub=None, **kw):
    return Supervisor(ALEX, Router(), {"policy_compliance": stub or _stub()}, store.audit, clock=lambda: DEMO_AS_OF,
                      members=MEMBERS, authorizer=store.authorizer, reminder_dir=tmp_path / "ics" if tmp_path else None, **kw)


def _events(audit, cid):
    return [(e["action"], e["actor_id"], e["decision"]) for e in audit.events(cid) if e["resource"]
            and e["resource"].startswith("reminder:")]


def test_calendar_request_is_a_reminder_not_a_refusal():
    r = rule_route(REQUEST)
    assert r.reminder_requested and not r.external_action_requested and r.dispatch_plan
    r = rule_route(f"Remind me to pay the county tax bill by {DUE.isoformat()}")
    assert r.reminder_requested and not r.external_action_requested
    for text in ("Please pay the county tax bill", "remind me and then transfer $500 to savings"):
        assert rule_route(text).external_action_requested


def test_without_a_checkpointer_the_draft_is_listed_and_nothing_else_happens(seeded_store, tmp_path):
    out = _sup(seeded_store, tmp_path).run(REQUEST, correlation_id="j_plain")
    [action] = out["briefing"]["actions_for_approval"]
    assert action["reminder"]["status"] == "draft" and "__interrupt__" not in out
    assert not (tmp_path / "ics").exists()
    assert _events(seeded_store.audit, "j_plain") == [("draft_reminder", "alex", "info")]


def test_draft_approve_creates_the_ics(seeded_store, tmp_path):
    sup, saver = _sup(seeded_store, tmp_path), InMemorySaver()
    out = sup.run(REQUEST, correlation_id="j_ok", checkpointer=saver)
    [intr] = out["__interrupt__"]
    assert out["status"] == "awaiting_reminder_approval" and intr.value["kind"] == "reminder_approval"
    assert not (tmp_path / "ics").exists()  # nothing exists until someone approves
    done = sup.resume("j_ok", {"approver": "jordan", "approved": True}, checkpointer=saver)
    [action] = done["briefing"]["actions_for_approval"]
    rid = action["reminder"]["reminder_id"]
    assert done["status"] == "reminder_created" and action["reminder"]["status"] == "created"
    path = Path(done["reminder_file"])
    assert path.parent == tmp_path / "ics" and path.name.startswith(rid + "-") and path.suffix == ".ics"
    ics = path.read_bytes()
    assert f"DTSTART;VALUE=DATE:{DUE.strftime('%Y%m%d')}".encode() in ics and b"chunk_policy_1" in ics
    assert [a for a, _, _ in _events(seeded_store.audit, "j_ok")] == ["draft_reminder", "approve", "create_reminder"]
    assert seeded_store.audit.verify() > 0


@pytest.mark.parametrize("approver", ["sam", "casey_cpa", "mallory", ""])
def test_unauthorized_approver_is_refused_audited_and_no_file_is_made(seeded_store, tmp_path, approver):
    sup, saver = _sup(seeded_store, tmp_path), InMemorySaver()
    sup.run(REQUEST, correlation_id="j_deny", checkpointer=saver)
    out = sup.resume("j_deny", {"approver": approver, "approved": True}, checkpointer=saver)
    assert out["reminder_denied_attempts"] == 1 and out["__interrupt__"] and not (tmp_path / "ics").exists()
    assert _events(seeded_store.audit, "j_deny")[-1] == ("approve", approver or "unknown", "deny")
    assert sup.resume("j_deny", {"approver": "alex", "approved": True}, checkpointer=saver)["status"] == "reminder_created"


def test_rejected_reminder_produces_no_ics(seeded_store, tmp_path):
    sup, saver = _sup(seeded_store, tmp_path), InMemorySaver()
    sup.run(REQUEST, correlation_id="j_rej", checkpointer=saver)
    done = sup.resume("j_rej", {"approver": "jordan", "approved": False, "note": "no"}, checkpointer=saver)
    assert done["status"] == "reminder_rejected" and done["briefing"]["actions_for_approval"][0]["reminder"]["status"] == "rejected"
    assert not (tmp_path / "ics").exists() and "reminder_file" not in done


def test_resume_does_not_duplicate_audit_events(seeded_store, tmp_path):
    sup, saver = _sup(seeded_store, tmp_path), InMemorySaver()
    sup.run(REQUEST, correlation_id="j_dup", checkpointer=saver)
    sup.resume("j_dup", {"approver": "sam", "approved": True}, checkpointer=saver)  # denied: node re-runs
    sup.resume("j_dup", {"approver": "alex", "approved": True}, checkpointer=saver)
    events = [(e["action"], e["decision"]) for e in seeded_store.audit.events("j_dup")]
    assert events.count(("draft_reminder", "info")) == 1 and events.count(("route", "info")) == 1
    assert events.count(("dispatch", "info")) == 1 and events.count(("create_reminder", "allow")) == 1
    assert [e for e in events if e[0] == "approve"] == [("approve", "deny"), ("approve", "allow")]


def test_approved_without_an_export_directory_stays_approved(seeded_store):
    sup, saver = _sup(seeded_store, None), InMemorySaver()
    sup.run(REQUEST, correlation_id="j_nodir", checkpointer=saver)
    done = sup.resume("j_nodir", {"approver": "alex", "approved": True}, checkpointer=saver)
    assert done["briefing"]["actions_for_approval"][0]["reminder"]["status"] == "approved"
    assert ("create_reminder", "alex", "deny") in _events(seeded_store.audit, "j_nodir")


def test_no_evidence_or_no_date_means_no_draft_and_an_open_question(seeded_store, tmp_path):
    out = _sup(seeded_store, tmp_path, _stub(evidence=())).run(REQUEST, correlation_id="j_noev", checkpointer=InMemorySaver())
    assert "__interrupt__" not in out and out["briefing"]["actions_for_approval"] == []
    assert any("evidence" in q for q in out["briefing"]["questions"])
    out = _sup(seeded_store, tmp_path).run("add my renewal to my calendar", correlation_id="j_nodate")
    assert any("deadline" in q for q in out["briefing"]["questions"])
    past = _sup(seeded_store, tmp_path).run("add my renewal on 2020-01-01 to my calendar", correlation_id="j_past")
    assert past["briefing"]["actions_for_approval"] == [] and any("past" in q for q in past["briefing"]["questions"])


def test_newline_in_the_request_cannot_inject_a_property(seeded_store, tmp_path):
    sup, saver = _sup(seeded_store, tmp_path), InMemorySaver()
    sup.run(f"add renewal on {DUE.isoformat()} to my calendar\nATTENDEE:mailto:evil@example.com", correlation_id="j_inj",
            checkpointer=saver)
    done = sup.resume("j_inj", {"approver": "alex", "approved": True}, checkpointer=saver)
    ics = Path(done["reminder_file"]).read_bytes()
    assert b"ATTENDEE" in ics and b"\r\nATTENDEE" not in ics and all(n in ALLOWED_PROPS for n in
        [ln.split(":", 1)[0].split(";", 1)[0] for ln in unfold(ics)])


def test_money_movement_is_still_refused_even_with_a_reminder_phrase(seeded_store, tmp_path):
    stub = _stub()
    out = _sup(seeded_store, tmp_path, stub).run("remind me and transfer $5,000 to brokerage", correlation_id="j_move")
    assert out["status"] == "refused" and stub.calls == [] and out["briefing"]["actions_for_approval"] == []


def test_reminder_tool_is_not_on_any_specialist_allowlist():
    names = {t["function"]["name"] for t in [*LEDGER_TOOLS, *POLICY_TOOLS]}
    assert not any("remind" in n for n in names)


# --- review fixes: H1 rejected review, H2 refusal vocabulary, M5 dates, LOW items ------------------------------------

import os  # noqa: E402
from datetime import timedelta, timezone  # noqa: E402

from langchain_core.messages import AIMessage  # noqa: E402

from household_agent.agents import MAX_DENIED_ATTEMPTS, DispatchItem, RoutingDecision, flagged_evidence_ids  # noqa: E402
from household_agent.reminders import due_date_from_request  # noqa: E402
from test_agents import Scripted  # noqa: E402

POLICY = DispatchItem(agent="policy_compliance", task="p", depends_on_ledger=False)


def _fixed(**kw):
    decision = RoutingDecision(**kw)
    return lambda _r: (decision, "rules")


def _two_results(bad_status="needs_review"):
    """Specialist whose result depends on the task text: 'good' completes with chunk_policy_1; 'bad' has `bad_status`."""
    def run(a):
        good = a.task == "good"
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent=a.assigned_agent,
                           status="completed" if good else bad_status, needs_review=not good,
                           evidence_ids=["chunk_policy_1"] if good else ["chunk_untrusted"])
    return run


def _review_job(store, tmp_path):
    plan = [DispatchItem(agent="policy_compliance", task="good", depends_on_ledger=False),
            DispatchItem(agent="policy_compliance", task="bad", depends_on_ledger=False)]
    router = _fixed(intent="tax_context", dispatch_plan=plan, reminder_requested=True)
    return Supervisor(ALEX, router, {"policy_compliance": _two_results()}, store.audit, clock=lambda: DEMO_AS_OF,
                      members=MEMBERS, authorizer=store.authorizer, reminder_dir=tmp_path / "ics")


def test_rejected_review_skips_the_reminder_and_keeps_its_status(seeded_store, tmp_path):
    sup, saver = _review_job(seeded_store, tmp_path), InMemorySaver()
    out = sup.run(REQUEST, correlation_id="j_h1r", checkpointer=saver)
    assert out["status"] == "awaiting_review"
    done = sup.resume("j_h1r", {"reviewer": "alex", "accepted": False, "note": "no"}, checkpointer=saver)
    assert done["status"] == "review_rejected" and "reminder" not in done and "__interrupt__" not in done
    assert done["briefing"]["actions_for_approval"] == []
    assert any("no reminder was drafted" in q for q in done["briefing"]["questions"])
    actions = [e["action"] for e in seeded_store.audit.events("j_h1r")]
    assert "draft_reminder" not in actions and "create_reminder" not in actions
    review = [e for e in seeded_store.audit.events("j_h1r") if e["action"] == "review"][-1]
    assert json.loads(review["details_json"])["reminder_skipped"] is True
    assert not (tmp_path / "ics").exists()


def test_accepted_review_drafts_a_reminder_from_completed_results_only(seeded_store, tmp_path):
    sup, saver = _review_job(seeded_store, tmp_path), InMemorySaver()
    sup.run(REQUEST, correlation_id="j_h1a", checkpointer=saver)
    done = sup.resume("j_h1a", {"reviewer": "alex", "accepted": True}, checkpointer=saver)
    assert done["status"] == "awaiting_reminder_approval"
    assert done["reminder"]["source_evidence_ids"] == ["chunk_policy_1"]  # the needs_review result's evidence is not used
    final = sup.resume("j_h1a", {"approver": "alex", "approved": True}, checkpointer=saver)
    assert final["status"] == "reminder_created"


def test_evidence_from_flagged_documents_is_not_used_for_a_reminder(seeded_store):
    def run(a):
        if a.task == "ledger":
            return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent="ledger_analyst",
                               status="completed", evidence_ids=["calc_run_bad"],
                               outputs={"data_quality": [{"document_id": "doc_bad", "status": "unreconciled",
                                                          "source_name": "bad.csv", "issue_codes": ["x"]}],
                                        "evidence_documents": {"calc_run_bad": ["doc_bad"]}})
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent="policy_compliance",
                           status="completed", evidence_ids=["chunk_policy_1", "calc_run_bad"],
                           outputs={"evidence_documents": {"calc_run_bad": ["doc_bad"]}})
    plan = [DispatchItem(agent="ledger_analyst", task="ledger"), DispatchItem(agent="policy_compliance", task="p", depends_on_ledger=False)]
    sup = Supervisor(ALEX, _fixed(intent="historical_comparison_and_tax_context", dispatch_plan=plan, reminder_requested=True),
                     {"ledger_analyst": run, "policy_compliance": run}, seeded_store.audit, clock=lambda: DEMO_AS_OF)
    out = sup.run(REQUEST, correlation_id="j_h1f")
    assert out["reminder"]["source_evidence_ids"] == ["chunk_policy_1"]


@settings(max_examples=60, deadline=None)
@given(st.dictionaries(st.sampled_from(["e1", "e2", "e3"]), st.lists(st.sampled_from(["d1", "d2", "d3"]), max_size=2)),
       st.sets(st.sampled_from(["d1", "d2"])))
def test_flagged_evidence_is_exactly_what_traces_to_a_flagged_document(mapping, flagged):
    ids = flagged_evidence_ids([{"outputs": {"evidence_documents": mapping}}], flagged)
    assert ids == flagged | {e for e, docs in mapping.items() if flagged & set(docs)}


# H2

def _model_router(**flags):
    reply = RoutingDecision(intent="tax_context", dispatch_plan=[POLICY], **flags).model_dump_json()
    return Router(Scripted(steps=[AIMessage(content=reply)]))


@pytest.mark.parametrize("text", [
    f"Remind me on {DUE.isoformat()} about the renewal and pay the invoice now",
    "Remind me about the renewal and send money to the landlord",
    "set a reminder for the renewal and wire $500 to savings",
    "Remind me about the bill and then pay it", "Remind me in January and venmo Sam",
    "Please pay the county tax bill", "move funds to brokerage"])
def test_money_movement_is_refused_with_or_without_a_reminder_phrase(text):
    assert rule_route(text).external_action_requested


@pytest.mark.parametrize("text", [
    f"Remind me to pay the county tax bill by {DUE.isoformat()}", REQUEST, "Why did our wireless bill go up?",
    "Was the October payment unusual?", "remind me about paying the insurance on 2027-01-15"])
def test_reminders_and_questions_about_payments_are_not_refused(text):
    assert not rule_route(text).external_action_requested


def test_a_model_yes_survives_a_rules_no_and_a_rules_yes_survives_a_model_no(seeded_store, tmp_path):
    decision, source = _model_router(external_action_requested=True, reminder_requested=True)(REQUEST)
    assert source == "model" and decision.external_action_requested is True  # rules say no, model says yes
    stub = _stub()
    sup = Supervisor(ALEX, _model_router(external_action_requested=True), {"policy_compliance": stub},
                     seeded_store.audit, clock=lambda: DEMO_AS_OF)
    assert sup.run(REQUEST, correlation_id="j_h2m")["status"] == "refused" and stub.calls == []
    pay = f"Remind me on {DUE.isoformat()} about the renewal and pay the invoice now"
    decision, _ = _model_router(external_action_requested=False)(pay)
    assert decision.external_action_requested is True
    out = Supervisor(ALEX, _model_router(external_action_requested=False), {"policy_compliance": stub},
                     seeded_store.audit, clock=lambda: DEMO_AS_OF).run(pay, correlation_id="j_h2p")
    assert out["status"] == "refused" and stub.calls == [] and "reminder" not in out


def test_a_plain_reminder_with_a_model_router_is_still_allowed(seeded_store, tmp_path):
    sup = Supervisor(ALEX, _model_router(), {"policy_compliance": _stub()}, seeded_store.audit, clock=lambda: DEMO_AS_OF)
    out = sup.run(REQUEST, correlation_id="j_h2ok")
    assert out["status"] != "refused" and out["reminder"]["status"] == "draft"


# M5

def test_the_date_after_remind_wins_over_an_earlier_date():
    text = "compare the 2025-12-01 bill; remind me on 2026-12-15"
    assert due_date_from_request(text, TODAY) == date(2026, 12, 15)
    assert due_date_from_request("remind me about the 2025-12-01 bill on 2026-12-15", TODAY) == date(2026, 12, 15)
    assert due_date_from_request("on 2026-12-15 remind me", TODAY) == date(2026, 12, 15)  # only one date at all


def test_several_dates_with_no_clear_reminder_date_are_refused_with_a_clear_message(seeded_store):
    for text in ("add 2026-12-01 and 2026-12-15 to my calendar", "remind me about 2026-12-01 and 2026-12-15"):
        with pytest.raises(ReminderError, match="several dates"):
            due_date_from_request(text, TODAY)
        assert extract_due_date(text, TODAY) is None
    out = _sup(seeded_store, None).run("add 2026-12-01 and 2026-12-15 to my calendar", correlation_id="j_amb")
    assert out["briefing"]["actions_for_approval"] == [] and any("several dates" in q for q in out["briefing"]["questions"])


@settings(max_examples=60, deadline=None)
@given(st.dates(min_value=date(2027, 1, 1), max_value=date(2035, 12, 31)),
       st.dates(min_value=date(2020, 1, 1), max_value=date(2026, 12, 31)))
def test_the_reminder_date_is_found_whatever_other_dates_are_mentioned(due, other):
    assert due_date_from_request(f"compare the {other.isoformat()} bill and remind me on {due.isoformat()}", TODAY) == due


def test_the_job_date_comes_from_the_caller_not_the_utc_clock(seeded_store):
    """Clock says 03:00 UTC on Nov 10, but the member's local day is still Nov 9."""
    late = datetime(2026, 11, 10, 3, 0, tzinfo=UTC)
    request = "add my renewal on 2026-11-09 to my calendar"
    mk = lambda: Supervisor(ALEX, Router(), {"policy_compliance": _stub()}, seeded_store.audit, clock=lambda: late)  # noqa: E731
    refused = mk().run(request, correlation_id="j_utc")
    assert any("past" in q for q in refused["briefing"]["questions"])  # without a local date the UTC date is used
    ok = mk().run(request, correlation_id="j_local", today=date(2026, 11, 9))
    assert ok["today"] == "2026-11-09" and ok["reminder"]["due_date"] == "2026-11-09"
    assert due_date_from_request("remind me on March 3", date(2026, 11, 9)) == date(2027, 3, 3)
    assert due_date_from_request("remind me on March 3", date(2026, 1, 9)) == date(2026, 3, 3)


# LOW (a) DTSTAMP

def test_dtstamp_is_converted_to_utc():
    plus5 = datetime(2026, 11, 10, 2, 0, 0, tzinfo=timezone(timedelta(hours=5)))
    assert "DTSTAMP:20261109T210000Z" in unfold(build_ics(make(), plus5))
    assert "DTSTAMP:20261109T120000Z" in unfold(build_ics(make(), datetime(2026, 11, 9, 12, 0)))  # naive = UTC


@settings(max_examples=40, deadline=None)
@given(st.integers(min_value=-14, max_value=14))
def test_dtstamp_names_the_same_instant_in_any_zone(hours):
    local = STAMP.astimezone(timezone(timedelta(hours=hours)))
    assert "DTSTAMP:20261109T120000Z" in unfold(build_ics(make(), local))


# LOW (b) a reminder request with nothing to dispatch

def test_a_reminder_request_with_an_empty_plan_gets_an_explicit_answer(seeded_store):
    sup = Supervisor(ALEX, _fixed(intent="out_of_scope", reminder_requested=True), {}, seeded_store.audit,
                     clock=lambda: DEMO_AS_OF)
    out = sup.run(REQUEST, correlation_id="j_empty")
    assert out["briefing"]["actions_for_approval"] == [] and any("evidence" in q for q in out["briefing"]["questions"])
    assert ("draft_reminder", "alex", "deny") in [(e["action"], e["actor_id"], e["decision"])
                                                  for e in seeded_store.audit.events("j_empty")]


# LOW (c) export safety

def test_export_refuses_symlinks_and_bad_ids(tmp_path):
    d = approved()
    real = tmp_path / "real"
    real.mkdir()
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real, target_is_directory=True)
    with pytest.raises(ReminderError, match="symbolic link"):
        export_ics(d, link_dir, STAMP)
    out = tmp_path / "out"
    out.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    (out / ics_filename(d)).symlink_to(victim)
    with pytest.raises(ReminderError, match="symbolic link"):
        export_ics(d, out, STAMP)
    assert victim.read_text() == "keep me"
    for bad in ("../../etc/x", "rem_" + "g" * 16, "rem_abc", "REM_" + "0" * 16, ""):
        with pytest.raises(ReminderError, match="reminder_id"):
            export_ics(d.model_copy(update={"reminder_id": bad}), tmp_path / "never", STAMP)
    assert not (tmp_path / "never").exists() and os.listdir(real) == []


@settings(max_examples=60, deadline=None)
@given(st.text(max_size=30))
def test_only_well_formed_reminder_ids_make_a_file_name(rid):
    import re
    d = approved().model_copy(update={"reminder_id": rid})
    if re.fullmatch(r"rem_[0-9a-f]{16}", rid):
        assert ics_filename(d).startswith(rid) and "/" not in ics_filename(d)
    else:
        with pytest.raises(ReminderError):
            ics_filename(d)


def test_two_jobs_with_the_same_content_write_different_files(tmp_path):
    a, b = approved(make(correlation_id="job_a")), approved(make(correlation_id="job_b"))
    assert a.reminder_id == b.reminder_id and ics_filename(a) != ics_filename(b)
    pa, _ = export_ics(a, tmp_path, STAMP)
    pb, _ = export_ics(b, tmp_path, STAMP)
    assert pa != pb and sorted(os.listdir(tmp_path)) == sorted([pa.name, pb.name])


def test_a_symlinked_export_target_is_audited_and_the_reminder_stays_approved(seeded_store, tmp_path):
    sup, saver = _sup(seeded_store, tmp_path), InMemorySaver()
    out = sup.run(REQUEST, correlation_id="j_link", checkpointer=saver)
    draft = ReminderDraft.model_validate(out["__interrupt__"][0].value["reminder"])
    (tmp_path / "ics").mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    (tmp_path / "ics" / ics_filename(draft)).symlink_to(victim)
    done = sup.resume("j_link", {"approver": "alex", "approved": True}, checkpointer=saver)
    assert victim.read_text() == "keep me" and "reminder_file" not in done
    assert done["briefing"]["actions_for_approval"][0]["reminder"]["status"] == "approved"
    assert ("create_reminder", "alex", "deny") in _events(seeded_store.audit, "j_link")


# LOW (d) denied-attempt caps

def test_repeated_denied_review_attempts_block_the_job(seeded_store, tmp_path):
    sup, saver = _review_job(seeded_store, tmp_path), InMemorySaver()
    sup.run(REQUEST, correlation_id="j_cap_r", checkpointer=saver)
    for n in range(1, MAX_DENIED_ATTEMPTS):
        out = sup.resume("j_cap_r", {"reviewer": "mallory", "accepted": True}, checkpointer=saver)
        assert out["review_denied_attempts"] == n and out["__interrupt__"] and out["status"] == "awaiting_review"
    out = sup.resume("j_cap_r", {"reviewer": "mallory", "accepted": True}, checkpointer=saver)
    assert "__interrupt__" not in out and out["status"] == "needs_review" and "reminder" not in out
    assert out["briefing"]["review"]["blocked"] is True and out["briefing"]["actions_for_approval"] == []
    reasons = [e["reason"] for e in seeded_store.audit.events("j_cap_r") if e["action"] == "review"]
    assert len(reasons) == MAX_DENIED_ATTEMPTS + 1 and "blocked" in reasons[-1]
    assert not (tmp_path / "ics").exists()


def test_repeated_denied_reminder_approvals_stop_the_wait_and_never_create_the_file(seeded_store, tmp_path):
    sup, saver = _sup(seeded_store, tmp_path), InMemorySaver()
    sup.run(REQUEST, correlation_id="j_cap_a", checkpointer=saver)
    for n in range(1, MAX_DENIED_ATTEMPTS):
        out = sup.resume("j_cap_a", {"approver": "mallory", "approved": True}, checkpointer=saver)
        assert out["reminder_denied_attempts"] == n and out["__interrupt__"]
    out = sup.resume("j_cap_a", {"approver": "mallory", "approved": True}, checkpointer=saver)
    assert "__interrupt__" not in out and out["status"] == "needs_review"
    assert out["briefing"]["actions_for_approval"][0]["reminder"]["status"] == "draft"
    assert not (tmp_path / "ics").exists()
    denies = [e for e in _events(seeded_store.audit, "j_cap_a") if e[0] == "approve" and e[2] == "deny"]
    assert len(denies) == MAX_DENIED_ATTEMPTS + 1
