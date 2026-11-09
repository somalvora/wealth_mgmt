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
    assert path.read_bytes() == content and path.name == f"{d.reminder_id}.ics"
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
    ics = (tmp_path / "ics" / f"{rid}.ics").read_bytes()
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
