import json
import sqlite3

from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from household_agent.agents import DispatchItem, RoutingDecision
from household_agent.briefing import DISCLAIMER
from household_agent.vertical_slice import (
    Check,
    PlaybackChatModel,
    SliceReport,
    _final,
    ledger_playback,
    main,
    policy_playback,
    run_vertical_slice,
)

CHECKS = ["ingest_and_reconcile", "three_agent_trace", "provenance_chain", "authorization_denial",
          "consent_revocation_denial", "unreconciled_block", "stale_source_escalation", "planning_reproducible",
          "cap_breach_hard_pruned", "critic_discrepancy_review", "draft_briefing", "interrupt_and_resume",
          "no_external_actions", "audit_trail"]


TRACE = RoutingDecision(intent="historical_comparison_and_tax_context", dispatch_plan=[
    DispatchItem(agent="ledger_analyst", task="compare"), DispatchItem(agent="policy_compliance", task="rules")])
PLAN = RoutingDecision(intent="allocation_planning", approval_required=True)
REFUSE = RoutingDecision(intent="out_of_scope", external_action_requested=True)


class RouterModel(BaseChatModel):
    """A router model: ``decide(request_text)`` returns its RoutingDecision."""

    decide: object

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = self.decide(str(messages[-1].content)).model_dump_json()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @property
    def _llm_type(self) -> str:
        return "router"


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


def _models(router=None, ledger=ledger_playback, policy=policy_playback):
    made = {"router": router, "ledger_analyst": ledger, "policy_compliance": policy}
    return lambda role: made[role]() if callable(made[role]) else made[role]


# --- happy path ------------------------------------------------------------------------

def test_offline_slice_meets_every_exit_criterion(tmp_path):
    report = run_vertical_slice(tmp_path)
    assert [c.name for c in report.checks] == CHECKS
    assert report.passed, [c for c in report.checks if not c.passed]
    assert report.mode == "offline playback models" and report.approval_thread_id
    assert all(c.evidence_ids for c in report.checks)


def test_trace_links_source_hashes_to_the_calculation_and_audit(tmp_path):
    report = run_vertical_slice(tmp_path)
    chain = _check(report, "provenance_chain")
    run_id, *rest = chain.evidence_ids
    conn = sqlite3.connect(tmp_path / "ledger.db")
    corr, inputs = conn.execute("SELECT correlation_id, input_ids_json FROM calculation_runs WHERE run_id = ?",
                                (run_id,)).fetchone()
    assert corr == report.correlation_id and json.loads(inputs) == rest[:2]
    hashes = conn.execute(f"SELECT source_hash FROM documents WHERE document_id IN ({','.join('?' * 2)})",
                          rest[2:]).fetchall()
    assert len(hashes) == 2 and all(h.startswith("sha256:") for (h,) in hashes)


def test_artifacts_are_kept_in_the_workdir(tmp_path):
    report = run_vertical_slice(tmp_path)
    assert {"ledger.db", "checkpoints.db", "chroma", "briefing_2026-10.md"} <= {p.name for p in tmp_path.iterdir()}
    md = (tmp_path / "briefing_2026-10.md").read_text()
    assert md == report.briefing_markdown and md.rstrip().endswith(DISCLAIMER)
    assert "$450.00 (16.36%) higher" in md and "waiting for a household admin" in md
    assert "acct_jordan_card" not in md


def test_resume_does_not_rerun_the_planning_step(tmp_path):
    report = run_vertical_slice(tmp_path)
    conn = sqlite3.connect(tmp_path / "ledger.db")
    actions = [a for (a,) in conn.execute("SELECT action FROM audit_events WHERE correlation_id = ? ORDER BY rowid",
                                          (report.approval_thread_id,))]
    assert actions == ["plan_allocation", "approve", "approve"]


def test_cli_writes_the_report_and_exits_zero(tmp_path, capsys):
    out = tmp_path / "slice.json"
    assert main(["--workdir", str(tmp_path / "w"), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["passed"] is True and [c["name"] for c in data["checks"]] == CHECKS
    assert "PASS  interrupt_and_resume" in capsys.readouterr().out


def _honest(text):
    return PLAN if "allocate" in text else REFUSE if "transfer" in text.lower() else TRACE


def test_model_factory_path_is_used_for_router_and_specialists(tmp_path):
    report = run_vertical_slice(tmp_path, models=_models(router=RouterModel(decide=_honest)))
    assert report.mode == "live models" and report.passed, [c for c in report.checks if not c.passed]


# --- negative paths --------------------------------------------------------------------

def test_a_ledger_model_that_skips_the_tools_fails_the_trace_but_not_the_guardrails(tmp_path):
    lazy = lambda: PlaybackChatModel(steps=[_final({"text": "Tax went up a lot.", "evidence_ids": []})])  # noqa: E731
    report = run_vertical_slice(tmp_path, models=_models(ledger=lazy))
    failed = {c.name for c in report.checks if not c.passed}
    assert not report.passed and {"three_agent_trace", "provenance_chain"} <= failed
    for name in ("authorization_denial", "consent_revocation_denial", "unreconciled_block",
                 "stale_source_escalation", "cap_breach_hard_pruned", "no_external_actions"):
        assert _check(report, name).passed, name


def test_router_model_that_hides_a_transfer_is_still_refused(tmp_path):
    def lie(text):
        return RoutingDecision(intent="out_of_scope") if "transfer" in text.lower() else _honest(text)
    report = run_vertical_slice(tmp_path, models=_models(router=RouterModel(decide=lie)))
    assert report.passed and _check(report, "no_external_actions").passed


def test_misrouted_allocation_request_fails_the_approval_check_without_crashing(tmp_path):
    report = run_vertical_slice(tmp_path, models=_models(router=RouterModel(decide=lambda _t: TRACE)))
    assert not _check(report, "interrupt_and_resume").passed and not report.passed
    assert _check(report, "three_agent_trace").passed and _check(report, "audit_trail").passed


def test_report_with_no_checks_or_a_failure_is_not_passed():
    assert not SliceReport("v", "m", "c", "t").passed
    ok, bad = Check("a", "x", True, ""), Check("b", "y", False, "")
    assert SliceReport("v", "m", "c", "t", [ok]).passed and not SliceReport("v", "m", "c", "t", [ok, bad]).passed
    assert json.loads(SliceReport("v", "m", "c", "t", [ok, bad]).to_json())["passed"] is False


def test_playback_past_its_script_returns_no_claims():
    model = PlaybackChatModel(steps=[])
    assert json.loads(model.invoke("hi").content) == {"claims": []}


# --- properties --------------------------------------------------------------------------

@settings(max_examples=3, deadline=None)
@given(st.from_regex(r"corr_[a-z0-9]{1,12}", fullmatch=True))
def test_any_correlation_id_passes_and_tags_every_job_event(tmp_path_factory, corr):
    workdir = tmp_path_factory.mktemp("slice")
    report = run_vertical_slice(workdir, correlation_id=corr)
    assert report.passed and report.correlation_id == corr
    conn = sqlite3.connect(workdir / "ledger.db")
    corrs = {c for (c,) in conn.execute("SELECT DISTINCT correlation_id FROM audit_events")}
    assert corrs == {corr, f"{corr}_plan", f"{corr}_external", report.approval_thread_id}
