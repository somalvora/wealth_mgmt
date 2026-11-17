"""The chat demo: upload -> ask -> approve -> audit, over the real system with offline stand-in models."""

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from household_agent.ingest import parse_statement
from household_agent.models import Action
from household_agent.sample_statements import samples
from household_agent.synthetic import DEMO_AS_OF
from household_agent.webapp import STATIC, DemoService, create_app

TAX_QUESTION = ("I paid $3,200 to the San Mateo County Tax Collector in October. Is this unusual compared with last "
                "year, and how should it be accounted for?")


@pytest.fixture
def service(tmp_path):
    svc = DemoService(tmp_path)
    yield svc
    svc.close()


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def upload(client, name, member="somal", content=None):
    data = content if content is not None else samples()[name].content
    return client.post("/api/upload", data={"member": member}, files={"file": (name, data, "application/pdf")})


def test_state_lists_members_samples_and_the_seeded_history(client):
    body = client.get("/api/state").json()
    assert {m["id"] for m in body["members"]} == {"somal", "sayali", "viaan"}
    assert len(body["samples"]) == 5 and body["model"] == "offline playback"
    assert "2026_10_rental_checking.pdf" not in [x["name"] for x in body["samples"]]  # preloaded, not an upload step
    assert [d["name"] for d in body["documents"]] == ["2025_10_rental_checking.csv", "2026_10_rental_checking.csv"]


def test_the_page_is_served_and_never_writes_model_text_as_html(client):
    page = client.get("/")
    assert page.status_code == 200 and "<title>" in page.text
    assert "innerHTML" not in (STATIC / "demo.html").read_text()  # model and PDF text are untrusted


def test_every_sample_pdf_is_deterministic_and_parses_or_fails_as_labelled():
    a, b = samples(), samples()
    assert all(a[n].content == b[n].content for n in a)
    for name in ("2026_10_rental_checking.pdf", "2026_10_somal_checking.pdf", "2026_10_sayali_card.pdf",
                 "2026_09_somal_checking_tampered.pdf", "2026_09_somal_checking_injection.pdf"):
        assert parse_statement(a[name].content, name).rows


def test_a_good_pdf_reconciles_and_is_listed(client):
    r = upload(client, "2026_10_somal_checking.pdf")
    assert r.status_code == 200 and r.json()["outcome"] == "reconciled"
    assert "2026_10_somal_checking.pdf" in [d["name"] for d in client.get("/api/state").json()["documents"]]


def test_a_tampered_closing_balance_is_stored_unreconciled(client):
    r = upload(client, "2026_09_somal_checking_tampered.pdf")
    assert r.status_code == 200 and r.json()["outcome"] == "unreconciled" and r.json()["issues"]


def test_a_scanned_page_is_rejected_for_having_no_text_layer(client):
    r = upload(client, "scanned_statement.pdf")
    assert r.status_code == 422 and r.json()["outcome"] == "rejected" and "text layer" in r.json()["reason"]


def test_somal_cannot_upload_into_sayalis_private_card_but_sayali_can(client):
    refused = upload(client, "2026_10_sayali_card.pdf", member="somal")
    assert refused.status_code == 403 and refused.json()["outcome"] == "refused"
    assert upload(client, "2026_10_sayali_card.pdf", member="sayali").json()["outcome"] == "reconciled"


def test_the_same_file_twice_is_a_duplicate_not_an_error(client):
    assert upload(client, "2026_10_somal_checking.pdf").status_code == 200
    again = upload(client, "2026_10_somal_checking.pdf")
    assert again.status_code == 409 and again.json()["outcome"] == "duplicate"


def test_unknown_members_and_bad_inputs_are_422(client):
    assert upload(client, "2026_10_somal_checking.pdf", member="mallory").status_code == 422
    assert upload(client, "x.pdf", content=b"not a statement at all").status_code == 422
    assert client.post("/api/chat", json={"member": "somal", "message": ""}).status_code == 422
    assert client.post("/api/chat", json={"member": "somal", "message": "x" * 2001}).status_code == 422
    assert client.post("/api/chat", json={"member": "casey_cpa", "message": "hi"}).status_code == 422
    assert client.post("/api/chat", json={"member": "somal", "message": "hi", "correlation_id": "a b/../c"}).status_code == 422
    assert client.get("/api/trace/a%20b").status_code in (404, 422)


def test_a_client_trace_id_cannot_join_an_existing_trace(client):
    assert client.post("/api/chat", json={"member": "somal", "message": "hi", "correlation_id": "corr_web_seed"}).status_code == 422
    first = {"member": "somal", "message": "Please transfer $5,000 to savings.", "correlation_id": "corr_ui_x1"}
    assert client.post("/api/chat", json=first).status_code == 200
    assert client.post("/api/chat", json=first).status_code == 422  # already used


def test_a_statement_for_another_household_is_rejected_not_a_server_error(service):
    from household_agent.sample_statements import _pdf, statement_lines
    lines = statement_lines("acct_somal_checking", "2026-10-01", "2026-10-31", 100000, [("2026-10-02", "X", -1000)])
    lines[0] = "Household ID: hh_other"
    code, body = service.upload("somal", "other.pdf", _pdf(lines, "t"))
    assert code == 422 and body["outcome"] == "rejected"


def test_an_oversize_csv_is_rejected_not_truncated(client):
    r = client.post("/api/upload", data={"member": "somal"}, files={"file": ("big.csv", b"x" * (5 * 1024 * 1024 + 5), "text/csv")})
    assert r.status_code == 422 and "larger than" in r.json()["reason"]


def test_a_reused_workdir_starts_again_and_lists_its_documents(tmp_path):
    first = DemoService(tmp_path)
    first.upload("somal", "2026_10_somal_checking.pdf", samples()["2026_10_somal_checking.pdf"].content)
    first.close()
    again = DemoService(tmp_path)
    try:
        assert {d["name"] for d in again.documents} >= {"2025_10_rental_checking.csv", "2026_10_somal_checking.pdf"}
    finally:
        again.close()


def test_allocation_plans_use_only_what_the_asker_may_read(client):
    upload(client, "2026_10_somal_checking.pdf")
    body = client.post("/api/chat", json={"member": "viaan", "message": "We have a surplus this month; how should we allocate it?"}).json()
    assert not (body.get("approval") or {}).get("plan", {}).get("allocation_minor")  # a dependent sees none of Somal's balances


def test_an_oversize_upload_is_rejected_not_parsed(client):
    r = upload(client, "big.pdf", content=b"%PDF-" + b"0" * (5 * 1024 * 1024 + 10))
    assert r.status_code == 422 and "larger than" in r.json()["reason"]


def test_sample_downloads(client):
    ok = client.get("/api/samples/2026_10_rental_checking.pdf")
    assert ok.status_code == 200 and ok.content.startswith(b"%PDF-") and ok.headers["content-type"] == "application/pdf"
    assert client.get("/api/samples/nope.pdf").status_code == 404
    assert client.get("/api/samples/..%2Fpyproject.toml").status_code == 404


def test_the_tax_question_is_answered_with_evidence_with_no_upload(client):
    body = client.post("/api/chat", json={"member": "somal", "message": TAX_QUESTION, "correlation_id": "corr_ui_t1"}).json()
    assert body["status"] in ("completed", "needs_review") and body["intent"] == "historical_comparison_and_tax_context"
    assert body["statements"] and all(s["evidence_ids"] for s in body["statements"])
    assert {a["agent"] for a in body["agents"]} == {"ledger_analyst", "policy_compliance"}
    assert [e["action"] for e in body["trace"]][0] == "route" and body["external_action"].startswith("none")
    assert client.get("/api/trace/corr_ui_t1").json()["trace"] == body["trace"]


def test_a_money_movement_request_is_refused(client):
    body = client.post("/api/chat", json={"member": "somal", "message": "Please transfer $5,000 from checking to the brokerage."}).json()
    assert body["status"] == "refused" and not body["statements"]


def test_an_unrelated_question_is_out_of_scope_and_the_page_says_so(client):
    body = client.post("/api/chat", json={"member": "somal", "message": "What's the weather in Boston?"}).json()
    assert body["intent"] == "out_of_scope" and not body["statements"] and not body["approval"]
    assert "outside what I can help with" in (STATIC / "demo.html").read_text()


def test_allocation_waits_for_an_adult_and_a_dependent_cannot_approve(client):
    for name in ("2026_10_somal_checking.pdf", "2026_10_sayali_card.pdf"):
        upload(client, name, member="sayali" if "sayali" in name else "somal")
    body = client.post("/api/chat", json={"member": "somal", "message": "We have a surplus this month; how should we allocate it?"}).json()
    assert body["status"] == "awaiting_approval" and body["approval"]["plan"]
    thread = body["approval"]["thread_id"]
    viaan = client.post("/api/approve", json={"thread_id": thread, "approver": "viaan"}).json()
    assert viaan["refused"] and viaan["status"] == "awaiting_approval"
    sayali = client.post("/api/approve", json={"thread_id": thread, "approver": "sayali", "note": "ok"}).json()
    assert not sayali["refused"] and sayali["status"] == "approved"
    again = client.post("/api/approve", json={"thread_id": thread, "approver": "sayali"}).json()
    assert again["refused"] and again["status"] == "approved"  # nothing is waiting any more
    assert client.post("/api/approve", json={"thread_id": "appr_missing", "approver": "sayali"}).json()["status"] == "not_found"
    assert client.post("/api/approve", json={"thread_id": "a/b", "approver": "sayali"}).status_code == 422


def test_audit_chain_verifies_then_reports_tampering(client, service):
    upload(client, "2026_10_somal_checking.pdf")
    ok = client.get("/api/audit").json()
    assert ok["intact"] and ok["events"] > 0
    service.conn.execute("DROP TRIGGER audit_no_update")  # test-only: lift the guard to simulate tampering
    service.conn.execute("UPDATE audit_events SET reason = 'edited' WHERE seq = 1")
    service.conn.commit()
    bad = client.get("/api/audit").json()
    assert bad["intact"] is False and bad["reason"]


def test_no_current_statements_says_so_instead_of_a_misleading_reserve_message(tmp_path):
    svc = DemoService(tmp_path, history=False)  # only last October is on file: nothing to allocate yet
    client = TestClient(create_app(svc))
    body = client.post("/api/chat", json={"member": "somal", "message": "We have a surplus this month; how should we allocate it?"}).json()
    assert body["approval"]["status"] == "not_eligible" and "Upload this month" in body["approval"]["note"]
    svc.close()


def test_with_two_years_of_history_an_allocation_is_planned_from_the_latest_statements(client):
    body = client.post("/api/chat", json={"member": "somal", "message": "We have a surplus this month; how should we allocate it?"}).json()
    assert body["approval"]["status"] == "awaiting_approval" and body["approval"]["note"] is None


def test_ledger_reads_in_the_trace_name_the_agent(client):
    body = client.post("/api/chat", json={"member": "somal", "message": TAX_QUESTION}).json()
    reads = [e for e in body["trace"] if e["action"] == "read"]
    assert reads and all(e["agent"] for e in reads)


@pytest.mark.parametrize("intent,expect_note", [("historical_comparison_and_tax_context", True), ("statement_ingestion", False)])
def test_a_ledger_analyst_with_no_figures_is_called_out_in_the_briefing(intent, expect_note):
    from household_agent.agents import NO_LEDGER_FIGURES, AgentResult, compose_briefing
    empty = AgentResult(task_id="t", correlation_id="c", agent="ledger_analyst", status="budget_exhausted").model_dump()
    out = compose_briefing({"results": [empty], "routing": {"intent": intent}})
    assert (NO_LEDGER_FIGURES in out["questions"]) is expect_note and out["needs_review"]


def test_a_crashed_ledger_analyst_is_not_reported_as_a_missing_statement():
    from household_agent.agents import NO_LEDGER_FIGURES, AgentResult, compose_briefing
    crashed = AgentResult(task_id="t", correlation_id="c", agent="ledger_analyst", status="failed",
                          needs_review=True, errors=["GoogleRateLimitError: 429"]).model_dump()
    out = compose_briefing({"results": [crashed], "routing": {"intent": "historical_comparison"}})
    assert NO_LEDGER_FIGURES not in out["questions"] and out["needs_review"]


def test_the_no_figures_note_does_not_assume_the_missing_statement_is_this_months():
    from household_agent.agents import NO_LEDGER_FIGURES
    assert "past month or year" in NO_LEDGER_FIGURES and "this month's" not in NO_LEDGER_FIGURES


@pytest.fixture(scope="module")
def shared_service(tmp_path_factory):
    svc = DemoService(tmp_path_factory.mktemp("web"))
    yield svc
    svc.close()


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(content=st.binary(max_size=300) | st.binary(max_size=100).map(lambda b: b"%PDF-1.4\n" + b))
def test_property_no_upload_is_ever_a_server_error(shared_service, content):
    code, body = shared_service.upload("somal", "fuzz.pdf", content)
    assert code in (200, 403, 409, 422) and body["name"] == "fuzz.pdf"


# --- demo additions: chips, injection sample, audit list, consent panel, evaluation panel ------------------------

def test_every_suggestion_chip_names_a_real_member(client):
    body = client.get("/api/state").json()
    members = {m["id"] for m in body["members"]}
    assert len(body["suggestions"]) >= 8 and all(s["member"] in members for s in body["suggestions"])


def test_the_injection_statement_reconciles_but_its_text_is_flagged_in_the_audit_list(client):
    r = upload(client, "2026_09_somal_checking_injection.pdf")
    assert r.status_code == 200 and r.json()["outcome"] == "reconciled"
    recent = client.get("/api/audit").json()["recent"]
    flagged = [e for e in recent if e["flagged"]]
    assert flagged and flagged[0]["action"] == "index"
    assert recent[0]["seq"] > recent[-1]["seq"]  # newest first
    assert not any(e["flagged"] for e in client.get("/api/audit").json()["recent"] if e["action"] == "upload")


def test_the_injection_and_tampered_samples_can_both_be_uploaded(client):
    assert upload(client, "2026_09_somal_checking_tampered.pdf").json()["outcome"] == "unreconciled"
    assert upload(client, "2026_09_somal_checking_injection.pdf").json()["outcome"] == "reconciled"


def read(service, actor, account="acct_sayali_card"):
    return bool(service.store.authorizer.authorize(actor, account, Action.READ, "historical_comparison", DEMO_AS_OF))


def test_sharing_a_private_account_opens_it_and_revoking_closes_it_again(client, service):
    somal, sayali = service.actor("somal"), service.actor("sayali")
    assert not read(service, somal)
    shared = client.post("/api/consent/share", json={"member": "sayali", "grantee": "somal", "account_id": "acct_sayali_card"})
    assert shared.status_code == 200 and shared.json()["outcome"] == "shared"
    assert read(service, somal)
    view = client.get("/api/consent").json()
    assert [(g["grantor"], g["grantee"], g["active"]) for g in view["shares"]] == [("sayali", "somal", True)]
    # a share is read-only: it never lets Somal write to the account
    assert not service.store.authorizer.authorize(somal, "acct_sayali_card", Action.UPLOAD, "statement_ingestion", DEMO_AS_OF)
    grant_id = shared.json()["grant_id"]
    revoked = client.post("/api/consent/revoke", json={"member": "sayali", "grant_id": grant_id})
    assert revoked.status_code == 200 and not read(service, somal)
    assert client.get("/api/consent").json()["shares"][0]["active"] is False
    actions = [e["action"] for e in client.get("/api/audit").json()["recent"]]
    assert "consent_grant" in actions and "consent_revoke" in actions


def test_only_the_owner_may_share_and_only_the_grantor_may_revoke(client, service):
    other = client.post("/api/consent/share", json={"member": "somal", "grantee": "sayali", "account_id": "acct_sayali_card"})
    assert other.status_code == 403 and other.json()["outcome"] == "refused"
    kid = client.post("/api/consent/share", json={"member": "viaan", "grantee": "somal", "account_id": "acct_somal_checking"})
    assert kid.status_code == 403
    to_kid = client.post("/api/consent/share", json={"member": "somal", "grantee": "viaan", "account_id": "acct_somal_checking"})
    assert to_kid.status_code == 422
    grant = client.post("/api/consent/share", json={"member": "sayali", "grantee": "somal", "account_id": "acct_sayali_card"}).json()
    steal = client.post("/api/consent/revoke", json={"member": "somal", "grant_id": grant["grant_id"]})
    assert steal.status_code == 403 and read(service, service.actor("somal"))
    assert client.post("/api/consent/revoke", json={"member": "sayali", "grant_id": "grant_cpa_rental"}).status_code == 422


def test_shares_survive_a_restart_of_the_service(tmp_path):
    first = DemoService(tmp_path)
    code, _ = first.share("sayali", "somal", "acct_sayali_card")
    first.close()
    again = DemoService(tmp_path)
    try:
        assert code == 200 and read(again, again.actor("somal"))
    finally:
        again.close()


def test_the_evaluation_panel_runs_the_offline_gates_and_caches_the_result(client):
    assert client.get("/api/eval").json()["report"] is None
    report = client.post("/api/eval/run").json()["report"]
    assert report["passed"] and len(report["gates"]) == 13 and report["scenarios"] > 200
    assert all(g["label"] and g["label"] != g["name"] for g in report["gates"])
    assert client.get("/api/eval").json()["report"]["gates"] == report["gates"]


def test_asking_about_someone_elses_private_account_is_refused_out_loud_until_it_is_shared(client):
    def ask(text="How much did Sayali's card spend at Safeway in September 2026?"):
        return client.post("/api/chat", json={"member": "somal", "message": text}).json()

    body = ask()
    assert body["status"] == "denied" and not body["statements"] and body["agents"] == []
    assert "sayali" in body["alerts"][0]["text"].lower() and "not shared" in body["alerts"][0]["text"]
    deny = [e for e in body["trace"] if e["decision"] == "deny"]
    assert deny and deny[0]["resource"] == "account:acct_sayali_card"
    grant = client.post("/api/consent/share", json={"member": "sayali", "grantee": "somal", "account_id": "acct_sayali_card"}).json()
    assert ask()["status"] != "denied"
    client.post("/api/consent/revoke", json={"member": "sayali", "grant_id": grant["grant_id"]})
    assert ask()["status"] == "denied"
    assert ask("What did somal spend at Safeway?")["status"] != "denied"  # your own name is never a reason to refuse
