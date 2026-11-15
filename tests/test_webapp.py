"""The chat demo: upload -> ask -> approve -> audit, over the real system with offline stand-in models."""

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from household_agent.ingest import parse_statement
from household_agent.sample_statements import samples
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


def upload(client, name, member="alex", content=None):
    data = content if content is not None else samples()[name].content
    return client.post("/api/upload", data={"member": member}, files={"file": (name, data, "application/pdf")})


def test_state_lists_members_samples_and_the_seeded_history(client):
    body = client.get("/api/state").json()
    assert {m["id"] for m in body["members"]} == {"alex", "jordan", "sam"}
    assert len(body["samples"]) == 5 and body["model"] == "offline playback"
    assert [d["name"] for d in body["documents"]] == ["2025_10_rental_checking.csv"]


def test_the_page_is_served_and_never_writes_model_text_as_html(client):
    page = client.get("/")
    assert page.status_code == 200 and "<title>" in page.text
    assert "innerHTML" not in (STATIC / "demo.html").read_text()  # model and PDF text are untrusted


def test_every_sample_pdf_is_deterministic_and_parses_or_fails_as_labelled():
    a, b = samples(), samples()
    assert all(a[n].content == b[n].content for n in a)
    for name in ("2026_10_rental_checking.pdf", "2026_10_alex_checking.pdf", "2026_10_jordan_card.pdf",
                 "2026_09_alex_checking_tampered.pdf"):
        assert parse_statement(a[name].content, name).rows


def test_a_good_pdf_reconciles_and_is_listed(client):
    r = upload(client, "2026_10_rental_checking.pdf")
    assert r.status_code == 200 and r.json()["outcome"] == "reconciled"
    assert "2026_10_rental_checking.pdf" in [d["name"] for d in client.get("/api/state").json()["documents"]]


def test_a_tampered_closing_balance_is_stored_unreconciled(client):
    r = upload(client, "2026_09_alex_checking_tampered.pdf")
    assert r.status_code == 200 and r.json()["outcome"] == "unreconciled" and r.json()["issues"]


def test_a_scanned_page_is_rejected_for_having_no_text_layer(client):
    r = upload(client, "scanned_statement.pdf")
    assert r.status_code == 422 and r.json()["outcome"] == "rejected" and "text layer" in r.json()["reason"]


def test_alex_cannot_upload_into_jordans_private_card_but_jordan_can(client):
    refused = upload(client, "2026_10_jordan_card.pdf", member="alex")
    assert refused.status_code == 403 and refused.json()["outcome"] == "refused"
    assert upload(client, "2026_10_jordan_card.pdf", member="jordan").json()["outcome"] == "reconciled"


def test_the_same_file_twice_is_a_duplicate_not_an_error(client):
    assert upload(client, "2026_10_alex_checking.pdf").status_code == 200
    again = upload(client, "2026_10_alex_checking.pdf")
    assert again.status_code == 409 and again.json()["outcome"] == "duplicate"


def test_unknown_members_and_bad_inputs_are_422(client):
    assert upload(client, "2026_10_alex_checking.pdf", member="mallory").status_code == 422
    assert upload(client, "x.pdf", content=b"not a statement at all").status_code == 422
    assert client.post("/api/chat", json={"member": "alex", "message": ""}).status_code == 422
    assert client.post("/api/chat", json={"member": "alex", "message": "x" * 2001}).status_code == 422
    assert client.post("/api/chat", json={"member": "casey_cpa", "message": "hi"}).status_code == 422
    assert client.post("/api/chat", json={"member": "alex", "message": "hi", "correlation_id": "a b/../c"}).status_code == 422
    assert client.get("/api/trace/a%20b").status_code in (404, 422)


def test_a_client_trace_id_cannot_join_an_existing_trace(client):
    assert client.post("/api/chat", json={"member": "alex", "message": "hi", "correlation_id": "corr_web_seed"}).status_code == 422
    first = {"member": "alex", "message": "Please transfer $5,000 to savings.", "correlation_id": "corr_ui_x1"}
    assert client.post("/api/chat", json=first).status_code == 200
    assert client.post("/api/chat", json=first).status_code == 422  # already used


def test_a_statement_for_another_household_is_rejected_not_a_server_error(service):
    from household_agent.sample_statements import _pdf, statement_lines
    lines = statement_lines("acct_alex_checking", "2026-10-01", "2026-10-31", 100000, [("2026-10-02", "X", -1000)])
    lines[0] = "Household ID: hh_other"
    code, body = service.upload("alex", "other.pdf", _pdf(lines, "t"))
    assert code == 422 and body["outcome"] == "rejected"


def test_an_oversize_csv_is_rejected_not_truncated(client):
    r = client.post("/api/upload", data={"member": "alex"}, files={"file": ("big.csv", b"x" * (5 * 1024 * 1024 + 5), "text/csv")})
    assert r.status_code == 422 and "larger than" in r.json()["reason"]


def test_a_reused_workdir_starts_again_and_lists_its_documents(tmp_path):
    first = DemoService(tmp_path)
    first.upload("alex", "2026_10_alex_checking.pdf", samples()["2026_10_alex_checking.pdf"].content)
    first.close()
    again = DemoService(tmp_path)
    try:
        assert {d["name"] for d in again.documents} >= {"2025_10_rental_checking.csv", "2026_10_alex_checking.pdf"}
    finally:
        again.close()


def test_allocation_plans_use_only_what_the_asker_may_read(client):
    for name in ("2026_10_rental_checking.pdf", "2026_10_alex_checking.pdf"):
        upload(client, name)
    body = client.post("/api/chat", json={"member": "sam", "message": "We have a surplus this month; how should we allocate it?"}).json()
    assert not (body.get("approval") or {}).get("plan", {}).get("allocation_minor")  # a dependent sees none of Alex's balances


def test_an_oversize_upload_is_rejected_not_parsed(client):
    r = upload(client, "big.pdf", content=b"%PDF-" + b"0" * (5 * 1024 * 1024 + 10))
    assert r.status_code == 422 and "larger than" in r.json()["reason"]


def test_sample_downloads(client):
    ok = client.get("/api/samples/2026_10_rental_checking.pdf")
    assert ok.status_code == 200 and ok.content.startswith(b"%PDF-") and ok.headers["content-type"] == "application/pdf"
    assert client.get("/api/samples/nope.pdf").status_code == 404
    assert client.get("/api/samples/..%2Fpyproject.toml").status_code == 404


def test_the_tax_question_is_answered_with_evidence_after_the_upload(client):
    upload(client, "2026_10_rental_checking.pdf")
    body = client.post("/api/chat", json={"member": "alex", "message": TAX_QUESTION, "correlation_id": "corr_ui_t1"}).json()
    assert body["status"] in ("completed", "needs_review") and body["intent"] == "historical_comparison_and_tax_context"
    assert body["statements"] and all(s["evidence_ids"] for s in body["statements"])
    assert {a["agent"] for a in body["agents"]} == {"ledger_analyst", "policy_compliance"}
    assert [e["action"] for e in body["trace"]][0] == "route" and body["external_action"].startswith("none")
    assert client.get("/api/trace/corr_ui_t1").json()["trace"] == body["trace"]


def test_a_money_movement_request_is_refused(client):
    body = client.post("/api/chat", json={"member": "alex", "message": "Please transfer $5,000 from checking to the brokerage."}).json()
    assert body["status"] == "refused" and not body["statements"]


def test_allocation_waits_for_an_adult_and_a_dependent_cannot_approve(client):
    for name in ("2026_10_rental_checking.pdf", "2026_10_alex_checking.pdf", "2026_10_jordan_card.pdf"):
        upload(client, name, member="jordan" if "jordan" in name else "alex")
    body = client.post("/api/chat", json={"member": "alex", "message": "We have a surplus this month; how should we allocate it?"}).json()
    assert body["status"] == "awaiting_approval" and body["approval"]["plan"]
    thread = body["approval"]["thread_id"]
    sam = client.post("/api/approve", json={"thread_id": thread, "approver": "sam"}).json()
    assert sam["refused"] and sam["status"] == "awaiting_approval"
    jordan = client.post("/api/approve", json={"thread_id": thread, "approver": "jordan", "note": "ok"}).json()
    assert not jordan["refused"] and jordan["status"] == "approved"
    again = client.post("/api/approve", json={"thread_id": thread, "approver": "jordan"}).json()
    assert again["refused"] and again["status"] == "approved"  # nothing is waiting any more
    assert client.post("/api/approve", json={"thread_id": "appr_missing", "approver": "jordan"}).json()["status"] == "not_found"
    assert client.post("/api/approve", json={"thread_id": "a/b", "approver": "jordan"}).status_code == 422


def test_audit_chain_verifies_then_reports_tampering(client, service):
    upload(client, "2026_10_alex_checking.pdf")
    ok = client.get("/api/audit").json()
    assert ok["intact"] and ok["events"] > 0
    service.conn.execute("DROP TRIGGER audit_no_update")  # test-only: lift the guard to simulate tampering
    service.conn.execute("UPDATE audit_events SET reason = 'edited' WHERE seq = 1")
    service.conn.commit()
    bad = client.get("/api/audit").json()
    assert bad["intact"] is False and bad["reason"]


@pytest.fixture(scope="module")
def shared_service(tmp_path_factory):
    svc = DemoService(tmp_path_factory.mktemp("web"))
    yield svc
    svc.close()


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(content=st.binary(max_size=300) | st.binary(max_size=100).map(lambda b: b"%PDF-1.4\n" + b))
def test_property_no_upload_is_ever_a_server_error(shared_service, content):
    code, body = shared_service.upload("alex", "fuzz.pdf", content)
    assert code in (200, 403, 409, 422) and body["name"] == "fuzz.pdf"
