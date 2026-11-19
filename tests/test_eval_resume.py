"""The resumable live routing eval: a failed call is never an answer, a partly finished run produces no score, and
a finished checkpoint is scored without asking the model anything again."""

import json
from types import SimpleNamespace

import pytest

from household_agent import eval_resume as resume
from household_agent.agents import Router, RoutingDecision, rule_route
from household_agent.evals import generate_scenarios, main

ROUTING = [s for s in generate_scenarios(0) if s.area == "routing"]
N = len({s.payload["request"] for s in ROUTING})


class FakeModel:
    """A hosted model. ``dies_after`` replies, then every call raises (an exhausted quota); ``reply`` shapes the answers."""

    def __init__(self, dies_after: int | None = None, reply=None):
        self.calls = self.answered = 0
        self.dies_after = dies_after
        self.reply = reply or (lambda request: rule_route(request).model_dump_json())

    def invoke(self, messages, *a, **k):
        self.calls += 1
        if self.dies_after is not None and self.answered >= self.dies_after:
            raise RuntimeError("429 RESOURCE_EXHAUSTED secret-provider-detail")
        self.answered += 1
        return SimpleNamespace(text=self.reply(messages[1].content))


def _run(monkeypatch, model, ck, *extra, capsys=None):
    monkeypatch.setattr("household_agent.llm.get_chat_model", lambda spec: model)
    code = main(["--model", "x:y", "--only-area", "routing", "--resume", str(ck), *extra])
    return code, (capsys.readouterr() if capsys else None)


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_quota_that_dies_midway_stops_cleanly_and_prints_no_score(monkeypatch, capsys, tmp_path):
    ck = tmp_path / "live.jsonl"
    model = FakeModel(dies_after=10)
    code, out = _run(monkeypatch, model, ck, capsys=capsys)
    assert code == resume.EXIT_PAUSED == 3
    assert "routing_accuracy" not in out.out and "PASS" not in out.out  # the keyword rules never stand in for the model
    assert "10/55" in out.out and "never counted as answers" in out.out and "secret-provider-detail" not in out.out + out.err
    assert "run the same command again" in out.err
    recs = _records(ck)
    assert [r["type"] for r in recs].count("answer") == 10 and [r["type"] for r in recs].count("unavailable") == 3
    assert model.calls == 13  # 10 answers, then 3 failures in a row, then it stopped asking


def test_the_next_run_asks_only_what_is_missing_and_a_finished_file_costs_nothing(monkeypatch, capsys, tmp_path):
    ck, report = tmp_path / "live.jsonl", tmp_path / "report.json"
    _run(monkeypatch, FakeModel(dies_after=10), ck)
    healthy = FakeModel()
    code, out = _run(monkeypatch, healthy, ck, "--out", str(report), capsys=capsys)
    assert code == 0 and healthy.calls == N - 10  # nothing asked twice
    assert "PASS  routing_accuracy" in out.out and "scored from 55 recorded model replies" in out.out
    data = json.loads(report.read_text())
    assert data["live_routing"] == {"checkpoint": str(ck), "model": "x:y", "replies": N, "unusable_replies": 0,
                                    "failed_calls_not_counted": 3}
    assert data["metrics"]["router_fallbacks"] == 0 and data["metrics"]["router_model_calls"] == N
    third = FakeModel()
    code, _ = _run(monkeypatch, third, ck, capsys=capsys)
    assert code == 0 and third.calls == 0


def test_a_model_that_never_answers_records_no_answers_and_gives_no_score(monkeypatch, capsys, tmp_path):
    ck = tmp_path / "live.jsonl"
    code, out = _run(monkeypatch, FakeModel(dies_after=0), ck, capsys=capsys)
    assert code == 3 and "0/55" in out.out and "routing_accuracy" not in out.out
    assert all(r["type"] != "answer" for r in _records(ck))


def test_the_score_comes_from_the_model_not_from_the_rules(monkeypatch, capsys, tmp_path):
    wrong = FakeModel(reply=lambda request: RoutingDecision(intent="out_of_scope").model_dump_json())
    code, out = _run(monkeypatch, wrong, tmp_path / "live.jsonl", capsys=capsys)
    assert code == 1 and "FAIL  routing_accuracy" in out.out  # the rules alone would score 1.0


def test_an_unusable_reply_is_recorded_once_and_scored_as_a_schema_failure(monkeypatch, capsys, tmp_path):
    ck, victim = tmp_path / "live.jsonl", ROUTING[7].payload["request"]
    model = FakeModel(reply=lambda r: "I think this is about groceries" if r == victim
                      else rule_route(r).model_dump_json())
    code, out = _run(monkeypatch, model, ck, capsys=capsys)
    assert code == 1 and "FAIL  schema_validity" in out.out and "1 unusable" in out.out
    assert any(r.get("kind") == "invalid_reply" and r["decision"] is None for r in _records(ck))
    again = FakeModel()
    _run(monkeypatch, again, ck)
    assert again.calls == 0  # the model did answer; asking again would only spend quota


def test_a_statement_ingestion_intent_from_the_model_is_an_unusable_reply(monkeypatch, capsys, tmp_path):
    model = FakeModel(reply=lambda r: RoutingDecision(intent="statement_ingestion").model_dump_json())
    code, out = _run(monkeypatch, model, tmp_path / "live.jsonl", capsys=capsys)
    assert code == 1 and "55 unusable" in out.out


def test_max_calls_leaves_quota_for_other_use(monkeypatch, capsys, tmp_path):
    ck = tmp_path / "live.jsonl"
    model = FakeModel()
    code, out = _run(monkeypatch, model, ck, "--max-calls", "5", capsys=capsys)
    assert code == 3 and model.calls == 5 and "Stopped at --max-calls" in out.out


def test_consecutive_failures_limit_is_a_setting(monkeypatch, tmp_path):
    model = FakeModel(dies_after=0)
    assert _run(monkeypatch, model, tmp_path / "live.jsonl", "--max-consecutive-failures", "1")[0] == 3
    assert model.calls == 1


def test_a_success_resets_the_failure_streak(tmp_path):
    flaky = SimpleNamespace(calls=0)

    class Flaky(FakeModel):  # fails twice, answers, fails twice, answers, ...
        def invoke(self, messages, *a, **k):
            flaky.calls += 1
            if flaky.calls % 3:
                raise RuntimeError("503")
            return super().invoke(messages)

    ck = resume.Checkpoint.open(str(tmp_path / "f.jsonl"), "x:y")
    run = resume.collect(ROUTING, Router(Flaky()), ck, max_consecutive_failures=3, max_calls=9)
    assert run.stopped == "max_calls" and run.answered == 3 and run.unavailable == 6


def test_a_changed_router_prompt_or_model_refuses_to_mix_answers(monkeypatch, capsys, tmp_path):
    ck = tmp_path / "live.jsonl"
    _run(monkeypatch, FakeModel(dies_after=3), ck)
    monkeypatch.setattr(resume, "ROUTER_PROMPT", "a different prompt")
    code, out = _run(monkeypatch, FakeModel(), ck, capsys=capsys)
    assert code == 2 and "different router prompt" in out.err and "routing_accuracy" not in out.out
    monkeypatch.undo()
    monkeypatch.setattr("household_agent.llm.get_chat_model", lambda spec: FakeModel())
    assert main(["--model", "other:model", "--only-area", "routing", "--resume", str(ck)]) == 2
    assert "was made with model 'x:y'" in capsys.readouterr().err


def test_a_torn_last_line_is_forgiven_but_a_damaged_middle_is_not(tmp_path):
    path = tmp_path / "live.jsonl"
    ck = resume.Checkpoint.open(str(path), "x:y")
    ck.record_answer("hello", "route_000", rule_route("hello"), "model")
    with open(path, "a") as fh:
        fh.write('{"type": "answer", "key": "abc", "kin')  # a crash mid-write
    assert len(resume.Checkpoint.open(str(path), "x:y").answers) == 1
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], "garbage", *lines[1:]]) + "\n")
    with pytest.raises(resume.CheckpointMismatch, match="line 2 is not valid JSON"):
        resume.Checkpoint.open(str(path), "x:y")


def test_a_file_that_is_not_a_checkpoint_is_refused(tmp_path):
    path = tmp_path / "other.jsonl"
    path.write_text('{"hello": "world"}\n')
    with pytest.raises(resume.CheckpointMismatch, match="no header"):
        resume.Checkpoint.open(str(path), "x:y")


def test_one_call_per_distinct_request_text(tmp_path):
    ck = resume.Checkpoint.open(str(tmp_path / "f.jsonl"), "x:y")
    twin = SimpleNamespace(scenario_id="route_900", payload={"request": ROUTING[0].payload["request"]})
    assert len(ck.pending([*ROUTING, twin])) == N


def test_arguments_are_checked_before_anything_runs(monkeypatch, tmp_path):
    ck = str(tmp_path / "f.jsonl")
    for argv in (["--resume", ck], ["--model", "x:y", "--resume", ck],  # no --model / no --only-area
                 ["--model", "x:y", "--only-area", "retrieval", "--resume", ck],
                 ["--model", "x:y", "--only-area", "routing", "--resume", ck, "--max-calls", "0"],
                 ["--model", "x:y", "--only-area", "routing", "--resume", ck, "--planner-model", "x:z"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 2


def test_router_ask_tells_a_failed_call_from_an_unusable_reply():
    request = "how much did we spend on groceries"
    assert Router(FakeModel(dies_after=0)).ask(request) == (None, "unavailable")
    assert Router(FakeModel(reply=lambda r: "nonsense")).ask(request) == (None, "invalid_reply")
    decision, why = Router(FakeModel()).ask(request)
    assert why == "model" and decision == rule_route(request)
