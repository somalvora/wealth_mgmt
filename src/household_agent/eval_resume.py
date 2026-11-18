"""A live routing eval that can be finished across days without asking the model the same question twice.

A hosted model with a small daily quota cannot answer all the routing questions in one sitting. ``--resume FILE`` keeps a
checkpoint (JSON lines) of the requests the model really answered, so each run asks only what is still missing and stops
cleanly when the quota runs out.

What counts as an answer (the point of this module):

- the model replied and the reply is usable                      -> recorded, scored as the model's decision
- the model replied but the reply is not usable (bad JSON, an intent a host never lets a model choose)
                                                                  -> recorded, scored as an invalid reply (a schema failure)
- the call failed (quota, 429/503, dropped connection)           -> NOT an answer: noted as an attempt, asked again next run

The keyword rules never stand in for a missing answer. A failed run therefore cannot show a healthy score, because the final
report is only produced when every request has a real model reply on file. A run that is cut short exits with code 3.

The checkpoint remembers the model and a hash of the router prompt. Changing either invalidates it (the model's earlier
answers were to different instructions), so the run refuses to mix them; start a new file instead.
"""

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from household_agent.agents import ROUTER_PROMPT, Router, RoutingDecision, rule_route

EXIT_PAUSED = 3  # distinct from 1 (a gate failed) and 2 (preflight or early stop)
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3


class CheckpointMismatch(RuntimeError):
    """The checkpoint was made with a different model or router prompt, so its answers do not belong to this run."""


def request_key(request: str) -> str:
    return hashlib.sha256(request.encode()).hexdigest()[:16]


def prompt_fingerprint() -> str:
    return hashlib.sha256(ROUTER_PROMPT.encode()).hexdigest()[:16]


@dataclass
class Checkpoint:
    path: str
    model: str
    prompt: str
    answers: dict[str, dict[str, Any]] = field(default_factory=dict)  # request key -> record
    unavailable: int = 0

    @classmethod
    def open(cls, path: str, model: str, prompt: str | None = None) -> "Checkpoint":
        ck = cls(path, model, prompt or prompt_fingerprint())
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            ck._append({"type": "header", "model": model, "router_prompt": ck.prompt, "created": _now()})
            return ck
        with open(path) as fh:
            lines = fh.read().splitlines()
        records = []
        for i, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1:  # only a torn final line (a crash mid-write) is forgiven
                    raise CheckpointMismatch(f"{path} line {i + 1} is not valid JSON; the file was edited or damaged") from None
        header = records[0] if records else {}
        if header.get("type") != "header":
            raise CheckpointMismatch(f"{path} has no header line; it is not a routing checkpoint")
        if header.get("model") != model:
            raise CheckpointMismatch(f"{path} was made with model {header.get('model')!r}, this run uses {model!r}")
        if header.get("router_prompt") != ck.prompt:
            raise CheckpointMismatch(f"{path} was made with a different router prompt; its answers were to other "
                                     "instructions. Start a new checkpoint file.")
        for r in records[1:]:
            if r.get("type") == "answer":
                ck.answers[r["key"]] = r
            elif r.get("type") == "unavailable":
                ck.unavailable += 1
        return ck

    def _append(self, record: dict[str, Any]) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            os.fsync(fh.fileno())  # a quota stop or a crash right after must not lose an answer already paid for

    def record_answer(self, request: str, scenario_id: str, decision: RoutingDecision | None, why: str) -> None:
        rec = {"type": "answer", "key": request_key(request), "scenario_id": scenario_id,
               "kind": "model" if decision is not None else why,
               "decision": decision.model_dump(mode="json") if decision is not None else None, "at": _now()}
        self._append(rec)
        self.answers[rec["key"]] = rec

    def record_unavailable(self, scenario_id: str) -> None:
        self._append({"type": "unavailable", "scenario_id": scenario_id, "at": _now()})
        self.unavailable += 1

    def pending(self, scenarios: Sequence[Any]) -> list[Any]:
        """The scenarios whose request has no answer yet (one model call per distinct request text)."""
        seen: set[str] = set()
        out = []
        for s in scenarios:
            key = request_key(s.payload["request"])
            if key not in self.answers and key not in seen:
                seen.add(key)
                out.append(s)
        return out

    def invalid_replies(self) -> int:
        return sum(r["kind"] != "model" for r in self.answers.values())

    def replay(self) -> Callable[[str], tuple[RoutingDecision, str]]:
        """A router that answers from the file. A model reply replays as the model's decision. An unusable reply replays as
        the keyword rules' decision tagged ``rules_fallback``, which the harness already scores as an invalid reply."""
        def route(request: str) -> tuple[RoutingDecision, str]:
            rec = self.answers[request_key(request)]  # KeyError here would mean a run was scored before it was complete
            if rec["decision"] is None:
                return rule_route(request), "rules_fallback"
            return RoutingDecision.model_validate(rec["decision"]), "model"
        return route


@dataclass
class Collected:
    asked: int = 0  # model calls made this run
    answered: int = 0  # of those, how many produced an answer on file (usable or not)
    unavailable: int = 0
    stopped: str | None = None  # "consecutive_failures" | "max_calls" | None


def collect(scenarios: Sequence[Any], router: Router, ck: Checkpoint, *,
            max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES, max_calls: int | None = None,
            progress: Callable[[int, int, Any], None] | None = None) -> Collected:
    """Ask the model every question that still has no answer; stop at the first sign the quota is gone."""
    todo = ck.pending(scenarios)
    run = Collected()
    streak = 0
    for i, s in enumerate(todo, 1):
        if max_calls is not None and run.asked >= max_calls:
            run.stopped = "max_calls"
            break
        if progress:
            progress(i, len(todo), s)
        decision, why = router.ask(s.payload["request"])
        run.asked += 1
        if why == "unavailable":  # no answer exists, so nothing is recorded as one
            ck.record_unavailable(s.scenario_id)
            run.unavailable += 1
            streak += 1
            if streak >= max_consecutive_failures:
                run.stopped = "consecutive_failures"
                break
            continue
        streak = 0
        ck.record_answer(s.payload["request"], s.scenario_id, decision, why)
        run.answered += 1
    return run


def summary(scenarios: Sequence[Any], ck: Checkpoint, run: Collected) -> str:
    keys = {request_key(s.payload["request"]) for s in scenarios}
    done = len(keys & ck.answers.keys())
    line = (f"live routing: {done}/{len(keys)} distinct requests have a real model reply on file "
            f"({ck.invalid_replies()} unusable); this run asked {run.asked}, got {run.answered} replies, "
            f"{run.unavailable} calls unavailable ({ck.unavailable} in all runs, never counted as answers)")
    if run.stopped == "consecutive_failures":
        line += ". Stopped: the model stopped answering (quota?). Run the same command again later."
    elif run.stopped == "max_calls":
        line += ". Stopped at --max-calls. Run the same command again to continue."
    return line


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
