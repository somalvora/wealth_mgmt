"""Draft reminders and local .ics export (design §3.2 ``draft_reminder``, §5.8 approvals; Supervisor-only).

    draft ──approve──► approved ──export──► created        draft ──reject──► rejected

- ``draft_reminder`` builds a ``ReminderDraft`` and nothing else. It checks the due date against the injected
  date, caps and cleans the title, and requires source evidence IDs that exist in the job's own evidence.
- A reminder is an external-effect action, so it needs explicit approval from an authorized household member
  (the Supervisor pauses with ``interrupt()``, see ``agents.py``) before it is "created".
- "Created" only means: an RFC 5545 ``.ics`` file written under a directory the caller names. **No calendar is
  connected, nothing is sent anywhere, and there are no recurring events.** The user opens the file themselves.
"""

import hashlib
import re
import unicodedata
from collections.abc import Collection, Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from household_agent.audit import AuditLog

MAX_TITLE_LEN = 100
MAX_EVIDENCE_IDS = 5
ReminderStatus = Literal["draft", "approved", "rejected", "created"]
_ALLOWED = {"draft": {"approved", "rejected"}, "approved": {"created"}, "rejected": set(), "created": set()}
PRODID = "-//household-agent//reminders//EN"
UID_DOMAIN = "reminder.household-agent.local"


class ReminderError(ValueError):
    """A reminder request that must not become a draft (or a transition that is not allowed)."""


def _has_control(text: str) -> bool:
    # Cc: C0/C1 controls incl. CR/LF/NUL; Zl/Zp: U+2028/U+2029 line and paragraph separators.
    return any(unicodedata.category(ch) in ("Cc", "Zl", "Zp") for ch in text)


class ReminderDraft(BaseModel):
    model_config = ConfigDict(frozen=True)

    reminder_id: str
    title: str
    due_date: date
    source_evidence_ids: tuple[str, ...] = Field(min_length=1)
    household_id: str
    requested_by: str
    status: ReminderStatus = "draft"
    correlation_id: str

    @field_validator("title")
    @classmethod
    def _title(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title is empty")
        if len(v) > MAX_TITLE_LEN:
            raise ValueError(f"title is longer than {MAX_TITLE_LEN} characters")
        if _has_control(v):
            raise ValueError("title contains control characters or line breaks")
        return v

    @field_validator("source_evidence_ids")
    @classmethod
    def _evidence(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if any(not e.strip() or _has_control(e) for e in v):
            raise ValueError("evidence ID is empty or contains control characters")
        return v


def reminder_id_for(household_id: str, title: str, due_date: date, evidence_ids: Iterable[str]) -> str:
    """Deterministic ID from the reminder's content (not from who asked or when)."""
    payload = "\x1f".join([household_id, title, due_date.isoformat(), *sorted(set(evidence_ids))])
    return "rem_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def draft_reminder(*, title: str, due_date: date, evidence_ids: Iterable[str], job_evidence_ids: Collection[str],
                   household_id: str, requested_by: str, correlation_id: str, today: date) -> ReminderDraft:
    """The ``draft_reminder`` tool: validate and return a ``draft``. Raises ``ReminderError`` on any problem."""
    if due_date < today:
        raise ReminderError(f"due date {due_date.isoformat()} is in the past (today is {today.isoformat()})")
    ids = tuple(dict.fromkeys(evidence_ids))
    if not ids:
        raise ReminderError("a reminder needs at least one source evidence ID")
    unknown = [i for i in ids if i not in set(job_evidence_ids)]
    if unknown:
        raise ReminderError(f"evidence not found in this job: {sorted(unknown)}")
    try:
        return ReminderDraft(reminder_id=reminder_id_for(household_id, title, due_date, ids), title=title,
                             due_date=due_date, source_evidence_ids=ids, household_id=household_id,
                             requested_by=requested_by, status="draft", correlation_id=correlation_id)
    except ValueError as exc:  # pydantic ValidationError is a ValueError
        raise ReminderError(str(exc)) from exc


def transition(draft: ReminderDraft, to: ReminderStatus) -> ReminderDraft:
    if to not in _ALLOWED[draft.status]:
        raise ReminderError(f"cannot move a reminder from {draft.status} to {to}")
    return draft.model_copy(update={"status": to})


def audit_reminder(audit: AuditLog, draft: ReminderDraft, *, actor_id: str, action: str, decision: str,
                   reason: str, at: datetime, **details) -> None:
    audit.append(correlation_id=draft.correlation_id, actor_id=actor_id, household_id=draft.household_id,
                 acting_agent="supervisor", action=action, purpose="reminder", decision=decision, reason=reason,
                 resource=f"reminder:{draft.reminder_id}", details=details, at=at)


# --- extracting a deadline from a request (used by the Supervisor) ------------------------

_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], 1)}
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_NAMED = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b", re.I)


def extract_due_date(text: str, today: date) -> date | None:
    """First ISO date or "Month D[, YYYY]" in ``text``; a year-less date means the next such day."""
    try:
        if m := _ISO.search(text):
            return date(int(m[1]), int(m[2]), int(m[3]))
        if m := _NAMED.search(text):
            month, day = _MONTHS[m[1].lower()], int(m[2])
            if m[3]:
                return date(int(m[3]), month, day)
            d = date(today.year, month, day)
            return d if d >= today else date(today.year + 1, month, day)
    except ValueError:
        return None
    return None


def title_from_request(request: str) -> str:
    """A single-line title from free text: control characters become spaces, whitespace collapses, length is capped."""
    cleaned = "".join(" " if _has_control(ch) else ch for ch in request)
    return " ".join(cleaned.split())[:MAX_TITLE_LEN].rstrip()


# --- RFC 5545 export (pure string building; nothing is sent anywhere) ----------------------

def _escape(text: str) -> str:
    text = "".join(" " if _has_control(ch) else ch for ch in text)  # defence in depth; titles are already clean
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")


def _fold(line: str) -> list[bytes]:
    """Fold to at most 75 octets per physical line (continuations start with one space), never splitting a character."""
    out: list[bytes] = []
    current, limit = b"", 75
    for ch in line:
        enc = ch.encode()
        if len(current) + len(enc) > limit:
            out.append(current)
            current, limit = b" ", 75
        current += enc
    out.append(current)
    return out


def build_ics(draft: ReminderDraft, stamp: datetime) -> bytes:
    """The ``.ics`` text for a reminder: one all-day VEVENT, CRLF line endings, deterministic UID."""
    description = "Source evidence: " + ", ".join(draft.source_evidence_ids)
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", f"PRODID:{PRODID}", "CALSCALE:GREGORIAN",
        "BEGIN:VEVENT",
        f"UID:{draft.reminder_id}@{UID_DOMAIN}",
        f"DTSTAMP:{stamp.strftime('%Y%m%dT%H%M%SZ')}",
        f"DTSTART;VALUE=DATE:{draft.due_date.strftime('%Y%m%d')}",
        f"SUMMARY:{_escape(draft.title)}",
        f"DESCRIPTION:{_escape(description)}",
        "END:VEVENT", "END:VCALENDAR",
    ]
    return b"".join(part + b"\r\n" for line in lines for part in _fold(line))


def export_ics(draft: ReminderDraft, directory: Path, stamp: datetime) -> tuple[Path, bytes]:
    """Write an *approved* reminder's ``.ics`` under ``directory`` and return ``(path, bytes)``. Only approved
    reminders are exported, so a draft or rejected one never produces a file."""
    if draft.status != "approved":
        raise ReminderError(f"only an approved reminder can be exported (status is {draft.status})")
    content = build_ics(draft, stamp)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{draft.reminder_id}.ics"
    path.write_bytes(content)
    return path, content
