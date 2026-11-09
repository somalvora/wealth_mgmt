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
import os
import re
import unicodedata
from collections.abc import Collection, Iterable
from datetime import UTC, date, datetime
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
_REMINDER_ID = re.compile(r"rem_[0-9a-f]{16}")


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


_LEAD = re.compile(r"\b(?:on|by|before|due(?:\s+on)?|until|for)\s+(?:the\s+)?$", re.I)
_REMIND = re.compile(r"\bremind\w*", re.I)


def _mentions(text: str, today: date) -> list[tuple[int, date]]:
    """Every valid date in ``text`` as (start offset, date), in order. A year-less date means the next such day."""
    found: list[tuple[int, date]] = []
    for m in _ISO.finditer(text):
        try:
            found.append((m.start(), date(int(m[1]), int(m[2]), int(m[3]))))
        except ValueError:
            pass
    for m in _NAMED.finditer(text):
        try:
            month, day = _MONTHS[m[1].lower()], int(m[2])
            if m[3]:
                found.append((m.start(), date(int(m[3]), month, day)))
            else:
                d = date(today.year, month, day)
                found.append((m.start(), d if d >= today else date(today.year + 1, month, day)))
        except ValueError:
            pass
    return sorted(found)


def due_date_from_request(text: str, today: date) -> date:
    """The reminder's date. One date: that one. Several: prefer those after the word "remind"/"reminder", then the
    one introduced by on/by/before/due/until/for; if that still leaves more than one, refuse (``ReminderError``)
    rather than guess. ``today`` is supplied by the caller (the job's date), never read from the system clock."""
    mentions = _mentions(text, today)
    if not mentions:
        raise ReminderError("no deadline date found in the request (use YYYY-MM-DD)")
    remind = _REMIND.search(text)
    pool = [m for m in mentions if remind and m[0] >= remind.end()] or mentions
    if len({d for _, d in pool}) > 1:
        led = [m for m in pool if _LEAD.search(text[max(0, m[0] - 14):m[0]])]
        pool = led if len({d for _, d in led}) == 1 else pool
    dates = sorted({d for _, d in pool})
    if len(dates) > 1:
        raise ReminderError("several dates in the request and none clearly follows 'remind me' "
                            f"({', '.join(d.isoformat() for d in dates)}); say 'remind me on YYYY-MM-DD'")
    return dates[0]


def extract_due_date(text: str, today: date) -> date | None:
    """``due_date_from_request`` that returns ``None`` instead of raising (no date, or an ambiguous request)."""
    try:
        return due_date_from_request(text, today)
    except ReminderError:
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
        f"DTSTAMP:{(stamp.astimezone(UTC) if stamp.tzinfo else stamp).strftime('%Y%m%dT%H%M%SZ')}",  # naive = UTC
        f"DTSTART;VALUE=DATE:{draft.due_date.strftime('%Y%m%d')}",
        f"SUMMARY:{_escape(draft.title)}",
        f"DESCRIPTION:{_escape(description)}",
        "END:VEVENT", "END:VCALENDAR",
    ]
    return b"".join(part + b"\r\n" for line in lines for part in _fold(line))


def ics_filename(draft: ReminderDraft) -> str:
    """``<reminder_id>-<8 hex of the job's correlation ID>.ics``: two jobs with identical content get different files."""
    if not _REMINDER_ID.fullmatch(draft.reminder_id):
        raise ReminderError("reminder_id is not in the expected form (rem_ + 16 hex digits)")
    return f"{draft.reminder_id}-{hashlib.sha256(draft.correlation_id.encode()).hexdigest()[:8]}.ics"


def export_ics(draft: ReminderDraft, directory: Path, stamp: datetime) -> tuple[Path, bytes]:
    """Write an *approved* reminder's ``.ics`` under ``directory`` and return ``(path, bytes)``. Only approved
    reminders are exported, so a draft or rejected one never produces a file. Refuses a symlinked directory or
    target, and a ``reminder_id`` that is not ``rem_`` + 16 hex digits (checked before any path is built)."""
    if draft.status != "approved":
        raise ReminderError(f"only an approved reminder can be exported (status is {draft.status})")
    name = ics_filename(draft)
    if directory.is_symlink():
        raise ReminderError("export directory is a symbolic link; refusing to write")
    content = build_ics(draft, stamp)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if path.is_symlink():
        raise ReminderError("export target is a symbolic link; refusing to write")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o644)
    with os.fdopen(fd, "wb") as fh:
        fh.write(content)
    return path, content
