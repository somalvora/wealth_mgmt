"""Deny-by-default authorization (design §7, "Enforceable Authorization Policy").

``authorize(actor, account, action, purpose, at)`` is called by tools and storage, never
by agent prompts. Owners act on their own accounts; everyone else needs consent from every
owner of the account, scoped to the exact action and purpose and valid at that moment.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime

from household_agent.models import Account, Action, Role

POLICY_VERSION = "authz_1.0"

DELEGATE_ACTIONS = frozenset({Action.READ, Action.REPORT, Action.REPORT_AGGREGATE})


class AuthorizationError(PermissionError):
    def __init__(self, decision: "Decision"):
        super().__init__(decision.reason)
        self.decision = decision


@dataclass(frozen=True)
class Actor:
    member_id: str
    household_id: str
    role: Role


@dataclass(frozen=True)
class ConsentGrant:
    grant_id: str
    household_id: str
    grantor_id: str
    grantee_id: str
    account_ids: frozenset[str]
    actions: frozenset[Action]
    purposes: frozenset[str]
    effective_at: datetime
    expires_at: datetime | None
    version: str
    revoked_at: datetime | None = None

    def active_at(self, at: datetime) -> bool:
        if at < self.effective_at:
            return False
        if self.expires_at is not None and at >= self.expires_at:
            return False
        return self.revoked_at is None or at < self.revoked_at


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    policy_version: str = POLICY_VERSION
    consent_version: str | None = None

    def __bool__(self) -> bool:
        return self.allowed


class Authorizer:
    def __init__(self, accounts: Iterable[Account], grants: Iterable[ConsentGrant] = ()):
        self._accounts = {a.account_id: a for a in accounts}
        self._grants = {g.grant_id: g for g in grants}

    def account(self, account_id: str) -> Account | None:
        return self._accounts.get(account_id)

    def add_grant(self, grant: ConsentGrant) -> None:
        self._grants[grant.grant_id] = grant

    def revoke(self, grant_id: str, at: datetime) -> None:
        self._grants[grant_id] = replace(self._grants[grant_id], revoked_at=at)

    def authorize(
        self, actor: Actor, account_id: str, action: Action, purpose: str, at: datetime
    ) -> Decision:
        account = self._accounts.get(account_id)
        if account is None:
            return Decision(False, f"unknown account {account_id}")
        if actor.household_id != account.household_id:
            return Decision(False, "actor is not in the account's household")
        if actor.role is Role.DEPENDENT:
            return Decision(False, "dependents have no ledger access")
        if actor.role is not Role.DELEGATE and actor.member_id in account.owner_member_ids:
            return Decision(True, "account owner")
        if actor.role is Role.DELEGATE and action not in DELEGATE_ACTIONS:
            return Decision(False, f"delegates may not {action}")

        consenting = {}
        for g in self._grants.values():
            if (
                g.grantee_id == actor.member_id
                and g.household_id == account.household_id
                and g.grantor_id in account.owner_member_ids
                and account_id in g.account_ids
                and action in g.actions
                and purpose in g.purposes
                and g.active_at(at)
                and (actor.role is not Role.DELEGATE or g.expires_at is not None)
            ):
                consenting[g.grantor_id] = g.version
        missing = account.owner_member_ids - consenting.keys()
        if missing:
            return Decision(False, f"no active consent for {action}/{purpose} from {sorted(missing)}")
        version = ",".join(f"{k}:{v}" for k, v in sorted(consenting.items()))
        return Decision(True, "consent", consent_version=version)
