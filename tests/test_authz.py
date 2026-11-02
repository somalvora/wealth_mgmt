from datetime import UTC, datetime, timedelta

import pytest

from household_agent.authz import Actor, Authorizer, ConsentGrant
from household_agent.models import Action, Role
from household_agent.synthetic import ACCOUNTS, ALEX, CASEY_CPA, HOUSEHOLD, JORDAN, SAM, cpa_grant

NOW = datetime(2026, 11, 1, tzinfo=UTC)


@pytest.fixture
def authz():
    return Authorizer(ACCOUNTS)


def test_owner_can_read_own_private_account(authz):
    assert authz.authorize(ALEX, "acct_alex_checking", Action.READ, "budgeting", NOW)


def test_other_adult_cannot_read_private_account(authz):
    d = authz.authorize(JORDAN, "acct_alex_checking", Action.READ, "budgeting", NOW)
    assert not d and "alex" in d.reason


def test_admin_role_does_not_override_private_consent(authz):
    # Alex is household admin but not an owner of Jordan's card.
    assert not authz.authorize(ALEX, "acct_jordan_card", Action.READ, "budgeting", NOW)


def test_joint_owners_can_read_joint_account(authz):
    assert authz.authorize(JORDAN, "acct_rental_checking", Action.READ, "budgeting", NOW)
    assert authz.authorize(ALEX, "acct_rental_checking", Action.READ, "budgeting", NOW)


def test_unknown_account_and_other_household_denied(authz):
    assert not authz.authorize(ALEX, "acct_missing", Action.READ, "budgeting", NOW)
    outsider = Actor("alex", "hh_other", Role.ADULT_MEMBER)
    assert not authz.authorize(outsider, "acct_alex_checking", Action.READ, "budgeting", NOW)


def test_dependent_denied(authz):
    assert not authz.authorize(SAM, "acct_rental_checking", Action.READ, "allowance", NOW)


def test_joint_account_delegate_needs_consent_from_every_owner(authz):
    authz.add_grant(cpa_grant(NOW + timedelta(days=30)))  # only Alex consented
    d = authz.authorize(CASEY_CPA, "acct_rental_checking", Action.READ, "tax_preparation", NOW)
    assert not d and "jordan" in d.reason

    authz.add_grant(ConsentGrant(
        grant_id="grant_cpa_rental_jordan", household_id=HOUSEHOLD, grantor_id="jordan", grantee_id="casey_cpa",
        account_ids=frozenset({"acct_rental_checking"}), actions=frozenset({Action.READ}),
        purposes=frozenset({"tax_preparation"}), effective_at=NOW - timedelta(days=1),
        expires_at=NOW + timedelta(days=30), version="cv_2",
    ))
    d = authz.authorize(CASEY_CPA, "acct_rental_checking", Action.READ, "tax_preparation", NOW)
    assert d and d.consent_version == "alex:cv_1,jordan:cv_2"


def _both_owner_grants(authz, expires):
    authz.add_grant(cpa_grant(expires))
    authz.add_grant(ConsentGrant(
        grant_id="g_j", household_id=HOUSEHOLD, grantor_id="jordan", grantee_id="casey_cpa",
        account_ids=frozenset({"acct_rental_checking"}), actions=frozenset({Action.READ}),
        purposes=frozenset({"tax_preparation"}), effective_at=NOW - timedelta(days=1),
        expires_at=expires, version="cv_2",
    ))


def test_consent_is_scoped_to_action_and_purpose(authz):
    _both_owner_grants(authz, NOW + timedelta(days=30))
    assert not authz.authorize(CASEY_CPA, "acct_rental_checking", Action.READ, "marketing", NOW)
    assert not authz.authorize(CASEY_CPA, "acct_rental_checking", Action.EXPORT, "tax_preparation", NOW)


def test_delegate_can_never_write(authz):
    _both_owner_grants(authz, NOW + timedelta(days=30))
    assert not authz.authorize(CASEY_CPA, "acct_rental_checking", Action.WRITE, "tax_preparation", NOW)


def test_expired_and_revoked_consent_denied(authz):
    _both_owner_grants(authz, NOW + timedelta(days=1))
    assert authz.authorize(CASEY_CPA, "acct_rental_checking", Action.READ, "tax_preparation", NOW)
    assert not authz.authorize(CASEY_CPA, "acct_rental_checking", Action.READ, "tax_preparation", NOW + timedelta(days=2))

    authz.revoke("grant_cpa_rental", NOW + timedelta(hours=1))
    assert not authz.authorize(CASEY_CPA, "acct_rental_checking", Action.READ, "tax_preparation", NOW + timedelta(hours=2))


def test_delegate_grant_without_expiry_is_ignored(authz):
    _both_owner_grants(authz, None)
    assert not authz.authorize(CASEY_CPA, "acct_rental_checking", Action.READ, "tax_preparation", NOW)
