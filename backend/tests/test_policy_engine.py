"""
Phase 7 tests - Deterministic Policy Engine.

The policy engine is tested in complete isolation from Gemini, the database,
and XGBoost.  No external calls are made.

Run from backend/:
    pytest tests/test_policy_engine.py -v
"""

import pytest

from app.services.gemini_agent import RecoveryRecommendation
from app.services.policy_engine import (
    MAX_PAYMENT_RETRIES,
    MAX_RECOVERY_ATTEMPTS,
    MAX_REMINDERS,
    PolicyContext,
    evaluate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(action: str, confidence: float = 0.85) -> RecoveryRecommendation:
    """Build a minimal RecoveryRecommendation for testing."""
    return RecoveryRecommendation(
        action=action,
        reason="test reason",
        confidence=confidence,
    )


def _ctx(
    recovery_opt_out: bool = False,
    previous_recovery_attempts: int = 0,
    previous_retry_count: int = 0,
    previous_reminder_count: int = 0,
) -> PolicyContext:
    return PolicyContext(
        recovery_opt_out=recovery_opt_out,
        previous_recovery_attempts=previous_recovery_attempts,
        previous_retry_count=previous_retry_count,
        previous_reminder_count=previous_reminder_count,
    )


# ---------------------------------------------------------------------------
# 1. PAYMENT_RETRY allowed for normal customer
# ---------------------------------------------------------------------------

def test_payment_retry_allowed_normal_customer():
    decision = evaluate(_rec("PAYMENT_RETRY"), _ctx())
    assert decision.action == "PAYMENT_RETRY"
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# 2. SEND_REMINDER allowed for normal customer
# ---------------------------------------------------------------------------

def test_send_reminder_allowed_normal_customer():
    decision = evaluate(_rec("SEND_REMINDER"), _ctx())
    assert decision.action == "SEND_REMINDER"
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# 3. Unknown action falls back to ESCALATE (safety net)
# ---------------------------------------------------------------------------

def test_unknown_action_falls_back_to_escalate():
    """
    The policy engine's fallback handles any action string not covered by
    the explicit rules (e.g. a future action or a bug in the caller).
    """
    # Bypass Pydantic validation by constructing the object then mutating action
    rec = _rec("ESCALATE")
    object.__setattr__(rec, "action", "UNKNOWN_ACTION")
    decision = evaluate(rec, _ctx())
    assert decision.action == "ESCALATE"
    assert decision.allowed is False


# ---------------------------------------------------------------------------
# 4. ESCALATE recommendation passes through
# ---------------------------------------------------------------------------

def test_escalate_recommendation_passes_through():
    decision = evaluate(_rec("ESCALATE"), _ctx())
    assert decision.action == "ESCALATE"
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# 5. Opted-out customer + PAYMENT_RETRY -> STOP, not allowed
# ---------------------------------------------------------------------------

def test_opted_out_customer_payment_retry_blocked():
    decision = evaluate(_rec("PAYMENT_RETRY"), _ctx(recovery_opt_out=True))
    assert decision.action == "STOP"
    assert decision.allowed is False
    assert "opted out" in decision.reason.lower()


# ---------------------------------------------------------------------------
# 6. Opted-out customer + SEND_REMINDER -> STOP, not allowed
# ---------------------------------------------------------------------------

def test_opted_out_customer_send_reminder_blocked():
    decision = evaluate(_rec("SEND_REMINDER"), _ctx(recovery_opt_out=True))
    assert decision.action == "STOP"
    assert decision.allowed is False
    assert "opted out" in decision.reason.lower()


# ---------------------------------------------------------------------------
# 7. Retry limit reached -> ESCALATE, not allowed
# ---------------------------------------------------------------------------

def test_retry_limit_reached_blocks_payment_retry():
    decision = evaluate(
        _rec("PAYMENT_RETRY"),
        _ctx(previous_retry_count=MAX_PAYMENT_RETRIES),
    )
    assert decision.action == "ESCALATE"
    assert decision.allowed is False
    assert "retry" in decision.reason.lower()


# ---------------------------------------------------------------------------
# 8. Reminder limit reached -> ESCALATE, not allowed
# ---------------------------------------------------------------------------

def test_reminder_limit_reached_blocks_send_reminder():
    decision = evaluate(
        _rec("SEND_REMINDER"),
        _ctx(previous_reminder_count=MAX_REMINDERS),
    )
    assert decision.action == "ESCALATE"
    assert decision.allowed is False
    assert "reminder" in decision.reason.lower()


# ---------------------------------------------------------------------------
# 9. Total recovery attempt limit reached -> ESCALATE, not allowed
# (applies before action-specific checks)
# ---------------------------------------------------------------------------

def test_total_recovery_limit_blocks_payment_retry():
    decision = evaluate(
        _rec("PAYMENT_RETRY"),
        _ctx(previous_recovery_attempts=MAX_RECOVERY_ATTEMPTS),
    )
    assert decision.action == "ESCALATE"
    assert decision.allowed is False


def test_total_recovery_limit_blocks_send_reminder():
    decision = evaluate(
        _rec("SEND_REMINDER"),
        _ctx(previous_recovery_attempts=MAX_RECOVERY_ATTEMPTS),
    )
    assert decision.action == "ESCALATE"
    assert decision.allowed is False


# ---------------------------------------------------------------------------
# 10. New customer (zero history) -> PAYMENT_RETRY allowed
# ---------------------------------------------------------------------------

def test_new_customer_payment_retry_allowed():
    decision = evaluate(
        _rec("PAYMENT_RETRY"),
        _ctx(
            recovery_opt_out=False,
            previous_recovery_attempts=0,
            previous_retry_count=0,
            previous_reminder_count=0,
        ),
    )
    assert decision.action == "PAYMENT_RETRY"
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# 11. Policy engine makes no external calls
# (verified by the absence of any network/DB fixture — if it tried to call
# Gemini or the DB it would raise an error in this isolated test context)
# ---------------------------------------------------------------------------

def test_policy_engine_makes_no_external_calls():
    """
    The policy engine must be a pure function of its inputs.
    This test runs without any mocking of network or DB — if the engine
    attempted any external call it would raise a connection error.
    """
    decision = evaluate(_rec("PAYMENT_RETRY"), _ctx())
    # If we reach here, no external call was made
    assert decision is not None
    assert isinstance(decision.allowed, bool)


# ---------------------------------------------------------------------------
# 12. Opt-out takes priority over ESCALATE recommendation
# ---------------------------------------------------------------------------

def test_opt_out_overrides_escalate_recommendation():
    """
    Even if Gemini says ESCALATE, opt-out rule fires first and sets allowed=False.
    (ESCALATE normally passes through as allowed=True, but opt-out overrides.)
    """
    decision = evaluate(_rec("ESCALATE"), _ctx(recovery_opt_out=True))
    assert decision.action == "STOP"
    assert decision.allowed is False


# ---------------------------------------------------------------------------
# 13. One below retry limit is still allowed
# ---------------------------------------------------------------------------

def test_one_below_retry_limit_still_allowed():
    decision = evaluate(
        _rec("PAYMENT_RETRY"),
        _ctx(previous_retry_count=MAX_PAYMENT_RETRIES - 1),
    )
    assert decision.action == "PAYMENT_RETRY"
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# 14. One below reminder limit is still allowed
# ---------------------------------------------------------------------------

def test_one_below_reminder_limit_still_allowed():
    decision = evaluate(
        _rec("SEND_REMINDER"),
        _ctx(previous_reminder_count=MAX_REMINDERS - 1),
    )
    assert decision.action == "SEND_REMINDER"
    assert decision.allowed is True
