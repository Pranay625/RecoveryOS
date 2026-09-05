"""
RecoveryOS - Phase 7: Deterministic Policy Engine

Receives Gemini's RecoveryRecommendation plus payment/customer context
and returns a deterministic PolicyDecision.

Architectural role:
    ML predicts.
    Gemini recommends.
    Policy Engine authorizes.
    Razorpay executes.  (next phase)

This module MUST NOT:
  - call Gemini or any LLM
  - call any external API
  - use probabilistic reasoning
  - use Gemini's confidence score as an authorization gate

Rules are evaluated in explicit, documented priority order.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.services.gemini_agent import RecoveryRecommendation

# ---------------------------------------------------------------------------
# Business-rule constants
# Defined here as named constants — never scattered as magic numbers.
# ---------------------------------------------------------------------------

MAX_RECOVERY_ATTEMPTS = 3   # total recovery attempts across all actions
MAX_PAYMENT_RETRIES   = 2   # PAYMENT_RETRY actions specifically
MAX_REMINDERS         = 2   # SEND_REMINDER actions specifically

# ---------------------------------------------------------------------------
# Policy models
# ---------------------------------------------------------------------------

PolicyAction = Literal["PAYMENT_RETRY", "SEND_REMINDER", "ESCALATE", "STOP"]


class PolicyDecision(BaseModel):
    action: PolicyAction
    allowed: bool
    reason: str


# ---------------------------------------------------------------------------
# Input context (plain dataclass-style — no DB objects cross this boundary)
# ---------------------------------------------------------------------------

class PolicyContext:
    """
    Carries the customer/history information the policy engine needs.
    Constructed by the router from the features dict and the Customer ORM
    object (if it exists).  No SQLAlchemy objects are passed into the engine.
    """

    def __init__(
        self,
        recovery_opt_out: bool,
        previous_recovery_attempts: int,
        previous_retry_count: int,
        previous_reminder_count: int,
    ) -> None:
        self.recovery_opt_out          = recovery_opt_out
        self.previous_recovery_attempts = previous_recovery_attempts
        self.previous_retry_count      = previous_retry_count
        self.previous_reminder_count   = previous_reminder_count


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

def evaluate(
    recommendation: RecoveryRecommendation,
    ctx: PolicyContext,
) -> PolicyDecision:
    """
    Apply deterministic business rules to Gemini's recommendation and
    return a PolicyDecision.

    Rules are evaluated in strict priority order:

        Rule 1  — Customer opt-out (highest priority)
        Rule 2  — STOP recommendation
        Rule 3  — ESCALATE recommendation
        Rule 4  — Total recovery attempt limit
        Rule 5  — PAYMENT_RETRY specific checks
        Rule 6  — SEND_REMINDER specific checks

    The engine never modifies XGBoost probabilities or Gemini's reasoning.
    It only determines whether the recommended action is permitted.
    """

    action = recommendation.action

    # ------------------------------------------------------------------
    # Rule 1 — Customer opt-out (overrides everything)
    # ------------------------------------------------------------------
    if ctx.recovery_opt_out:
        return PolicyDecision(
            action="STOP",
            allowed=False,
            reason=(
                "Recovery action blocked: this customer has opted out of "
                "all automated recovery actions."
            ),
        )

    # ------------------------------------------------------------------
    # Rule 2 — Gemini recommends STOP
    # ------------------------------------------------------------------
    if action == "STOP":
        return PolicyDecision(
            action="STOP",
            allowed=True,
            reason="No further automated recovery is recommended.",
        )

    # ------------------------------------------------------------------
    # Rule 3 — Gemini recommends ESCALATE
    # ------------------------------------------------------------------
    if action == "ESCALATE":
        return PolicyDecision(
            action="ESCALATE",
            allowed=True,
            reason="Escalation to human review is recommended.",
        )

    # ------------------------------------------------------------------
    # Rule 4 — Total recovery attempt limit
    # Applies before action-specific checks.
    # ------------------------------------------------------------------
    if ctx.previous_recovery_attempts >= MAX_RECOVERY_ATTEMPTS:
        return PolicyDecision(
            action="ESCALATE",
            allowed=False,
            reason=(
                f"Recovery action blocked: the maximum number of recovery "
                f"attempts ({MAX_RECOVERY_ATTEMPTS}) has already been reached "
                f"for this customer. Escalating to human review."
            ),
        )

    # ------------------------------------------------------------------
    # Rule 5 — PAYMENT_RETRY
    # ------------------------------------------------------------------
    if action == "PAYMENT_RETRY":
        if ctx.previous_retry_count >= MAX_PAYMENT_RETRIES:
            return PolicyDecision(
                action="ESCALATE",
                allowed=False,
                reason=(
                    f"Payment retry blocked: the maximum number of payment "
                    f"retries ({MAX_PAYMENT_RETRIES}) has already been reached "
                    f"for this customer. Escalating to human review."
                ),
            )
        return PolicyDecision(
            action="PAYMENT_RETRY",
            allowed=True,
            reason=(
                "Payment retry is permitted: the customer has not opted out "
                "and recovery limits have not been reached."
            ),
        )

    # ------------------------------------------------------------------
    # Rule 6 — SEND_REMINDER
    # ------------------------------------------------------------------
    if action == "SEND_REMINDER":
        if ctx.previous_reminder_count >= MAX_REMINDERS:
            return PolicyDecision(
                action="ESCALATE",
                allowed=False,
                reason=(
                    f"Reminder blocked: the maximum number of reminders "
                    f"({MAX_REMINDERS}) has already been sent to this customer. "
                    f"Escalating to human review."
                ),
            )
        return PolicyDecision(
            action="SEND_REMINDER",
            allowed=True,
            reason=(
                "Reminder is permitted: the customer has not opted out "
                "and reminder limits have not been reached."
            ),
        )

    # ------------------------------------------------------------------
    # Fallback — unknown action (should never reach here given Pydantic
    # validation on RecoveryRecommendation, but be explicit)
    # ------------------------------------------------------------------
    return PolicyDecision(
        action="ESCALATE",
        allowed=False,
        reason=f"Unknown recommended action '{action}'. Escalating for safety.",
    )
