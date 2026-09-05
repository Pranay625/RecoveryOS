"""
RecoveryOS - Phase 6: Live Inference Feature Engineering

Computes the same 14 ML features used during XGBoost training, but from
a live database query rather than a full historical scan.

Semantics are kept strictly identical to app/ml/feature_engineering.py:
  - payments.status == "captured"  means a successful payment
  - recovery_attempts.status == "completed"  means a successful recovery
  - failure_reason defaults to "none" when NULL
  - days_since_last_success == -1 when no prior successful payment exists
  - all historical features use a strict cutoff: only records with
    created_at < cutoff are included (no future leakage)

Cutoff for live inference:
  The caller passes `as_of` (defaults to datetime.now(UTC)).
  This represents "the moment the recovery decision is being made."
  All historical queries use created_at < as_of.
  The current payment being evaluated is NOT yet in the database, so
  there is no need to exclude it by payment_id.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.models.customer import Customer
from app.models.payment import Payment
from app.models.recovery_attempt import RecoveryAttempt

# ---------------------------------------------------------------------------
# Zero-history defaults (new customer / cold start)
# Must match the conventions used in training feature engineering.
# ---------------------------------------------------------------------------

ZERO_HISTORY: dict = {
    "total_previous_transactions": 0,
    "success_rate": 0.0,
    "average_transaction_amount": 0.0,
    "days_since_last_success": -1,
    "previous_recovery_attempts": 0,
    "recovery_success_rate": 0.0,
    "previous_retry_count": 0,
    "previous_reminder_count": 0,
    "transactions_last_30_days": 0,
    "successful_transactions_last_30_days": 0,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_inference_features(
    db: Session,
    customer_id: str,
    amount: float,
    payment_method: str,
    failure_reason: str | None,
    attempt_number: int,
    as_of: datetime | None = None,
) -> dict:
    """
    Return the 14 non-action features expected by predict_recovery_probabilities().

    Args:
        db              - active SQLAlchemy session
        customer_id     - customer to look up; if not found, zero-history is used
        amount          - current payment amount
        payment_method  - current payment method
        failure_reason  - current failure reason (None becomes "none")
        attempt_number  - current attempt number
        as_of           - inference cutoff timestamp; defaults to now(UTC).
                          Only records with created_at < as_of are included.
                          The current payment is not yet in the DB, so no
                          payment_id exclusion is needed.

    Returns:
        dict with exactly 14 keys matching FEATURE_COLUMNS in the training pipeline.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)

    # Normalise failure_reason to match training convention
    normalised_failure_reason = failure_reason or "none"

    # Current payment features — always from the request
    current_features = {
        "amount": float(amount),
        "payment_method": payment_method,
        "failure_reason": normalised_failure_reason,
        "attempt_number": attempt_number,
    }

    # Check whether the customer exists
    customer = db.query(Customer).filter(
        Customer.customer_id == customer_id
    ).first()

    if customer is None:
        # New customer — use zero-history defaults
        return {**current_features, **ZERO_HISTORY}

    # ------------------------------------------------------------------
    # Existing customer — query historical records before cutoff
    # ------------------------------------------------------------------
    prior_payments: list[Payment] = (
        db.query(Payment)
        .filter(
            and_(
                Payment.customer_id == customer_id,
                Payment.created_at < as_of,
            )
        )
        .order_by(Payment.created_at)
        .all()
    )

    total_previous_transactions = len(prior_payments)

    if prior_payments:
        captured = [p for p in prior_payments if p.status == "captured"]
        success_rate = round(len(captured) / total_previous_transactions, 6)
        average_transaction_amount = round(
            sum(float(p.amount) for p in prior_payments) / total_previous_transactions,
            2,
        )
        if captured:
            last_success_dt = max(p.created_at for p in captured)
            # Ensure both datetimes are comparable (both naive or both aware)
            last_success_dt = _normalise_dt(last_success_dt)
            as_of_normalised = _normalise_dt(as_of)
            days_since_last_success = (as_of_normalised - last_success_dt).days
        else:
            days_since_last_success = -1
    else:
        success_rate = 0.0
        average_transaction_amount = 0.0
        days_since_last_success = -1

    # 30-day window
    window_start = as_of - timedelta(days=30)
    recent = [
        p for p in prior_payments
        if _normalise_dt(p.created_at) >= _normalise_dt(window_start)
    ]
    transactions_last_30_days = len(recent)
    successful_transactions_last_30_days = sum(
        1 for p in recent if p.status == "captured"
    )

    # ------------------------------------------------------------------
    # Previous recovery attempts for this customer (via their payments)
    # ------------------------------------------------------------------
    # Collect all payment_ids belonging to this customer
    customer_payment_ids = [p.payment_id for p in prior_payments]

    if customer_payment_ids:
        prior_recoveries: list[RecoveryAttempt] = (
            db.query(RecoveryAttempt)
            .filter(
                and_(
                    RecoveryAttempt.payment_id.in_(customer_payment_ids),
                    RecoveryAttempt.created_at < as_of,
                )
            )
            .order_by(RecoveryAttempt.created_at)
            .all()
        )
    else:
        prior_recoveries = []

    previous_recovery_attempts = len(prior_recoveries)

    if prior_recoveries:
        completed = sum(1 for r in prior_recoveries if r.status == "completed")
        recovery_success_rate = round(completed / previous_recovery_attempts, 6)
    else:
        recovery_success_rate = 0.0

    previous_retry_count = sum(
        1 for r in prior_recoveries if r.action == "PAYMENT_RETRY"
    )
    previous_reminder_count = sum(
        1 for r in prior_recoveries if r.action == "SEND_REMINDER"
    )

    return {
        **current_features,
        "total_previous_transactions": total_previous_transactions,
        "success_rate": success_rate,
        "average_transaction_amount": average_transaction_amount,
        "days_since_last_success": days_since_last_success,
        "previous_recovery_attempts": previous_recovery_attempts,
        "recovery_success_rate": recovery_success_rate,
        "previous_retry_count": previous_retry_count,
        "previous_reminder_count": previous_reminder_count,
        "transactions_last_30_days": transactions_last_30_days,
        "successful_transactions_last_30_days": successful_transactions_last_30_days,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_dt(dt: datetime) -> datetime:
    """
    Ensure datetime is timezone-aware (UTC).
    MySQL returns naive datetimes; as_of may be tz-aware.
    Normalise both to naive UTC for safe subtraction.
    """
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None) - dt.utcoffset()
    return dt
