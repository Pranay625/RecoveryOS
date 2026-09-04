"""
RecoveryOS — Phase 4A: Feature Engineering

Builds a Pandas training DataFrame from the MySQL database.
One row per historical recovery attempt.
All features are computed using ONLY information available
BEFORE that recovery attempt's created_at timestamp.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import pandas as pd

from app.db.session import SessionLocal
from app.models.payment import Payment
from app.models.recovery_attempt import RecoveryAttempt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_training_dataframe() -> pd.DataFrame:
    """
    Load payments and recovery attempts from MySQL, then build one
    training row per recovery attempt using only prior information.

    Returns a DataFrame with columns:
        amount, payment_method, failure_reason, attempt_number,
        total_previous_transactions, success_rate,
        average_transaction_amount, days_since_last_success,
        previous_recovery_attempts, recovery_success_rate,
        previous_retry_count, previous_reminder_count,
        transactions_last_30_days, successful_transactions_last_30_days,
        action, target
    """
    db = SessionLocal()
    try:
        payments, recoveries = _load_data(db)
    finally:
        db.close()

    return _build_rows(payments, recoveries)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(db):
    """Load all payments and recovery attempts, sorted chronologically."""
    payments = (
        db.query(Payment)
        .order_by(Payment.created_at)
        .all()
    )
    recoveries = (
        db.query(RecoveryAttempt)
        .order_by(RecoveryAttempt.created_at)
        .all()
    )
    return payments, recoveries


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _build_rows(payments: list, recoveries: list) -> pd.DataFrame:
    """
    Process recovery attempts in chronological order.
    For each attempt, compute features from the state of the world
    that existed strictly before that attempt's created_at.
    """

    # ------------------------------------------------------------------
    # Build a lookup: payment_id → Payment object
    # ------------------------------------------------------------------
    payment_by_id: dict[str, Payment] = {p.payment_id: p for p in payments}

    # ------------------------------------------------------------------
    # Build a lookup: payment_id → customer_id
    # ------------------------------------------------------------------
    customer_of: dict[str, str] = {
        p.payment_id: p.customer_id for p in payments
    }

    # ------------------------------------------------------------------
    # Per-customer ordered payment history (sorted by created_at).
    # We'll use this to answer "what payments existed before time T?"
    # ------------------------------------------------------------------
    customer_payments: dict[str, list[Payment]] = defaultdict(list)
    for p in payments:                          # already sorted
        customer_payments[p.customer_id].append(p)

    # ------------------------------------------------------------------
    # Per-customer ordered recovery history (sorted by created_at).
    # We'll use this to answer "what recoveries existed before time T?"
    # ------------------------------------------------------------------
    customer_recoveries: dict[str, list[RecoveryAttempt]] = defaultdict(list)
    for r in recoveries:                        # already sorted
        cid = customer_of.get(r.payment_id)
        if cid:
            customer_recoveries[cid].append(r)

    # ------------------------------------------------------------------
    # Build one row per recovery attempt
    # ------------------------------------------------------------------
    rows: list[dict] = []

    for rec in recoveries:
        payment = payment_by_id.get(rec.payment_id)
        if payment is None:
            continue                            # orphaned record — skip

        customer_id = payment.customer_id
        cutoff = rec.created_at                 # strict "before this moment"

        # ---- current payment features --------------------------------
        amount = float(payment.amount)
        payment_method = payment.payment_method
        failure_reason = payment.failure_reason or "none"
        attempt_number = payment.attempt_number

        # ---- customer payment history (before cutoff) ----------------
        prior_payments = [
            p for p in customer_payments[customer_id]
            if p.created_at < cutoff and p.payment_id != payment.payment_id
        ]

        total_previous_transactions = len(prior_payments)

        if prior_payments:
            captured = [p for p in prior_payments if p.status == "captured"]
            success_rate = len(captured) / total_previous_transactions
            average_transaction_amount = (
                sum(float(p.amount) for p in prior_payments)
                / total_previous_transactions
            )
            if captured:
                last_success_dt = max(p.created_at for p in captured)
                days_since_last_success = (cutoff - last_success_dt).days
            else:
                days_since_last_success = -1
        else:
            success_rate = 0.0
            average_transaction_amount = 0.0
            days_since_last_success = -1

        # ---- recent behavior (30-day window before cutoff) -----------
        window_start = cutoff - timedelta(days=30)
        recent = [
            p for p in prior_payments
            if p.created_at >= window_start
        ]
        transactions_last_30_days = len(recent)
        successful_transactions_last_30_days = sum(
            1 for p in recent if p.status == "captured"
        )

        # ---- previous recovery history (before cutoff) ---------------
        prior_recoveries = [
            r for r in customer_recoveries[customer_id]
            if r.created_at < cutoff and r.recovery_id != rec.recovery_id
        ]

        previous_recovery_attempts = len(prior_recoveries)

        if prior_recoveries:
            completed = sum(
                1 for r in prior_recoveries if r.status == "completed"
            )
            recovery_success_rate = completed / previous_recovery_attempts
        else:
            recovery_success_rate = 0.0

        previous_retry_count = sum(
            1 for r in prior_recoveries if r.action == "PAYMENT_RETRY"
        )
        previous_reminder_count = sum(
            1 for r in prior_recoveries if r.action == "SEND_REMINDER"
        )

        # ---- label columns -------------------------------------------
        action = rec.action
        target = 1 if rec.status == "completed" else 0

        rows.append({
            "amount": amount,
            "payment_method": payment_method,
            "failure_reason": failure_reason,
            "attempt_number": attempt_number,
            "total_previous_transactions": total_previous_transactions,
            "success_rate": round(success_rate, 6),
            "average_transaction_amount": round(average_transaction_amount, 2),
            "days_since_last_success": days_since_last_success,
            "previous_recovery_attempts": previous_recovery_attempts,
            "recovery_success_rate": round(recovery_success_rate, 6),
            "previous_retry_count": previous_retry_count,
            "previous_reminder_count": previous_reminder_count,
            "transactions_last_30_days": transactions_last_30_days,
            "successful_transactions_last_30_days": successful_transactions_last_30_days,
            "action": action,
            "target": target,
        })

    return pd.DataFrame(rows)


def build_training_dataframe_with_customers() -> tuple[pd.DataFrame, list[str]]:
    """
    Same as build_training_dataframe() but also returns a parallel list of
    customer_id values (one per row) so the training script can perform a
    group-aware train/validation/test split without leaking customer_id
    into the feature matrix.

    Returns:
        df           — the standard 16-column training DataFrame
        customer_ids — list[str] of length len(df), same row order
    """
    db = SessionLocal()
    try:
        payments, recoveries = _load_data(db)
    finally:
        db.close()

    payment_by_id: dict[str, Payment] = {p.payment_id: p for p in payments}
    customer_of: dict[str, str] = {p.payment_id: p.customer_id for p in payments}

    customer_payments: dict[str, list[Payment]] = defaultdict(list)
    for p in payments:
        customer_payments[p.customer_id].append(p)

    customer_recoveries: dict[str, list[RecoveryAttempt]] = defaultdict(list)
    for r in recoveries:
        cid = customer_of.get(r.payment_id)
        if cid:
            customer_recoveries[cid].append(r)

    rows: list[dict] = []
    customer_ids: list[str] = []

    for rec in recoveries:
        payment = payment_by_id.get(rec.payment_id)
        if payment is None:
            continue

        customer_id = payment.customer_id
        cutoff = rec.created_at

        amount = float(payment.amount)
        payment_method = payment.payment_method
        failure_reason = payment.failure_reason or "none"
        attempt_number = payment.attempt_number

        prior_payments = [
            p for p in customer_payments[customer_id]
            if p.created_at < cutoff and p.payment_id != payment.payment_id
        ]
        total_previous_transactions = len(prior_payments)

        if prior_payments:
            captured = [p for p in prior_payments if p.status == "captured"]
            success_rate = len(captured) / total_previous_transactions
            average_transaction_amount = (
                sum(float(p.amount) for p in prior_payments) / total_previous_transactions
            )
            if captured:
                last_success_dt = max(p.created_at for p in captured)
                days_since_last_success = (cutoff - last_success_dt).days
            else:
                days_since_last_success = -1
        else:
            success_rate = 0.0
            average_transaction_amount = 0.0
            days_since_last_success = -1

        window_start = cutoff - timedelta(days=30)
        recent = [p for p in prior_payments if p.created_at >= window_start]
        transactions_last_30_days = len(recent)
        successful_transactions_last_30_days = sum(
            1 for p in recent if p.status == "captured"
        )

        prior_recoveries = [
            r for r in customer_recoveries[customer_id]
            if r.created_at < cutoff and r.recovery_id != rec.recovery_id
        ]
        previous_recovery_attempts = len(prior_recoveries)

        if prior_recoveries:
            completed = sum(1 for r in prior_recoveries if r.status == "completed")
            recovery_success_rate = completed / previous_recovery_attempts
        else:
            recovery_success_rate = 0.0

        previous_retry_count = sum(
            1 for r in prior_recoveries if r.action == "PAYMENT_RETRY"
        )
        previous_reminder_count = sum(
            1 for r in prior_recoveries if r.action == "SEND_REMINDER"
        )

        rows.append({
            "amount": amount,
            "payment_method": payment_method,
            "failure_reason": failure_reason,
            "attempt_number": attempt_number,
            "total_previous_transactions": total_previous_transactions,
            "success_rate": round(success_rate, 6),
            "average_transaction_amount": round(average_transaction_amount, 2),
            "days_since_last_success": days_since_last_success,
            "previous_recovery_attempts": previous_recovery_attempts,
            "recovery_success_rate": round(recovery_success_rate, 6),
            "previous_retry_count": previous_retry_count,
            "previous_reminder_count": previous_reminder_count,
            "transactions_last_30_days": transactions_last_30_days,
            "successful_transactions_last_30_days": successful_transactions_last_30_days,
            "action": rec.action,
            "target": 1 if rec.status == "completed" else 0,
        })
        customer_ids.append(customer_id)

    return pd.DataFrame(rows), customer_ids
