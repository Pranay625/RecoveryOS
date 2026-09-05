"""
Phase 10 regression tests - Runtime/historical data contamination guard.

Verifies that PAY_RT_* and REC_RT_* rows are never included in the
historical ML features computed by build_inference_features().

All tests use an in-memory SQLite database so no real MySQL connection
is required.

Run from backend/:
    pytest tests/test_inference_contamination.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.models.customer import Customer
from app.models.payment import Payment
from app.models.recovery_attempt import RecoveryAttempt
from app.ml.inference_features import build_inference_features

# ---------------------------------------------------------------------------
# In-memory SQLite engine + session factory
# ---------------------------------------------------------------------------

engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
)

# SQLite does not enforce FK constraints by default; enable them so the
# schema behaves closer to MySQL.
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(conn, _record):
    conn.execute("PRAGMA foreign_keys=ON")

Base.metadata.create_all(engine)
TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_DT = datetime(2024, 1, 1, 12, 0, 0)   # naive UTC baseline
_AS_OF    = datetime(2024, 6, 1, 12, 0, 0)   # inference cutoff (well after all fixtures)


def _dt(days_offset: int) -> datetime:
    return _BASE_DT + timedelta(days=days_offset)


def _customer(db: Session, cid: str) -> Customer:
    c = Customer(
        customer_id=cid,
        name="Test User",
        email="test@example.com",
        contact="9999999999",
        recovery_opt_out=False,
    )
    db.add(c)
    return c


def _payment(
    db: Session,
    payment_id: str,
    customer_id: str,
    status: str = "captured",
    amount: float = 1000.0,
    days_offset: int = 0,
) -> Payment:
    p = Payment(
        payment_id=payment_id,
        customer_id=customer_id,
        amount=amount,
        currency="INR",
        payment_method="upi",
        status=status,
        failure_reason=None,
        attempt_number=1,
        created_at=_dt(days_offset),
    )
    db.add(p)
    return p


def _recovery(
    db: Session,
    recovery_id: str,
    payment_id: str,
    action: str = "PAYMENT_RETRY",
    status: str = "completed",
    days_offset: int = 1,
) -> RecoveryAttempt:
    r = RecoveryAttempt(
        recovery_id=recovery_id,
        payment_id=payment_id,
        action=action,
        reason="test",
        ml_probability=0.8,
        status=status,
        created_at=_dt(days_offset),
    )
    db.add(r)
    return r


def _features(db: Session, customer_id: str) -> dict:
    return build_inference_features(
        db=db,
        customer_id=customer_id,
        amount=500.0,
        payment_method="upi",
        failure_reason="insufficient_funds",
        attempt_number=1,
        as_of=_AS_OF,
    )


# ---------------------------------------------------------------------------
# Test 1 — runtime payment excluded from total_previous_transactions
# ---------------------------------------------------------------------------

def test_runtime_payment_excluded_from_transaction_count():
    """PAY_RT_* must not appear in total_previous_transactions."""
    db = TestingSession()
    try:
        _customer(db, "CUST_T1")
        _payment(db, "PAY_001", "CUST_T1", status="captured", days_offset=10)
        _payment(db, "PAY_002", "CUST_T1", status="captured", days_offset=20)
        db.commit()

        features_before = _features(db, "CUST_T1")

        # Add a runtime payment
        _payment(db, "PAY_RT_001", "CUST_T1", status="pending", days_offset=30)
        db.commit()

        features_after = _features(db, "CUST_T1")

        assert features_before["total_previous_transactions"] == 2
        assert features_after["total_previous_transactions"] == 2, (
            "PAY_RT_* must not increment total_previous_transactions"
        )
        assert features_before == features_after
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 2 — runtime recovery attempt excluded from previous_recovery_attempts
# ---------------------------------------------------------------------------

def test_runtime_recovery_excluded_from_recovery_count():
    """REC_RT_* must not appear in previous_recovery_attempts."""
    db = TestingSession()
    try:
        _customer(db, "CUST_T2")
        _payment(db, "PAY_010", "CUST_T2", status="failed", days_offset=5)
        _recovery(db, "REC_010", "PAY_010", action="PAYMENT_RETRY",
                  status="completed", days_offset=6)
        db.commit()

        features_before = _features(db, "CUST_T2")

        # Add a runtime payment + runtime recovery
        _payment(db, "PAY_RT_010", "CUST_T2", status="pending", days_offset=40)
        _recovery(db, "REC_RT_010", "PAY_RT_010", action="PAYMENT_RETRY",
                  status="initiated", days_offset=40)
        db.commit()

        features_after = _features(db, "CUST_T2")

        assert features_before["previous_recovery_attempts"] == 1
        assert features_after["previous_recovery_attempts"] == 1, (
            "REC_RT_* must not increment previous_recovery_attempts"
        )
        assert features_before == features_after
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 3 — failed runtime payment excluded from success_rate
# ---------------------------------------------------------------------------

def test_failed_runtime_payment_excluded_from_success_rate():
    """A PAY_RT_* with status='failed' must not affect success_rate."""
    db = TestingSession()
    try:
        _customer(db, "CUST_T3")
        _payment(db, "PAY_020", "CUST_T3", status="captured", days_offset=5)
        _payment(db, "PAY_021", "CUST_T3", status="captured", days_offset=10)
        db.commit()

        features_before = _features(db, "CUST_T3")

        # Add a failed runtime payment — would dilute success_rate if included
        _payment(db, "PAY_RT_020", "CUST_T3", status="failed", days_offset=50)
        db.commit()

        features_after = _features(db, "CUST_T3")

        assert features_before["success_rate"] == 1.0
        assert features_after["success_rate"] == 1.0, (
            "Failed PAY_RT_* must not dilute success_rate"
        )
        assert features_before == features_after
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 4 — captured runtime payment excluded from total and success_rate
# ---------------------------------------------------------------------------

def test_captured_runtime_payment_excluded():
    """
    A PAY_RT_* with status='captured' must not affect
    total_previous_transactions or success_rate.
    Filtering only failed/pending would still be wrong.
    """
    db = TestingSession()
    try:
        _customer(db, "CUST_T4")
        _payment(db, "PAY_030", "CUST_T4", status="captured", amount=1000.0, days_offset=5)
        _payment(db, "PAY_031", "CUST_T4", status="failed",   amount=2000.0, days_offset=10)
        db.commit()

        features_before = _features(db, "CUST_T4")

        # Add a captured runtime payment — would inflate count and skew average
        _payment(db, "PAY_RT_030", "CUST_T4", status="captured",
                 amount=9999.0, days_offset=50)
        db.commit()

        features_after = _features(db, "CUST_T4")

        assert features_before["total_previous_transactions"] == 2
        assert features_after["total_previous_transactions"] == 2, (
            "Captured PAY_RT_* must not increment total_previous_transactions"
        )
        assert features_before["success_rate"] == 0.5
        assert features_after["success_rate"] == 0.5, (
            "Captured PAY_RT_* must not change success_rate"
        )
        assert features_before == features_after
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 5 — historical failed payment still counts
# ---------------------------------------------------------------------------

def test_historical_failed_payment_still_counts():
    """
    The PAY_RT_* filter must NOT accidentally remove legitimate historical
    PAY_* records with status='failed'. Failed historical payments must
    continue contributing to feature calculations.
    """
    db = TestingSession()
    try:
        _customer(db, "CUST_T5")
        _payment(db, "PAY_040", "CUST_T5", status="captured", amount=1000.0, days_offset=5)
        _payment(db, "PAY_041", "CUST_T5", status="failed",   amount=500.0,  days_offset=10)
        _payment(db, "PAY_042", "CUST_T5", status="failed",   amount=750.0,  days_offset=15)
        db.commit()

        features = _features(db, "CUST_T5")

        # All 3 historical payments must be counted
        assert features["total_previous_transactions"] == 3, (
            "Historical failed payments must be included in total_previous_transactions"
        )
        # 1 captured out of 3
        assert abs(features["success_rate"] - round(1 / 3, 6)) < 1e-6, (
            "Historical failed payments must be included in success_rate denominator"
        )
        # Average includes all 3 amounts: (1000 + 500 + 750) / 3
        expected_avg = round((1000.0 + 500.0 + 750.0) / 3, 2)
        assert features["average_transaction_amount"] == expected_avg, (
            "Historical failed payments must be included in average_transaction_amount"
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 6 — repeated demo execution stability (A == B == C)
# ---------------------------------------------------------------------------

def test_repeated_demo_execution_stability():
    """
    Simulates multiple demo executions for the same customer.
    Each execution creates PAY_RT_* + REC_RT_* rows.
    All three feature snapshots must be identical.
    """
    db = TestingSession()
    try:
        _customer(db, "CUST_T6")
        # Historical baseline: 3 payments, 2 recoveries
        _payment(db, "PAY_050", "CUST_T6", status="captured", amount=1200.0, days_offset=5)
        _payment(db, "PAY_051", "CUST_T6", status="failed",   amount=800.0,  days_offset=15)
        _payment(db, "PAY_052", "CUST_T6", status="captured", amount=1500.0, days_offset=25)
        _recovery(db, "REC_050", "PAY_050", action="PAYMENT_RETRY",
                  status="completed", days_offset=6)
        _recovery(db, "REC_051", "PAY_051", action="SEND_REMINDER",
                  status="completed", days_offset=16)
        db.commit()

        # Snapshot A — before any demo execution
        features_a = _features(db, "CUST_T6")

        # First demo execution
        _payment(db, "PAY_RT_050", "CUST_T6", status="pending", days_offset=60)
        _recovery(db, "REC_RT_050", "PAY_RT_050", action="PAYMENT_RETRY",
                  status="initiated", days_offset=60)
        db.commit()

        # Snapshot B — after first demo execution
        features_b = _features(db, "CUST_T6")

        # Second demo execution
        _payment(db, "PAY_RT_051", "CUST_T6", status="captured", days_offset=70)
        _recovery(db, "REC_RT_051", "PAY_RT_051", action="PAYMENT_RETRY",
                  status="completed", days_offset=70)
        db.commit()

        # Snapshot C — after second demo execution
        features_c = _features(db, "CUST_T6")

        # All historical features must be identical across all three snapshots
        historical_keys = [
            "total_previous_transactions",
            "success_rate",
            "average_transaction_amount",
            "days_since_last_success",
            "previous_recovery_attempts",
            "recovery_success_rate",
            "previous_retry_count",
            "previous_reminder_count",
            "transactions_last_30_days",
            "successful_transactions_last_30_days",
        ]
        for key in historical_keys:
            assert features_a[key] == features_b[key] == features_c[key], (
                f"Feature '{key}' changed across demo executions: "
                f"A={features_a[key]}, B={features_b[key]}, C={features_c[key]}"
            )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 7 — /predict XGBoost input unchanged after runtime execution
# ---------------------------------------------------------------------------

def test_predict_xgboost_input_stable_after_execute():
    """
    Verifies that build_inference_features() returns the same historical
    feature dict before and after runtime rows are added, simulating
    what /predict would feed to XGBoost.
    """
    db = TestingSession()
    try:
        _customer(db, "CUST_T7")
        _payment(db, "PAY_060", "CUST_T7", status="captured", amount=2000.0, days_offset=10)
        _payment(db, "PAY_061", "CUST_T7", status="failed",   amount=1500.0, days_offset=20)
        _recovery(db, "REC_060", "PAY_060", action="SEND_REMINDER",
                  status="completed", days_offset=11)
        db.commit()

        features_before_execute = _features(db, "CUST_T7")

        # Simulate /execute creating runtime rows
        _payment(db, "PAY_RT_060", "CUST_T7", status="pending", days_offset=80)
        _recovery(db, "REC_RT_060", "PAY_RT_060", action="PAYMENT_RETRY",
                  status="initiated", days_offset=80)
        db.commit()

        features_after_execute = _features(db, "CUST_T7")

        # The XGBoost input (all historical features) must be identical
        assert features_before_execute == features_after_execute, (
            "XGBoost feature input changed after /execute created runtime rows"
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 8 — runtime retry/reminder counts excluded from policy context features
# ---------------------------------------------------------------------------

def test_runtime_recovery_counts_excluded_from_retry_and_reminder_counts():
    """
    previous_retry_count and previous_reminder_count must not include
    REC_RT_* rows, since those are runtime state, not historical behavior.
    """
    db = TestingSession()
    try:
        _customer(db, "CUST_T8")
        _payment(db, "PAY_070", "CUST_T8", status="failed", days_offset=5)
        _recovery(db, "REC_070", "PAY_070", action="PAYMENT_RETRY",
                  status="completed", days_offset=6)
        db.commit()

        features_before = _features(db, "CUST_T8")

        # Add runtime rows with both action types
        _payment(db, "PAY_RT_070", "CUST_T8", status="pending", days_offset=40)
        _recovery(db, "REC_RT_070", "PAY_RT_070", action="PAYMENT_RETRY",
                  status="initiated", days_offset=40)
        _payment(db, "PAY_RT_071", "CUST_T8", status="pending", days_offset=41)
        _recovery(db, "REC_RT_071", "PAY_RT_071", action="SEND_REMINDER",
                  status="initiated", days_offset=41)
        db.commit()

        features_after = _features(db, "CUST_T8")

        assert features_before["previous_retry_count"] == 1
        assert features_after["previous_retry_count"] == 1, (
            "REC_RT_* PAYMENT_RETRY must not increment previous_retry_count"
        )
        assert features_before["previous_reminder_count"] == 0
        assert features_after["previous_reminder_count"] == 0, (
            "REC_RT_* SEND_REMINDER must not increment previous_reminder_count"
        )
    finally:
        db.close()
