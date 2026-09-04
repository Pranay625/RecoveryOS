"""
Phase 2 tests — database models and relationships.
Run from backend/ with:
    pytest tests/test_phase2_models.py -v
"""
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect, text

from app.db.session import engine, SessionLocal
from app.models import Customer, Payment, RecoveryAttempt, AuditLog


def uid():
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db():
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


# ---------------------------------------------------------------------------
# 1. MySQL connection
# ---------------------------------------------------------------------------

def test_db_connection():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1


# ---------------------------------------------------------------------------
# 2. All four tables exist
# ---------------------------------------------------------------------------

def test_all_tables_exist():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    for expected in ("customers", "payments", "recovery_attempts", "audit_logs"):
        assert expected in tables, f"Table '{expected}' not found"


# ---------------------------------------------------------------------------
# 3. Customer → Payment (one-to-many)
# ---------------------------------------------------------------------------

def test_customer_multiple_payments(db):
    customer = Customer(
        customer_id=uid(), name="Test User", email="test@example.com", contact="9999999999"
    )
    db.add(customer)
    db.flush()

    for _ in range(3):
        db.add(Payment(
            payment_id=uid(),
            customer_id=customer.customer_id,
            amount=500.00,
            currency="INR",
            payment_method="upi",
            status="success",
        ))
    db.flush()
    db.refresh(customer)

    assert len(customer.payments) == 3


# ---------------------------------------------------------------------------
# 4. Payment → RecoveryAttempt (one-to-many)
# ---------------------------------------------------------------------------

def test_payment_multiple_recovery_attempts(db):
    customer = Customer(
        customer_id=uid(), name="User2", email="u2@example.com", contact="8888888888"
    )
    db.add(customer)
    db.flush()

    payment = Payment(
        payment_id=uid(),
        customer_id=customer.customer_id,
        amount=1000.00,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="insufficient_funds",
    )
    db.add(payment)
    db.flush()

    for action in ("PAYMENT_RETRY", "SEND_REMINDER", "ESCALATE"):
        db.add(RecoveryAttempt(
            recovery_id=uid(),
            payment_id=payment.payment_id,
            action=action,
            reason="test reason",
            status="pending",
        ))
    db.flush()
    db.refresh(payment)

    assert len(payment.recovery_attempts) == 3


# ---------------------------------------------------------------------------
# 5. RecoveryAttempt → AuditLog
# ---------------------------------------------------------------------------

def test_recovery_attempt_audit_logs(db):
    customer = Customer(
        customer_id=uid(), name="User3", email="u3@example.com", contact="7777777777"
    )
    db.add(customer)
    db.flush()

    payment = Payment(
        payment_id=uid(),
        customer_id=customer.customer_id,
        amount=250.00,
        currency="INR",
        payment_method="netbanking",
        status="failed",
    )
    db.add(payment)
    db.flush()

    recovery = RecoveryAttempt(
        recovery_id=uid(),
        payment_id=payment.payment_id,
        action="PAYMENT_RETRY",
        reason="retry after failure",
        status="pending",
    )
    db.add(recovery)
    db.flush()

    db.add(AuditLog(
        audit_id=uid(),
        payment_id=payment.payment_id,
        recovery_id=recovery.recovery_id,
        event="RECOVERY_INITIATED",
    ))
    db.flush()
    db.refresh(recovery)

    assert len(recovery.audit_logs) == 1


# ---------------------------------------------------------------------------
# 6. AuditLog.recovery_id is nullable (payment-level event)
# ---------------------------------------------------------------------------

def test_audit_log_nullable_recovery_id(db):
    customer = Customer(
        customer_id=uid(), name="User4", email="u4@example.com", contact="6666666666"
    )
    db.add(customer)
    db.flush()

    payment = Payment(
        payment_id=uid(),
        customer_id=customer.customer_id,
        amount=100.00,
        currency="INR",
        payment_method="wallet",
        status="failed",
    )
    db.add(payment)
    db.flush()

    log = AuditLog(
        audit_id=uid(),
        payment_id=payment.payment_id,
        recovery_id=None,
        event="PAYMENT_FAILED",
    )
    db.add(log)
    db.flush()

    assert log.recovery_id is None


# ---------------------------------------------------------------------------
# 7. Payment.razorpay fields are nullable
# ---------------------------------------------------------------------------

def test_payment_nullable_razorpay_fields(db):
    customer = Customer(
        customer_id=uid(), name="User5", email="u5@example.com", contact="5555555555"
    )
    db.add(customer)
    db.flush()

    payment = Payment(
        payment_id=uid(),
        customer_id=customer.customer_id,
        amount=750.00,
        currency="INR",
        payment_method="upi",
        status="failed",
        razorpay_order_id=None,
        razorpay_payment_id=None,
    )
    db.add(payment)
    db.flush()

    assert payment.razorpay_order_id is None
    assert payment.razorpay_payment_id is None


# ---------------------------------------------------------------------------
# 8. FK violation — payment with nonexistent customer_id is rejected
# ---------------------------------------------------------------------------

def test_fk_violation_rejected(db):
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError):
        db.add(Payment(
            payment_id=uid(),
            customer_id="nonexistent_customer",
            amount=100.00,
            currency="INR",
            payment_method="upi",
            status="failed",
        ))
        db.flush()
    db.rollback()
