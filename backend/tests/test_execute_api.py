"""
Phase 8 tests - POST /api/recovery/execute

All external dependencies (DB, XGBoost, Gemini, Razorpay) are mocked.
No real network calls or database connections are required.

Run from backend/:
    pytest tests/test_execute_api.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app
from app.services.razorpay_service import RazorpayOrder, RazorpayServiceError

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

VALID_REQUEST = {
    "customer_id": "CUST_000001",
    "amount": 2499.0,
    "currency": "INR",
    "payment_method": "upi",
    "failure_reason": "insufficient_funds",
    "attempt_number": 1,
}

MOCK_FEATURES = {
    "amount": 2499.0,
    "payment_method": "upi",
    "failure_reason": "insufficient_funds",
    "attempt_number": 1,
    "total_previous_transactions": 15,
    "success_rate": 0.8,
    "average_transaction_amount": 2800.0,
    "days_since_last_success": 7,
    "previous_recovery_attempts": 1,
    "recovery_success_rate": 0.5,
    "previous_retry_count": 0,
    "previous_reminder_count": 0,
    "transactions_last_30_days": 4,
    "successful_transactions_last_30_days": 3,
}

MOCK_ML_PREDICTIONS = {"PAYMENT_RETRY": 0.8944, "SEND_REMINDER": 0.7278}

FAKE_RZ_ORDER = RazorpayOrder(
    id="order_TEST_abc123",
    amount=249900,
    currency="INR",
    receipt="rcvros_test",
    status="created",
)


def _mock_gemini(action: str, confidence: float = 0.85):
    return MagicMock(action=action, confidence=confidence, reason="test reason")


def _make_mock_db(customer=None):
    """Return a mock DB session with customer lookup pre-configured."""
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = customer
    return mock_db


def _override_db(mock_db):
    """Return a FastAPI dependency override that yields mock_db."""
    def _get_mock_db():
        yield mock_db
    return _get_mock_db


# ---------------------------------------------------------------------------
# 1. Allowed PAYMENT_RETRY — Razorpay order is created, runtime rows written
# ---------------------------------------------------------------------------

def test_payment_retry_allowed_creates_razorpay_order():
    mock_customer = MagicMock()
    mock_customer.recovery_opt_out = False
    mock_db = _make_mock_db(mock_customer)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        with patch("app.routers.recovery.build_inference_features",
                   return_value=MOCK_FEATURES), \
             patch("app.routers.recovery.predict_recovery_probabilities",
                   return_value=MOCK_ML_PREDICTIONS), \
             patch("app.routers.recovery._get_agent") as mock_get_agent, \
             patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:

            mock_get_agent.return_value.recommend.return_value = _mock_gemini("PAYMENT_RETRY")
            mock_rz_svc = MagicMock()
            mock_rz_svc.create_order.return_value = FAKE_RZ_ORDER
            mock_get_rz.return_value = mock_rz_svc

            response = client.post("/api/recovery/execute", json=VALID_REQUEST)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    data = response.json()

    assert data["action"] == "PAYMENT_RETRY"
    assert data["allowed"] is True
    assert data["execution_status"] == "READY_FOR_PAYMENT"
    assert data["razorpay_order_id"] == "order_TEST_abc123"
    assert data["amount_paise"] == 249900
    assert data["currency"] == "INR"
    assert data["runtime_payment_id"] is not None
    assert data["runtime_recovery_id"] is not None

    mock_rz_svc.create_order.assert_called_once()
    call_kwargs = mock_rz_svc.create_order.call_args.kwargs
    assert call_kwargs["amount_inr"] == 2499.0
    assert call_kwargs["currency"] == "INR"

    mock_db.commit.assert_called_once()


# ---------------------------------------------------------------------------
# 2. Blocked PAYMENT_RETRY — Razorpay is NOT called, no DB writes
# ---------------------------------------------------------------------------

def test_blocked_payment_retry_does_not_call_razorpay():
    mock_customer = MagicMock()
    mock_customer.recovery_opt_out = False
    mock_db = _make_mock_db(mock_customer)

    features_at_limit = {**MOCK_FEATURES, "previous_retry_count": 2,
                         "previous_recovery_attempts": 2}

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        with patch("app.routers.recovery.build_inference_features",
                   return_value=features_at_limit), \
             patch("app.routers.recovery.predict_recovery_probabilities",
                   return_value=MOCK_ML_PREDICTIONS), \
             patch("app.routers.recovery._get_agent") as mock_get_agent, \
             patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:

            mock_get_agent.return_value.recommend.return_value = _mock_gemini("PAYMENT_RETRY")

            response = client.post("/api/recovery/execute", json=VALID_REQUEST)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    data = response.json()

    assert data["allowed"] is False
    assert data["execution_status"] == "BLOCKED"
    assert data["razorpay_order_id"] is None

    mock_get_rz.assert_not_called()
    mock_db.commit.assert_not_called()


# ---------------------------------------------------------------------------
# 3. SEND_REMINDER — Razorpay is NOT called
# ---------------------------------------------------------------------------

def test_send_reminder_does_not_call_razorpay():
    mock_customer = MagicMock()
    mock_customer.recovery_opt_out = False
    mock_db = _make_mock_db(mock_customer)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        with patch("app.routers.recovery.build_inference_features",
                   return_value=MOCK_FEATURES), \
             patch("app.routers.recovery.predict_recovery_probabilities",
                   return_value=MOCK_ML_PREDICTIONS), \
             patch("app.routers.recovery._get_agent") as mock_get_agent, \
             patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:

            mock_get_agent.return_value.recommend.return_value = _mock_gemini("SEND_REMINDER")

            response = client.post("/api/recovery/execute", json=VALID_REQUEST)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    data = response.json()

    assert data["action"] == "SEND_REMINDER"
    assert data["allowed"] is True
    assert data["execution_status"] == "REMINDER_REQUIRED"
    assert data["razorpay_order_id"] is None
    assert data["amount_paise"] is None

    mock_get_rz.assert_not_called()
    mock_db.commit.assert_called_once()


# ---------------------------------------------------------------------------
# 4. ESCALATE — Razorpay is NOT called
# ---------------------------------------------------------------------------

def test_escalate_does_not_call_razorpay():
    mock_customer = MagicMock()
    mock_customer.recovery_opt_out = False
    mock_db = _make_mock_db(mock_customer)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        with patch("app.routers.recovery.build_inference_features",
                   return_value=MOCK_FEATURES), \
             patch("app.routers.recovery.predict_recovery_probabilities",
                   return_value={"PAYMENT_RETRY": 0.42, "SEND_REMINDER": 0.38}), \
             patch("app.routers.recovery._get_agent") as mock_get_agent, \
             patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:

            mock_get_agent.return_value.recommend.return_value = _mock_gemini("ESCALATE")

            response = client.post("/api/recovery/execute", json=VALID_REQUEST)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    data = response.json()

    assert data["action"] == "ESCALATE"
    assert data["allowed"] is True
    assert data["execution_status"] == "MANUAL_REVIEW"
    assert data["razorpay_order_id"] is None

    mock_get_rz.assert_not_called()
    mock_db.commit.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Opted-out customer — execution is blocked
# ---------------------------------------------------------------------------

def test_opted_out_customer_is_blocked():
    mock_customer = MagicMock()
    mock_customer.recovery_opt_out = True
    mock_db = _make_mock_db(mock_customer)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        with patch("app.routers.recovery.build_inference_features",
                   return_value=MOCK_FEATURES), \
             patch("app.routers.recovery.predict_recovery_probabilities",
                   return_value=MOCK_ML_PREDICTIONS), \
             patch("app.routers.recovery._get_agent") as mock_get_agent, \
             patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:

            mock_get_agent.return_value.recommend.return_value = _mock_gemini("PAYMENT_RETRY")

            response = client.post("/api/recovery/execute", json=VALID_REQUEST)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    data = response.json()

    assert data["allowed"] is False
    assert data["execution_status"] == "BLOCKED"
    assert "opted out" in data["policy_reason"].lower()

    mock_get_rz.assert_not_called()
    mock_db.commit.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Retry limit prevents execution
# ---------------------------------------------------------------------------

def test_retry_limit_prevents_execution():
    mock_customer = MagicMock()
    mock_customer.recovery_opt_out = False
    mock_db = _make_mock_db(mock_customer)

    features_at_limit = {**MOCK_FEATURES, "previous_retry_count": 2,
                         "previous_recovery_attempts": 2}

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        with patch("app.routers.recovery.build_inference_features",
                   return_value=features_at_limit), \
             patch("app.routers.recovery.predict_recovery_probabilities",
                   return_value=MOCK_ML_PREDICTIONS), \
             patch("app.routers.recovery._get_agent") as mock_get_agent, \
             patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:

            mock_get_agent.return_value.recommend.return_value = _mock_gemini("PAYMENT_RETRY")

            response = client.post("/api/recovery/execute", json=VALID_REQUEST)
    finally:
        app.dependency_overrides.pop(get_db, None)

    data = response.json()
    assert data["allowed"] is False
    assert data["execution_status"] == "BLOCKED"
    mock_get_rz.assert_not_called()
    mock_db.commit.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Reminder limit prevents execution
# ---------------------------------------------------------------------------

def test_reminder_limit_prevents_execution():
    mock_customer = MagicMock()
    mock_customer.recovery_opt_out = False
    mock_db = _make_mock_db(mock_customer)

    features_at_limit = {**MOCK_FEATURES, "previous_reminder_count": 2,
                         "previous_recovery_attempts": 2}

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        with patch("app.routers.recovery.build_inference_features",
                   return_value=features_at_limit), \
             patch("app.routers.recovery.predict_recovery_probabilities",
                   return_value=MOCK_ML_PREDICTIONS), \
             patch("app.routers.recovery._get_agent") as mock_get_agent, \
             patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:

            mock_get_agent.return_value.recommend.return_value = _mock_gemini("SEND_REMINDER")

            response = client.post("/api/recovery/execute", json=VALID_REQUEST)
    finally:
        app.dependency_overrides.pop(get_db, None)

    data = response.json()
    assert data["allowed"] is False
    assert data["execution_status"] == "BLOCKED"
    mock_get_rz.assert_not_called()
    mock_db.commit.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Frontend cannot bypass policy — extra fields are silently ignored
# ---------------------------------------------------------------------------

def test_frontend_cannot_submit_allowed_true():
    """
    The execute request schema has no allowed/action fields.
    Extra fields submitted by the frontend are silently ignored by Pydantic.
    The backend re-runs the full pipeline independently.
    """
    tampered_request = {
        **VALID_REQUEST,
        "allowed": True,
        "action": "PAYMENT_RETRY",
        "policy_decision": {"action": "PAYMENT_RETRY", "allowed": True, "reason": "hacked"},
    }

    mock_customer = MagicMock()
    mock_customer.recovery_opt_out = False
    mock_db = _make_mock_db(mock_customer)

    features_at_limit = {**MOCK_FEATURES, "previous_retry_count": 2,
                         "previous_recovery_attempts": 2}

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        with patch("app.routers.recovery.build_inference_features",
                   return_value=features_at_limit), \
             patch("app.routers.recovery.predict_recovery_probabilities",
                   return_value=MOCK_ML_PREDICTIONS), \
             patch("app.routers.recovery._get_agent") as mock_get_agent, \
             patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:

            mock_get_agent.return_value.recommend.return_value = _mock_gemini("PAYMENT_RETRY")

            response = client.post("/api/recovery/execute", json=tampered_request)
    finally:
        app.dependency_overrides.pop(get_db, None)

    # Policy still blocks because retry limit is reached — extra fields ignored
    data = response.json()
    assert data["allowed"] is False
    assert data["execution_status"] == "BLOCKED"
    mock_get_rz.assert_not_called()


# ---------------------------------------------------------------------------
# 9. Razorpay failure returns 502, no DB commit
# ---------------------------------------------------------------------------

def test_razorpay_failure_returns_502_no_db_commit():
    mock_customer = MagicMock()
    mock_customer.recovery_opt_out = False
    mock_db = _make_mock_db(mock_customer)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        with patch("app.routers.recovery.build_inference_features",
                   return_value=MOCK_FEATURES), \
             patch("app.routers.recovery.predict_recovery_probabilities",
                   return_value=MOCK_ML_PREDICTIONS), \
             patch("app.routers.recovery._get_agent") as mock_get_agent, \
             patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:

            mock_get_agent.return_value.recommend.return_value = _mock_gemini("PAYMENT_RETRY")
            mock_rz_svc = MagicMock()
            mock_rz_svc.create_order.side_effect = RazorpayServiceError(
                "order creation failed: BadRequestError"
            )
            mock_get_rz.return_value = mock_rz_svc

            response = client.post("/api/recovery/execute", json=VALID_REQUEST)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 502
    mock_db.commit.assert_not_called()
    assert "KEY_SECRET" not in response.text
    assert "rzp_" not in response.text


# ---------------------------------------------------------------------------
# 10. Runtime execution state is recorded correctly
# ---------------------------------------------------------------------------

def test_runtime_execution_state_recorded():
    mock_customer = MagicMock()
    mock_customer.recovery_opt_out = False
    mock_db = _make_mock_db(mock_customer)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        with patch("app.routers.recovery.build_inference_features",
                   return_value=MOCK_FEATURES), \
             patch("app.routers.recovery.predict_recovery_probabilities",
                   return_value=MOCK_ML_PREDICTIONS), \
             patch("app.routers.recovery._get_agent") as mock_get_agent, \
             patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:

            mock_get_agent.return_value.recommend.return_value = _mock_gemini("PAYMENT_RETRY")
            mock_rz_svc = MagicMock()
            mock_rz_svc.create_order.return_value = FAKE_RZ_ORDER
            mock_get_rz.return_value = mock_rz_svc

            response = client.post("/api/recovery/execute", json=VALID_REQUEST)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    data = response.json()

    assert data["runtime_payment_id"].startswith("PAY_RT_")
    assert data["runtime_recovery_id"].startswith("REC_RT_")

    # Three db.add() calls: Payment, RecoveryAttempt, AuditLog
    assert mock_db.add.call_count == 3
    mock_db.commit.assert_called_once()

    # Inspect the Payment object that was added
    added_payment = mock_db.add.call_args_list[0][0][0]
    assert added_payment.status == "pending"
    assert added_payment.razorpay_order_id == "order_TEST_abc123"
    assert added_payment.customer_id == "CUST_000001"

    # Inspect the RecoveryAttempt object
    added_recovery = mock_db.add.call_args_list[1][0][0]
    assert added_recovery.status == "initiated"
    assert added_recovery.action == "PAYMENT_RETRY"
    assert added_recovery.payment_id == data["runtime_payment_id"]


# ---------------------------------------------------------------------------
# 11. Synthetic historical records are NOT modified
# ---------------------------------------------------------------------------

def test_synthetic_historical_records_not_modified():
    """
    /execute must only add NEW rows via db.add().
    It must never call db.query(...).update() on existing rows.
    """
    mock_customer = MagicMock()
    mock_customer.recovery_opt_out = False
    mock_db = _make_mock_db(mock_customer)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        with patch("app.routers.recovery.build_inference_features",
                   return_value=MOCK_FEATURES), \
             patch("app.routers.recovery.predict_recovery_probabilities",
                   return_value=MOCK_ML_PREDICTIONS), \
             patch("app.routers.recovery._get_agent") as mock_get_agent, \
             patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:

            mock_get_agent.return_value.recommend.return_value = _mock_gemini("PAYMENT_RETRY")
            mock_rz_svc = MagicMock()
            mock_rz_svc.create_order.return_value = FAKE_RZ_ORDER
            mock_get_rz.return_value = mock_rz_svc

            client.post("/api/recovery/execute", json=VALID_REQUEST)
    finally:
        app.dependency_overrides.pop(get_db, None)

    # No update() call should appear anywhere in the mock's call history
    for c in mock_db.method_calls:
        assert "update" not in str(c), f"Unexpected update call: {c}"


# ---------------------------------------------------------------------------
# 12. Existing /predict behavior is unchanged — read-only, no DB writes
# ---------------------------------------------------------------------------

def test_predict_endpoint_unchanged():
    mock_customer = MagicMock()
    mock_customer.recovery_opt_out = False
    mock_db = _make_mock_db(mock_customer)

    app.dependency_overrides[get_db] = _override_db(mock_db)
    try:
        with patch("app.routers.recovery.build_inference_features",
                   return_value=MOCK_FEATURES), \
             patch("app.routers.recovery.predict_recovery_probabilities",
                   return_value=MOCK_ML_PREDICTIONS), \
             patch("app.routers.recovery._get_agent") as mock_get_agent:

            mock_get_agent.return_value.recommend.return_value = _mock_gemini("PAYMENT_RETRY")

            response = client.post("/api/recovery/predict", json=VALID_REQUEST)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    data = response.json()

    assert "customer_id" in data
    assert "customer_exists" in data
    assert "ml_predictions" in data
    assert "recommended_action" in data
    assert "confidence" in data
    assert "reason" in data
    assert "policy" in data
    assert "action" in data["policy"]
    assert "allowed" in data["policy"]

    # /predict must NEVER write to the DB
    mock_db.add.assert_not_called()
    mock_db.commit.assert_not_called()
