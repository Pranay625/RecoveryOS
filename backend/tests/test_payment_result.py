"""
Phase 9 tests - Payment result handling.

Tests for:
  - RazorpayService.verify_payment_signature()
  - POST /api/recovery/payment-result
  - POST /api/recovery/payment-failed

All external dependencies are mocked. No real network calls or DB connections.

Run from backend/:
    pytest tests/test_payment_result.py -v
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app
from app.services.razorpay_service import RazorpayService, RazorpayServiceError

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

VALID_RESULT_REQUEST = {
    "razorpay_payment_id": "pay_TEST_abc123",
    "razorpay_order_id":   "order_TEST_xyz789",
    "razorpay_signature":  "valid_sig_abc",
}

VALID_FAILED_REQUEST = {
    "razorpay_order_id":   "order_TEST_xyz789",
    "error_code":          "BAD_REQUEST_ERROR",
    "error_description":   "Payment failed due to insufficient funds",
}


def _make_runtime_payment(
    payment_id="PAY_RT_AAAA1111",
    order_id="order_TEST_xyz789",
    status="pending",
    razorpay_payment_id=None,
):
    p = MagicMock()
    p.payment_id          = payment_id
    p.razorpay_order_id   = order_id
    p.razorpay_payment_id = razorpay_payment_id
    p.status              = status
    p.customer_id         = "CUST_000001"
    return p


def _make_runtime_recovery(
    recovery_id="REC_RT_BBBB2222",
    payment_id="PAY_RT_AAAA1111",
    status="initiated",
):
    r = MagicMock()
    r.recovery_id = recovery_id
    r.payment_id  = payment_id
    r.status      = status
    r.completed_at = None
    return r


def _make_mock_db(payment=None, recovery=None):
    mock_db = MagicMock()

    # Payment query: filter by razorpay_order_id
    mock_db.query.return_value.filter.return_value.first.return_value = payment

    # Recovery query: filter + order_by + first
    mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = recovery

    return mock_db


def _override_db(mock_db):
    def _get():
        yield mock_db
    return _get


# ===========================================================================
# SECTION 1 — RazorpayService.verify_payment_signature()
# ===========================================================================

class TestVerifyPaymentSignature:

    # 1. Valid signature is accepted (SDK raises no exception)
    def test_valid_signature_accepted(self):
        svc = RazorpayService()
        mock_client = MagicMock()
        mock_client.utility.verify_payment_signature.return_value = None
        svc._client = mock_client

        # Should not raise
        svc.verify_payment_signature(
            razorpay_order_id="order_abc",
            razorpay_payment_id="pay_abc",
            razorpay_signature="sig_abc",
        )
        mock_client.utility.verify_payment_signature.assert_called_once_with({
            "razorpay_order_id":   "order_abc",
            "razorpay_payment_id": "pay_abc",
            "razorpay_signature":  "sig_abc",
        })

    # 2. Invalid signature raises RazorpayServiceError
    def test_invalid_signature_raises_service_error(self):
        svc = RazorpayService()
        mock_client = MagicMock()
        mock_client.utility.verify_payment_signature.side_effect = Exception(
            "SignatureVerificationError"
        )
        svc._client = mock_client

        with pytest.raises(RazorpayServiceError, match="verification failed"):
            svc.verify_payment_signature("order_x", "pay_x", "bad_sig")

    # 3. Error message does NOT contain the secret
    def test_error_does_not_expose_secret(self):
        secret = "rzp_secret_should_not_appear"
        svc = RazorpayService()
        mock_client = MagicMock()
        mock_client.utility.verify_payment_signature.side_effect = Exception(
            f"auth={secret}"
        )
        svc._client = mock_client

        try:
            svc.verify_payment_signature("o", "p", "s")
        except RazorpayServiceError as exc:
            assert secret not in str(exc)


# ===========================================================================
# SECTION 2 — POST /api/recovery/payment-result
# ===========================================================================

class TestPaymentResult:

    # 4. Valid signed payment updates runtime Payment status to "captured"
    def test_valid_payment_updates_payment_status(self):
        payment  = _make_runtime_payment()
        recovery = _make_runtime_recovery()
        mock_db  = _make_mock_db(payment, recovery)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            with patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:
                mock_rz = MagicMock()
                mock_rz.verify_payment_signature.return_value = None
                mock_get_rz.return_value = mock_rz

                response = client.post("/api/recovery/payment-result",
                                       json=VALID_RESULT_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 200
        assert payment.status == "captured"
        assert payment.razorpay_payment_id == "pay_TEST_abc123"

    # 5. Valid signed payment updates runtime RecoveryAttempt to "completed"
    def test_valid_payment_updates_recovery_status(self):
        payment  = _make_runtime_payment()
        recovery = _make_runtime_recovery()
        mock_db  = _make_mock_db(payment, recovery)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            with patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:
                mock_get_rz.return_value.verify_payment_signature.return_value = None

                client.post("/api/recovery/payment-result", json=VALID_RESULT_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert recovery.status == "completed"
        assert recovery.completed_at is not None

    # 6. AuditLog is created on success
    def test_audit_log_created_on_success(self):
        payment  = _make_runtime_payment()
        recovery = _make_runtime_recovery()
        mock_db  = _make_mock_db(payment, recovery)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            with patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:
                mock_get_rz.return_value.verify_payment_signature.return_value = None

                client.post("/api/recovery/payment-result", json=VALID_RESULT_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        mock_db.add.assert_called_once()
        added = mock_db.add.call_args[0][0]
        assert added.event == "PAYMENT_SUCCESS"
        assert "pay_TEST_abc123" in (added.policy_result or "")

    # 7. Correct Razorpay IDs are stored in the response
    def test_correct_ids_in_response(self):
        payment  = _make_runtime_payment()
        recovery = _make_runtime_recovery()
        mock_db  = _make_mock_db(payment, recovery)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            with patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:
                mock_get_rz.return_value.verify_payment_signature.return_value = None

                response = client.post("/api/recovery/payment-result",
                                       json=VALID_RESULT_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["razorpay_payment_id"] == "pay_TEST_abc123"
        assert data["razorpay_order_id"]   == "order_TEST_xyz789"
        assert data["runtime_payment_id"]  == "PAY_RT_AAAA1111"
        assert data["runtime_recovery_id"] == "REC_RT_BBBB2222"

    # 8. Missing runtime order returns 404
    def test_missing_order_returns_404(self):
        mock_db = _make_mock_db(payment=None)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            with patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:
                mock_get_rz.return_value.verify_payment_signature.return_value = None

                response = client.post("/api/recovery/payment-result",
                                       json=VALID_RESULT_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 404

    # 9. Historical PAY_* payment cannot be modified
    def test_historical_payment_cannot_be_modified(self):
        historical_payment = _make_runtime_payment(payment_id="PAY_000001")
        mock_db = _make_mock_db(historical_payment)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            with patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:
                mock_get_rz.return_value.verify_payment_signature.return_value = None

                response = client.post("/api/recovery/payment-result",
                                       json=VALID_RESULT_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 400
        assert "historical" in response.json()["detail"].lower()
        # Status must not have been mutated
        assert historical_payment.status == "pending"

    # 10. Missing runtime RecoveryAttempt returns 404
    def test_missing_recovery_attempt_returns_404(self):
        payment  = _make_runtime_payment()
        mock_db  = _make_mock_db(payment, recovery=None)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            with patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:
                mock_get_rz.return_value.verify_payment_signature.return_value = None

                response = client.post("/api/recovery/payment-result",
                                       json=VALID_RESULT_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 404

    # 11. Duplicate successful callback is safe (idempotent)
    def test_duplicate_success_callback_is_idempotent(self):
        # Payment already captured from a previous callback
        payment  = _make_runtime_payment(
            status="captured",
            razorpay_payment_id="pay_TEST_abc123",
        )
        recovery = _make_runtime_recovery(status="completed")
        mock_db  = _make_mock_db(payment, recovery)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            with patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:
                mock_get_rz.return_value.verify_payment_signature.return_value = None

                response = client.post("/api/recovery/payment-result",
                                       json=VALID_RESULT_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 200
        assert response.json()["status"] == "SUCCESS"
        # No new DB writes on duplicate
        mock_db.add.assert_not_called()
        mock_db.commit.assert_not_called()

    # Invalid signature returns 400, no DB mutation
    def test_invalid_signature_returns_400_no_db_write(self):
        payment  = _make_runtime_payment()
        recovery = _make_runtime_recovery()
        mock_db  = _make_mock_db(payment, recovery)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            with patch("app.routers.recovery._get_razorpay_service") as mock_get_rz:
                mock_get_rz.return_value.verify_payment_signature.side_effect = (
                    RazorpayServiceError("Payment signature verification failed: SignatureVerificationError")
                )

                response = client.post("/api/recovery/payment-result",
                                       json=VALID_RESULT_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 400
        mock_db.add.assert_not_called()
        mock_db.commit.assert_not_called()
        assert payment.status == "pending"


# ===========================================================================
# SECTION 3 — POST /api/recovery/payment-failed
# ===========================================================================

class TestPaymentFailed:

    # 12. Failure updates runtime Payment to "failed"
    def test_failure_updates_payment_status(self):
        payment  = _make_runtime_payment()
        recovery = _make_runtime_recovery()
        mock_db  = _make_mock_db(payment, recovery)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            response = client.post("/api/recovery/payment-failed",
                                   json=VALID_FAILED_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 200
        assert payment.status == "failed"

    # 13. Failure updates runtime RecoveryAttempt to "failed"
    def test_failure_updates_recovery_status(self):
        payment  = _make_runtime_payment()
        recovery = _make_runtime_recovery()
        mock_db  = _make_mock_db(payment, recovery)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            client.post("/api/recovery/payment-failed", json=VALID_FAILED_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert recovery.status == "failed"
        assert recovery.completed_at is not None

    # 14. Failure AuditLog is created
    def test_failure_audit_log_created(self):
        payment  = _make_runtime_payment()
        recovery = _make_runtime_recovery()
        mock_db  = _make_mock_db(payment, recovery)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            client.post("/api/recovery/payment-failed", json=VALID_FAILED_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        mock_db.add.assert_called_once()
        added = mock_db.add.call_args[0][0]
        assert added.event == "PAYMENT_FAILED"
        assert "order_TEST_xyz789" in (added.policy_result or "")

    # 15. Missing runtime order returns 404
    def test_missing_order_returns_404(self):
        mock_db = _make_mock_db(payment=None)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            response = client.post("/api/recovery/payment-failed",
                                   json=VALID_FAILED_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 404

    # 16. Historical PAY_* payment cannot be modified
    def test_historical_payment_cannot_be_modified(self):
        historical_payment = _make_runtime_payment(payment_id="PAY_000001")
        mock_db = _make_mock_db(historical_payment)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            response = client.post("/api/recovery/payment-failed",
                                   json=VALID_FAILED_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 400
        assert "historical" in response.json()["detail"].lower()
        assert historical_payment.status == "pending"

    # 17. Duplicate failure callback is safe (idempotent)
    def test_duplicate_failure_callback_is_idempotent(self):
        payment  = _make_runtime_payment(status="failed")
        recovery = _make_runtime_recovery(status="failed")
        mock_db  = _make_mock_db(payment, recovery)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            response = client.post("/api/recovery/payment-failed",
                                   json=VALID_FAILED_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 200
        assert response.json()["status"] == "FAILED"
        # No new DB writes on duplicate
        mock_db.add.assert_not_called()
        mock_db.commit.assert_not_called()

    # 18. Failure endpoint does NOT invoke ML, Gemini, or policy engine
    def test_failure_does_not_invoke_ml_gemini_policy(self):
        payment  = _make_runtime_payment()
        recovery = _make_runtime_recovery()
        mock_db  = _make_mock_db(payment, recovery)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            with patch("app.routers.recovery.build_inference_features") as mock_fe, \
                 patch("app.routers.recovery.predict_recovery_probabilities") as mock_xgb, \
                 patch("app.routers.recovery._get_agent") as mock_gemini, \
                 patch("app.routers.recovery.policy_evaluate") as mock_policy:

                client.post("/api/recovery/payment-failed", json=VALID_FAILED_REQUEST)

                mock_fe.assert_not_called()
                mock_xgb.assert_not_called()
                mock_gemini.assert_not_called()
                mock_policy.assert_not_called()
        finally:
            app.dependency_overrides.pop(get_db, None)

    # Failure response shape is correct
    def test_failure_response_shape(self):
        payment  = _make_runtime_payment()
        recovery = _make_runtime_recovery()
        mock_db  = _make_mock_db(payment, recovery)

        app.dependency_overrides[get_db] = _override_db(mock_db)
        try:
            response = client.post("/api/recovery/payment-failed",
                                   json=VALID_FAILED_REQUEST)
        finally:
            app.dependency_overrides.pop(get_db, None)

        data = response.json()
        assert data["status"]               == "FAILED"
        assert data["razorpay_order_id"]    == "order_TEST_xyz789"
        assert data["runtime_payment_id"]   == "PAY_RT_AAAA1111"
        assert data["runtime_recovery_id"]  == "REC_RT_BBBB2222"
