"""
Phase 6 tests - Recovery Prediction API.

All external dependencies (DB, XGBoost, Gemini) are mocked.
No real API calls or database connections are required.

Run from backend/:
    pytest tests/test_recovery_api.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.policy_engine import PolicyDecision

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

MOCK_FEATURES_EXISTING = {
    "amount": 2499.0,
    "payment_method": "upi",
    "failure_reason": "insufficient_funds",
    "attempt_number": 1,
    "total_previous_transactions": 15,
    "success_rate": 0.8,
    "average_transaction_amount": 2800.0,
    "days_since_last_success": 7,
    "previous_recovery_attempts": 2,
    "recovery_success_rate": 0.5,
    "previous_retry_count": 1,
    "previous_reminder_count": 1,
    "transactions_last_30_days": 4,
    "successful_transactions_last_30_days": 3,
}

MOCK_FEATURES_NEW = {
    "amount": 2499.0,
    "payment_method": "upi",
    "failure_reason": "insufficient_funds",
    "attempt_number": 1,
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

MOCK_ML_PREDICTIONS = {"PAYMENT_RETRY": 0.8944, "SEND_REMINDER": 0.7278}

MOCK_GEMINI_RECOMMENDATION = MagicMock(
    action="PAYMENT_RETRY",
    confidence=0.89,
    reason="Strong retry probability based on customer history.",
)

MOCK_POLICY_DECISION = PolicyDecision(
    action="PAYMENT_RETRY",
    allowed=True,
    reason="Payment retry is permitted.",
)


def _mock_db_session():
    """Return a mock DB session that yields a MagicMock."""
    mock_session = MagicMock()
    return mock_session


# ---------------------------------------------------------------------------
# 1. Existing customer - full pipeline runs correctly
# ---------------------------------------------------------------------------

def test_existing_customer_full_pipeline():
    mock_customer = MagicMock()
    mock_customer.customer_id = "CUST_000001"

    with patch("app.routers.recovery.build_inference_features",
               return_value=MOCK_FEATURES_EXISTING) as mock_features, \
         patch("app.routers.recovery.predict_recovery_probabilities",
               return_value=MOCK_ML_PREDICTIONS) as mock_xgb, \
         patch("app.routers.recovery._get_agent") as mock_get_agent, \
         patch("app.routers.recovery.get_db") as mock_get_db:

        # DB returns an existing customer
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_customer
        mock_get_db.return_value = iter([mock_db])

        mock_agent = MagicMock()
        mock_agent.recommend.return_value = MOCK_GEMINI_RECOMMENDATION
        mock_get_agent.return_value = mock_agent

        response = client.post("/api/recovery/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == "CUST_000001"
    assert data["customer_exists"] is True
    assert data["ml_predictions"]["PAYMENT_RETRY"] == 0.8944
    assert data["ml_predictions"]["SEND_REMINDER"] == 0.7278
    assert data["recommended_action"] == "PAYMENT_RETRY"
    assert data["confidence"] == 0.89
    assert data["reason"]
    # Policy field is present in response
    assert "policy" in data
    assert "action" in data["policy"]
    assert "allowed" in data["policy"]
    assert "reason" in data["policy"]

    # Verify XGBoost was called with the features from inference_features
    mock_xgb.assert_called_once_with(MOCK_FEATURES_EXISTING)

    # Verify Gemini received the actual XGBoost probabilities
    gemini_call_ctx = mock_agent.recommend.call_args[0][0]
    assert gemini_call_ctx.ml_predictions.PAYMENT_RETRY == 0.8944
    assert gemini_call_ctx.ml_predictions.SEND_REMINDER == 0.7278


# ---------------------------------------------------------------------------
# 2. New customer - zero-history features, pipeline still runs
# ---------------------------------------------------------------------------

def test_new_customer_zero_history():
    new_customer_request = {**VALID_REQUEST, "customer_id": "brand_new_customer"}

    with patch("app.routers.recovery.build_inference_features",
               return_value=MOCK_FEATURES_NEW) as mock_features, \
         patch("app.routers.recovery.predict_recovery_probabilities",
               return_value=MOCK_ML_PREDICTIONS) as mock_xgb, \
         patch("app.routers.recovery._get_agent") as mock_get_agent, \
         patch("app.routers.recovery.get_db") as mock_get_db:

        # DB returns no customer
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_get_db.return_value = iter([mock_db])

        mock_agent = MagicMock()
        mock_agent.recommend.return_value = MOCK_GEMINI_RECOMMENDATION
        mock_get_agent.return_value = mock_agent

        response = client.post("/api/recovery/predict", json=new_customer_request)

    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == "brand_new_customer"
    assert data["customer_exists"] is False
    assert "policy" in data

    # XGBoost still ran
    mock_xgb.assert_called_once()

    # Gemini still received predictions
    mock_agent.recommend.assert_called_once()

    # Zero-history features were used
    gemini_call_ctx = mock_agent.recommend.call_args[0][0]
    assert gemini_call_ctx.customer_history.total_previous_transactions == 0
    assert gemini_call_ctx.customer_history.success_rate == 0.0
    assert gemini_call_ctx.customer_history.days_since_last_success == -1


# ---------------------------------------------------------------------------
# 3. Gemini receives exact XGBoost probabilities (not hardcoded)
# ---------------------------------------------------------------------------

def test_gemini_receives_exact_xgboost_probabilities():
    distinct_predictions = {"PAYMENT_RETRY": 0.7531, "SEND_REMINDER": 0.4219}

    with patch("app.routers.recovery.build_inference_features",
               return_value=MOCK_FEATURES_EXISTING), \
         patch("app.routers.recovery.predict_recovery_probabilities",
               return_value=distinct_predictions), \
         patch("app.routers.recovery._get_agent") as mock_get_agent, \
         patch("app.routers.recovery.get_db") as mock_get_db:

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = MagicMock()
        mock_get_db.return_value = iter([mock_db])

        mock_agent = MagicMock()
        mock_agent.recommend.return_value = MOCK_GEMINI_RECOMMENDATION
        mock_get_agent.return_value = mock_agent

        client.post("/api/recovery/predict", json=VALID_REQUEST)

    ctx = mock_agent.recommend.call_args[0][0]
    assert ctx.ml_predictions.PAYMENT_RETRY == 0.7531
    assert ctx.ml_predictions.SEND_REMINDER == 0.4219


# ---------------------------------------------------------------------------
# 4. Invalid amount is rejected by Pydantic
# ---------------------------------------------------------------------------

def test_invalid_amount_rejected():
    bad_request = {**VALID_REQUEST, "amount": -100.0}
    response = client.post("/api/recovery/predict", json=bad_request)
    assert response.status_code == 422


def test_zero_amount_rejected():
    bad_request = {**VALID_REQUEST, "amount": 0}
    response = client.post("/api/recovery/predict", json=bad_request)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 5. Invalid attempt_number is rejected
# ---------------------------------------------------------------------------

def test_invalid_attempt_number_rejected():
    bad_request = {**VALID_REQUEST, "attempt_number": 0}
    response = client.post("/api/recovery/predict", json=bad_request)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 6. /health endpoint still works
# ---------------------------------------------------------------------------

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 7. Gemini error returns 502
# ---------------------------------------------------------------------------

def test_gemini_error_returns_502():
    from app.services.gemini_agent import GeminiAgentError

    with patch("app.routers.recovery.build_inference_features",
               return_value=MOCK_FEATURES_EXISTING), \
         patch("app.routers.recovery.predict_recovery_probabilities",
               return_value=MOCK_ML_PREDICTIONS), \
         patch("app.routers.recovery._get_agent") as mock_get_agent, \
         patch("app.routers.recovery.get_db") as mock_get_db:

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = MagicMock()
        mock_get_db.return_value = iter([mock_db])

        mock_agent = MagicMock()
        mock_agent.recommend.side_effect = GeminiAgentError("API key missing")
        mock_get_agent.return_value = mock_agent

        response = client.post("/api/recovery/predict", json=VALID_REQUEST)

    assert response.status_code == 502


# ---------------------------------------------------------------------------
# 8. Missing model returns 503
# ---------------------------------------------------------------------------

def test_missing_model_returns_503():
    with patch("app.routers.recovery.build_inference_features",
               return_value=MOCK_FEATURES_EXISTING), \
         patch("app.routers.recovery.predict_recovery_probabilities",
               side_effect=FileNotFoundError("model not found")), \
         patch("app.routers.recovery.get_db") as mock_get_db:

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = MagicMock()
        mock_get_db.return_value = iter([mock_db])

        response = client.post("/api/recovery/predict", json=VALID_REQUEST)

    assert response.status_code == 503


# ---------------------------------------------------------------------------
# 9. Empty customer_id is rejected
# ---------------------------------------------------------------------------

def test_empty_customer_id_rejected():
    bad_request = {**VALID_REQUEST, "customer_id": ""}
    response = client.post("/api/recovery/predict", json=bad_request)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Phase 7 additions — policy engine integration
# ---------------------------------------------------------------------------

def _make_pipeline_patches(mock_customer=None, features=None, predictions=None,
                           gemini_action="PAYMENT_RETRY"):
    """Helper: returns a context manager tuple for the standard pipeline mocks."""
    from unittest.mock import patch, MagicMock
    features = features or MOCK_FEATURES_EXISTING
    predictions = predictions or MOCK_ML_PREDICTIONS
    gemini_rec = MagicMock(action=gemini_action, confidence=0.85,
                           reason="test reason")
    return features, predictions, gemini_rec


# 10. Policy field is present and correct for an allowed recommendation
def test_policy_field_present_and_allowed():
    with patch("app.routers.recovery.build_inference_features",
               return_value=MOCK_FEATURES_EXISTING), \
         patch("app.routers.recovery.predict_recovery_probabilities",
               return_value=MOCK_ML_PREDICTIONS), \
         patch("app.routers.recovery._get_agent") as mock_get_agent, \
         patch("app.routers.recovery.get_db") as mock_get_db:

        mock_db = MagicMock()
        mock_customer = MagicMock()
        mock_customer.recovery_opt_out = False
        mock_db.query.return_value.filter.return_value.first.return_value = mock_customer
        mock_get_db.return_value = iter([mock_db])

        mock_agent = MagicMock()
        mock_agent.recommend.return_value = MagicMock(
            action="PAYMENT_RETRY", confidence=0.85, reason="good history"
        )
        mock_get_agent.return_value = mock_agent

        response = client.post("/api/recovery/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    data = response.json()
    assert data["policy"]["action"] == "PAYMENT_RETRY"
    assert data["policy"]["allowed"] is True
    # Gemini recommendation is preserved separately
    assert data["recommended_action"] == "PAYMENT_RETRY"


# 11. Opted-out customer: policy blocks action, Gemini recommendation preserved
def test_opted_out_customer_policy_blocks():
    with patch("app.routers.recovery.build_inference_features",
               return_value=MOCK_FEATURES_EXISTING), \
         patch("app.routers.recovery.predict_recovery_probabilities",
               return_value=MOCK_ML_PREDICTIONS), \
         patch("app.routers.recovery._get_agent") as mock_get_agent, \
         patch("app.routers.recovery.get_db") as mock_get_db, \
         patch("app.routers.recovery.policy_evaluate") as mock_policy:

        mock_db = MagicMock()
        mock_customer = MagicMock()
        mock_customer.recovery_opt_out = True   # <-- opted out
        mock_db.query.return_value.filter.return_value.first.return_value = mock_customer
        mock_get_db.return_value = iter([mock_db])

        mock_agent = MagicMock()
        mock_agent.recommend.return_value = MagicMock(
            action="PAYMENT_RETRY", confidence=0.9, reason="high probability"
        )
        mock_get_agent.return_value = mock_agent

        # Policy engine is called with the opted-out customer — mock its output
        from app.services.policy_engine import PolicyDecision
        mock_policy.return_value = PolicyDecision(
            action="STOP",
            allowed=False,
            reason="Recovery action blocked: this customer has opted out of all automated recovery actions.",
        )

        response = client.post("/api/recovery/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    data = response.json()
    # Gemini still recommended PAYMENT_RETRY
    assert data["recommended_action"] == "PAYMENT_RETRY"
    # Policy blocked it
    assert data["policy"]["action"] == "STOP"
    assert data["policy"]["allowed"] is False
    assert "opted out" in data["policy"]["reason"].lower()


# 12. Retry limit exceeded: policy escalates, Gemini recommendation preserved
def test_retry_limit_exceeded_policy_escalates():
    features_at_limit = {
        **MOCK_FEATURES_EXISTING,
        "previous_retry_count": 2,       # == MAX_PAYMENT_RETRIES
        "previous_recovery_attempts": 2,
    }

    with patch("app.routers.recovery.build_inference_features",
               return_value=features_at_limit), \
         patch("app.routers.recovery.predict_recovery_probabilities",
               return_value=MOCK_ML_PREDICTIONS), \
         patch("app.routers.recovery._get_agent") as mock_get_agent, \
         patch("app.routers.recovery.get_db") as mock_get_db:

        mock_db = MagicMock()
        mock_customer = MagicMock()
        mock_customer.recovery_opt_out = False
        mock_db.query.return_value.filter.return_value.first.return_value = mock_customer
        mock_get_db.return_value = iter([mock_db])

        mock_agent = MagicMock()
        mock_agent.recommend.return_value = MagicMock(
            action="PAYMENT_RETRY", confidence=0.88, reason="retry recommended"
        )
        mock_get_agent.return_value = mock_agent

        response = client.post("/api/recovery/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    data = response.json()
    assert data["recommended_action"] == "PAYMENT_RETRY"
    assert data["policy"]["action"] == "ESCALATE"
    assert data["policy"]["allowed"] is False


# 13. New customer: policy allows PAYMENT_RETRY (zero history, no opt-out)
def test_new_customer_policy_allows_payment_retry():
    with patch("app.routers.recovery.build_inference_features",
               return_value=MOCK_FEATURES_NEW), \
         patch("app.routers.recovery.predict_recovery_probabilities",
               return_value=MOCK_ML_PREDICTIONS), \
         patch("app.routers.recovery._get_agent") as mock_get_agent, \
         patch("app.routers.recovery.get_db") as mock_get_db:

        mock_db = MagicMock()
        # No customer found
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_get_db.return_value = iter([mock_db])

        mock_agent = MagicMock()
        mock_agent.recommend.return_value = MagicMock(
            action="PAYMENT_RETRY", confidence=0.75, reason="new customer retry"
        )
        mock_get_agent.return_value = mock_agent

        response = client.post("/api/recovery/predict",
                               json={**VALID_REQUEST, "customer_id": "new_cust"})

    assert response.status_code == 200
    data = response.json()
    assert data["policy"]["action"] == "PAYMENT_RETRY"
    assert data["policy"]["allowed"] is True


# ---------------------------------------------------------------------------
# Phase 8 additions — ESCALATE as a Gemini action
# ---------------------------------------------------------------------------

# 14. Gemini returns ESCALATE — policy allows it, recommended_action preserved
def test_gemini_escalate_reaches_policy_and_is_allowed():
    with patch("app.routers.recovery.build_inference_features",
               return_value=MOCK_FEATURES_EXISTING), \
         patch("app.routers.recovery.predict_recovery_probabilities",
               return_value={"PAYMENT_RETRY": 0.42, "SEND_REMINDER": 0.39}), \
         patch("app.routers.recovery._get_agent") as mock_get_agent, \
         patch("app.routers.recovery.get_db") as mock_get_db:

        mock_db = MagicMock()
        mock_customer = MagicMock()
        mock_customer.recovery_opt_out = False
        mock_db.query.return_value.filter.return_value.first.return_value = mock_customer
        mock_get_db.return_value = iter([mock_db])

        mock_agent = MagicMock()
        mock_agent.recommend.return_value = MagicMock(
            action="ESCALATE", confidence=0.70,
            reason="Both probabilities are weak; human review needed."
        )
        mock_get_agent.return_value = mock_agent

        response = client.post("/api/recovery/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    data = response.json()
    # Gemini recommendation preserved
    assert data["recommended_action"] == "ESCALATE"
    # Policy allows ESCALATE
    assert data["policy"]["action"] == "ESCALATE"
    assert data["policy"]["allowed"] is True


# 15. Gemini returns SEND_REMINDER — policy allows it
def test_gemini_send_reminder_policy_allows():
    with patch("app.routers.recovery.build_inference_features",
               return_value=MOCK_FEATURES_EXISTING), \
         patch("app.routers.recovery.predict_recovery_probabilities",
               return_value=MOCK_ML_PREDICTIONS), \
         patch("app.routers.recovery._get_agent") as mock_get_agent, \
         patch("app.routers.recovery.get_db") as mock_get_db:

        mock_db = MagicMock()
        mock_customer = MagicMock()
        mock_customer.recovery_opt_out = False
        mock_db.query.return_value.filter.return_value.first.return_value = mock_customer
        mock_get_db.return_value = iter([mock_db])

        mock_agent = MagicMock()
        mock_agent.recommend.return_value = MagicMock(
            action="SEND_REMINDER", confidence=0.73, reason="reminder appropriate"
        )
        mock_get_agent.return_value = mock_agent

        response = client.post("/api/recovery/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    data = response.json()
    assert data["recommended_action"] == "SEND_REMINDER"
    assert data["policy"]["action"] == "SEND_REMINDER"
    assert data["policy"]["allowed"] is True


# 16. Gemini recommends PAYMENT_RETRY but retry limit is reached — policy escalates
def test_gemini_payment_retry_does_not_bypass_retry_limit():
    features_at_retry_limit = {
        **MOCK_FEATURES_EXISTING,
        "previous_retry_count": 2,        # == MAX_PAYMENT_RETRIES
        "previous_recovery_attempts": 2,
    }

    with patch("app.routers.recovery.build_inference_features",
               return_value=features_at_retry_limit), \
         patch("app.routers.recovery.predict_recovery_probabilities",
               return_value=MOCK_ML_PREDICTIONS), \
         patch("app.routers.recovery._get_agent") as mock_get_agent, \
         patch("app.routers.recovery.get_db") as mock_get_db:

        mock_db = MagicMock()
        mock_customer = MagicMock()
        mock_customer.recovery_opt_out = False
        mock_db.query.return_value.filter.return_value.first.return_value = mock_customer
        mock_get_db.return_value = iter([mock_db])

        mock_agent = MagicMock()
        mock_agent.recommend.return_value = MagicMock(
            action="PAYMENT_RETRY", confidence=0.88, reason="retry recommended"
        )
        mock_get_agent.return_value = mock_agent

        response = client.post("/api/recovery/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    data = response.json()
    # Gemini said PAYMENT_RETRY but policy must block it
    assert data["recommended_action"] == "PAYMENT_RETRY"
    assert data["policy"]["action"] == "ESCALATE"
    assert data["policy"]["allowed"] is False


# 17. Gemini recommends SEND_REMINDER but reminder limit is reached — policy escalates
def test_gemini_send_reminder_does_not_bypass_reminder_limit():
    features_at_reminder_limit = {
        **MOCK_FEATURES_EXISTING,
        "previous_reminder_count": 2,     # == MAX_REMINDERS
        "previous_recovery_attempts": 2,
    }

    with patch("app.routers.recovery.build_inference_features",
               return_value=features_at_reminder_limit), \
         patch("app.routers.recovery.predict_recovery_probabilities",
               return_value=MOCK_ML_PREDICTIONS), \
         patch("app.routers.recovery._get_agent") as mock_get_agent, \
         patch("app.routers.recovery.get_db") as mock_get_db:

        mock_db = MagicMock()
        mock_customer = MagicMock()
        mock_customer.recovery_opt_out = False
        mock_db.query.return_value.filter.return_value.first.return_value = mock_customer
        mock_get_db.return_value = iter([mock_db])

        mock_agent = MagicMock()
        mock_agent.recommend.return_value = MagicMock(
            action="SEND_REMINDER", confidence=0.72, reason="reminder recommended"
        )
        mock_get_agent.return_value = mock_agent

        response = client.post("/api/recovery/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    data = response.json()
    # Gemini said SEND_REMINDER but policy must block it
    assert data["recommended_action"] == "SEND_REMINDER"
    assert data["policy"]["action"] == "ESCALATE"
    assert data["policy"]["allowed"] is False
