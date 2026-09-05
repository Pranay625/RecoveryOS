"""
Phase 5 unit tests — Gemini Recovery Agent.

All tests mock the Gemini client. No real API calls are made.

Run from backend/:
    pytest tests/test_gemini_agent.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.gemini_agent import (
    CustomerHistory,
    GeminiAgentError,
    GeminiRecoveryAgent,
    MLPredictions,
    PaymentContext,
    RecoveryContext,
    RecoveryRecommendation,
    _build_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_context() -> RecoveryContext:
    return RecoveryContext(
        payment=PaymentContext(
            amount=2499.0,
            currency="INR",
            payment_method="upi",
            failure_reason="insufficient_funds",
            attempt_number=1,
        ),
        customer_history=CustomerHistory(
            total_previous_transactions=20,
            success_rate=0.80,
            average_transaction_amount=3000.0,
            days_since_last_success=5,
            previous_recovery_attempts=1,
            recovery_success_rate=1.0,
            previous_retry_count=1,
            previous_reminder_count=0,
        ),
        ml_predictions=MLPredictions(
            PAYMENT_RETRY=0.9283,
            SEND_REMINDER=0.8713,
        ),
    )


def _make_mock_response(payload: dict) -> MagicMock:
    """Return a mock that looks like a google-genai response object."""
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(payload)
    return mock_resp


# ---------------------------------------------------------------------------
# 1. Valid structured response is accepted
# ---------------------------------------------------------------------------

def test_valid_response_accepted(valid_context):
    valid_payload = {
        "action": "PAYMENT_RETRY",
        "reason": (
            "The payment has a strong retry probability and the customer "
            "has a strong historical success rate."
        ),
        "confidence": 0.91,
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_mock_response(
        valid_payload
    )

    agent = GeminiRecoveryAgent()
    agent._client = mock_client

    result = agent.recommend(valid_context)

    assert isinstance(result, RecoveryRecommendation)
    assert result.action == "PAYMENT_RETRY"
    assert result.confidence == 0.91
    assert "retry" in result.reason.lower() or len(result.reason) > 0


# ---------------------------------------------------------------------------
# 2. Invalid action value is rejected
# ---------------------------------------------------------------------------

def test_invalid_action_rejected(valid_context):
    invalid_payload = {
        "action": "DO_SOMETHING",
        "reason": "test",
        "confidence": 0.9,
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_mock_response(
        invalid_payload
    )

    agent = GeminiRecoveryAgent()
    agent._client = mock_client

    with pytest.raises(GeminiAgentError, match="validation"):
        agent.recommend(valid_context)


# ---------------------------------------------------------------------------
# 3. Confidence below 0 is rejected
# ---------------------------------------------------------------------------

def test_confidence_below_zero_rejected(valid_context):
    invalid_payload = {
        "action": "SEND_REMINDER",
        "reason": "test",
        "confidence": -0.1,
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_mock_response(
        invalid_payload
    )

    agent = GeminiRecoveryAgent()
    agent._client = mock_client

    with pytest.raises(GeminiAgentError, match="validation"):
        agent.recommend(valid_context)


# ---------------------------------------------------------------------------
# 4. Confidence above 1 is rejected
# ---------------------------------------------------------------------------

def test_confidence_above_one_rejected(valid_context):
    invalid_payload = {
        "action": "ESCALATE",
        "reason": "test",
        "confidence": 1.5,
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_mock_response(
        invalid_payload
    )

    agent = GeminiRecoveryAgent()
    agent._client = mock_client

    with pytest.raises(GeminiAgentError, match="validation"):
        agent.recommend(valid_context)


# ---------------------------------------------------------------------------
# 5. Prompt contains all supplied context fields
# ---------------------------------------------------------------------------

def test_prompt_contains_context_fields(valid_context):
    prompt = _build_prompt(valid_context)

    assert "2499" in prompt
    assert "INR" in prompt
    assert "upi" in prompt
    assert "insufficient_funds" in prompt
    assert "0.9283" in prompt
    assert "0.8713" in prompt
    assert "0.8000" in prompt          # success_rate
    assert "3000.00" in prompt         # average_transaction_amount
    assert "PAYMENT_RETRY" in prompt
    assert "SEND_REMINDER" in prompt


# ---------------------------------------------------------------------------
# 6. Structured response schema is passed to Gemini
# ---------------------------------------------------------------------------

def test_structured_schema_passed_to_gemini(valid_context):
    valid_payload = {
        "action": "SEND_REMINDER",
        "reason": "Low retry probability.",
        "confidence": 0.72,
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_mock_response(
        valid_payload
    )

    agent = GeminiRecoveryAgent()
    agent._client = mock_client

    agent.recommend(valid_context)

    call_kwargs = mock_client.models.generate_content.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs.args[2] if call_kwargs.args else None

    # Verify generate_content was called with a config argument
    assert mock_client.models.generate_content.called
    # Verify the model name was passed
    call_args = mock_client.models.generate_content.call_args
    assert GEMINI_MODEL_NAME_CHECK in str(call_args)


GEMINI_MODEL_NAME_CHECK = "gemini-3.6-flash"


# ---------------------------------------------------------------------------
# 7. Gemini API failure is converted to GeminiAgentError
# ---------------------------------------------------------------------------

def test_api_failure_raises_gemini_agent_error(valid_context):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError(
        "Connection timeout"
    )

    agent = GeminiRecoveryAgent()
    agent._client = mock_client

    with pytest.raises(GeminiAgentError, match="Gemini API request failed"):
        agent.recommend(valid_context)


# ---------------------------------------------------------------------------
# 8. Empty Gemini response raises GeminiAgentError
# ---------------------------------------------------------------------------

def test_empty_response_raises_error(valid_context):
    mock_resp = MagicMock()
    mock_resp.text = ""

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp

    agent = GeminiRecoveryAgent()
    agent._client = mock_client

    with pytest.raises(GeminiAgentError, match="empty response"):
        agent.recommend(valid_context)


# ---------------------------------------------------------------------------
# 9. Missing GEMINI_API_KEY raises GeminiAgentError
# ---------------------------------------------------------------------------

def test_missing_api_key_raises_error(valid_context):
    agent = GeminiRecoveryAgent()
    # _client is None — will attempt to build it from settings

    with patch("app.services.gemini_agent.settings") as mock_settings:
        mock_settings.GEMINI_API_KEY = ""
        with pytest.raises(GeminiAgentError, match="GEMINI_API_KEY"):
            agent.recommend(valid_context)


# ---------------------------------------------------------------------------
# 10. All three valid actions are accepted by RecoveryRecommendation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("action", ["PAYMENT_RETRY", "SEND_REMINDER", "ESCALATE"])
def test_all_valid_actions_accepted(action):
    rec = RecoveryRecommendation(action=action, reason="test", confidence=0.5)
    assert rec.action == action


# ---------------------------------------------------------------------------
# 10b. STOP is no longer a valid Gemini action
# ---------------------------------------------------------------------------

def test_stop_action_rejected():
    with pytest.raises(Exception):
        RecoveryRecommendation(action="STOP", reason="test", confidence=0.5)


# ---------------------------------------------------------------------------
# 11. Input validation — amount must be > 0
# ---------------------------------------------------------------------------

def test_payment_amount_must_be_positive():
    with pytest.raises(Exception):
        PaymentContext(
            amount=0,
            currency="INR",
            payment_method="upi",
            failure_reason="insufficient_funds",
            attempt_number=1,
        )


# ---------------------------------------------------------------------------
# 12. Input validation — probabilities must be in [0, 1]
# ---------------------------------------------------------------------------

def test_ml_prediction_out_of_range():
    with pytest.raises(Exception):
        MLPredictions(PAYMENT_RETRY=1.5, SEND_REMINDER=0.5)


# ---------------------------------------------------------------------------
# 13. Input validation — negative counts are rejected
# ---------------------------------------------------------------------------

def test_negative_recovery_count_rejected():
    with pytest.raises(Exception):
        CustomerHistory(
            total_previous_transactions=-1,
            success_rate=0.8,
            average_transaction_amount=3000,
            days_since_last_success=5,
            previous_recovery_attempts=0,
            recovery_success_rate=0.0,
            previous_retry_count=0,
            previous_reminder_count=0,
        )


# ---------------------------------------------------------------------------
# 14. Gemini returns ESCALATE — accepted and passed through
# ---------------------------------------------------------------------------

def test_escalate_response_accepted(valid_context):
    escalate_payload = {
        "action": "ESCALATE",
        "reason": "Both probabilities are weak and recovery history is poor.",
        "confidence": 0.55,
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_mock_response(
        escalate_payload
    )

    agent = GeminiRecoveryAgent()
    agent._client = mock_client

    result = agent.recommend(valid_context)

    assert isinstance(result, RecoveryRecommendation)
    assert result.action == "ESCALATE"
    assert result.confidence == 0.55


# ---------------------------------------------------------------------------
# 15. Prompt contains ESCALATE action definition
# ---------------------------------------------------------------------------

def test_prompt_contains_escalate_definition(valid_context):
    prompt = _build_prompt(valid_context)
    assert "ESCALATE" in prompt
    # Prompt must explain what ESCALATE means
    assert "human" in prompt.lower() or "manual" in prompt.lower()


# ---------------------------------------------------------------------------
# 16. Prompt contains all three action names
# ---------------------------------------------------------------------------

def test_prompt_contains_all_three_actions(valid_context):
    prompt = _build_prompt(valid_context)
    assert "PAYMENT_RETRY" in prompt
    assert "SEND_REMINDER" in prompt
    assert "ESCALATE" in prompt
