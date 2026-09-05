"""
RecoveryOS — Phase 5: Gemini Recovery Agent

Accepts a validated payment/customer context plus XGBoost probabilities,
sends them to Gemini with a structured-output schema, and returns a
validated recovery recommendation.

This module is intentionally independent from app/ml/predict.py.
It does NOT call XGBoost — the caller is responsible for supplying the
XGBoost probabilities as part of the request context.

Flow:
    RecoveryContext (Pydantic)
        ↓
    GeminiRecoveryAgent.recommend()
        ↓
    RecoveryRecommendation (Pydantic)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import settings

# ---------------------------------------------------------------------------
# Application-level exception
# ---------------------------------------------------------------------------

class GeminiAgentError(Exception):
    """
    Raised for any failure in the Gemini recovery agent:
        - missing API key
        - network / API error
        - empty or invalid response
        - structured-output validation failure
    """


# ---------------------------------------------------------------------------
# Gemini model selection
# ---------------------------------------------------------------------------

# gemini-3.6-flash is the current stable Flash model supported by the
# google-genai SDK and available on the free tier as of mid-2025.
# Model name sourced directly from the live API error response.
GEMINI_MODEL = "gemini-3.6-flash"

# ---------------------------------------------------------------------------
# System instruction
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are the RecoveryOS revenue recovery decision agent.

Your job is to recommend the most appropriate recovery action for a failed \
payment using ONLY the supplied payment context, customer history, and \
XGBoost model predictions.

Available actions:

PAYMENT_RETRY:
Recommend attempting payment recovery again.

SEND_REMINDER:
Recommend contacting/reminding the customer.

ESCALATE:
Recommend human review.

STOP:
Recommend no further automated recovery.

Rules:

1. Do not invent customer information.
2. Do not invent payment information.
3. Do not modify the payment amount, currency, payment method, transaction \
details, or customer history.
4. Do not execute any payment or external API operation.
5. Do not claim that a payment was recovered.
6. Do not override information supplied by the backend.
7. Treat XGBoost probabilities as evidence, not guaranteed facts about the future.
8. Consider the complete supplied context when choosing an action.
9. When the evidence is ambiguous or insufficient, prefer ESCALATE rather \
than inventing certainty.
10. Return only the structured decision.
11. You are making a recommendation only. A separate deterministic policy \
engine will later determine whether the recommendation is permitted.\
"""

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PaymentContext(BaseModel):
    amount: float = Field(..., gt=0, description="Payment amount in the given currency")
    currency: str = Field(..., description="Currency code, e.g. INR")
    payment_method: str = Field(..., description="e.g. upi, card, netbanking, wallet")
    failure_reason: str = Field(..., description="Reason the payment failed")
    attempt_number: int = Field(..., ge=1, description="Which attempt this is")


class CustomerHistory(BaseModel):
    total_previous_transactions: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    average_transaction_amount: float = Field(..., ge=0.0)
    days_since_last_success: int = Field(
        ..., ge=-1, description="-1 means no prior successful payment"
    )
    previous_recovery_attempts: int = Field(..., ge=0)
    recovery_success_rate: float = Field(..., ge=0.0, le=1.0)
    previous_retry_count: int = Field(..., ge=0)
    previous_reminder_count: int = Field(..., ge=0)


class MLPredictions(BaseModel):
    PAYMENT_RETRY: float = Field(..., ge=0.0, le=1.0)
    SEND_REMINDER: float = Field(..., ge=0.0, le=1.0)


class RecoveryContext(BaseModel):
    payment: PaymentContext
    customer_history: CustomerHistory
    ml_predictions: MLPredictions


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

RecoveryAction = Literal["PAYMENT_RETRY", "SEND_REMINDER", "ESCALATE", "STOP"]


class RecoveryRecommendation(BaseModel):
    action: RecoveryAction
    reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(ctx: RecoveryContext) -> str:
    p  = ctx.payment
    ch = ctx.customer_history
    ml = ctx.ml_predictions

    return f"""\
Analyse the following failed payment and recommend a recovery action.

=== PAYMENT ===
Amount          : {p.amount} {p.currency}
Payment method  : {p.payment_method}
Failure reason  : {p.failure_reason}
Attempt number  : {p.attempt_number}

=== CUSTOMER HISTORY ===
Total previous transactions     : {ch.total_previous_transactions}
Historical success rate         : {ch.success_rate:.4f}
Average transaction amount      : {ch.average_transaction_amount:.2f} {p.currency}
Days since last success         : {ch.days_since_last_success} \
(-1 = no prior success)
Previous recovery attempts      : {ch.previous_recovery_attempts}
Recovery success rate           : {ch.recovery_success_rate:.4f}
Previous PAYMENT_RETRY count    : {ch.previous_retry_count}
Previous SEND_REMINDER count    : {ch.previous_reminder_count}

=== XGBOOST MODEL PREDICTIONS ===
P(success | PAYMENT_RETRY)  : {ml.PAYMENT_RETRY:.4f}
P(success | SEND_REMINDER)  : {ml.SEND_REMINDER:.4f}

Based on the above, recommend the single best recovery action.
Return a structured response with action, reason, and confidence.\
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class GeminiRecoveryAgent:
    """
    Wraps the google-genai client and exposes a single recommend() method.

    The client is instantiated lazily so that import-time failures only
    occur when the agent is actually used, not when the module is imported.
    """

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise GeminiAgentError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file before using the Gemini agent."
            )

        try:
            from google import genai  # noqa: PLC0415
        except ImportError as exc:
            raise GeminiAgentError(
                "google-genai package is not installed. "
                "Run: pip install google-genai"
            ) from exc

        self._client = genai.Client(api_key=api_key)
        return self._client

    def recommend(self, ctx: RecoveryContext) -> RecoveryRecommendation:
        """
        Send the recovery context to Gemini and return a validated
        RecoveryRecommendation.

        Raises:
            GeminiAgentError — for any API, network, or validation failure.
        """
        client = self._get_client()
        prompt = _build_prompt(ctx)

        try:
            from google.genai import types  # noqa: PLC0415

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=RecoveryRecommendation,
                    # Disable tool use — the agent must not call external APIs
                    tools=[],
                ),
            )
        except GeminiAgentError:
            raise
        except Exception as exc:
            raise GeminiAgentError(
                f"Gemini API request failed: {exc}"
            ) from exc

        # Extract and validate the structured response
        raw = getattr(response, "text", None)
        if not raw:
            raise GeminiAgentError(
                "Gemini returned an empty response. "
                "Check your API key and model availability."
            )

        try:
            recommendation = RecoveryRecommendation.model_validate_json(raw)
        except Exception as exc:
            raise GeminiAgentError(
                f"Gemini response failed Pydantic validation: {exc}\n"
                f"Raw response: {raw!r}"
            ) from exc

        return recommendation


# ---------------------------------------------------------------------------
# Module-level singleton — importable directly
# ---------------------------------------------------------------------------

agent = GeminiRecoveryAgent()
