"""
RecoveryOS — LLM Recovery Agent (Groq)

Accepts a validated payment/customer context plus XGBoost probabilities,
sends them to the Groq API with a structured JSON prompt, and returns a
validated recovery recommendation.

This module is intentionally independent from app/ml/predict.py.
It does NOT call XGBoost — the caller is responsible for supplying the
XGBoost probabilities as part of the request context.

Flow:
    RecoveryContext (Pydantic)
        ↓
    GeminiRecoveryAgent.recommend()   ← name kept for import compatibility
        ↓
    RecoveryRecommendation (Pydantic)

Public names are unchanged so that all existing imports, tests, and the
router continue to work without modification.
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
    Raised for any failure in the LLM recovery agent:
        - missing API key
        - network / API error
        - empty or invalid response
        - structured-output validation failure

    Name kept as GeminiAgentError for import compatibility.
    """


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-20b"


# ---------------------------------------------------------------------------
# System instruction
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are the RecoveryOS revenue recovery decision agent.

Your job is to recommend the most appropriate recovery action for a failed \
payment using ONLY the supplied payment context, customer history, and \
XGBoost model predictions.

You MUST choose exactly one of these three actions:

PAYMENT_RETRY:
Recommend another automated payment retry. Use when the XGBoost retry \
probability is strong, the customer has a good payment history, and \
previous recovery attempts have not been exhausted.

SEND_REMINDER:
Recommend contacting or reminding the customer. Use when a reminder is \
more appropriate than an immediate retry, e.g. the customer has a \
reasonable history but the retry probability is weak.

ESCALATE:
Recommend human or manual intervention instead of another automated \
recovery action. Use when:
  - the recovery history is poor (low recovery success rate)
  - repeated recovery attempts have already failed
  - the customer has a strong pattern of unsuccessful recovery
  - both XGBoost probabilities are weak or ambiguous (below ~0.5)
  - the previous retry or reminder counts are high
  - the situation clearly requires human judgement
  - continuing automated recovery is unlikely to succeed

Do NOT use any action name other than PAYMENT_RETRY, SEND_REMINDER, or ESCALATE.

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
10. Return ONLY valid JSON matching this exact schema — no markdown, no prose:
    {"action": "<PAYMENT_RETRY|SEND_REMINDER|ESCALATE>", "reason": "<string>", "confidence": <float 0-1>}
11. You are making a recommendation only. A separate deterministic policy \
engine will later determine whether the recommendation is permitted.\
"""


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PaymentContext(BaseModel):
    amount: float = Field(
        ...,
        gt=0,
        description="Payment amount in the given currency",
    )
    currency: str = Field(
        ...,
        description="Currency code, e.g. INR",
    )
    payment_method: str = Field(
        ...,
        description="e.g. upi, card, netbanking, wallet",
    )
    failure_reason: str = Field(
        ...,
        description="Reason the payment failed",
    )
    attempt_number: int = Field(
        ...,
        ge=1,
        description="Which attempt this is",
    )


class CustomerHistory(BaseModel):
    total_previous_transactions: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    average_transaction_amount: float = Field(..., ge=0.0)
    days_since_last_success: int = Field(
        ...,
        ge=-1,
        description="-1 means no prior successful payment",
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

RecoveryAction = Literal[
    "PAYMENT_RETRY",
    "SEND_REMINDER",
    "ESCALATE",
]


class RecoveryRecommendation(BaseModel):
    action: RecoveryAction
    reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(ctx: RecoveryContext) -> str:
    p = ctx.payment
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
Days since last success         : {ch.days_since_last_success} (-1 = no prior success)
Previous recovery attempts      : {ch.previous_recovery_attempts}
Recovery success rate           : {ch.recovery_success_rate:.4f}
Previous PAYMENT_RETRY count    : {ch.previous_retry_count}
Previous SEND_REMINDER count    : {ch.previous_reminder_count}

=== XGBOOST MODEL PREDICTIONS ===
P(success | PAYMENT_RETRY)  : {ml.PAYMENT_RETRY:.4f}
P(success | SEND_REMINDER)  : {ml.SEND_REMINDER:.4f}

=== AVAILABLE ACTIONS ===
PAYMENT_RETRY  - recommend another automated payment retry
SEND_REMINDER  - recommend contacting/reminding the customer
ESCALATE       - recommend human/manual intervention instead of another automated recovery action

Choose ESCALATE if the recovery history is poor, probabilities are weak \
(below ~0.5), repeated attempts have failed, or human judgement is needed.

Based on the above, recommend the single best recovery action.

Return ONLY valid JSON with exactly these three keys:
action
reason
confidence

The JSON must have this structure:
{{"action": "PAYMENT_RETRY", "reason": "your explanation", "confidence": 0.95}}

Do not include markdown, code fences, or any additional text.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class GeminiRecoveryAgent:
    """
    Wraps the Groq client and exposes a single recommend() method.

    Class name kept as GeminiRecoveryAgent for import compatibility with
    the router and all existing tests.

    The client is instantiated lazily so that import-time failures only
    occur when the agent is actually used, not when the module is imported.
    """

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        api_key = settings.GROQ_API_KEY

        if not api_key:
            raise GeminiAgentError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file before using the LLM agent."
            )

        try:
            from groq import Groq  # noqa: PLC0415
        except ImportError as exc:
            raise GeminiAgentError(
                "groq package is not installed. "
                "Run: pip install groq"
            ) from exc

        self._client = Groq(api_key=api_key)

        return self._client

    def recommend(
        self,
        ctx: RecoveryContext,
    ) -> RecoveryRecommendation:
        """
        Send the recovery context to Groq and return a validated
        RecoveryRecommendation.

        Raises:
            GeminiAgentError — for any API, network, or validation failure.
        """

        client = self._get_client()
        prompt = _build_prompt(ctx)

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": _SYSTEM_INSTRUCTION,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,

                # GPT-OSS is a reasoning model. Disable reasoning so that
                # the response content contains our JSON recommendation.
                include_reasoning=False,

                # Give the model enough completion space for the JSON
                # response while keeping the response small.
                max_completion_tokens=2048,

                # Ask Groq for JSON output.
                response_format={
                    "type": "json_object"
                },
            )

        except Exception as exc:
            raise GeminiAgentError(
                f"Groq API request failed: {exc}"
            ) from exc

        raw = (
            response.choices[0].message.content
            if response.choices
            else None
        )

        if not raw:
            raise GeminiAgentError(
                "Groq returned an empty response. "
                "Check your API key and model availability."
            )

        raw = raw.strip()

        try:
            recommendation = RecoveryRecommendation.model_validate_json(raw)

        except Exception as exc:
            raise GeminiAgentError(
                f"Groq response failed Pydantic validation: {exc}\n"
                f"Raw response: {raw!r}"
            ) from exc

        return recommendation


# ---------------------------------------------------------------------------
# Module-level singleton — importable directly
# ---------------------------------------------------------------------------

agent = GeminiRecoveryAgent()