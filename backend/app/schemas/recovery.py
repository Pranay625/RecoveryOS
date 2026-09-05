"""
RecoveryOS - Phase 6/7/8: Recovery API Schemas

Request and response Pydantic models for:
  POST /api/recovery/predict
  POST /api/recovery/execute

The frontend only supplies current payment context; all historical
features are derived by the backend from the database.
"""

from pydantic import BaseModel, Field


class RecoveryPredictionRequest(BaseModel):
    customer_id: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)
    currency: str = Field(..., min_length=1)
    payment_method: str = Field(..., min_length=1)
    failure_reason: str = Field(..., min_length=1)
    attempt_number: int = Field(..., ge=1)


class MLPredictionResult(BaseModel):
    PAYMENT_RETRY: float
    SEND_REMINDER: float


class PolicyDecisionResult(BaseModel):
    """
    The deterministic policy engine's authorization decision.

    action  — the action the policy engine authorizes (may differ from
              Gemini's recommended_action when a rule blocks it)
    allowed — True if the recommended action is permitted to execute
    reason  — human-readable explanation of the policy decision
    """
    action: str
    allowed: bool
    reason: str


class RecoveryPredictionResponse(BaseModel):
    customer_id: str
    customer_exists: bool
    ml_predictions: MLPredictionResult
    # Gemini's recommendation — preserved as-is, never overwritten
    recommended_action: str
    confidence: float
    reason: str
    # Policy engine's authorization decision
    policy: PolicyDecisionResult


# ---------------------------------------------------------------------------
# Execute schemas  (Phase 8)
# ---------------------------------------------------------------------------

# RecoveryExecuteRequest intentionally reuses the same fields as
# RecoveryPredictionRequest.  The frontend cannot submit a pre-approved
# action or allowed=true — the backend re-runs the full decision pipeline.
RecoveryExecuteRequest = RecoveryPredictionRequest


class RecoveryExecuteResponse(BaseModel):
    """
    Response from POST /api/recovery/execute.

    execution_status values:
      READY_FOR_PAYMENT  — Razorpay order created, frontend should open Checkout
      REMINDER_REQUIRED  — policy approved SEND_REMINDER, frontend should show UI
      MANUAL_REVIEW      — policy approved ESCALATE
      BLOCKED            — policy blocked the action (allowed=false)
    """
    customer_id: str
    action: str
    allowed: bool
    execution_status: str
    policy_reason: str
    # PAYMENT_RETRY only — None for all other actions
    razorpay_order_id: str | None = None
    amount_paise: int | None = None
    currency: str | None = None
    # Runtime tracking — None when action is blocked
    runtime_payment_id: str | None = None
    runtime_recovery_id: str | None = None


# ---------------------------------------------------------------------------
# Payment-result schemas  (Phase 9)
# ---------------------------------------------------------------------------

class PaymentResultRequest(BaseModel):
    """
    Sent by the frontend after a successful Razorpay Checkout.
    The backend verifies the signature independently — the frontend
    cannot assert that the payment succeeded.
    """
    razorpay_payment_id: str = Field(..., min_length=1)
    razorpay_order_id:   str = Field(..., min_length=1)
    razorpay_signature:  str = Field(..., min_length=1)


class PaymentResultResponse(BaseModel):
    status: str                    # "SUCCESS"
    razorpay_payment_id: str
    razorpay_order_id: str
    runtime_payment_id: str
    runtime_recovery_id: str


class PaymentFailedRequest(BaseModel):
    """
    Sent by the frontend when Razorpay Checkout fires payment.failed.
    No signature is available for failed payments, so we only record
    the failure — we do NOT verify a signature here.
    """
    razorpay_order_id: str = Field(..., min_length=1)
    error_code:        str | None = Field(default=None)
    error_description: str | None = Field(default=None)


class PaymentFailedResponse(BaseModel):
    status: str                    # "FAILED"
    razorpay_order_id: str
    runtime_payment_id: str
    runtime_recovery_id: str
