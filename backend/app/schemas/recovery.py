"""
RecoveryOS - Phase 6/7: Recovery API Schemas

Request and response Pydantic models for POST /api/recovery/predict.
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
