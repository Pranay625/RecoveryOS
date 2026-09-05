"""
RecoveryOS - Phase 6/7: Recovery Prediction Router

POST /api/recovery/predict

Flow:
    RecoveryPredictionRequest
            |
            v
    DB lookup (customer exists?)
            |
            v
    build_inference_features()         [app/ml/inference_features.py]
            |
            v
    predict_recovery_probabilities()   [app/ml/predict.py]
            |
            v
    RecoveryContext -> GeminiRecoveryAgent.recommend()
            |
            v
    PolicyContext -> policy_engine.evaluate()
            |
            v
    RecoveryPredictionResponse

This endpoint is recommendation + authorization only.
It does NOT execute payments, create recovery attempts, or call Razorpay.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.ml.inference_features import build_inference_features
from app.ml.predict import predict_recovery_probabilities
from app.models.customer import Customer
from app.schemas.recovery import (
    MLPredictionResult,
    PolicyDecisionResult,
    RecoveryPredictionRequest,
    RecoveryPredictionResponse,
)
from app.services.gemini_agent import (
    CustomerHistory,
    GeminiAgentError,
    GeminiRecoveryAgent,
    MLPredictions,
    PaymentContext,
    RecoveryContext,
)
from app.services.policy_engine import PolicyContext, evaluate as policy_evaluate

router = APIRouter(prefix="/api/recovery", tags=["recovery"])

# Lazy singleton — loaded once on first request, not at import time
_gemini_agent: GeminiRecoveryAgent | None = None


def _get_agent() -> GeminiRecoveryAgent:
    global _gemini_agent
    if _gemini_agent is None:
        _gemini_agent = GeminiRecoveryAgent()
    return _gemini_agent


@router.post("/predict", response_model=RecoveryPredictionResponse)
def predict_recovery(
    request: RecoveryPredictionRequest,
    db: Session = Depends(get_db),
):
    """
    Given a failed payment context, return an AI-powered recovery recommendation
    and a deterministic policy authorization decision.

    The backend:
      1. Looks up the customer in the database.
      2. Derives historical ML features (zero-history for new customers).
      3. Runs the saved XGBoost pipeline to get recovery probabilities.
      4. Passes the full context + probabilities to the Gemini agent.
      5. Passes Gemini's recommendation to the deterministic policy engine.
      6. Returns both the Gemini recommendation and the policy decision.
    """
    # ------------------------------------------------------------------
    # 1. Check whether customer exists
    # ------------------------------------------------------------------
    customer = db.query(Customer).filter(
        Customer.customer_id == request.customer_id
    ).first()
    customer_exists = customer is not None

    # ------------------------------------------------------------------
    # 2. Build the 14 ML features from DB history (or zero-history)
    # ------------------------------------------------------------------
    try:
        features = build_inference_features(
            db=db,
            customer_id=request.customer_id,
            amount=request.amount,
            payment_method=request.payment_method,
            failure_reason=request.failure_reason,
            attempt_number=request.attempt_number,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature engineering failed: {exc}",
        )

    # ------------------------------------------------------------------
    # 3. XGBoost inference — probabilities come from the saved pipeline
    # ------------------------------------------------------------------
    try:
        ml_predictions = predict_recovery_probabilities(features)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not available. Run: python -m scripts.train_recovery_model",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"XGBoost inference failed: {exc}",
        )

    # ------------------------------------------------------------------
    # 4. Construct RecoveryContext — XGBoost values flow directly in
    # ------------------------------------------------------------------
    ctx = RecoveryContext(
        payment=PaymentContext(
            amount=request.amount,
            currency=request.currency,
            payment_method=request.payment_method,
            failure_reason=request.failure_reason,
            attempt_number=request.attempt_number,
        ),
        customer_history=CustomerHistory(
            total_previous_transactions=features["total_previous_transactions"],
            success_rate=features["success_rate"],
            average_transaction_amount=features["average_transaction_amount"],
            days_since_last_success=features["days_since_last_success"],
            previous_recovery_attempts=features["previous_recovery_attempts"],
            recovery_success_rate=features["recovery_success_rate"],
            previous_retry_count=features["previous_retry_count"],
            previous_reminder_count=features["previous_reminder_count"],
        ),
        ml_predictions=MLPredictions(
            # These are the actual XGBoost outputs — never hardcoded
            PAYMENT_RETRY=ml_predictions["PAYMENT_RETRY"],
            SEND_REMINDER=ml_predictions["SEND_REMINDER"],
        ),
    )

    # ------------------------------------------------------------------
    # 5. Gemini recommendation
    # ------------------------------------------------------------------
    try:
        recommendation = _get_agent().recommend(ctx)
    except GeminiAgentError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini agent error: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during Gemini inference: {exc}",
        )

    # ------------------------------------------------------------------
    # 6. Deterministic policy engine
    # No SQLAlchemy objects cross into the policy engine — only plain values.
    # ------------------------------------------------------------------
    policy_ctx = PolicyContext(
        recovery_opt_out=(customer.recovery_opt_out if customer else False),
        previous_recovery_attempts=features["previous_recovery_attempts"],
        previous_retry_count=features["previous_retry_count"],
        previous_reminder_count=features["previous_reminder_count"],
    )
    policy_decision = policy_evaluate(recommendation, policy_ctx)

    # ------------------------------------------------------------------
    # 7. Return structured response
    # Gemini's recommendation is preserved as-is.
    # Policy decision is shown alongside it — not overwriting it.
    # ------------------------------------------------------------------
    return RecoveryPredictionResponse(
        customer_id=request.customer_id,
        customer_exists=customer_exists,
        ml_predictions=MLPredictionResult(
            PAYMENT_RETRY=ml_predictions["PAYMENT_RETRY"],
            SEND_REMINDER=ml_predictions["SEND_REMINDER"],
        ),
        recommended_action=recommendation.action,
        confidence=recommendation.confidence,
        reason=recommendation.reason,
        policy=PolicyDecisionResult(
            action=policy_decision.action,
            allowed=policy_decision.allowed,
            reason=policy_decision.reason,
        ),
    )
