"""
RecoveryOS - Phase 6/7/8: Recovery Router

POST /api/recovery/predict   — recommendation + policy decision (read-only)
POST /api/recovery/execute   — policy-authorized execution + runtime state

Shared decision pipeline (used by both endpoints):
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
    PolicyDecision  (authoritative)

/predict  — returns the decision, never executes anything.
/execute  — executes only if policy.allowed == True, records runtime state.

SYNTHETIC DATA RULE:
    /execute NEVER modifies existing synthetic Payment or RecoveryAttempt rows.
    Runtime executions create NEW Payment rows (status="pending") and NEW
    RecoveryAttempt rows linked to them.  Historical rows are untouched.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.ml.inference_features import build_inference_features
from app.ml.predict import predict_recovery_probabilities
from app.models.audit_log import AuditLog
from app.models.customer import Customer
from app.models.payment import Payment
from app.models.recovery_attempt import RecoveryAttempt
from app.schemas.recovery import (
    MLPredictionResult,
    PaymentFailedRequest,
    PaymentFailedResponse,
    PaymentResultRequest,
    PaymentResultResponse,
    PolicyDecisionResult,
    RecoveryExecuteRequest,
    RecoveryExecuteResponse,
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
    RecoveryRecommendation,
)
from app.services.policy_engine import PolicyContext, PolicyDecision, evaluate as policy_evaluate
from app.services.razorpay_service import RazorpayServiceError
from app.services import razorpay_service as _razorpay_module


def _get_razorpay_service():
    return _razorpay_module.razorpay_service

router = APIRouter(prefix="/api/recovery", tags=["recovery"])

# Lazy singleton — loaded once on first request, not at import time
_gemini_agent: GeminiRecoveryAgent | None = None


def _get_agent() -> GeminiRecoveryAgent:
    global _gemini_agent
    if _gemini_agent is None:
        _gemini_agent = GeminiRecoveryAgent()
    return _gemini_agent


# ---------------------------------------------------------------------------
# Shared decision pipeline
# ---------------------------------------------------------------------------

class _DecisionResult:
    """Plain container for the shared pipeline output."""
    __slots__ = (
        "customer", "customer_exists", "features",
        "ml_predictions", "recommendation", "policy_decision",
    )

    def __init__(
        self,
        customer,
        customer_exists: bool,
        features: dict,
        ml_predictions: dict,
        recommendation: RecoveryRecommendation,
        policy_decision: PolicyDecision,
    ) -> None:
        self.customer        = customer
        self.customer_exists = customer_exists
        self.features        = features
        self.ml_predictions  = ml_predictions
        self.recommendation  = recommendation
        self.policy_decision = policy_decision


def _run_decision_pipeline(
    request: RecoveryPredictionRequest,
    db: Session,
) -> _DecisionResult:
    """
    Run the full ML → Gemini → Policy pipeline and return all intermediate
    results.  Raises HTTPException on any failure.  Does NOT write to the DB.
    """
    # 1. Customer lookup
    customer = db.query(Customer).filter(
        Customer.customer_id == request.customer_id
    ).first()

    # 2. Feature engineering
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

    # 3. XGBoost
    try:
        ml_preds = predict_recovery_probabilities(features)
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

    # 4. Build RecoveryContext
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
            PAYMENT_RETRY=ml_preds["PAYMENT_RETRY"],
            SEND_REMINDER=ml_preds["SEND_REMINDER"],
        ),
    )

    # 5. Gemini recommendation
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

    # 6. Deterministic policy engine
    policy_ctx = PolicyContext(
        recovery_opt_out=(customer.recovery_opt_out if customer else False),
        previous_recovery_attempts=features["previous_recovery_attempts"],
        previous_retry_count=features["previous_retry_count"],
        previous_reminder_count=features["previous_reminder_count"],
    )
    policy_decision = policy_evaluate(recommendation, policy_ctx)

    return _DecisionResult(
        customer=customer,
        customer_exists=(customer is not None),
        features=features,
        ml_predictions=ml_preds,
        recommendation=recommendation,
        policy_decision=policy_decision,
    )


# ---------------------------------------------------------------------------
# POST /api/recovery/predict
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=RecoveryPredictionResponse)
def predict_recovery(
    request: RecoveryPredictionRequest,
    db: Session = Depends(get_db),
):
    """
    Return an AI-powered recovery recommendation and deterministic policy
    authorization decision.  Read-only — never executes anything.
    """
    d = _run_decision_pipeline(request, db)

    return RecoveryPredictionResponse(
        customer_id=request.customer_id,
        customer_exists=d.customer_exists,
        ml_predictions=MLPredictionResult(
            PAYMENT_RETRY=d.ml_predictions["PAYMENT_RETRY"],
            SEND_REMINDER=d.ml_predictions["SEND_REMINDER"],
        ),
        recommended_action=d.recommendation.action,
        confidence=d.recommendation.confidence,
        reason=d.recommendation.reason,
        policy=PolicyDecisionResult(
            action=d.policy_decision.action,
            allowed=d.policy_decision.allowed,
            reason=d.policy_decision.reason,
        ),
    )


# ---------------------------------------------------------------------------
# POST /api/recovery/execute
# ---------------------------------------------------------------------------

@router.post("/execute", response_model=RecoveryExecuteResponse)
def execute_recovery(
    request: RecoveryExecuteRequest,
    db: Session = Depends(get_db),
):
    """
    Run the full decision pipeline and execute the policy-authorized action.

    The frontend cannot submit a pre-approved action or bypass policy.
    The backend re-runs ML → Gemini → Policy on every call.

    PAYMENT_RETRY (allowed):
        Creates a Razorpay Test Mode Order.
        Creates a runtime Payment row (status="pending") — NOT a synthetic row.
        Creates a RecoveryAttempt row linked to the runtime payment.
        Returns order_id for the frontend to open Razorpay Checkout.

    SEND_REMINDER (allowed):
        Creates runtime Payment + RecoveryAttempt rows.
        Returns REMINDER_REQUIRED — frontend handles the UI.

    ESCALATE (allowed):
        Creates runtime Payment + RecoveryAttempt rows.
        Returns MANUAL_REVIEW.

    Blocked (allowed=False):
        No Razorpay call.  No DB writes.  Returns BLOCKED.
    """
    d = _run_decision_pipeline(request, db)
    pd = d.policy_decision

    # ------------------------------------------------------------------
    # Policy blocked — return immediately, no side effects
    # ------------------------------------------------------------------
    if not pd.allowed:
        return RecoveryExecuteResponse(
            customer_id=request.customer_id,
            action=pd.action,
            allowed=False,
            execution_status="BLOCKED",
            policy_reason=pd.reason,
        )

    # ------------------------------------------------------------------
    # Policy allowed — determine execution path
    # ------------------------------------------------------------------
    action = pd.action

    razorpay_order_id: str | None = None
    amount_paise: int | None = None

    if action == "PAYMENT_RETRY":
        receipt = f"rcvros_{request.customer_id}_{int(datetime.now(timezone.utc).timestamp())}"
        try:
            rz_order = _get_razorpay_service().create_order(
                amount_inr=request.amount,
                currency=request.currency,
                receipt=receipt,
            )
        except RazorpayServiceError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Razorpay order creation failed: {exc}",
            )
        razorpay_order_id = rz_order.id
        amount_paise      = rz_order.amount

    # ------------------------------------------------------------------
    # Runtime DB records
    # New Payment row (status="pending") — completely separate from
    # synthetic historical payments.  Synthetic rows are never touched.
    # ------------------------------------------------------------------
    runtime_payment_id  = f"PAY_RT_{uuid.uuid4().hex[:16].upper()}"
    runtime_recovery_id = f"REC_RT_{uuid.uuid4().hex[:16].upper()}"
    now = datetime.now(timezone.utc)

    runtime_payment = Payment(
        payment_id=runtime_payment_id,
        customer_id=request.customer_id,
        razorpay_order_id=razorpay_order_id,   # set for PAYMENT_RETRY, None otherwise
        razorpay_payment_id=None,               # filled after Checkout completes
        amount=request.amount,
        currency=request.currency,
        payment_method=request.payment_method,
        status="pending",
        failure_reason=request.failure_reason,
        attempt_number=request.attempt_number,
        created_at=now,
    )
    db.add(runtime_payment)

    runtime_recovery = RecoveryAttempt(
        recovery_id=runtime_recovery_id,
        payment_id=runtime_payment_id,
        action=action,
        reason=d.recommendation.reason,
        ml_probability=d.ml_predictions.get(action),
        status="initiated",
        created_at=now,
        completed_at=None,
    )
    db.add(runtime_recovery)

    audit = AuditLog(
        audit_id=f"AUD_RT_{uuid.uuid4().hex[:16].upper()}",
        payment_id=runtime_payment_id,
        recovery_id=runtime_recovery_id,
        event=f"EXECUTE_{action}",
        agent_reason=d.recommendation.reason,
        policy_result=pd.reason,
        timestamp=now,
    )
    db.add(audit)

    db.commit()

    # ------------------------------------------------------------------
    # Build response
    # ------------------------------------------------------------------
    execution_status_map = {
        "PAYMENT_RETRY": "READY_FOR_PAYMENT",
        "SEND_REMINDER": "REMINDER_REQUIRED",
        "ESCALATE":      "MANUAL_REVIEW",
    }

    return RecoveryExecuteResponse(
        customer_id=request.customer_id,
        action=action,
        allowed=True,
        execution_status=execution_status_map.get(action, "UNKNOWN"),
        policy_reason=pd.reason,
        razorpay_order_id=razorpay_order_id,
        amount_paise=amount_paise,
        currency=request.currency if action == "PAYMENT_RETRY" else None,
        runtime_payment_id=runtime_payment_id,
        runtime_recovery_id=runtime_recovery_id,
    )


# ---------------------------------------------------------------------------
# Shared helper — locate a runtime Payment by razorpay_order_id
# ---------------------------------------------------------------------------

_RUNTIME_PAYMENT_PREFIX = "PAY_RT_"
_RUNTIME_RECOVERY_PREFIX = "REC_RT_"


def _get_runtime_payment(db: Session, razorpay_order_id: str) -> Payment:
    """
    Locate a runtime Payment row by razorpay_order_id.

    Raises HTTPException:
      404 — no payment found with that order ID
      400 — payment exists but is a synthetic historical row (PAY_* prefix)
    """
    payment = db.query(Payment).filter(
        Payment.razorpay_order_id == razorpay_order_id
    ).first()

    if payment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No payment found for Razorpay order ID: {razorpay_order_id}",
        )

    if not payment.payment_id.startswith(_RUNTIME_PAYMENT_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The referenced payment is a historical record and cannot be modified.",
        )

    return payment


def _get_runtime_recovery(db: Session, payment_id: str) -> RecoveryAttempt:
    """
    Locate the runtime RecoveryAttempt linked to a runtime Payment.

    Raises HTTPException:
      404 — no recovery attempt found for this payment
      400 — recovery attempt exists but is not a runtime row
    """
    recovery = (
        db.query(RecoveryAttempt)
        .filter(RecoveryAttempt.payment_id == payment_id)
        .order_by(RecoveryAttempt.created_at.desc())
        .first()
    )

    if recovery is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No recovery attempt found for payment: {payment_id}",
        )

    if not recovery.recovery_id.startswith(_RUNTIME_RECOVERY_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The referenced recovery attempt is a historical record and cannot be modified.",
        )

    return recovery


# ---------------------------------------------------------------------------
# POST /api/recovery/payment-result
# ---------------------------------------------------------------------------

@router.post("/payment-result", response_model=PaymentResultResponse)
def payment_result(
    request: PaymentResultRequest,
    db: Session = Depends(get_db),
):
    """
    Called by the frontend after a successful Razorpay Checkout.

    Flow:
      1. Verify the Razorpay signature server-side (HMAC-SHA256, no network call).
      2. Locate the runtime Payment by razorpay_order_id.
      3. Guard against modifying synthetic historical rows.
      4. Idempotency: if already marked captured, return success without re-writing.
      5. Locate the corresponding runtime RecoveryAttempt.
      6. Update Payment  → status="captured", razorpay_payment_id stored.
      7. Update RecoveryAttempt → status="completed", completed_at set.
      8. Create AuditLog.
      9. Commit once.
    """
    # ------------------------------------------------------------------
    # Step 1 — Signature verification (no DB access, no network call)
    # ------------------------------------------------------------------
    try:
        _get_razorpay_service().verify_payment_signature(
            razorpay_order_id=request.razorpay_order_id,
            razorpay_payment_id=request.razorpay_payment_id,
            razorpay_signature=request.razorpay_signature,
        )
    except RazorpayServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payment verification failed: {exc}",
        )

    # ------------------------------------------------------------------
    # Step 2 & 3 — Locate runtime Payment (raises 404/400 if invalid)
    # ------------------------------------------------------------------
    payment = _get_runtime_payment(db, request.razorpay_order_id)

    # ------------------------------------------------------------------
    # Step 4 — Idempotency: already captured → return success as-is
    # ------------------------------------------------------------------
    if payment.status == "captured":
        recovery = _get_runtime_recovery(db, payment.payment_id)
        return PaymentResultResponse(
            status="SUCCESS",
            razorpay_payment_id=payment.razorpay_payment_id or request.razorpay_payment_id,
            razorpay_order_id=request.razorpay_order_id,
            runtime_payment_id=payment.payment_id,
            runtime_recovery_id=recovery.recovery_id,
        )

    # ------------------------------------------------------------------
    # Step 5 — Locate runtime RecoveryAttempt
    # ------------------------------------------------------------------
    recovery = _get_runtime_recovery(db, payment.payment_id)

    # ------------------------------------------------------------------
    # Steps 6-9 — Update records and commit
    # ------------------------------------------------------------------
    now = datetime.now(timezone.utc)

    try:
        payment.status              = "captured"
        payment.razorpay_payment_id = request.razorpay_payment_id
        # razorpay_order_id is already set — do not overwrite

        recovery.status       = "completed"
        recovery.completed_at = now

        db.add(AuditLog(
            audit_id=f"AUD_RT_{uuid.uuid4().hex[:16].upper()}",
            payment_id=payment.payment_id,
            recovery_id=recovery.recovery_id,
            event="PAYMENT_SUCCESS",
            agent_reason=None,
            policy_result=(
                f"razorpay_order_id={request.razorpay_order_id} "
                f"razorpay_payment_id={request.razorpay_payment_id}"
            ),
            timestamp=now,
        ))

        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while recording payment success.",
        )

    return PaymentResultResponse(
        status="SUCCESS",
        razorpay_payment_id=request.razorpay_payment_id,
        razorpay_order_id=request.razorpay_order_id,
        runtime_payment_id=payment.payment_id,
        runtime_recovery_id=recovery.recovery_id,
    )


# ---------------------------------------------------------------------------
# POST /api/recovery/payment-failed
# ---------------------------------------------------------------------------

@router.post("/payment-failed", response_model=PaymentFailedResponse)
def payment_failed(
    request: PaymentFailedRequest,
    db: Session = Depends(get_db),
):
    """
    Called by the frontend when Razorpay Checkout fires payment.failed.

    No signature is available for failed payments — we record the failure
    only.  We do NOT trigger another ML/Gemini/policy evaluation here.

    Flow:
      1. Locate the runtime Payment by razorpay_order_id.
      2. Guard against modifying synthetic historical rows.
      3. Idempotency: if already marked failed, return failure response as-is.
      4. Locate the corresponding runtime RecoveryAttempt.
      5. Update Payment  → status="failed".
      6. Update RecoveryAttempt → status="failed", completed_at set.
      7. Create AuditLog.
      8. Commit once.

    Does NOT invoke ML, Gemini, or the policy engine.
    """
    # ------------------------------------------------------------------
    # Steps 1 & 2 — Locate runtime Payment (raises 404/400 if invalid)
    # ------------------------------------------------------------------
    payment = _get_runtime_payment(db, request.razorpay_order_id)

    # ------------------------------------------------------------------
    # Step 3 — Idempotency: already failed → return as-is
    # ------------------------------------------------------------------
    if payment.status == "failed":
        recovery = _get_runtime_recovery(db, payment.payment_id)
        return PaymentFailedResponse(
            status="FAILED",
            razorpay_order_id=request.razorpay_order_id,
            runtime_payment_id=payment.payment_id,
            runtime_recovery_id=recovery.recovery_id,
        )

    # ------------------------------------------------------------------
    # Step 4 — Locate runtime RecoveryAttempt
    # ------------------------------------------------------------------
    recovery = _get_runtime_recovery(db, payment.payment_id)

    # ------------------------------------------------------------------
    # Steps 5-8 — Update records and commit
    # ------------------------------------------------------------------
    now = datetime.now(timezone.utc)

    failure_detail = " | ".join(filter(None, [
        request.error_code,
        request.error_description,
    ])) or "payment.failed"

    try:
        payment.status = "failed"

        recovery.status       = "failed"
        recovery.completed_at = now

        db.add(AuditLog(
            audit_id=f"AUD_RT_{uuid.uuid4().hex[:16].upper()}",
            payment_id=payment.payment_id,
            recovery_id=recovery.recovery_id,
            event="PAYMENT_FAILED",
            agent_reason=failure_detail,
            policy_result=f"razorpay_order_id={request.razorpay_order_id}",
            timestamp=now,
        ))

        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while recording payment failure.",
        )

    return PaymentFailedResponse(
        status="FAILED",
        razorpay_order_id=request.razorpay_order_id,
        runtime_payment_id=payment.payment_id,
        runtime_recovery_id=recovery.recovery_id,
    )
