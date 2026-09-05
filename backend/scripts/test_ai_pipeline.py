"""
RecoveryOS - Phase 5B: XGBoost -> Gemini Integration Test

Proves that the actual output of the saved XGBoost pipeline is passed
directly into the Gemini recovery agent - no manually supplied probabilities.

Flow:
    Hardcoded test features
            |
            v
    predict_recovery_probabilities()   [app/ml/predict.py]
            |
            v
    Actual recovery_model.joblib
            |
            v
    { "PAYMENT_RETRY": <real>, "SEND_REMINDER": <real> }
            |
            v
    MLPredictions -> RecoveryContext
            |
            v
    GeminiRecoveryAgent.recommend()    [app/services/gemini_agent.py]
            |
            v
    RecoveryRecommendation

Run from backend/:
    python -m scripts.test_ai_pipeline
"""

import sys

from app.ml.predict import predict_recovery_probabilities
from app.services.gemini_agent import (
    CustomerHistory,
    GeminiAgentError,
    GeminiRecoveryAgent,
    MLPredictions,
    PaymentContext,
    RecoveryContext,
)

# ---------------------------------------------------------------------------
# Test input - 14 non-action features only.
# These are realistic hardcoded values used solely as test input.
# They do NOT come from the database.
# ---------------------------------------------------------------------------

FEATURES = {
    "amount": 2499.0,
    "payment_method": "upi",
    "failure_reason": "insufficient_funds",
    "attempt_number": 1,
    "total_previous_transactions": 10,
    "success_rate": 0.8,
    "average_transaction_amount": 2100.0,
    "days_since_last_success": 5,
    "previous_recovery_attempts": 2,
    "recovery_success_rate": 0.5,
    "previous_retry_count": 1,
    "previous_reminder_count": 1,
    "transactions_last_30_days": 4,
    "successful_transactions_last_30_days": 3,
}


def main():
    print("=" * 55)
    print("  RecoveryOS AI Pipeline Integration Test")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Step 1 - XGBoost inference
    # ------------------------------------------------------------------
    print("\nStep 1: Running XGBoost inference...")

    try:
        ml_predictions = predict_recovery_probabilities(FEATURES)
    except Exception as exc:
        print("\nXGBoost inference failed.")
        print(f"  Error: {exc}")
        sys.exit(1)

    print("\n=== XGBOOST OUTPUT ===")
    print(f"  PAYMENT_RETRY : {ml_predictions['PAYMENT_RETRY']}")
    print(f"  SEND_REMINDER : {ml_predictions['SEND_REMINDER']}")

    # ------------------------------------------------------------------
    # Step 2 - Construct RecoveryContext
    # The XGBoost values flow directly into MLPredictions - not hardcoded.
    # ------------------------------------------------------------------
    print("\nStep 2: Constructing RecoveryContext from XGBoost output...")

    context = RecoveryContext(
        payment=PaymentContext(
            amount=FEATURES["amount"],
            currency="INR",
            payment_method=FEATURES["payment_method"],
            failure_reason=FEATURES["failure_reason"],
            attempt_number=FEATURES["attempt_number"],
        ),
        customer_history=CustomerHistory(
            total_previous_transactions=FEATURES["total_previous_transactions"],
            success_rate=FEATURES["success_rate"],
            average_transaction_amount=FEATURES["average_transaction_amount"],
            days_since_last_success=FEATURES["days_since_last_success"],
            previous_recovery_attempts=FEATURES["previous_recovery_attempts"],
            recovery_success_rate=FEATURES["recovery_success_rate"],
            previous_retry_count=FEATURES["previous_retry_count"],
            previous_reminder_count=FEATURES["previous_reminder_count"],
        ),
        ml_predictions=MLPredictions(
            # These values come directly from XGBoost - not manually entered.
            PAYMENT_RETRY=ml_predictions["PAYMENT_RETRY"],
            SEND_REMINDER=ml_predictions["SEND_REMINDER"],
        ),
    )

    # ------------------------------------------------------------------
    # Step 3 - Verify handoff: XGBoost values are in the context
    # ------------------------------------------------------------------
    print("\nStep 3: Verifying XGBoost -> RecoveryContext handoff...")

    assert context.ml_predictions.PAYMENT_RETRY == ml_predictions["PAYMENT_RETRY"], (
        f"PAYMENT_RETRY mismatch: context has "
        f"{context.ml_predictions.PAYMENT_RETRY} but XGBoost returned "
        f"{ml_predictions['PAYMENT_RETRY']}"
    )
    assert context.ml_predictions.SEND_REMINDER == ml_predictions["SEND_REMINDER"], (
        f"SEND_REMINDER mismatch: context has "
        f"{context.ml_predictions.SEND_REMINDER} but XGBoost returned "
        f"{ml_predictions['SEND_REMINDER']}"
    )

    print("  XGBoost values confirmed in RecoveryContext.")

    # ------------------------------------------------------------------
    # Step 4 - Gemini inference
    # ------------------------------------------------------------------
    print("\nStep 4: Sending context to Gemini...")

    print("\n=== GEMINI INPUT ===")
    print(f"  PAYMENT_RETRY : {context.ml_predictions.PAYMENT_RETRY}")
    print(f"  SEND_REMINDER : {context.ml_predictions.SEND_REMINDER}")

    try:
        gemini_agent = GeminiRecoveryAgent()
        recommendation = gemini_agent.recommend(context)
    except GeminiAgentError as exc:
        print("\nGemini inference failed.")
        print(f"  Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print("\nGemini inference failed.")
        print(f"  Unexpected error: {exc}")
        sys.exit(1)

    print("\n=== GEMINI OUTPUT ===")
    print(f"  Action     : {recommendation.action}")
    print(f"  Reason     : {recommendation.reason}")
    print(f"  Confidence : {recommendation.confidence}")

    # ------------------------------------------------------------------
    # Step 5 - Final assertions
    # ------------------------------------------------------------------
    print("\nStep 5: Running final assertions...")

    assert recommendation.action in {
        "PAYMENT_RETRY",
        "SEND_REMINDER",
        "ESCALATE",
        "STOP",
    }, f"Unexpected action: {recommendation.action!r}"

    assert 0.0 <= recommendation.confidence <= 1.0, (
        f"Confidence out of range: {recommendation.confidence}"
    )

    assert recommendation.reason, "Reason must not be empty"

    print("  All assertions passed.")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print("  RecoveryOS AI Pipeline Test")
    print(f"{'='*55}")
    print()
    print("  Raw test features")
    print("          |")
    print("          v")
    print("      XGBoost")
    print("          |")
    print(f"          v  PAYMENT_RETRY = {ml_predictions['PAYMENT_RETRY']}")
    print(f"             SEND_REMINDER = {ml_predictions['SEND_REMINDER']}")
    print("          |")
    print("          v")
    print("       Gemini")
    print("          |")
    print(f"          v  ACTION     = {recommendation.action}")
    print(f"             CONFIDENCE = {recommendation.confidence}")
    print()
    print(f"{'='*55}")
    print("  END-TO-END AI PIPELINE PASSED")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
