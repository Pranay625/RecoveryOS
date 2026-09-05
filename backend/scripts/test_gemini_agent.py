"""
RecoveryOS — Phase 5: Groq Agent Manual Test

Makes a REAL Groq API call using GROQ_API_KEY from the environment.

Run from backend:
    python -m scripts.test_gemini_agent
"""

import json
import sys

from app.core.config import settings
from app.services.gemini_agent import (
    CustomerHistory,
    GeminiAgentError,
    GeminiRecoveryAgent,
    MLPredictions,
    PaymentContext,
    RecoveryContext,
)


def main():
    if not settings.GROQ_API_KEY:
        print(
            "ERROR: GROQ_API_KEY is not set.\n"
            "Add it to your .env file:\n"
            "    GROQ_API_KEY=your_key_here"
        )
        sys.exit(1)

    print("=" * 55)
    print("  RecoveryOS — Groq Agent Manual Test")
    print("=" * 55)

    ctx = RecoveryContext(
        payment=PaymentContext(
            amount=2499,
            currency="INR",
            payment_method="upi",
            failure_reason="insufficient_funds",
            attempt_number=1,
        ),
        customer_history=CustomerHistory(
            total_previous_transactions=20,
            success_rate=0.80,
            average_transaction_amount=3000,
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

    print("\nSending recovery context to Groq...")
    print(f"  Payment  : INR {ctx.payment.amount} via {ctx.payment.payment_method}")
    print(f"  Failure  : {ctx.payment.failure_reason}")
    print(
        f"  XGBoost  : RETRY={ctx.ml_predictions.PAYMENT_RETRY}  "
        f"REMINDER={ctx.ml_predictions.SEND_REMINDER}"
    )

    try:
        agent = GeminiRecoveryAgent()
        recommendation = agent.recommend(ctx)
    except GeminiAgentError as exc:
        print(f"\nGroq agent error: {exc}")
        sys.exit(1)

    print("\nGroq Recovery Recommendation:")
    print(json.dumps(recommendation.model_dump(), indent=4))

    assert recommendation.action in (
        "PAYMENT_RETRY",
        "SEND_REMINDER",
        "ESCALATE",
    ), f"Unexpected action: {recommendation.action}"

    assert 0.0 <= recommendation.confidence <= 1.0, (
        f"Confidence out of range: {recommendation.confidence}"
    )

    assert len(recommendation.reason) > 0, "Reason is empty"

    print("\n  All assertions passed.")
    print(f"  Action     : {recommendation.action}")
    print(f"  Confidence : {recommendation.confidence}")
    print("=" * 55)


if __name__ == "__main__":
    main()