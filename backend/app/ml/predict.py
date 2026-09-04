"""
RecoveryOS — Phase 4B: Inference

Loads the saved sklearn Pipeline (preprocessing + XGBoost) and exposes
a reusable prediction function for the recovery probability of both
supported actions.

The saved Pipeline already contains the fitted ColumnTransformer, so
preprocessing is never duplicated here — we simply pass a DataFrame
with the same feature schema used during training.
"""

from __future__ import annotations

import os

import joblib
import pandas as pd

# Path to the saved pipeline artifact
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "recovery_model.joblib"
)

# Cached pipeline — loaded once on first call
_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {_MODEL_PATH}. "
                "Run: python -m scripts.train_recovery_model"
            )
        _pipeline = joblib.load(_MODEL_PATH)
    return _pipeline


def predict_recovery_probabilities(features: dict) -> dict[str, float]:
    """
    Given the current payment/customer context as a flat dict of feature
    values (without action), return the predicted recovery probability for
    each supported action.

    The function constructs two rows from the same context — one per action —
    and passes both through the saved Pipeline in a single call.

    Args:
        features: dict containing the 14 non-action feature values:
            amount, payment_method, failure_reason, attempt_number,
            total_previous_transactions, success_rate,
            average_transaction_amount, days_since_last_success,
            previous_recovery_attempts, recovery_success_rate,
            previous_retry_count, previous_reminder_count,
            transactions_last_30_days, successful_transactions_last_30_days

    Returns:
        {
            "PAYMENT_RETRY": 0.81,
            "SEND_REMINDER": 0.43,
        }
    """
    pipeline = _load_pipeline()

    actions = ["PAYMENT_RETRY", "SEND_REMINDER"]

    # Build one row per action — same payment/customer context, different action.
    # The Pipeline's ColumnTransformer handles one-hot encoding of action
    # exactly as it did during training (handle_unknown="ignore" covers any
    # unseen values safely).
    rows = []
    for action in actions:
        row = dict(features)
        row["action"] = action
        rows.append(row)

    X = pd.DataFrame(rows)
    probabilities = pipeline.predict_proba(X)[:, 1]

    return {
        action: round(float(prob), 4)
        for action, prob in zip(actions, probabilities)
    }


def predict_recovery_probability(features: pd.DataFrame) -> float:
    """
    Single-row prediction variant.

    Args:
        features: a single-row DataFrame that already includes the `action`
                  column set to the desired action value.

    Returns:
        float — probability of successful recovery for that action.
    """
    pipeline = _load_pipeline()
    return round(float(pipeline.predict_proba(features)[0, 1]), 4)
