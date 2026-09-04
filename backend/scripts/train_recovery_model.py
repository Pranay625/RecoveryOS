"""
RecoveryOS — Phase 4B: XGBoost Training Pipeline

Trains a single XGBClassifier that estimates:
    P(recovery succeeds | payment/customer context, action)

The model takes `action` as an input feature (one-hot encoded alongside
payment_method and failure_reason), so a single model handles both:
    PAYMENT_RETRY  →  P(success | retry)
    SEND_REMINDER  →  P(success | reminder)

Run from backend/:
    python -m scripts.train_recovery_model
"""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from app.ml.feature_engineering import build_training_dataframe_with_customers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

# 14 raw features fed into the model.
# customer_id is NEVER included here — it is only used for splitting.
FEATURE_COLUMNS = [
    "amount",
    "payment_method",
    "failure_reason",
    "attempt_number",
    "total_previous_transactions",
    "success_rate",
    "average_transaction_amount",
    "days_since_last_success",
    "previous_recovery_attempts",
    "recovery_success_rate",
    "previous_retry_count",
    "previous_reminder_count",
    "transactions_last_30_days",
    "successful_transactions_last_30_days",
    # action is the 15th model input — one-hot encoded with the categoricals
    "action",
]

NUMERICAL_FEATURES = [
    "amount",
    "attempt_number",
    "total_previous_transactions",
    "success_rate",
    "average_transaction_amount",
    "days_since_last_success",
    "previous_recovery_attempts",
    "recovery_success_rate",
    "previous_retry_count",
    "previous_reminder_count",
    "transactions_last_30_days",
    "successful_transactions_last_30_days",
]

# action is encoded here alongside the other categoricals so the model
# learns action-specific recovery probability from a single classifier.
CATEGORICAL_FEATURES = ["payment_method", "failure_reason", "action"]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "recovery_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "recovery_model_metadata.json")


# ---------------------------------------------------------------------------
# Group-aware train / validation / test split
# ---------------------------------------------------------------------------

def group_aware_split(
    df: pd.DataFrame,
    customer_ids: list[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Split rows so that ALL rows belonging to the same customer land in the
    same partition.  This prevents the model from seeing a customer's later
    recovery attempts during training while that customer's earlier attempts
    are in the test set — a subtle but real form of data leakage.

    customer_id is used ONLY here and is never added to the feature matrix.

    Strategy:
        1. Collect unique customers and shuffle them with a fixed seed.
        2. Assign customers to train / val / test by cumulative fraction.
        3. Filter df rows by customer membership.
    """
    rng = np.random.default_rng(random_state)

    cid_series = pd.Series(customer_ids, index=df.index)
    unique_customers = np.array(cid_series.unique())
    rng.shuffle(unique_customers)

    n = len(unique_customers)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_customers = set(unique_customers[:train_end])
    val_customers   = set(unique_customers[train_end:val_end])
    test_customers  = set(unique_customers[val_end:])

    train_mask = cid_series.isin(train_customers)
    val_mask   = cid_series.isin(val_customers)
    test_mask  = cid_series.isin(test_customers)

    X = df[FEATURE_COLUMNS]
    y = df["target"]

    return (
        X[train_mask], X[val_mask], X[test_mask],
        y[train_mask], y[val_mask], y[test_mask],
    )


# ---------------------------------------------------------------------------
# Preprocessing + model pipeline
# ---------------------------------------------------------------------------

def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """
    ColumnTransformer:
        numerical  → passed through as-is (XGBoost handles scale natively)
        categorical → OneHotEncoder(handle_unknown="ignore")
                      covers payment_method, failure_reason, AND action

    The saved Pipeline object contains both preprocessing and the classifier,
    so inference never needs to replicate preprocessing logic separately.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute and print a full set of classification metrics."""
    y_pred  = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)
    auc  = roc_auc_score(y, y_proba)
    ll   = log_loss(y, y_proba)
    cm   = confusion_matrix(y, y_pred)

    print(f"\n{'─'*50}")
    print(f"  {name} metrics")
    print(f"{'─'*50}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1         : {f1:.4f}")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  Log Loss   : {ll:.4f}")
    print(f"  Confusion matrix (rows=actual, cols=predicted):")
    print(f"    {cm[0]}  ← actual 0 (failed)")
    print(f"    {cm[1]}  ← actual 1 (completed)")

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "log_loss": round(ll, 4),
    }


def evaluate_by_action(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """
    Evaluate predictions separately for PAYMENT_RETRY and SEND_REMINDER.
    This verifies the model learned genuinely different recovery probabilities
    for the two interventions rather than ignoring the action feature.
    """
    results = {}
    print(f"\n{'─'*50}")
    print("  Action-specific evaluation (test set)")
    print(f"{'─'*50}")

    for action in ("PAYMENT_RETRY", "SEND_REMINDER"):
        mask = X_test["action"] == action
        X_sub = X_test[mask]
        y_sub = y_test[mask]

        if len(X_sub) == 0:
            print(f"  {action}: no examples in test set")
            continue

        y_proba = pipeline.predict_proba(X_sub)[:, 1]
        actual_rate = float(y_sub.mean())
        avg_prob    = float(y_proba.mean())

        auc_str = "n/a"
        auc_val = None
        if len(y_sub.unique()) == 2:
            auc_val = round(float(roc_auc_score(y_sub, y_proba)), 4)
            auc_str = str(auc_val)

        print(f"\n  {action}")
        print(f"    Examples          : {len(X_sub)}")
        print(f"    Actual success    : {actual_rate:.4f}")
        print(f"    Avg predicted P   : {avg_prob:.4f}")
        print(f"    ROC-AUC           : {auc_str}")

        results[action] = {
            "n": len(X_sub),
            "actual_success_rate": round(actual_rate, 4),
            "avg_predicted_probability": round(avg_prob, 4),
            "roc_auc": auc_val,
        }

    return results


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train():
    print("=" * 55)
    print("  RecoveryOS — XGBoost Training Pipeline")
    print("=" * 55)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\nLoading training data from database...")
    df, customer_ids = build_training_dataframe_with_customers()

    print(f"  Total rows          : {len(df)}")
    print(f"  Unique customers    : {len(set(customer_ids))}")
    print(f"\n  Target distribution:")
    print(f"    completed (1)     : {(df['target']==1).sum()}")
    print(f"    failed    (0)     : {(df['target']==0).sum()}")
    print(f"\n  Action distribution:")
    print(f"    PAYMENT_RETRY     : {(df['action']=='PAYMENT_RETRY').sum()}")
    print(f"    SEND_REMINDER     : {(df['action']=='SEND_REMINDER').sum()}")

    # ------------------------------------------------------------------
    # 2. Group-aware split (customer_id used only here, never as a feature)
    # ------------------------------------------------------------------
    print("\nPerforming group-aware train/val/test split (70/15/15)...")
    X_train, X_val, X_test, y_train, y_val, y_test = group_aware_split(
        df, customer_ids
    )
    print(f"  Train rows          : {len(X_train)}")
    print(f"  Validation rows     : {len(X_val)}")
    print(f"  Test rows           : {len(X_test)}")

    # ------------------------------------------------------------------
    # 3. Class imbalance — computed from training split ONLY
    # ------------------------------------------------------------------
    neg_train = int((y_train == 0).sum())
    pos_train = int((y_train == 1).sum())
    ratio = neg_train / pos_train if pos_train > 0 else 1.0
    # Only apply scale_pos_weight if imbalance is meaningful (ratio > 1.5)
    scale_pos_weight = round(ratio, 4) if ratio > 1.5 else 1.0
    print(f"\n  Class imbalance (train only):")
    print(f"    Negative (failed)   : {neg_train}")
    print(f"    Positive (completed): {pos_train}")
    print(f"    scale_pos_weight    : {scale_pos_weight}")

    # ------------------------------------------------------------------
    # 4. Build and train pipeline
    # ------------------------------------------------------------------
    print("\nTraining XGBoost pipeline...")
    pipeline = build_pipeline(scale_pos_weight)
    pipeline.fit(X_train, y_train)
    print("  Training complete.")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    val_metrics  = evaluate("Validation", pipeline, X_val,  y_val)
    test_metrics = evaluate("Test",       pipeline, X_test, y_test)

    # Predicted probability distribution on test set
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    print(f"\n  Predicted probability distribution (test set):")
    print(f"    min    : {test_proba.min():.4f}")
    print(f"    mean   : {test_proba.mean():.4f}")
    print(f"    median : {float(np.median(test_proba)):.4f}")
    print(f"    max    : {test_proba.max():.4f}")

    # Action-specific evaluation
    action_metrics = evaluate_by_action(pipeline, X_test, y_test)

    # ------------------------------------------------------------------
    # 6. Save pipeline
    # ------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")

    # ------------------------------------------------------------------
    # 7. Save metadata
    # ------------------------------------------------------------------
    # Recover action categories from the fitted encoder
    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    action_idx  = CATEGORICAL_FEATURES.index("action")
    action_categories = list(cat_encoder.categories_[action_idx])

    metadata = {
        "model_version": "1.0.0",
        "training_row_count": len(X_train),
        "total_row_count": len(df),
        "feature_columns": FEATURE_COLUMNS,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "action_categories": action_categories,
        "random_seed": RANDOM_STATE,
        "split": {
            "strategy": "group_aware_by_customer",
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
            "train_frac": 0.70,
            "val_frac": 0.15,
            "test_frac": 0.15,
        },
        "class_imbalance": {
            "neg_train": neg_train,
            "pos_train": pos_train,
            "scale_pos_weight": scale_pos_weight,
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "action_metrics": action_metrics,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved → {METADATA_PATH}")

    # ------------------------------------------------------------------
    # 8. Quick inference smoke-test
    # ------------------------------------------------------------------
    print("\n  Smoke-test: predicting both actions for first test row...")
    sample_row = X_test.iloc[[0]].copy()

    probs = {}
    for action in ("PAYMENT_RETRY", "SEND_REMINDER"):
        row = sample_row.copy()
        row["action"] = action
        probs[action] = round(float(pipeline.predict_proba(row)[0, 1]), 4)

    print(f"    PAYMENT_RETRY  → {probs['PAYMENT_RETRY']}")
    print(f"    SEND_REMINDER  → {probs['SEND_REMINDER']}")

    assert 0.0 <= probs["PAYMENT_RETRY"] <= 1.0
    assert 0.0 <= probs["SEND_REMINDER"] <= 1.0
    print("  Probability range assertion passed.")

    # ------------------------------------------------------------------
    # 9. Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print("  Training Summary")
    print(f"{'='*55}")
    print(f"  Dataset size          : {len(df)}")
    print(f"  Train / Val / Test    : {len(X_train)} / {len(X_val)} / {len(X_test)}")
    print(f"  Val  ROC-AUC          : {val_metrics['roc_auc']}")
    print(f"  Test ROC-AUC          : {test_metrics['roc_auc']}")
    print(f"  Test F1               : {test_metrics['f1']}")
    print(f"  Test Log Loss         : {test_metrics['log_loss']}")
    print(f"  Model path            : {MODEL_PATH}")
    print(f"  Metadata path         : {METADATA_PATH}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    train()
