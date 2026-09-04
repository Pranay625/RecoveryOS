"""
Phase 4A validation script.

Run from backend/:
    python -m scripts.validate_features
"""

from app.db.session import SessionLocal
from app.models.recovery_attempt import RecoveryAttempt
from app.ml.feature_engineering import build_training_dataframe

EXPECTED_COLUMNS = [
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
    "action",
    "target",
]


def main():
    # ----------------------------------------------------------------
    # Count recovery attempts directly from DB
    # ----------------------------------------------------------------
    db = SessionLocal()
    try:
        total_recoveries = db.query(RecoveryAttempt).count()
    finally:
        db.close()

    print("Building training DataFrame...")
    df = build_training_dataframe()

    # ----------------------------------------------------------------
    # Basic info
    # ----------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  DataFrame shape      : {df.shape}")
    print(f"  Columns ({len(df.columns)})          : {list(df.columns)}")

    # ----------------------------------------------------------------
    # First 10 rows
    # ----------------------------------------------------------------
    print(f"\n--- First 10 rows ---")
    print(df.head(10).to_string(index=False))

    # ----------------------------------------------------------------
    # Distributions
    # ----------------------------------------------------------------
    print(f"\n--- Target distribution ---")
    print(df["target"].value_counts().to_string())
    print(f"  (0=failed, 1=completed)")

    print(f"\n--- Action distribution ---")
    print(df["action"].value_counts().to_string())

    # ----------------------------------------------------------------
    # Missing values
    # ----------------------------------------------------------------
    print(f"\n--- Missing value counts ---")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() if missing.any() else "  None")

    # ----------------------------------------------------------------
    # Assertions
    # ----------------------------------------------------------------
    print(f"\n{'='*55}")
    print("Running assertions...")

    errors = []

    # 1. Row count matches recovery attempts in DB
    if len(df) != total_recoveries:
        errors.append(
            f"Row count mismatch: DataFrame has {len(df)} rows "
            f"but DB has {total_recoveries} recovery attempts."
        )

    # 2. Exact column set
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    if extra_cols:
        errors.append(f"Unexpected extra columns: {extra_cols}")

    # 3. target contains only 0 and 1
    invalid_targets = set(df["target"].unique()) - {0, 1}
    if invalid_targets:
        errors.append(f"target contains unexpected values: {invalid_targets}")

    # 4. action contains only known values
    invalid_actions = set(df["action"].unique()) - {"PAYMENT_RETRY", "SEND_REMINDER"}
    if invalid_actions:
        errors.append(f"action contains unexpected values: {invalid_actions}")

    # 5. No forbidden leakage columns
    forbidden = {"customer_id", "payment_id", "recovery_id", "ml_probability"}
    leaked = forbidden & set(df.columns)
    if leaked:
        errors.append(f"Leakage columns present: {leaked}")

    # 6. success_rate and recovery_success_rate are in [0, 1]
    for col in ("success_rate", "recovery_success_rate"):
        if col in df.columns:
            out_of_range = df[(df[col] < 0) | (df[col] > 1)]
            if not out_of_range.empty:
                errors.append(f"{col} has values outside [0, 1]")

    # 7. days_since_last_success is either -1 or >= 0
    dsl = df["days_since_last_success"]
    bad_dsl = df[(dsl < -1)]
    if not bad_dsl.empty:
        errors.append("days_since_last_success has values < -1")

    if errors:
        print("\nASSERTION FAILURES:")
        for e in errors:
            print(f"  ✗ {e}")
        raise SystemExit(1)

    print("  All assertions passed.")
    print(f"\n{'='*55}")
    print(f"  Total recovery attempts in DB : {total_recoveries}")
    print(f"  Training rows produced        : {len(df)}")
    print(f"  Target=1 (completed)          : {(df['target']==1).sum()}")
    print(f"  Target=0 (failed)             : {(df['target']==0).sum()}")
    print(f"  PAYMENT_RETRY rows            : {(df['action']=='PAYMENT_RETRY').sum()}")
    print(f"  SEND_REMINDER rows            : {(df['action']=='SEND_REMINDER').sum()}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
