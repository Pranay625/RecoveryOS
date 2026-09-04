"""
Generate coherent synthetic data for RecoveryOS.

The generator creates:
    - customers
    - historical payments
    - recovery attempts
    - audit logs

The data is generated as a connected behavioral simulation rather than
as independent random rows.

Important:
    recovery_attempt.ml_probability is synthetic simulator metadata.
    It MUST NOT be used as an ML training feature.

Usage from backend/:

    python -m scripts.generate_synthetic_data --customers 10 --seed 42 --clear

    python -m scripts.generate_synthetic_data --customers 1000 --seed 42 --clear
"""

from __future__ import annotations

import argparse
import random
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Make the project root importable when this script is executed as a module.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.customer import Customer
from app.models.payment import Payment
from app.models.recovery_attempt import RecoveryAttempt
from app.models.audit_log import AuditLog


# ===========================================================================
# Configuration
# ===========================================================================

DEFAULT_CUSTOMERS = 1000

CURRENCIES = ["INR"]

PAYMENT_METHODS = [
    "card",
    "upi",
    "netbanking",
    "wallet",
]

FAILURE_REASONS = [
    "network_error",
    "insufficient_funds",
    "bank_declined",
    "authentication_failed",
    "expired_card",
]


# ===========================================================================
# Customer behavioral profiles
# ===========================================================================

PROFILES = {
    "reliable": {
        "reliability": 0.93,
        "activity": (25, 55),
        "spending_multiplier": 1.0,
        "retry_response": 0.82,
        "reminder_response": 0.62,
    },
    "frequent": {
        "reliability": 0.90,
        "activity": (45, 70),
        "spending_multiplier": 0.90,
        "retry_response": 0.78,
        "reminder_response": 0.72,
    },
    "inconsistent": {
        "reliability": 0.72,
        "activity": (20, 50),
        "spending_multiplier": 0.85,
        "retry_response": 0.55,
        "reminder_response": 0.50,
    },
    "high_risk": {
        "reliability": 0.55,
        "activity": (15, 40),
        "spending_multiplier": 0.75,
        "retry_response": 0.35,
        "reminder_response": 0.30,
    },
    "high_value": {
        "reliability": 0.84,
        "activity": (18, 45),
        "spending_multiplier": 2.2,
        "retry_response": 0.72,
        "reminder_response": 0.58,
    },
}


# ===========================================================================
# Utility functions
# ===========================================================================

def generate_id(prefix: str) -> str:
    """Generate a compact prefixed identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def choose_weighted(
    rng: random.Random,
    values: list[str],
    weights: list[float],
) -> str:
    return rng.choices(values, weights=weights, k=1)[0]


def generate_amount(
    rng: random.Random,
    spending_multiplier: float,
) -> Decimal:
    """
    Generate a transaction amount with a long-tail distribution.

    Values are intentionally bounded so synthetic data remains reasonable
    for a payment-recovery demo.
    """

    base = rng.lognormvariate(7.7, 0.75)
    amount = base * spending_multiplier

    amount = clamp(amount, 100.0, 15000.0)

    return Decimal(str(round(amount, 2)))


def choose_payment_method(
    rng: random.Random,
    profile_name: str,
) -> str:
    """
    Slightly vary payment-method preference by customer profile.
    """

    if profile_name == "reliable":
        weights = [0.35, 0.40, 0.15, 0.10]

    elif profile_name == "frequent":
        weights = [0.30, 0.45, 0.15, 0.10]

    elif profile_name == "inconsistent":
        weights = [0.40, 0.30, 0.15, 0.15]

    elif profile_name == "high_risk":
        weights = [0.45, 0.25, 0.15, 0.15]

    else:  # high_value
        weights = [0.50, 0.25, 0.20, 0.05]

    return choose_weighted(rng, PAYMENT_METHODS, weights)


def choose_failure_reason(
    rng: random.Random,
    payment_method: str,
    profile_name: str,
) -> str:
    """
    Generate a failure reason with some relationship to payment method.
    """

    if payment_method == "card":
        reasons = [
            "bank_declined",
            "expired_card",
            "authentication_failed",
            "insufficient_funds",
            "network_error",
        ]

        weights = [
            0.30,
            0.12,
            0.18,
            0.25,
            0.15,
        ]

    elif payment_method == "upi":
        reasons = [
            "insufficient_funds",
            "bank_declined",
            "authentication_failed",
            "network_error",
        ]

        weights = [
            0.30,
            0.25,
            0.10,
            0.35,
        ]

    elif payment_method == "netbanking":
        reasons = [
            "bank_declined",
            "authentication_failed",
            "network_error",
            "insufficient_funds",
        ]

        weights = [
            0.30,
            0.20,
            0.35,
            0.15,
        ]

    else:  # wallet
        reasons = [
            "insufficient_funds",
            "authentication_failed",
            "network_error",
            "bank_declined",
        ]

        weights = [
            0.30,
            0.15,
            0.40,
            0.15,
        ]

    # High-risk customers get a slightly greater chance of
    # financial/issuer-related failures.
    if profile_name == "high_risk":
        adjusted = list(weights)

        for index, reason in enumerate(reasons):
            if reason in {"bank_declined", "insufficient_funds"}:
                adjusted[index] += 0.05

        total = sum(adjusted)
        weights = [weight / total for weight in adjusted]

    return choose_weighted(rng, reasons, weights)


def calculate_payment_success_probability(
    rng: random.Random,
    profile_name: str,
    payment_method: str,
    amount: Decimal,
) -> float:
    """
    Generate a payment success probability.

    This is part of the synthetic world model only.
    """

    profile = PROFILES[profile_name]

    probability = profile["reliability"]

    # Payment-method effects.
    if payment_method == "upi":
        probability += 0.015

    elif payment_method == "card":
        probability += 0.005

    elif payment_method == "netbanking":
        probability -= 0.015

    elif payment_method == "wallet":
        probability -= 0.010

    # Very large payments are slightly more likely to fail.
    if amount >= 10000:
        probability -= 0.04

    elif amount >= 5000:
        probability -= 0.02

    # Small random variation.
    probability += rng.uniform(-0.025, 0.025)

    return clamp(probability, 0.20, 0.98)


def calculate_recovery_probability(
    profile_name: str,
    action: str,
    failure_reason: str,
    attempt_number: int,
    amount: Decimal,
) -> float:
    """
    Generate the hidden probability used by the simulator to decide
    whether a recovery attempt succeeds.

    IMPORTANT:
        This probability is NOT an ML feature.
        It represents the synthetic ground-truth mechanism.
    """

    profile = PROFILES[profile_name]

    if action == "PAYMENT_RETRY":
        probability = profile["retry_response"]

    elif action == "SEND_REMINDER":
        probability = profile["reminder_response"]

    else:
        return 0.0

    # Failure-specific effects.
    if failure_reason == "network_error":
        probability += 0.12 if action == "PAYMENT_RETRY" else 0.02

    elif failure_reason == "insufficient_funds":
        probability -= 0.08 if action == "PAYMENT_RETRY" else 0.02

    elif failure_reason == "bank_declined":
        probability -= 0.05

    elif failure_reason == "authentication_failed":
        probability -= 0.12 if action == "PAYMENT_RETRY" else 0.04

    elif failure_reason == "expired_card":
        probability -= 0.25 if action == "PAYMENT_RETRY" else 0.08

    # Repeated attempts become less effective.
    if attempt_number >= 2:
        probability -= 0.12

    # Very high-value payments are slightly harder to recover automatically.
    if amount >= 10000:
        probability -= 0.04

    return clamp(probability, 0.05, 0.95)


def choose_recovery_action(
    rng: random.Random,
    profile_name: str,
    failure_reason: str,
) -> str:
    """
    Choose a historical recovery action for the simulator.

    This is not the future LLM decision.
    The future system will decide using:
        XGBoost → Grok → Policy Engine.
    """

    profile = PROFILES[profile_name]

    retry_weight = profile["retry_response"]
    reminder_weight = profile["reminder_response"]

    if failure_reason == "network_error":
        retry_weight += 0.20

    elif failure_reason == "expired_card":
        retry_weight -= 0.30
        reminder_weight += 0.10

    elif failure_reason == "insufficient_funds":
        reminder_weight += 0.15

    return choose_weighted(
        rng,
        ["PAYMENT_RETRY", "SEND_REMINDER"],
        [max(retry_weight, 0.05), max(reminder_weight, 0.05)],
    )


# ===========================================================================
# Customer generation
# ===========================================================================

def generate_customer_profile(rng: random.Random) -> str:
    profile_names = list(PROFILES.keys())

    weights = [
        0.30,  # reliable
        0.20,  # frequent
        0.20,  # inconsistent
        0.15,  # high risk
        0.15,  # high value
    ]

    return choose_weighted(rng, profile_names, weights)


def generate_customer_name(
    rng: random.Random,
    customer_number: int,
) -> str:
    """
    Lightweight deterministic synthetic names.

    Faker is deliberately not required because names are not ML features.
    """

    first_names = [
        "Arun",
        "Vikram",
        "Rahul",
        "Karan",
        "Aditya",
        "Rohan",
        "Aman",
        "Nikhil",
        "Varun",
        "Sanjay",
        "Priya",
        "Ananya",
        "Sneha",
        "Meera",
        "Kavya",
        "Divya",
        "Neha",
        "Isha",
        "Pooja",
        "Aditi",
    ]

    last_names = [
        "Sharma",
        "Reddy",
        "Kumar",
        "Patel",
        "Rao",
        "Nair",
        "Iyer",
        "Mehta",
        "Singh",
        "Verma",
    ]

    return (
        f"{rng.choice(first_names)} "
        f"{rng.choice(last_names)} "
        f"{customer_number}"
    )


# ===========================================================================
# Database cleanup
# ===========================================================================

def clear_existing_data(session: Session) -> None:
    """
    Delete existing records in dependency order.

    --clear means the generated dataset becomes the total dataset.
    """

    print("Clearing existing synthetic data...")

    session.execute(delete(AuditLog))
    session.execute(delete(RecoveryAttempt))
    session.execute(delete(Payment))
    session.execute(delete(Customer))

    session.commit()

    print("Existing customer/payment/recovery/audit data cleared.")


# ===========================================================================
# Main generation logic
# ===========================================================================

def generate_dataset(
    session: Session,
    customer_count: int,
    seed: int,
) -> dict[str, int]:

    rng = random.Random(seed)

    reference_time = datetime.now()

    counts = {
        "customers": 0,
        "payments": 0,
        "captured_payments": 0,
        "failed_payments": 0,
        "recovery_attempts": 0,
        "successful_recoveries": 0,
        "failed_recoveries": 0,
        "audit_logs": 0,
    }

    for customer_number in range(1, customer_count + 1):

        # ---------------------------------------------------------------
        # 1. Create latent customer behavioral profile.
        #
        # This is intentionally NOT stored in the database.
        # It is only used by the simulator to create correlated behavior.
        # ---------------------------------------------------------------

        profile_name = generate_customer_profile(rng)
        profile = PROFILES[profile_name]

        customer_id = generate_id("CUST")

        customer = Customer(
            customer_id=customer_id,
            name=generate_customer_name(rng, customer_number),
            email=f"customer{customer_number}@example.com",
            contact=f"+919000{customer_number:06d}",
            created_at=reference_time - timedelta(days=rng.randint(180, 365)),
            recovery_opt_out=False,
        )

        session.add(customer)

        counts["customers"] += 1

        # ---------------------------------------------------------------
        # 2. Generate chronological payment history.
        # ---------------------------------------------------------------

        transaction_count = rng.randint(
            profile["activity"][0],
            profile["activity"][1],
        )

        # Generate timestamps first so the history is chronological.
        timestamps = []

        for _ in range(transaction_count):
            days_ago = rng.uniform(0, 180)
            timestamp = reference_time - timedelta(days=days_ago)
            timestamps.append(timestamp)

        timestamps.sort()

        for payment_index, payment_time in enumerate(timestamps):

            amount = generate_amount(
                rng,
                profile["spending_multiplier"],
            )

            payment_method = choose_payment_method(
                rng,
                profile_name,
            )

            success_probability = calculate_payment_success_probability(
                rng,
                profile_name,
                payment_method,
                amount,
            )

            captured = rng.random() < success_probability

            payment_id = generate_id("PAY")

            if captured:
                status = "captured"
                failure_reason = None

            else:
                status = "failed"

                failure_reason = choose_failure_reason(
                    rng,
                    payment_method,
                    profile_name,
                )

            payment = Payment(
                payment_id=payment_id,
                customer_id=customer_id,
                razorpay_order_id=None,
                razorpay_payment_id=None,
                amount=amount,
                currency="INR",
                payment_method=payment_method,
                status=status,
                failure_reason=failure_reason,
                attempt_number=1,
                created_at=payment_time,
            )

            session.add(payment)

            counts["payments"] += 1

            if captured:
                counts["captured_payments"] += 1
                continue

            counts["failed_payments"] += 1

            # -----------------------------------------------------------
            # 3. Failed payments may receive recovery attempts.
            # -----------------------------------------------------------

            # Some failures receive no automated recovery at all.
            if rng.random() > 0.75:
                continue

            recovery_attempt_count = 1

            # A subset gets a second recovery attempt.
            if rng.random() < 0.30:
                recovery_attempt_count = 2

            previous_action = None

            for recovery_index in range(recovery_attempt_count):

                attempt_number = recovery_index + 1

                # If there is a second attempt, prefer the alternate action.
                if recovery_index == 0:
                    action = choose_recovery_action(
                        rng,
                        profile_name,
                        failure_reason,
                    )
                else:
                    action = (
                        "SEND_REMINDER"
                        if previous_action == "PAYMENT_RETRY"
                        else "PAYMENT_RETRY"
                    )

                previous_action = action

                recovery_probability = calculate_recovery_probability(
                    profile_name,
                    action,
                    failure_reason,
                    attempt_number,
                    amount,
                )

                recovered = rng.random() < recovery_probability

                recovery_id = generate_id("REC")

                # Recovery is simulated some time after the payment failure.
                recovery_created_at = payment_time + timedelta(
                    hours=rng.randint(1, 48),
                    minutes=rng.randint(0, 59),
                )

                recovery_completed_at = recovery_created_at + timedelta(
                    minutes=rng.randint(1, 30)
                )

                # -------------------------------------------------------
                # IMPORTANT:
                #
                # ml_probability is simulator metadata.
                # It represents the hidden probability used to generate
                # the synthetic outcome.
                #
                # It MUST NOT be included in ML training features.
                # -------------------------------------------------------

                recovery_attempt = RecoveryAttempt(
                    recovery_id=recovery_id,
                    payment_id=payment_id,
                    action=action,
                    reason=f"synthetic_{failure_reason}",
                    ml_probability=round(
                        recovery_probability,
                        4,
                    ),
                    status="completed" if recovered else "failed",
                    created_at=recovery_created_at,
                    completed_at=recovery_completed_at,
                )

                session.add(recovery_attempt)

                counts["recovery_attempts"] += 1

                if recovered:
                    counts["successful_recoveries"] += 1
                    recovery_event = "recovery_succeeded"
                else:
                    counts["failed_recoveries"] += 1
                    recovery_event = "recovery_failed"

                # -------------------------------------------------------
                # 4. Audit log for every recovery attempt.
                # -------------------------------------------------------

                audit_log = AuditLog(
                    audit_id=generate_id("AUD"),
                    payment_id=payment_id,
                    recovery_id=recovery_id,
                    event=recovery_event,
                    agent_reason=(
                        f"Synthetic historical recovery using {action} "
                        f"for failure reason {failure_reason}."
                    ),
                    policy_result="approved",
                    timestamp=recovery_completed_at,
                )

                session.add(audit_log)

                counts["audit_logs"] += 1

                # If the first recovery succeeds, stop additional recovery.
                if recovered:
                    break

    session.commit()

    return counts


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic RecoveryOS data."
    )

    parser.add_argument(
        "--customers",
        type=int,
        default=DEFAULT_CUSTOMERS,
        help="Number of customers to generate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing customer/payment/recovery/audit data first.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.customers <= 0:
        raise ValueError("--customers must be greater than 0.")

    print("=" * 70)
    print("RecoveryOS Synthetic Data Generator")
    print("=" * 70)
    print(f"Customers requested : {args.customers}")
    print(f"Random seed         : {args.seed}")
    print(f"Clear existing data : {args.clear}")
    print()

    session = SessionLocal()

    try:
        if args.clear:
            clear_existing_data(session)

        counts = generate_dataset(
            session=session,
            customer_count=args.customers,
            seed=args.seed,
        )

        print()
        print("=" * 70)
        print("Generation complete")
        print("=" * 70)

        print(f"Customers           : {counts['customers']}")
        print(f"Payments            : {counts['payments']}")
        print(f"  Captured          : {counts['captured_payments']}")
        print(f"  Failed            : {counts['failed_payments']}")
        print(f"Recovery attempts   : {counts['recovery_attempts']}")
        print(f"  Successful        : {counts['successful_recoveries']}")
        print(f"  Failed            : {counts['failed_recoveries']}")
        print(f"Audit logs          : {counts['audit_logs']}")

        if counts["recovery_attempts"] > 0:
            recovery_rate = (
                counts["successful_recoveries"]
                / counts["recovery_attempts"]
            ) * 100

            print(
                f"Recovery success    : "
                f"{recovery_rate:.2f}%"
            )

        print("=" * 70)

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()


if __name__ == "__main__":
    main()

