
"""
RecoveryOS - Synthetic Data Generator

Generates coherent synthetic data for:
    customers
    payments
    recovery_attempts
    audit_logs

Run from backend/:

    python -m scripts.generate_synthetic_data --customers 1000 --seed 42 --clear

Important data semantics:
    payments.status
        = outcome of the payment attempt represented by that payment row.

    recovery_attempts.status
        = outcome of the RecoveryOS intervention.

Therefore:
    payment.status = "failed"
    recovery_attempt.status = "completed"

is valid. It means the original payment attempt failed,
but RecoveryOS subsequently recovered the payment.

ml_probability is simulator metadata only.
It must NEVER be used as an ML training feature.
"""

import argparse
import random
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import delete

from app.db.session import SessionLocal
from app.models import (
    Customer,
    Payment,
    RecoveryAttempt,
    AuditLog,
)


# ============================================================
# CONFIGURATION
# ============================================================

PAYMENT_METHODS = [
    "upi",
    "card",
    "netbanking",
    "wallet",
]

FAILURE_REASONS = [
    "insufficient_funds",
    "bank_declined",
    "network_error",
    "authentication_failed",
    "expired_card",
]

CURRENCIES = ["INR"]

PROFILE_TYPES = [
    "reliable",
    "frequent",
    "inconsistent",
    "high_risk",
    "high_value",
]


# ============================================================
# CUSTOMER PROFILES
# ============================================================

PROFILES = {
    "reliable": {
        "payment_count_range": (20, 60),
        "success_probability": 0.90,
        "recovery_probability": 0.80,
        "failure_weights": {
            "insufficient_funds": 0.10,
            "bank_declined": 0.20,
            "network_error": 0.30,
            "authentication_failed": 0.25,
            "expired_card": 0.15,
        },
        "amount_range": (500, 10000),
    },

    "frequent": {
        "payment_count_range": (40, 70),
        "success_probability": 0.84,
        "recovery_probability": 0.75,
        "failure_weights": {
            "insufficient_funds": 0.25,
            "bank_declined": 0.15,
            "network_error": 0.30,
            "authentication_failed": 0.20,
            "expired_card": 0.10,
        },
        "amount_range": (200, 6000),
    },

    "inconsistent": {
        "payment_count_range": (15, 45),
        "success_probability": 0.68,
        "recovery_probability": 0.55,
        "failure_weights": {
            "insufficient_funds": 0.30,
            "bank_declined": 0.25,
            "network_error": 0.20,
            "authentication_failed": 0.15,
            "expired_card": 0.10,
        },
        "amount_range": (500, 12000),
    },

    "high_risk": {
        "payment_count_range": (15, 40),
        "success_probability": 0.50,
        "recovery_probability": 0.35,
        "failure_weights": {
            "insufficient_funds": 0.40,
            "bank_declined": 0.30,
            "network_error": 0.10,
            "authentication_failed": 0.15,
            "expired_card": 0.05,
        },
        "amount_range": (300, 8000),
    },

    "high_value": {
        "payment_count_range": (10, 30),
        "success_probability": 0.78,
        "recovery_probability": 0.65,
        "failure_weights": {
            "insufficient_funds": 0.20,
            "bank_declined": 0.30,
            "network_error": 0.20,
            "authentication_failed": 0.20,
            "expired_card": 0.10,
        },
        "amount_range": (10000, 100000),
    },
}


# ============================================================
# HELPERS
# ============================================================

def weighted_choice(weights: dict) -> str:
    """Choose one item using a dictionary of probabilities."""
    items = list(weights.keys())
    probabilities = list(weights.values())
    return random.choices(items, weights=probabilities, k=1)[0]


def generate_customer_name(index: int) -> str:
    """Generate deterministic synthetic customer names."""
    first_names = [
        "Arun",
        "Rahul",
        "Vikram",
        "Kiran",
        "Aditya",
        "Rohan",
        "Nikhil",
        "Varun",
        "Aman",
        "Sanjay",
    ]

    last_names = [
        "Sharma",
        "Kumar",
        "Reddy",
        "Patel",
        "Rao",
        "Singh",
        "Mehta",
        "Iyer",
        "Nair",
        "Verma",
    ]

    first = first_names[index % len(first_names)]
    last = last_names[(index // len(first_names)) % len(last_names)]

    return f"{first} {last}"


def generate_amount(profile: dict) -> Decimal:
    low, high = profile["amount_range"]

    amount = random.uniform(low, high)

    # Round to nearest 50 for more realistic transaction amounts.
    amount = round(amount / 50) * 50

    return Decimal(str(amount)).quantize(Decimal("0.01"))


def generate_payment_method() -> str:
    return random.choice(PAYMENT_METHODS)


def generate_payment_id(index: int) -> str:
    return f"PAY_{index:08d}"


def generate_recovery_id(index: int) -> str:
    return f"REC_{index:08d}"


def generate_audit_id(index: int) -> str:
    return f"AUD_{index:08d}"


# ============================================================
# RECOVERY SIMULATION
# ============================================================

def simulate_recovery(
    profile: dict,
    failure_reason: str,
    attempt_number: int,
) -> tuple[str, float, str]:
    """
    Simulate RecoveryOS intervention outcome.

    Returns:
        action
        ml_probability
        status
    """

    base_probability = profile["recovery_probability"]

    # Different failures respond differently to interventions.
    retry_modifier = {
        "network_error": 0.20,
        "authentication_failed": 0.05,
        "insufficient_funds": -0.05,
        "bank_declined": -0.10,
        "expired_card": -0.20,
    }.get(failure_reason, 0.0)

    reminder_modifier = {
        "network_error": 0.05,
        "authentication_failed": 0.10,
        "insufficient_funds": 0.15,
        "bank_declined": 0.05,
        "expired_card": -0.15,
    }.get(failure_reason, 0.0)

    retry_probability = min(
        max(base_probability + retry_modifier, 0.05),
        0.95,
    )

    reminder_probability = min(
        max(base_probability + reminder_modifier, 0.05),
        0.95,
    )

    # Avoid repeatedly retrying.
    if attempt_number >= 2:
        action = "SEND_REMINDER"
        probability = reminder_probability
    else:
        # Simulate the recovery system choosing between interventions.
        if retry_probability >= reminder_probability:
            action = "PAYMENT_RETRY"
            probability = retry_probability
        else:
            action = "SEND_REMINDER"
            probability = reminder_probability

    # Small penalty for repeated attempts.
    probability -= (attempt_number - 1) * 0.10
    probability = min(max(probability, 0.05), 0.95)

    status = (
        "completed"
        if random.random() < probability
        else "failed"
    )

    return action, round(probability, 4), status


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_data(
    customer_count: int,
    seed: int,
    clear_existing: bool,
):
    random.seed(seed)

    db = SessionLocal()

    try:
        # ----------------------------------------------------
        # CLEAR EXISTING DATA
        # ----------------------------------------------------

        if clear_existing:
            print("Clearing existing synthetic data...")

            db.execute(delete(AuditLog))
            db.execute(delete(RecoveryAttempt))
            db.execute(delete(Payment))
            db.execute(delete(Customer))

            db.commit()

        # ----------------------------------------------------
        # COUNTERS
        # ----------------------------------------------------

        payment_counter = 1
        recovery_counter = 1
        audit_counter = 1

        customers_created = 0
        payments_created = 0
        recoveries_created = 0
        audits_created = 0

        # ----------------------------------------------------
        # GENERATE CUSTOMERS
        # ----------------------------------------------------

        for customer_index in range(1, customer_count + 1):

            customer_id = f"CUST_{customer_index:06d}"

            profile_type = random.choice(PROFILE_TYPES)
            profile = PROFILES[profile_type]

            customer = Customer(
                customer_id=customer_id,
                name=generate_customer_name(customer_index),
                email=f"customer{customer_index}@example.com",
                contact=f"90000{customer_index:05d}",
                recovery_opt_out=False,
            )

            db.add(customer)

            customers_created += 1

            # ------------------------------------------------
            # PAYMENT HISTORY
            # ------------------------------------------------

            payment_count = random.randint(
                *profile["payment_count_range"]
            )

            # Spread transactions across approximately 6 months.
            start_date = datetime.utcnow() - timedelta(days=180)

            previous_payments = []

            for payment_index in range(payment_count):

                created_at = start_date + timedelta(
                    days=random.randint(0, 180),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                )

                amount = generate_amount(profile)
                payment_method = generate_payment_method()

                is_successful = (
                    random.random()
                    < profile["success_probability"]
                )

                # --------------------------------------------
                # SUCCESSFUL PAYMENT
                # --------------------------------------------

                if is_successful:

                    payment = Payment(
                        payment_id=generate_payment_id(payment_counter),
                        customer_id=customer_id,
                        razorpay_order_id=None,
                        razorpay_payment_id=None,
                        amount=amount,
                        currency="INR",
                        payment_method=payment_method,
                        status="captured",
                        failure_reason=None,
                        attempt_number=1,
                        created_at=created_at,
                    )

                    db.add(payment)

                    previous_payments.append(payment)

                    payment_counter += 1
                    payments_created += 1

                    continue

                # --------------------------------------------
                # FAILED PAYMENT
                # --------------------------------------------

                failure_reason = weighted_choice(
                    profile["failure_weights"]
                )

                payment = Payment(
                    payment_id=generate_payment_id(payment_counter),
                    customer_id=customer_id,
                    razorpay_order_id=None,
                    razorpay_payment_id=None,
                    amount=amount,
                    currency="INR",
                    payment_method=payment_method,
                    status="failed",
                    failure_reason=failure_reason,
                    attempt_number=1,
                    created_at=created_at,
                )

                db.add(payment)
                previous_payments.append(payment)

                payment_counter += 1
                payments_created += 1

                # --------------------------------------------
                # RECOVERY ATTEMPTS
                # --------------------------------------------

                max_recovery_attempts = random.choice([1, 1, 1, 2])

                recovery_succeeded = False

                for attempt_number in range(
                    1,
                    max_recovery_attempts + 1,
                ):

                    if recovery_succeeded:
                        break

                    action, probability, recovery_status = (
                        simulate_recovery(
                            profile=profile,
                            failure_reason=failure_reason,
                            attempt_number=attempt_number,
                        )
                    )

                    recovery_id = generate_recovery_id(
                        recovery_counter
                    )

                    recovery_created_at = created_at + timedelta(
                        minutes=random.randint(5, 240)
                    )

                    completed_at = (
                        recovery_created_at
                        + timedelta(minutes=random.randint(1, 30))
                        if recovery_status == "completed"
                        else None
                    )

                    recovery_attempt = RecoveryAttempt(
                        recovery_id=recovery_id,
                        payment_id=payment.payment_id,
                        action=action,
                        reason=(
                            f"Synthetic recovery simulation for "
                            f"{failure_reason}"
                        ),
                        ml_probability=Decimal(
                            str(probability)
                        ),
                        status=recovery_status,
                        created_at=recovery_created_at,
                        completed_at=completed_at,
                    )

                    db.add(recovery_attempt)

                    recoveries_created += 1

                    # ----------------------------------------
                    # AUDIT LOG
                    # ----------------------------------------

                    audit_event = (
                        "recovery_completed"
                        if recovery_status == "completed"
                        else "recovery_failed"
                    )

                    policy_result = (
                        "approved"
                        if recovery_status == "completed"
                        else "approved_execution_failed"
                    )

                    audit_log = AuditLog(
                        audit_id=generate_audit_id(audit_counter),
                        payment_id=payment.payment_id,
                        recovery_id=recovery_id,
                        event=audit_event,
                        agent_reason=(
                            f"Selected {action} for "
                            f"{failure_reason}"
                        ),
                        policy_result=policy_result,
                        timestamp=recovery_created_at,
                    )

                    db.add(audit_log)

                    audits_created += 1

                    recovery_counter += 1
                    audit_counter += 1

                    # ----------------------------------------
                    # STOP AFTER SUCCESS
                    # ----------------------------------------

                    if recovery_status == "completed":
                        recovery_succeeded = True

            # ------------------------------------------------
            # COMMIT CUSTOMER BATCH
            # ------------------------------------------------

            db.commit()

            if customer_index % 100 == 0:
                print(
                    f"Generated {customer_index}/"
                    f"{customer_count} customers..."
                )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print("\nSynthetic data generation complete.")
        print("--------------------------------")
        print(f"Customers created       : {customers_created}")
        print(f"Payments created        : {payments_created}")
        print(f"Recovery attempts       : {recoveries_created}")
        print(f"Audit logs created      : {audits_created}")
        print("--------------------------------")
        print(f"Random seed             : {seed}")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic RecoveryOS data."
    )

    parser.add_argument(
        "--customers",
        type=int,
        default=1000,
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
        help="Clear existing data before generation.",
    )

    args = parser.parse_args()

    generate_data(
        customer_count=args.customers,
        seed=args.seed,
        clear_existing=args.clear,
    )


if __name__ == "__main__":
    main()

