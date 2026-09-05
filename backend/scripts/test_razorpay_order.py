"""
RecoveryOS - Phase 8: Razorpay Test Mode connectivity verification.

Makes ONE real API call to Razorpay Test Mode to confirm that:
  - The credentials in .env are valid.
  - RazorpayService.create_order() works end-to-end.

This script:
  - Does NOT modify the database.
  - Does NOT create RecoveryAttempt or AuditLog records.
  - Does NOT print RAZORPAY_KEY_SECRET or any credential value.
  - Does NOT implement any endpoint or application feature.

Run from backend/ (Windows):
    .venv/Scripts/python.exe scripts/test_razorpay_order.py
"""

import sys
import time

# Ensure backend/ is on the path when run as a plain script.
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.razorpay_service import RazorpayServiceError, razorpay_service

AMOUNT_INR = 10.0
CURRENCY   = "INR"
RECEIPT    = f"recoveryos_test_{int(time.time())}"

print("RecoveryOS - Razorpay Test Mode Verification")
print("---------------------------------------------")
print(f"Amount  : Rs. {AMOUNT_INR}")
print(f"Currency: {CURRENCY}")
print(f"Receipt : {RECEIPT}")
print()

try:
    order = razorpay_service.create_order(
        amount_inr=AMOUNT_INR,
        currency=CURRENCY,
        receipt=RECEIPT,
    )
except RazorpayServiceError as exc:
    # Safe message only — the service never embeds credentials in this exception.
    print(f"FAILED  : {exc}")
    sys.exit(1)
except Exception as exc:
    # Unexpected error — print only the type, not the message.
    print(f"FAILED  : Unexpected error ({type(exc).__name__})")
    sys.exit(1)

print("Razorpay Test Mode Order Created")
print(f"Order ID : {order.id}")
print(f"Amount   : {order.amount} paise  (Rs. {order.amount / 100:.2f})")
print(f"Currency : {order.currency}")
print(f"Status   : {order.status}")
print(f"Receipt  : {order.receipt}")
print()
print("No database changes were made.")
