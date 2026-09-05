"""
RecoveryOS - Phase 8/9: Razorpay Service

Responsible ONLY for communication with the Razorpay API.

This module:
  - Reads credentials from app settings (never hardcoded).
  - Initialises the Razorpay client lazily (no network call on import).
  - Exposes create_order() for creating a Razorpay Test Mode Order.
  - Exposes verify_payment_signature() for server-side Checkout verification.
  - Converts rupee amounts to paise (smallest INR unit) before sending.
  - Returns the raw Razorpay order dict for the caller to use.

This module does NOT:
  - Contain ML, Gemini, or policy logic.
  - Write to the database.
  - Create RecoveryAttempt or AuditLog records.
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from app.core.config import settings


# ---------------------------------------------------------------------------
# Application-level exception
# ---------------------------------------------------------------------------

class RazorpayServiceError(Exception):
    """
    Raised for any failure in the Razorpay service layer.
    The Razorpay secret is never included in the message.
    """


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

class RazorpayOrder(BaseModel):
    """Subset of the Razorpay order object needed by the frontend checkout."""
    id: str
    amount: int          # in smallest currency unit (paise for INR)
    currency: str
    receipt: str
    status: str


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class RazorpayService:
    """
    Thin wrapper around the Razorpay Python SDK.

    The client is initialised lazily so that:
      - importing this module never makes a network call
      - tests can inject a mock client via _client before calling methods
    """

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not settings.RAZORPAY_KEY_ID or not settings.RAZORPAY_KEY_SECRET:
            raise RazorpayServiceError(
                "RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set in .env "
                "before using the Razorpay service."
            )

        try:
            import razorpay  # noqa: PLC0415
        except ImportError as exc:
            raise RazorpayServiceError(
                "razorpay package is not installed. Run: pip install razorpay"
            ) from exc

        self._client = razorpay.Client(
            auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET)
        )
        return self._client

    def create_order(
        self,
        amount_inr: float,
        currency: str = "INR",
        receipt: str = "rcpt_recovery",
    ) -> RazorpayOrder:
        """
        Create a Razorpay Test Mode Order.

        Args:
            amount_inr: Amount in rupees (e.g. 2499.0). Converted to paise
                        internally before sending to Razorpay.
            currency:   ISO 4217 currency code. Defaults to "INR".
            receipt:    Merchant receipt identifier. Defaults to a generic value.

        Returns:
            RazorpayOrder with the fields needed by the frontend checkout.

        Raises:
            RazorpayServiceError: for missing credentials, SDK import failure,
                                  or any Razorpay API error.
        """
        client = self._get_client()

        # Razorpay expects the smallest currency unit.
        # For INR: 1 rupee = 100 paise. Use ceil to avoid rounding down.
        amount_paise = math.ceil(amount_inr * 100)

        order_data = {
            "amount": amount_paise,
            "currency": currency,
            "receipt": receipt,
            "payment_capture": 1,   # auto-capture on payment success
        }

        try:
            order = client.order.create(data=order_data)
        except Exception as exc:
            # Do not include exc details that might echo back credentials.
            raise RazorpayServiceError(
                f"Razorpay order creation failed: {type(exc).__name__}"
            ) from exc

        try:
            return RazorpayOrder(
                id=order["id"],
                amount=order["amount"],
                currency=order["currency"],
                receipt=order["receipt"],
                status=order["status"],
            )
        except (KeyError, TypeError) as exc:
            raise RazorpayServiceError(
                f"Unexpected Razorpay order response shape: {exc}"
            ) from exc

    def verify_payment_signature(
        self,
        razorpay_order_id: str,
        razorpay_payment_id: str,
        razorpay_signature: str,
    ) -> None:
        """
        Verify the Razorpay Checkout payment signature server-side.

        Uses the Razorpay SDK's utility.verify_payment_signature() which
        performs a local HMAC-SHA256 check — no network call is made.

        Args:
            razorpay_order_id:   The order ID returned by create_order().
            razorpay_payment_id: The payment ID returned by Razorpay Checkout.
            razorpay_signature:  The signature returned by Razorpay Checkout.

        Returns:
            None on success.

        Raises:
            RazorpayServiceError: if the signature is invalid or verification
                                  fails for any reason.  The key secret is
                                  never included in the error message.
        """
        client = self._get_client()
        params = {
            "razorpay_order_id":   razorpay_order_id,
            "razorpay_payment_id": razorpay_payment_id,
            "razorpay_signature":  razorpay_signature,
        }
        try:
            client.utility.verify_payment_signature(params)
        except Exception as exc:
            # The SDK raises SignatureVerificationError on mismatch.
            # We catch all exceptions to avoid leaking SDK internals or
            # credentials in the error message.
            raise RazorpayServiceError(
                f"Payment signature verification failed: {type(exc).__name__}"
            ) from exc


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

razorpay_service = RazorpayService()
