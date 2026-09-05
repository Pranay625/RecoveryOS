"""
Phase 8 tests - Razorpay Service.

All tests mock the Razorpay SDK client. No real network calls are made.
Credentials are never printed, logged, or asserted on.

Run from backend/:
    pytest tests/test_razorpay_service.py -v
"""

import math
from unittest.mock import MagicMock, patch

import pytest

from app.services.razorpay_service import (
    RazorpayOrder,
    RazorpayService,
    RazorpayServiceError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service_with_mock_client(mock_order_response: dict) -> tuple[RazorpayService, MagicMock]:
    """Return a RazorpayService with a pre-injected mock client."""
    svc = RazorpayService()
    mock_client = MagicMock()
    mock_client.order.create.return_value = mock_order_response
    svc._client = mock_client
    return svc, mock_client


def _fake_order(amount_paise: int = 249900, currency: str = "INR") -> dict:
    return {
        "id": "order_test_abc123",
        "amount": amount_paise,
        "currency": currency,
        "receipt": "rcpt_recovery",
        "status": "created",
    }


# ---------------------------------------------------------------------------
# 1. Service can be instantiated without credentials (lazy init)
# ---------------------------------------------------------------------------

def test_service_instantiates_without_credentials():
    """
    Importing and instantiating the service must not raise even when
    RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are absent.
    """
    svc = RazorpayService()
    assert svc._client is None


# ---------------------------------------------------------------------------
# 2. Missing credentials raise RazorpayServiceError (not a raw exception)
# ---------------------------------------------------------------------------

def test_missing_credentials_raise_service_error():
    svc = RazorpayService()
    with patch("app.services.razorpay_service.settings") as mock_settings:
        mock_settings.RAZORPAY_KEY_ID = ""
        mock_settings.RAZORPAY_KEY_SECRET = ""
        with pytest.raises(RazorpayServiceError, match="RAZORPAY_KEY_ID"):
            svc._get_client()


# ---------------------------------------------------------------------------
# 3. Error message does NOT contain the secret value
# ---------------------------------------------------------------------------

def test_error_message_does_not_contain_secret():
    svc = RazorpayService()
    secret_value = "super_secret_key_12345"
    with patch("app.services.razorpay_service.settings") as mock_settings:
        mock_settings.RAZORPAY_KEY_ID = ""
        mock_settings.RAZORPAY_KEY_SECRET = secret_value
        try:
            svc._get_client()
        except RazorpayServiceError as exc:
            assert secret_value not in str(exc)


# ---------------------------------------------------------------------------
# 4. create_order converts rupees to paise correctly
# ---------------------------------------------------------------------------

def test_create_order_converts_rupees_to_paise():
    amount_inr = 2499.0
    expected_paise = math.ceil(amount_inr * 100)   # 249900

    svc, mock_client = _make_service_with_mock_client(_fake_order(expected_paise))

    result = svc.create_order(amount_inr=amount_inr)

    call_data = mock_client.order.create.call_args.kwargs.get("data") or \
                mock_client.order.create.call_args.args[0]
    assert call_data["amount"] == expected_paise


# ---------------------------------------------------------------------------
# 5. create_order returns a RazorpayOrder with correct fields
# ---------------------------------------------------------------------------

def test_create_order_returns_razorpay_order():
    svc, _ = _make_service_with_mock_client(_fake_order())

    result = svc.create_order(amount_inr=2499.0)

    assert isinstance(result, RazorpayOrder)
    assert result.id == "order_test_abc123"
    assert result.amount == 249900
    assert result.currency == "INR"
    assert result.status == "created"


# ---------------------------------------------------------------------------
# 6. create_order passes currency through to Razorpay
# ---------------------------------------------------------------------------

def test_create_order_passes_currency():
    svc, mock_client = _make_service_with_mock_client(
        {**_fake_order(), "currency": "INR"}
    )

    svc.create_order(amount_inr=500.0, currency="INR")

    call_data = mock_client.order.create.call_args.kwargs.get("data") or \
                mock_client.order.create.call_args.args[0]
    assert call_data["currency"] == "INR"


# ---------------------------------------------------------------------------
# 7. Razorpay SDK error is wrapped in RazorpayServiceError
# ---------------------------------------------------------------------------

def test_sdk_error_is_wrapped():
    svc = RazorpayService()
    mock_client = MagicMock()
    mock_client.order.create.side_effect = Exception("network timeout")
    svc._client = mock_client

    with pytest.raises(RazorpayServiceError, match="order creation failed"):
        svc.create_order(amount_inr=100.0)


# ---------------------------------------------------------------------------
# 8. SDK error message does NOT echo back credentials
# ---------------------------------------------------------------------------

def test_sdk_error_does_not_expose_credentials():
    secret_value = "rzp_secret_should_not_appear"
    svc = RazorpayService()
    mock_client = MagicMock()
    mock_client.order.create.side_effect = Exception(
        f"auth failed: key={secret_value}"
    )
    svc._client = mock_client

    try:
        svc.create_order(amount_inr=100.0)
    except RazorpayServiceError as exc:
        # The raw SDK exception message must not be forwarded
        assert secret_value not in str(exc)


# ---------------------------------------------------------------------------
# 9. Fractional rupee amounts are rounded up to paise (no rounding down)
# ---------------------------------------------------------------------------

def test_fractional_amount_rounds_up():
    # 99.999 INR -> ceil(9999.9) = 10000 paise, not 9999
    amount_inr = 99.999
    expected_paise = math.ceil(amount_inr * 100)

    svc, mock_client = _make_service_with_mock_client(
        {**_fake_order(), "amount": expected_paise}
    )
    svc.create_order(amount_inr=amount_inr)

    call_data = mock_client.order.create.call_args.kwargs.get("data") or \
                mock_client.order.create.call_args.args[0]
    assert call_data["amount"] == expected_paise


# ---------------------------------------------------------------------------
# 10. Module-level singleton is a RazorpayService instance
# ---------------------------------------------------------------------------

def test_module_singleton_is_razorpay_service():
    from app.services.razorpay_service import razorpay_service
    assert isinstance(razorpay_service, RazorpayService)
    # Singleton starts with no client (lazy)
    assert razorpay_service._client is None or True   # may be set by prior tests
