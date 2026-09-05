// Mirrors backend app/schemas/recovery.py exactly

// ── /execute ─────────────────────────────────────────────────────────────────
// RecoveryExecuteRequest is identical to RecoveryPredictionRequest on the wire

export interface RecoveryExecuteResponse {
  customer_id: string;
  action: string;
  allowed: boolean;
  execution_status: string; // READY_FOR_PAYMENT | REMINDER_REQUIRED | MANUAL_REVIEW | BLOCKED
  policy_reason: string;
  razorpay_order_id: string | null;
  amount_paise: number | null;
  currency: string | null;
  runtime_payment_id: string | null;
  runtime_recovery_id: string | null;
}

// ── /payment-result ───────────────────────────────────────────────────────────

export interface PaymentResultRequest {
  razorpay_payment_id: string;
  razorpay_order_id: string;
  razorpay_signature: string;
}

export interface PaymentResultResponse {
  status: string;
  razorpay_payment_id: string;
  razorpay_order_id: string;
  runtime_payment_id: string;
  runtime_recovery_id: string;
}

// ── /payment-failed ───────────────────────────────────────────────────────────

export interface PaymentFailedRequest {
  razorpay_order_id: string;
  error_code: string | null;
  error_description: string | null;
}

export interface PaymentFailedResponse {
  status: string;
  razorpay_order_id: string;
  runtime_payment_id: string;
  runtime_recovery_id: string;
}

// ── /predict ──────────────────────────────────────────────────────────────────

export interface RecoveryPredictionRequest {
  customer_id: string;
  amount: number;
  currency: string;
  payment_method: string;
  failure_reason: string;
  attempt_number: number;
}

export interface MLPredictionResult {
  PAYMENT_RETRY: number;
  SEND_REMINDER: number;
}

export interface PolicyDecisionResult {
  action: string;
  allowed: boolean;
  reason: string;
}

export interface RecoveryPredictionResponse {
  customer_id: string;
  customer_exists: boolean;
  ml_predictions: MLPredictionResult;
  recommended_action: string;
  confidence: number;
  reason: string;
  policy: PolicyDecisionResult;
}
