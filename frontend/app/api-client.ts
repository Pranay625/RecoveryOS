import type {
  RecoveryPredictionRequest,
  RecoveryPredictionResponse,
  RecoveryExecuteResponse,
  PaymentResultRequest,
  PaymentResultResponse,
  PaymentFailedRequest,
  PaymentFailedResponse,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const b = await res.json();
      if (b?.detail) detail = String(b.detail);
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export function predict(req: RecoveryPredictionRequest) {
  return post<RecoveryPredictionResponse>("/api/recovery/predict", req);
}

export function execute(req: RecoveryPredictionRequest) {
  return post<RecoveryExecuteResponse>("/api/recovery/execute", req);
}

export function recordPaymentResult(req: PaymentResultRequest) {
  return post<PaymentResultResponse>("/api/recovery/payment-result", req);
}

export function recordPaymentFailed(req: PaymentFailedRequest) {
  return post<PaymentFailedResponse>("/api/recovery/payment-failed", req);
}
