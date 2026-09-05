"use client";

import { useState } from "react";
import { predict, execute, recordPaymentResult, recordPaymentFailed } from "./api-client";
import type {
  RecoveryPredictionRequest,
  RecoveryPredictionResponse,
  RecoveryExecuteResponse,
  PaymentResultResponse,
} from "./types";
import "./razorpay.d.ts";

// ── constants ────────────────────────────────────────────────────────────────

const PAYMENT_METHODS = ["upi", "card", "netbanking", "wallet", "emi"];
const FAILURE_REASONS = [
  "insufficient_funds",
  "card_declined",
  "bank_timeout",
  "invalid_details",
  "authentication_failed",
  "network_error",
  "other",
];

const DEFAULT_FORM: RecoveryPredictionRequest = {
  customer_id: "CUST_000001",
  amount: 2499,
  currency: "INR",
  payment_method: "upi",
  failure_reason: "insufficient_funds",
  attempt_number: 1,
};

const RZP_KEY = process.env.NEXT_PUBLIC_RAZORPAY_KEY_ID ?? "";

// ── payment flow state ───────────────────────────────────────────────────────

type PaymentPhase =
  | "IDLE"
  | "CREATING_ORDER"
  | "CHECKOUT_OPEN"
  | "VERIFYING"
  | "SUCCESS"
  | "FAILED";

interface PaymentState {
  phase: PaymentPhase;
  executeResponse: RecoveryExecuteResponse | null;
  successResponse: PaymentResultResponse | null;
  failureReason: string | null;
  error: string | null;
}

const PAYMENT_IDLE: PaymentState = {
  phase: "IDLE",
  executeResponse: null,
  successResponse: null,
  failureReason: null,
  error: null,
};

// ── helpers ──────────────────────────────────────────────────────────────────

function pct(v: number) {
  return (v * 100).toFixed(1) + "%";
}

function actionColor(action: string) {
  if (action === "PAYMENT_RETRY") return "text-emerald-400";
  if (action === "SEND_REMINDER") return "text-amber-400";
  if (action === "ESCALATE") return "text-rose-400";
  if (action === "STOP") return "text-zinc-400";
  return "text-zinc-300";
}

function actionLabel(action: string) {
  if (action === "PAYMENT_RETRY") return "Payment Retry";
  if (action === "SEND_REMINDER") return "Send Reminder";
  if (action === "ESCALATE") return "Escalate";
  if (action === "STOP") return "Stop";
  return action;
}

// ── sub-components ───────────────────────────────────────────────────────────

function ProbBar({ label, value }: { label: string; value: number }) {
  const w = Math.round(value * 100);
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-zinc-400">{label}</span>
        <span className="font-mono text-zinc-200">{pct(value)}</span>
      </div>
      <div className="h-2 rounded-full bg-zinc-700 overflow-hidden">
        <div
          className="h-full rounded-full bg-indigo-500"
          style={{ width: `${w}%` }}
        />
      </div>
    </div>
  );
}

function Card({
  title,
  children,
  className = "",
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-xl border border-zinc-700 bg-zinc-800/60 p-5 ${className}`}>
      <h3 className="text-xs font-semibold uppercase tracking-widest text-zinc-500 mb-4">
        {title}
      </h3>
      {children}
    </div>
  );
}

// ── PaymentRetryPanel ─────────────────────────────────────────────────────────

function PaymentRetryPanel({
  predictResult,
  form,
  payment,
  setPayment,
}: {
  predictResult: RecoveryPredictionResponse;
  form: RecoveryPredictionRequest;
  payment: PaymentState;
  setPayment: React.Dispatch<React.SetStateAction<PaymentState>>;
}) {
  const { policy } = predictResult;

  async function handleProceed() {
    if (payment.phase !== "IDLE") return;

    setPayment({ ...PAYMENT_IDLE, phase: "CREATING_ORDER" });

    let execResp: RecoveryExecuteResponse;
    try {
      execResp = await execute(form);
    } catch (err) {
      setPayment({
        ...PAYMENT_IDLE,
        phase: "FAILED",
        error: err instanceof Error ? err.message : "Execute failed",
      });
      return;
    }

    if (
      execResp.execution_status !== "READY_FOR_PAYMENT" ||
      !execResp.razorpay_order_id ||
      execResp.amount_paise == null ||
      !execResp.currency
    ) {
      setPayment({
        ...PAYMENT_IDLE,
        phase: "FAILED",
        error: `Execution not ready: ${execResp.execution_status} — ${execResp.policy_reason}`,
      });
      return;
    }

    if (!window.Razorpay) {
      setPayment({
        ...PAYMENT_IDLE,
        phase: "FAILED",
        error: "Razorpay SDK not loaded. Check your internet connection.",
      });
      return;
    }

    setPayment({ ...PAYMENT_IDLE, phase: "CHECKOUT_OPEN", executeResponse: execResp });

    const rzp = new window.Razorpay({
      key: RZP_KEY,
      amount: execResp.amount_paise,
      currency: execResp.currency,
      order_id: execResp.razorpay_order_id,
      name: "RecoveryOS",
      description: `Payment recovery for ${form.customer_id}`,
      prefill: { name: form.customer_id },
      theme: { color: "#6366f1" },

      handler: async (response) => {
        setPayment((p) => ({ ...p, phase: "VERIFYING" }));
        try {
          const result = await recordPaymentResult({
            razorpay_payment_id: response.razorpay_payment_id,
            razorpay_order_id: response.razorpay_order_id,
            razorpay_signature: response.razorpay_signature,
          });
          setPayment({
            ...PAYMENT_IDLE,
            phase: "SUCCESS",
            executeResponse: execResp,
            successResponse: result,
          });
        } catch (err) {
          setPayment({
            ...PAYMENT_IDLE,
            phase: "FAILED",
            executeResponse: execResp,
            error: err instanceof Error ? err.message : "Verification failed",
          });
        }
      },

      modal: {
        ondismiss: () => {
          // User closed the modal without paying — record as failed
          setPayment((p) => {
            if (p.phase !== "CHECKOUT_OPEN") return p;
            const orderId = execResp.razorpay_order_id!;
            recordPaymentFailed({
              razorpay_order_id: orderId,
              error_code: "MODAL_DISMISSED",
              error_description: "Customer closed the payment modal",
            }).catch(() => {/* best-effort */});
            return {
              ...PAYMENT_IDLE,
              phase: "FAILED",
              executeResponse: execResp,
              failureReason: "Payment cancelled",
            };
          });
        },
      },
    });

    rzp.on("payment.failed", async (response) => {
      const orderId = execResp.razorpay_order_id!;
      try {
        await recordPaymentFailed({
          razorpay_order_id: orderId,
          error_code: response.error.code,
          error_description: response.error.description,
        });
      } catch { /* best-effort */ }
      setPayment({
        ...PAYMENT_IDLE,
        phase: "FAILED",
        executeResponse: execResp,
        failureReason: response.error.description || response.error.code,
      });
    });

    rzp.open();
  }

  // ── SUCCESS state ──
  if (payment.phase === "SUCCESS" && payment.successResponse) {
    const s = payment.successResponse;
    return (
      <Card title="Final Action" className="border-emerald-700/60 bg-emerald-950/30">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-2xl">✅</span>
          <span className="text-lg font-bold text-emerald-400">Payment Successful</span>
        </div>
        <div className="space-y-1 text-xs font-mono text-zinc-400">
          <div><span className="text-zinc-500">payment_id </span>{s.razorpay_payment_id}</div>
          <div><span className="text-zinc-500">order_id   </span>{s.razorpay_order_id}</div>
          <div><span className="text-zinc-500">recovery   </span>{s.runtime_recovery_id}</div>
        </div>
        <p className="mt-3 text-xs text-zinc-500">
          Signature verified by backend. Recovery attempt marked completed.
        </p>
      </Card>
    );
  }

  // ── FAILED state ──
  if (payment.phase === "FAILED") {
    const reason = payment.failureReason ?? payment.error ?? "Unknown failure";
    return (
      <Card title="Final Action" className="border-rose-800/60 bg-rose-950/30">
        <div className="flex items-center gap-3 mb-3">
          <span className="text-2xl">❌</span>
          <span className="text-lg font-bold text-rose-400">Payment Failed</span>
        </div>
        <p className="text-sm text-zinc-300 mb-4">{reason}</p>
        <button
          onClick={() => setPayment(PAYMENT_IDLE)}
          className="text-xs text-zinc-400 underline hover:text-zinc-200"
        >
          Try again
        </button>
      </Card>
    );
  }

  // ── VERIFYING state ──
  if (payment.phase === "VERIFYING") {
    return (
      <Card title="Final Action" className="border-indigo-800/60 bg-indigo-950/30">
        <div className="flex items-center gap-3">
          <div className="w-5 h-5 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-indigo-300">Verifying payment with backend…</span>
        </div>
      </Card>
    );
  }

  // ── IDLE / CREATING_ORDER / CHECKOUT_OPEN ──
  const busy = payment.phase === "CREATING_ORDER" || payment.phase === "CHECKOUT_OPEN";
  const buttonLabel =
    payment.phase === "CREATING_ORDER"
      ? "Creating payment…"
      : payment.phase === "CHECKOUT_OPEN"
      ? "Checkout open…"
      : "Proceed to Payment";

  return (
    <Card title="Final Action" className="border-emerald-800/60 bg-emerald-950/30">
      <div className="flex items-center gap-3 mb-4">
        <span className="text-2xl">💳</span>
        <span className="text-lg font-bold text-emerald-400">Payment Retry Authorised</span>
      </div>
      <p className="text-sm text-zinc-300 mb-5">{policy.reason}</p>
      <button
        onClick={handleProceed}
        disabled={busy}
        className="w-full rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 text-sm transition-colors"
      >
        {busy && (
          <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2 align-middle" />
        )}
        {buttonLabel}
      </button>
    </Card>
  );
}

// ── ActionPanel ───────────────────────────────────────────────────────────────

function ActionPanel({
  result,
  form,
  payment,
  setPayment,
}: {
  result: RecoveryPredictionResponse;
  form: RecoveryPredictionRequest;
  payment: PaymentState;
  setPayment: React.Dispatch<React.SetStateAction<PaymentState>>;
}) {
  const { policy } = result;

  if (!policy.allowed) {
    return (
      <Card title="Final Action" className="border-rose-800/60 bg-rose-950/30">
        <div className="flex items-center gap-3 mb-3">
          <span className="text-2xl">🚫</span>
          <span className="text-lg font-bold text-rose-400">Action Blocked</span>
        </div>
        <p className="text-sm text-zinc-300">{policy.reason}</p>
      </Card>
    );
  }

  if (policy.action === "PAYMENT_RETRY") {
    return (
      <PaymentRetryPanel
        predictResult={result}
        form={form}
        payment={payment}
        setPayment={setPayment}
      />
    );
  }

  if (policy.action === "SEND_REMINDER") {
    return (
      <Card title="Final Action" className="border-amber-800/60 bg-amber-950/30">
        <div className="flex items-center gap-3 mb-3">
          <span className="text-2xl">🔔</span>
          <span className="text-lg font-bold text-amber-400">Reminder Recommended</span>
        </div>
        <p className="text-sm text-zinc-300 mb-3">{policy.reason}</p>
        <p className="text-xs text-zinc-500 italic">
          RecoveryOS is recommending this action. No notification has been sent.
        </p>
      </Card>
    );
  }

  if (policy.action === "ESCALATE") {
    return (
      <Card title="Final Action" className="border-rose-800/60 bg-rose-950/30">
        <div className="flex items-center gap-3 mb-3">
          <span className="text-2xl">⚠️</span>
          <span className="text-lg font-bold text-rose-400">Manual Review Required</span>
        </div>
        <p className="text-sm text-zinc-300">{policy.reason}</p>
      </Card>
    );
  }

  return (
    <Card title="Final Action">
      <p className="text-sm text-zinc-300">{policy.reason}</p>
    </Card>
  );
}

// ── main page ─────────────────────────────────────────────────────────────────

export default function Home() {
  const [form, setForm] = useState<RecoveryPredictionRequest>(DEFAULT_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RecoveryPredictionResponse | null>(null);
  const [payment, setPayment] = useState<PaymentState>(PAYMENT_IDLE);

  function set<K extends keyof RecoveryPredictionRequest>(
    key: K,
    value: RecoveryPredictionRequest[K]
  ) {
    setForm((f) => ({ ...f, [key]: value }));
    // Reset payment state when form changes
    setPayment(PAYMENT_IDLE);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setPayment(PAYMENT_IDLE);
    try {
      const res = await predict(form);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* ── header ── */}
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-sm font-bold">
          R
        </div>
        <div>
          <h1 className="text-base font-bold leading-none">RecoveryOS</h1>
          <p className="text-xs text-zinc-500 mt-0.5">AI-Powered Payment Recovery</p>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── left: input form ── */}
        <section>
          <h2 className="text-sm font-semibold text-zinc-400 uppercase tracking-widest mb-4">
            Recovery Analysis
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs text-zinc-400 mb-1">Customer ID</label>
              <input
                type="text"
                value={form.customer_id}
                onChange={(e) => set("customer_id", e.target.value)}
                required
                placeholder="CUST_000001"
                className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-indigo-500"
              />
            </div>

            <div className="grid grid-cols-3 gap-3">
              <div className="col-span-2">
                <label className="block text-xs text-zinc-400 mb-1">Amount</label>
                <input
                  type="number"
                  min={0.01}
                  step={0.01}
                  value={form.amount}
                  onChange={(e) => set("amount", parseFloat(e.target.value) || 0)}
                  required
                  className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-xs text-zinc-400 mb-1">Currency</label>
                <input
                  type="text"
                  value={form.currency}
                  onChange={(e) => set("currency", e.target.value)}
                  required
                  maxLength={3}
                  className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-100 uppercase focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs text-zinc-400 mb-1">Payment Method</label>
              <select
                value={form.payment_method}
                onChange={(e) => set("payment_method", e.target.value)}
                className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:border-indigo-500"
              >
                {PAYMENT_METHODS.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs text-zinc-400 mb-1">Failure Reason</label>
              <select
                value={form.failure_reason}
                onChange={(e) => set("failure_reason", e.target.value)}
                className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:border-indigo-500"
              >
                {FAILURE_REASONS.map((r) => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs text-zinc-400 mb-1">Attempt Number</label>
              <input
                type="number"
                min={1}
                step={1}
                value={form.attempt_number}
                onChange={(e) =>
                  set("attempt_number", Math.max(1, parseInt(e.target.value) || 1))
                }
                required
                className="w-full rounded-lg bg-zinc-800 border border-zinc-700 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:border-indigo-500"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-2.5 text-sm transition-colors"
            >
              {loading ? "Analysing…" : "Run Recovery Analysis"}
            </button>

            {error && (
              <div className="rounded-lg border border-rose-700 bg-rose-950/40 px-4 py-3 text-sm text-rose-300">
                <span className="font-semibold">Error: </span>{error}
              </div>
            )}
          </form>
        </section>

        {/* ── right: results ── */}
        <section className="space-y-4">
          {!result && !loading && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-8 text-center text-zinc-600 text-sm">
              Submit the form to run the ML → Gemini → Policy pipeline.
            </div>
          )}

          {loading && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-8 text-center">
              <div className="inline-block w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mb-3" />
              <p className="text-sm text-zinc-400">Running pipeline…</p>
            </div>
          )}

          {result && (
            <>
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <span className="font-mono text-zinc-300">{result.customer_id}</span>
                <span
                  className={`px-2 py-0.5 rounded-full border text-xs ${
                    result.customer_exists
                      ? "border-emerald-700 text-emerald-400 bg-emerald-950/40"
                      : "border-zinc-700 text-zinc-400"
                  }`}
                >
                  {result.customer_exists ? "existing customer" : "new customer"}
                </span>
              </div>

              <Card title="① XGBoost Probabilities">
                <div className="space-y-3">
                  <ProbBar label="PAYMENT_RETRY" value={result.ml_predictions.PAYMENT_RETRY} />
                  <ProbBar label="SEND_REMINDER" value={result.ml_predictions.SEND_REMINDER} />
                </div>
              </Card>

              <Card title="② Gemini AI Recommendation">
                <div className="flex items-start justify-between gap-4 mb-3">
                  <div>
                    <span className={`text-xl font-bold ${actionColor(result.recommended_action)}`}>
                      {actionLabel(result.recommended_action)}
                    </span>
                    <span className="ml-2 text-xs text-zinc-500">
                      {pct(result.confidence)} confidence
                    </span>
                  </div>
                </div>
                <p className="text-sm text-zinc-300 leading-relaxed">{result.reason}</p>
              </Card>

              <Card title="③ Policy Engine Decision">
                <div className="flex items-center gap-3 mb-3">
                  <span className={`text-lg font-bold ${actionColor(result.policy.action)}`}>
                    {actionLabel(result.policy.action)}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded-full text-xs font-semibold border ${
                      result.policy.allowed
                        ? "border-emerald-700 text-emerald-400 bg-emerald-950/40"
                        : "border-rose-700 text-rose-400 bg-rose-950/40"
                    }`}
                  >
                    {result.policy.allowed ? "ALLOWED" : "BLOCKED"}
                  </span>
                </div>
                <p className="text-sm text-zinc-300">{result.policy.reason}</p>
              </Card>

              <ActionPanel
                result={result}
                form={form}
                payment={payment}
                setPayment={setPayment}
              />
            </>
          )}
        </section>
      </main>
    </div>
  );
}
