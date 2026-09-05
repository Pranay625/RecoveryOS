
"use client";

import { useState } from "react";
import {
  predict,
  execute,
  recordPaymentResult,
  recordPaymentFailed,
} from "./api-client";
import type {
  RecoveryPredictionRequest,
  RecoveryPredictionResponse,
  RecoveryExecuteResponse,
  PaymentResultResponse,
} from "./types";

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

function formatAmount(amount: number, currency: string) {
  if (currency === "INR") {
    return `₹${amount.toLocaleString("en-IN", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 2,
    })}`;
  }

  return `${currency} ${amount.toLocaleString("en-IN", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  })}`;
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

// ── reusable UI ──────────────────────────────────────────────────────────────

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
    <div
      className={`rounded-2xl border border-zinc-700/80 bg-zinc-900/70 p-5 shadow-lg shadow-black/10 ${className}`}
    >
      <h3 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-500 mb-4">
        {title}
      </h3>
      {children}
    </div>
  );
}

function ProbBar({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  const w = Math.round(value * 100);

  return (
    <div>
      <div className="flex justify-between text-xs mb-2">
        <span className="text-zinc-400">{label}</span>
        <span className="font-mono font-semibold text-zinc-200">
          {pct(value)}
        </span>
      </div>

      <div className="h-2.5 rounded-full bg-zinc-800 overflow-hidden border border-zinc-700/50">
        <div
          className="h-full rounded-full bg-indigo-500 transition-all duration-700"
          style={{ width: `${w}%` }}
        />
      </div>
    </div>
  );
}

function PipelineStep({
  number,
  title,
  value,
  status,
}: {
  number: string;
  title: string;
  value: string;
  status?: "success" | "normal";
}) {
  return (
    <div className="flex items-center gap-3">
      <div
        className={`w-8 h-8 shrink-0 rounded-full flex items-center justify-center text-xs font-bold ${
          status === "success"
            ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
            : "bg-indigo-500/15 text-indigo-400 border border-indigo-500/30"
        }`}
      >
        {number}
      </div>

      <div className="min-w-0">
        <p className="text-[10px] uppercase tracking-widest text-zinc-500">
          {title}
        </p>
        <p
          className={`text-sm font-semibold ${
            status === "success" ? "text-emerald-400" : "text-zinc-200"
          }`}
        >
          {value}
        </p>
      </div>
    </div>
  );
}

// ── PaymentRetryPanel ────────────────────────────────────────────────────────

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

    setPayment({
      ...PAYMENT_IDLE,
      phase: "CHECKOUT_OPEN",
      executeResponse: execResp,
    });

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
            error:
              err instanceof Error ? err.message : "Verification failed",
          });
        }
      },

      modal: {
        ondismiss: () => {
          setPayment((p) => {
            if (p.phase !== "CHECKOUT_OPEN") return p;

            const orderId = execResp.razorpay_order_id!;

            recordPaymentFailed({
              razorpay_order_id: orderId,
              error_code: "MODAL_DISMISSED",
              error_description: "Customer closed the payment modal",
            }).catch(() => {});

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
      } catch {}

      setPayment({
        ...PAYMENT_IDLE,
        phase: "FAILED",
        executeResponse: execResp,
        failureReason:
          response.error.description || response.error.code,
      });
    });

    rzp.open();
  }

  // ── SUCCESS ────────────────────────────────────────────────────────────────

  if (payment.phase === "SUCCESS" && payment.successResponse) {
    const s = payment.successResponse;

    return (
      <div className="rounded-2xl border border-emerald-500/40 bg-gradient-to-br from-emerald-950/70 via-zinc-900 to-zinc-900 p-6 shadow-xl shadow-emerald-950/20">
        <div className="text-center mb-6">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500/15 border border-emerald-500/30">
            <span className="text-3xl">✓</span>
          </div>

          <p className="text-xs uppercase tracking-[0.25em] text-emerald-400 font-semibold">
            Payment Recovered
          </p>

          <div className="mt-2 text-4xl sm:text-5xl font-black tracking-tight text-white">
            {formatAmount(form.amount, form.currency)}
          </div>

          <p className="mt-2 text-sm text-zinc-400">
            Money successfully recovered through Razorpay
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mb-5">
          <div className="rounded-xl bg-zinc-950/60 border border-zinc-800 p-3">
            <p className="text-[10px] uppercase tracking-widest text-zinc-600">
              Payment ID
            </p>
            <p className="mt-1 text-[11px] font-mono text-zinc-300 break-all">
              {s.razorpay_payment_id}
            </p>
          </div>

          <div className="rounded-xl bg-zinc-950/60 border border-zinc-800 p-3">
            <p className="text-[10px] uppercase tracking-widest text-zinc-600">
              Order ID
            </p>
            <p className="mt-1 text-[11px] font-mono text-zinc-300 break-all">
              {s.razorpay_order_id}
            </p>
          </div>

          <div className="rounded-xl bg-zinc-950/60 border border-zinc-800 p-3">
            <p className="text-[10px] uppercase tracking-widest text-zinc-600">
              Recovery ID
            </p>
            <p className="mt-1 text-[11px] font-mono text-zinc-300 break-all">
              {s.runtime_recovery_id}
            </p>
          </div>
        </div>

        <div className="rounded-xl border border-emerald-800/50 bg-emerald-950/30 px-4 py-3">
          <div className="flex items-center gap-2 text-sm text-emerald-300">
            <span>✓</span>
            <span>Razorpay signature verified by backend</span>
          </div>

          <div className="flex items-center gap-2 text-sm text-emerald-300 mt-1">
            <span>✓</span>
            <span>Recovery attempt marked completed</span>
          </div>
        </div>
      </div>
    );
  }

  // ── FAILED ─────────────────────────────────────────────────────────────────

  if (payment.phase === "FAILED") {
    const reason =
      payment.failureReason ?? payment.error ?? "Unknown failure";

    return (
      <Card
        title="Final Result"
        className="border-rose-800/60 bg-rose-950/30"
      >
        <div className="text-center py-2">
          <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-full bg-rose-500/10 border border-rose-500/20">
            <span className="text-2xl">×</span>
          </div>

          <p className="text-xs uppercase tracking-[0.2em] text-rose-400 font-semibold">
            Recovery Unsuccessful
          </p>

          <div className="mt-1 text-3xl font-black text-white">
            ₹0 Recovered
          </div>

          <p className="mt-2 text-sm text-zinc-400">{reason}</p>

          <button
            onClick={() => setPayment(PAYMENT_IDLE)}
            className="mt-5 rounded-lg border border-zinc-700 px-4 py-2 text-xs font-semibold text-zinc-300 hover:bg-zinc-800 transition-colors"
          >
            Try Again
          </button>
        </div>
      </Card>
    );
  }

  // ── VERIFYING ──────────────────────────────────────────────────────────────

  if (payment.phase === "VERIFYING") {
    return (
      <Card
        title="Final Result"
        className="border-indigo-800/60 bg-indigo-950/30"
      >
        <div className="flex items-center gap-3">
          <div className="w-5 h-5 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
          <div>
            <p className="text-sm font-semibold text-indigo-300">
              Verifying payment
            </p>
            <p className="text-xs text-zinc-500 mt-0.5">
              Confirming Razorpay signature with backend…
            </p>
          </div>
        </div>
      </Card>
    );
  }

  // ── READY / CHECKOUT ───────────────────────────────────────────────────────

  const busy =
    payment.phase === "CREATING_ORDER" ||
    payment.phase === "CHECKOUT_OPEN";

  const buttonLabel =
    payment.phase === "CREATING_ORDER"
      ? "Creating payment order…"
      : payment.phase === "CHECKOUT_OPEN"
      ? "Checkout open…"
      : `Recover ${formatAmount(form.amount, form.currency)}`;

  return (
    <Card
      title="Final Action"
      className="border-emerald-800/60 bg-emerald-950/30"
    >
      <div className="flex items-start gap-3 mb-4">
        <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
          <span className="text-xl">↻</span>
        </div>

        <div>
          <p className="text-lg font-bold text-emerald-400">
            Payment Retry Authorised
          </p>
          <p className="text-xs text-zinc-500 mt-0.5">
            Policy engine has approved the recovery action.
          </p>
        </div>
      </div>

      <p className="text-sm text-zinc-300 leading-relaxed mb-5">
        {policy.reason}
      </p>

      <button
        onClick={handleProceed}
        disabled={busy}
        className="w-full rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold py-3.5 text-sm transition-all shadow-lg shadow-emerald-950/30"
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
      <Card
        title="Final Action"
        className="border-rose-800/60 bg-rose-950/30"
      >
        <div className="flex items-center gap-3 mb-3">
          <span className="text-2xl">🚫</span>
          <span className="text-lg font-bold text-rose-400">
            Action Blocked
          </span>
        </div>

        <p className="text-sm text-zinc-300 leading-relaxed">
          {policy.reason}
        </p>
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
      <Card
        title="Final Action"
        className="border-amber-800/60 bg-amber-950/30"
      >
        <div className="flex items-center gap-3 mb-3">
          <span className="text-2xl">🔔</span>
          <span className="text-lg font-bold text-amber-400">
            Reminder Recommended
          </span>
        </div>

        <p className="text-sm text-zinc-300 mb-3 leading-relaxed">
          {policy.reason}
        </p>

        <p className="text-xs text-zinc-500 italic">
          RecoveryOS is recommending this action. No notification has been sent.
        </p>
      </Card>
    );
  }

  if (policy.action === "ESCALATE") {
    return (
      <Card
        title="Final Action"
        className="border-rose-800/60 bg-rose-950/30"
      >
        <div className="flex items-center gap-3 mb-3">
          <span className="text-2xl">⚠️</span>
          <span className="text-lg font-bold text-rose-400">
            Manual Review Required
          </span>
        </div>

        <p className="text-sm text-zinc-300 leading-relaxed">
          {policy.reason}
        </p>
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
  const [form, setForm] =
    useState<RecoveryPredictionRequest>(DEFAULT_FORM);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] =
    useState<RecoveryPredictionResponse | null>(null);

  const [payment, setPayment] =
    useState<PaymentState>(PAYMENT_IDLE);

  function set<K extends keyof RecoveryPredictionRequest>(
    key: K,
    value: RecoveryPredictionRequest[K]
  ) {
    setForm((f) => ({ ...f, [key]: value }));
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
      setError(
        err instanceof Error ? err.message : "Unknown error"
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* ── HEADER ─────────────────────────────────────────────────────────── */}

      <header className="border-b border-zinc-800/80 bg-zinc-950/90 backdrop-blur">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center text-sm font-black shadow-lg shadow-indigo-950/40">
              R
            </div>

            <div>
              <h1 className="text-base font-bold leading-none">
                RecoveryOS
              </h1>

              <p className="text-[11px] text-zinc-500 mt-1">
                Intelligent Payment Recovery
              </p>
            </div>
          </div>

          <div className="hidden sm:flex items-center gap-2 text-[10px] uppercase tracking-widest text-zinc-600">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            Recovery Engine Online
          </div>
        </div>
      </header>

      {/* ── MAIN ───────────────────────────────────────────────────────────── */}

      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
        {/* HERO */}

        <div className="mb-8">
          <p className="text-xs uppercase tracking-[0.25em] text-indigo-400 font-semibold mb-2">
            Automated Payment Recovery
          </p>

          <h2 className="text-3xl sm:text-4xl font-black tracking-tight text-white">
            Recover failed payments intelligently.
          </h2>

          <p className="mt-3 max-w-2xl text-sm sm:text-base text-zinc-500 leading-relaxed">
            RecoveryOS combines machine learning, AI reasoning and
            deterministic policy controls to decide the safest recovery
            action for every failed payment.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[0.85fr_1.15fr] gap-6">
          {/* ── LEFT: INPUT ────────────────────────────────────────────────── */}

          <section>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-widest">
                Payment Context
              </h2>

              <span className="text-[10px] text-zinc-600">
                INPUT
              </span>
            </div>

            <form
              onSubmit={handleSubmit}
              className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5 space-y-4 shadow-xl shadow-black/10"
            >
              <div>
                <label className="block text-xs text-zinc-400 mb-1.5">
                  Customer ID
                </label>

                <input
                  type="text"
                  value={form.customer_id}
                  onChange={(e) =>
                    set("customer_id", e.target.value)
                  }
                  required
                  placeholder="CUST_000001"
                  className="w-full rounded-xl bg-zinc-950 border border-zinc-700 px-3.5 py-2.5 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-indigo-500 transition-colors"
                />
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div className="col-span-2">
                  <label className="block text-xs text-zinc-400 mb-1.5">
                    Amount at Risk
                  </label>

                  <input
                    type="number"
                    min={0.01}
                    step={0.01}
                    value={form.amount}
                    onChange={(e) =>
                      set(
                        "amount",
                        parseFloat(e.target.value) || 0
                      )
                    }
                    required
                    className="w-full rounded-xl bg-zinc-950 border border-zinc-700 px-3.5 py-2.5 text-sm text-zinc-100 focus:outline-none focus:border-indigo-500"
                  />
                </div>

                <div>
                  <label className="block text-xs text-zinc-400 mb-1.5">
                    Currency
                  </label>

                  <input
                    type="text"
                    value={form.currency}
                    onChange={(e) =>
                      set("currency", e.target.value.toUpperCase())
                    }
                    required
                    maxLength={3}
                    className="w-full rounded-xl bg-zinc-950 border border-zinc-700 px-3.5 py-2.5 text-sm text-zinc-100 uppercase focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs text-zinc-400 mb-1.5">
                  Payment Method
                </label>

                <select
                  value={form.payment_method}
                  onChange={(e) =>
                    set("payment_method", e.target.value)
                  }
                  className="w-full rounded-xl bg-zinc-950 border border-zinc-700 px-3.5 py-2.5 text-sm text-zinc-100 focus:outline-none focus:border-indigo-500"
                >
                  {PAYMENT_METHODS.map((m) => (
                    <option key={m} value={m}>
                      {m.toUpperCase()}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs text-zinc-400 mb-1.5">
                  Failure Reason
                </label>

                <select
                  value={form.failure_reason}
                  onChange={(e) =>
                    set("failure_reason", e.target.value)
                  }
                  className="w-full rounded-xl bg-zinc-950 border border-zinc-700 px-3.5 py-2.5 text-sm text-zinc-100 focus:outline-none focus:border-indigo-500"
                >
                  {FAILURE_REASONS.map((r) => (
                    <option key={r} value={r}>
                      {r.replaceAll("_", " ")}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs text-zinc-400 mb-1.5">
                  Attempt Number
                </label>

                <input
                  type="number"
                  min={1}
                  step={1}
                  value={form.attempt_number}
                  onChange={(e) =>
                    set(
                      "attempt_number",
                      Math.max(
                        1,
                        parseInt(e.target.value) || 1
                      )
                    )
                  }
                  required
                  className="w-full rounded-xl bg-zinc-950 border border-zinc-700 px-3.5 py-2.5 text-sm text-zinc-100 focus:outline-none focus:border-indigo-500"
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold py-3 text-sm transition-all shadow-lg shadow-indigo-950/30"
              >
                {loading ? "Analysing…" : "Run Recovery Analysis"}
              </button>

              {error && (
                <div className="rounded-xl border border-rose-700 bg-rose-950/40 px-4 py-3 text-sm text-rose-300">
                  <span className="font-semibold">Error: </span>
                  {error}
                </div>
              )}
            </form>

            {/* PIPELINE */}

            <div className="mt-4 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
              <p className="text-[10px] uppercase tracking-[0.2em] text-zinc-600 mb-4">
                Decision Pipeline
              </p>

              <div className="space-y-4">
                <PipelineStep
                  number="1"
                  title="Prediction"
                  value="XGBoost Recovery Model"
                />

                <div className="ml-4 h-3 border-l border-dashed border-zinc-700" />

                <PipelineStep
                  number="2"
                  title="Reasoning"
                  value="Groq AI Recovery Agent"
                />

                <div className="ml-4 h-3 border-l border-dashed border-zinc-700" />

                <PipelineStep
                  number="3"
                  title="Authorization"
                  value="Deterministic Policy Engine"
                />

                <div className="ml-4 h-3 border-l border-dashed border-zinc-700" />

                <PipelineStep
                  number="4"
                  title="Execution"
                  value="Razorpay Test Mode"
                  status="success"
                />
              </div>
            </div>
          </section>

          {/* ── RIGHT: RESULTS ─────────────────────────────────────────────── */}

          <section className="space-y-4">
            {!result && !loading && (
              <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-10 text-center">
                <div className="mx-auto mb-4 w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
                  <span className="text-xl">⌁</span>
                </div>

                <p className="text-sm font-semibold text-zinc-400">
                  Ready for analysis
                </p>

                <p className="mt-1 text-xs text-zinc-600">
                  Enter a failed payment to evaluate its recovery path.
                </p>
              </div>
            )}

            {loading && (
              <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-10 text-center">
                <div className="inline-block w-7 h-7 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mb-4" />

                <p className="text-sm font-semibold text-zinc-300">
                  Running RecoveryOS
                </p>

                <p className="text-xs text-zinc-600 mt-1">
                  XGBoost → Groq → Policy Engine
                </p>
              </div>
            )}

            {result && (
              <>
                {/* CUSTOMER + AT RISK */}

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-5">
                    <p className="text-[10px] uppercase tracking-[0.2em] text-zinc-600">
                      Customer
                    </p>

                    <div className="mt-2 flex items-center gap-2">
                      <span className="font-mono text-sm text-zinc-200">
                        {result.customer_id}
                      </span>

                      <span
                        className={`px-2 py-0.5 rounded-full border text-[10px] ${
                          result.customer_exists
                            ? "border-emerald-700 text-emerald-400 bg-emerald-950/40"
                            : "border-zinc-700 text-zinc-400"
                        }`}
                      >
                        {result.customer_exists
                          ? "EXISTING"
                          : "NEW"}
                      </span>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-indigo-500/20 bg-indigo-950/20 p-5">
                    <p className="text-[10px] uppercase tracking-[0.2em] text-indigo-400">
                      Amount at Risk
                    </p>

                    <p className="mt-1 text-3xl font-black text-white">
                      {formatAmount(form.amount, form.currency)}
                    </p>
                  </div>
                </div>

                {/* RECOVERY PROBABILITY */}

                <Card title="Recovery Probability">
                  <div className="mb-5">
                    <p className="text-xs text-zinc-500">
                      XGBoost estimates the likelihood of successful recovery.
                    </p>
                  </div>

                  <div className="space-y-4">
                    <ProbBar
                      label="Payment Retry"
                      value={result.ml_predictions.PAYMENT_RETRY}
                    />

                    <ProbBar
                      label="Send Reminder"
                      value={result.ml_predictions.SEND_REMINDER}
                    />
                  </div>

                  <div className="mt-5 pt-4 border-t border-zinc-800 flex items-center justify-between">
                    <span className="text-xs text-zinc-500">
                      Primary recovery probability
                    </span>

                    <span className="text-2xl font-black text-indigo-400">
                      {pct(result.ml_predictions.PAYMENT_RETRY)}
                    </span>
                  </div>
                </Card>

                {/* GROQ */}

                <Card
                  title="② Groq AI Recommendation"
                  className="border-indigo-800/50"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-4">
                    <div>
                      <p
                        className={`text-xl font-bold ${actionColor(
                          result.recommended_action
                        )}`}
                      >
                        {actionLabel(result.recommended_action)}
                      </p>

                      <p className="text-xs text-zinc-600 mt-1">
                        AI confidence: {pct(result.confidence)}
                      </p>
                    </div>

                    <div className="rounded-lg border border-indigo-700/40 bg-indigo-950/30 px-3 py-1.5 text-[10px] uppercase tracking-widest text-indigo-400">
                      Groq AI
                    </div>
                  </div>

                  <div className="rounded-xl bg-zinc-950/50 border border-zinc-800 p-4">
                    <p className="text-[10px] uppercase tracking-widest text-zinc-600 mb-2">
                      AI Reasoning
                    </p>

                    <p className="text-sm text-zinc-300 leading-relaxed">
                      {result.reason}
                    </p>
                  </div>
                </Card>

                {/* POLICY */}

                <Card title="③ Policy Engine Decision">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
                    <div>
                      <p
                        className={`text-lg font-bold ${actionColor(
                          result.policy.action
                        )}`}
                      >
                        {actionLabel(result.policy.action)}
                      </p>

                      <p className="text-xs text-zinc-600 mt-1">
                        Deterministic authorization layer
                      </p>
                    </div>

                    <span
                      className={`inline-flex w-fit px-3 py-1 rounded-full text-xs font-bold border ${
                        result.policy.allowed
                          ? "border-emerald-700 text-emerald-400 bg-emerald-950/40"
                          : "border-rose-700 text-rose-400 bg-rose-950/40"
                      }`}
                    >
                      {result.policy.allowed
                        ? "✓ ALLOWED"
                        : "✕ BLOCKED"}
                    </span>
                  </div>

                  <p className="text-sm text-zinc-300 leading-relaxed">
                    {result.policy.reason}
                  </p>
                </Card>

                {/* FINAL ACTION */}

                <ActionPanel
                  result={result}
                  form={form}
                  payment={payment}
                  setPayment={setPayment}
                />
              </>
            )}
          </section>
        </div>
      </main>

      {/* ── FOOTER ─────────────────────────────────────────────────────────── */}

      <footer className="max-w-6xl mx-auto px-6 py-8 text-center">
        <p className="text-[10px] uppercase tracking-[0.2em] text-zinc-700">
          RecoveryOS · ML Prediction · AI Reasoning · Deterministic Policy · Razorpay
        </p>
      </footer>
    </div>
  );
}
