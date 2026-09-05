# RecoveryOS

### Intelligent, Explainable & Policy-Controlled Payment Recovery

> **RecoveryOS turns failed payments into intelligent recovery decisions
> --- combining machine learning, AI reasoning, deterministic policy
> enforcement, and Razorpay payment execution in one controlled
> pipeline.**

------------------------------------------------------------------------

## 🚀 Why RecoveryOS?

A failed payment does not always mean the same thing.

A customer with a strong payment history may be a good candidate for
another payment attempt. Another customer may respond better to a
reminder. Repeatedly retrying a payment can also become
counterproductive when recovery limits have already been reached.

Traditional recovery systems often rely on static rules or blind
retries.

**RecoveryOS takes a decision-first approach.**

Instead of asking:

> "Should we retry every failed payment?"

RecoveryOS asks:

> **"Given this customer's behavior, this payment's context, the
> predicted recovery probability, and our business constraints --- what
> is the safest and most appropriate recovery action?"**

The system deliberately separates intelligence from authority:

``` text
                ┌──────────────────────┐
                │    Failed Payment    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Feature Engineering  │
                │ Customer + Payment   │
                │ Behavioral Context   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │       XGBoost        │
                │ Recovery Prediction  │
                └──────────┬───────────┘
                           │
                  probabilities +
                    payment context
                           │
                           ▼
                ┌──────────────────────┐
                │       Groq AI        │
                │ Recovery Agent       │
                │                      │
                │ Recommendation +     │
                │ Reason + Confidence  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Deterministic Policy │
                │       Engine         │
                │                      │
                │  Can this action     │
                │  actually happen?    │
                └──────────┬───────────┘
                           │
                     if authorized
                           │
                           ▼
                ┌──────────────────────┐
                │  Razorpay Test Mode  │
                │ Payment Execution    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Backend Verification │
                │ Signature + DB State │
                └──────────────────────┘
```

### The core principle

> **ML predicts. AI reasons. Policy authorizes. Razorpay executes.**

That separation is the foundation of RecoveryOS.

------------------------------------------------------------------------

# ✨ Key Features

-   **Behavior-aware payment recovery**
-   **XGBoost-based recovery probability prediction**
-   **Groq-powered contextual AI recovery recommendations**
-   **Structured AI outputs with action, reasoning, and confidence**
-   **Deterministic policy enforcement**
-   **Retry and recovery-attempt limits**
-   **Customer opt-out protection**
-   **Razorpay Test Mode integration**
-   **Server-side Razorpay signature verification**
-   **Runtime transaction isolation from historical ML data**
-   **Audit logging**
-   **Idempotent payment-result handling**
-   **FastAPI backend**
-   **Next.js frontend**
-   **SQLAlchemy + MySQL persistence**
-   **Swagger/OpenAPI API documentation**
-   **Automated backend test coverage**

------------------------------------------------------------------------

# 🧠 The Problem

Payment failure recovery is usually treated as an execution problem:

``` text
Payment failed
      ↓
Retry
```

But recovery is actually a **decision problem**.

A system needs to consider:

-   How has this customer behaved historically?
-   How frequently do their payments succeed?
-   What kinds of failures have occurred before?
-   Has recovery worked for this customer previously?
-   How many recovery attempts have already happened?
-   How many retries have already been performed?
-   Would a reminder be more appropriate?
-   Has the customer opted out of recovery?
-   Is the recommended action still within operational limits?

RecoveryOS combines these considerations into a single controlled
workflow.

------------------------------------------------------------------------

# 🎯 Objectives

RecoveryOS was designed around five objectives:

### 1. Replace blind retries with intelligent recovery

Use historical behavioral data to estimate which recovery strategies are
more likely to work.

### 2. Add contextual reasoning

A probability alone does not explain *why* an action is appropriate. The
AI layer interprets model predictions together with payment and customer
context.

### 3. Keep AI under deterministic control

The AI should never have unrestricted authority over payment execution.

The policy engine remains the final authorization layer.

### 4. Make payment execution verifiable

A successful frontend interaction is not treated as sufficient proof of
payment success. Razorpay results are verified on the backend before the
recovery is recorded as completed.

### 5. Keep inference data trustworthy

Runtime/demo transactions must not silently become training-like
historical signals and influence subsequent ML decisions.

------------------------------------------------------------------------

# 🏗️ System Architecture

## End-to-End Flow

``` text
Frontend
   │
   │ POST /predict
   ▼
Feature Engineering
   │
   ▼
XGBoost
   │
   │ recovery probabilities
   ▼
Groq Recovery Agent
   │
   │ action + reason + confidence
   ▼
Policy Engine
   │
   ├────────────── BLOCKED ──────────────► Frontend
   │
   ▼
/execute
   │
   ▼
Razorpay Test Mode Order
   │
   ▼
Razorpay Checkout
   │
   ├──────────── Payment Success
   │                    │
   │                    ▼
   │            /payment-result
   │                    │
   │                    ▼
   │            Signature Verification
   │                    │
   │                    ▼
   │              Database Update
   │
   └──────────── Payment Failure
                        │
                        ▼
                 /payment-failed
                        │
                        ▼
                  Database Update
```

------------------------------------------------------------------------

# 🔬 Layer 1 --- Machine Learning

## Why XGBoost?

The first question RecoveryOS needs to answer is quantitative:

> **"How likely is each recovery strategy to succeed?"**

This is a structured prediction problem involving customer behavior,
payment behavior, and historical recovery outcomes.

XGBoost was selected because it performs well on structured/tabular data
and can model nonlinear relationships between behavioral features.

The ML layer is intentionally focused on **prediction**, not business
authorization.

------------------------------------------------------------------------

## 📊 Historical Dataset

RecoveryOS uses a synthetic historical dataset containing:

-   Approximately **1,000 customers**
-   Approximately **38,000 historical payments**
-   Thousands of historical recovery attempts
-   Multiple behavioral customer profiles
-   Historical payment outcomes
-   Historical recovery outcomes

The dataset is designed to represent different customer/payment
behaviors rather than a single homogeneous population.

Historical records use IDs such as:

``` text
PAY_*
REC_*
```

Runtime/demo records use separate namespaces:

``` text
PAY_RT_*
REC_RT_*
AUD_RT_*
```

This distinction is important for keeping historical inference data
isolated from live demo activity.

------------------------------------------------------------------------

# 🧮 Feature Engineering

The model does not operate directly on raw database rows.

RecoveryOS converts historical behavior into structured inference
features.

The feature engineering layer captures information such as:

-   Payment success/failure behavior
-   Customer-level payment behavior
-   Recovery behavior
-   Previous recovery attempts
-   Retry history
-   Reminder history
-   Payment context
-   Attempt-related information
-   Behavioral patterns derived from historical records

This transforms transactional history into a representation suitable for
the ML model.

------------------------------------------------------------------------

# 🤖 Model Output

The XGBoost inference layer produces recovery probabilities such as:

``` json
{
  "PAYMENT_RETRY": 0.9283,
  "SEND_REMINDER": 0.8713
}
```

These probabilities answer:

> "How promising does each recovery strategy look based on the
> structured historical data?"

They do **not** authorize a payment.

------------------------------------------------------------------------

## 📈 Model Performance

The trained XGBoost model achieved:

``` text
ROC-AUC: 0.7242
```

This demonstrates meaningful predictive signal while also reflecting
that payment recovery is not a perfectly deterministic problem.

The goal of the model is not to pretend that recovery can be predicted
with certainty.

Its job is to provide a useful probabilistic signal to the downstream
decision system.

------------------------------------------------------------------------

# 🧠 Layer 2 --- Groq AI Recovery Agent

The second layer addresses a different question:

> **"Given the prediction and the surrounding context, what recovery
> action makes the most sense?"**

The AI agent is powered by the Groq API using:

``` text
openai/gpt-oss-20b
```

The service is accessed from the **backend**, never from the browser.

------------------------------------------------------------------------

# 📦 What Goes Into the AI?

The AI does not receive just the XGBoost score.

The recovery agent is given a structured context containing relevant
information such as:

### Payment context

-   Payment amount
-   Currency
-   Payment method
-   Failure reason
-   Attempt information

### Customer behavior

-   Historical payment behavior
-   Success/failure patterns
-   Recovery behavior
-   Previous recovery attempts

### ML signal

-   XGBoost recovery probabilities

### Recovery state

-   Previous retry count
-   Previous reminder count
-   Relevant recovery-attempt context

This gives the AI enough context to reason about the recommendation
rather than simply echoing the highest probability.

------------------------------------------------------------------------

# 📝 What Comes Out of the AI?

The AI is constrained to a small set of valid actions:

``` text
PAYMENT_RETRY
SEND_REMINDER
ESCALATE
```

The response is structured as:

``` json
{
  "action": "PAYMENT_RETRY",
  "reason": "The customer has a strong payment and recovery profile...",
  "confidence": 0.95
}
```

The response is validated against a strict Pydantic schema before it
reaches the policy layer.

This prevents free-form model output from becoming uncontrolled
application logic.

------------------------------------------------------------------------

# 🔐 Why Doesn't the AI Execute the Payment?

This is one of the most important architectural decisions in RecoveryOS.

The AI is probabilistic.

Payment authorization is a high-control operation.

Therefore:

``` text
AI Recommendation
       ≠
Payment Authorization
```

Instead:

``` text
AI
 ↓
Recommendation
 ↓
Policy Engine
 ↓
Authorization
 ↓
Razorpay
```

This means an AI hallucination, unexpected recommendation, or overly
aggressive suggestion cannot directly trigger a payment.

------------------------------------------------------------------------

# ⚖️ Layer 3 --- Deterministic Policy Engine

The policy engine is the system's final authority.

It evaluates whether the AI's recommendation is actually permitted.

## Current policy constraints

``` text
MAX_RECOVERY_ATTEMPTS = 3
MAX_PAYMENT_RETRIES   = 2
MAX_REMINDERS         = 2
```

The policy evaluates conditions in a deterministic priority order.

### 1. Customer opt-out

If the customer has opted out:

``` text
STOP
allowed = false
```

### 2. AI recommends ESCALATE

The system respects escalation:

``` text
ESCALATE
allowed = true
```

The frontend can present this as a manual-review decision rather than
automatically executing a payment.

### 3. Recovery-attempt limit

If previous recovery attempts have reached the maximum:

``` text
ESCALATE
allowed = false
```

### 4. Payment retry limit

If the AI recommends:

``` text
PAYMENT_RETRY
```

but the retry limit has already been reached:

``` text
ESCALATE
allowed = false
```

### 5. Reminder limit

If the AI recommends:

``` text
SEND_REMINDER
```

but the reminder limit has been reached:

``` text
ESCALATE
allowed = false
```

### 6. Unknown action

Unexpected actions are never silently executed.

They fail closed:

``` text
ESCALATE
allowed = false
```

------------------------------------------------------------------------

# 💡 Why This Architecture Matters

RecoveryOS intentionally creates a hierarchy:

  Layer                  Responsibility
  ---------------------- -----------------------------------------
  Feature Engineering    Convert history into behavioral signals
  XGBoost                Predict recovery probabilities
  Groq AI                Reason over predictions + context
  Policy Engine          Authorize or block the recommendation
  Razorpay               Execute an authorized payment
  Backend Verification   Verify the resulting payment state

This separation makes the system easier to reason about, test, audit,
and extend.

------------------------------------------------------------------------

# 💳 Razorpay Integration

RecoveryOS integrates with **Razorpay Test Mode** for payment execution.

The integration is real, but the demonstration does not move real money.

## Execution Flow

``` text
Policy = ALLOWED
       ↓
Create Razorpay Order
       ↓
Return order information to frontend
       ↓
Open Razorpay Checkout
       ↓
Customer completes test payment
       ↓
Razorpay returns payment information
       ↓
Backend verifies signature
       ↓
Database state updated
```

------------------------------------------------------------------------

# 🔒 Server-Side Payment Verification

The frontend is not trusted as the final source of payment truth.

For a successful payment, the backend verifies the Razorpay signature
using the order ID, payment ID, and signature returned by the checkout
flow.

Only after successful verification is the recovery attempt marked as
completed.

The backend records:

-   Razorpay order ID
-   Razorpay payment ID
-   Payment status
-   Recovery attempt status
-   Completion timestamp
-   Audit event

This creates a clear distinction between:

``` text
Checkout UI success
```

and

``` text
Backend-verified payment success
```

------------------------------------------------------------------------

# 🧾 Payment Failure Handling

RecoveryOS also implements a dedicated payment-failure path.

When a runtime payment fails:

``` text
Payment → failed
RecoveryAttempt → failed
AuditLog → PAYMENT_FAILED
```

The failure callback does **not** automatically invoke another ML + AI +
policy cycle.

This is deliberate.

It avoids turning a payment failure callback into an uncontrolled
recursive recovery loop.

A future version could implement controlled re-evaluation using explicit
runtime state.

------------------------------------------------------------------------

# 🗄️ Database Design

RecoveryOS uses:

``` text
MySQL
   +
SQLAlchemy
```

The main entities are:

### Customer

Stores customer-level identity and recovery state.

### Payment

Stores payment transactions and their outcomes.

Important runtime fields include:

-   Payment ID
-   Amount
-   Currency
-   Status
-   Razorpay Order ID
-   Razorpay Payment ID

### RecoveryAttempt

Tracks recovery actions and their lifecycle.

Examples:

``` text
initiated
completed
failed
```

### AuditLog

Provides an event trail for important recovery actions.

Examples include:

``` text
PAYMENT_SUCCESS
PAYMENT_FAILED
```

------------------------------------------------------------------------

# 🛡️ Runtime Data Isolation

A subtle but important problem appears in systems that combine
historical ML data with live execution.

Suppose a demo creates:

``` text
PAY_RT_ABC123
```

If that transaction is immediately treated as historical behavior, then
running the same customer through the model again could change the
model's input.

That creates **inference contamination**.

RecoveryOS prevents this by explicitly excluding runtime records from
the historical feature-engineering queries.

Historical records remain:

``` text
PAY_*
REC_*
```

Runtime records remain:

``` text
PAY_RT_*
REC_RT_*
AUD_RT_*
```

This ensures that repeated demos do not progressively alter the
historical behavioral signals used by the model.

------------------------------------------------------------------------

# 🔁 `/predict` vs `/execute`

RecoveryOS separates analysis from execution.

## `/predict`

The prediction endpoint runs:

``` text
Customer Lookup
      ↓
Feature Engineering
      ↓
XGBoost
      ↓
Groq
      ↓
Policy
```

It is read-only.

It does not create payment or recovery records.

This makes it safe for analysis and UI previews.

------------------------------------------------------------------------

## `/execute`

Execution intentionally reruns the decision pipeline rather than
trusting a client-supplied action.

The frontend sends the recovery input again.

The backend independently determines:

``` text
XGBoost
   ↓
Groq
   ↓
Policy
```

Only then can execution occur.

This prevents the frontend from simply saying:

``` json
{
  "action": "PAYMENT_RETRY",
  "allowed": true
}
```

and bypassing the policy engine.

The client provides the **request context**.

The server determines the **decision**.

------------------------------------------------------------------------

# 🧩 API Overview

## Recovery Analysis

``` http
POST /api/recovery/predict
```

Runs the prediction → AI → policy pipeline.

------------------------------------------------------------------------

## Recovery Execution

``` http
POST /api/recovery/execute
```

Reruns the decision pipeline and, if authorized, creates the Razorpay
recovery transaction.

------------------------------------------------------------------------

## Payment Success

``` http
POST /api/recovery/payment-result
```

Verifies the Razorpay payment signature and completes the runtime
recovery attempt.

------------------------------------------------------------------------

## Payment Failure

``` http
POST /api/recovery/payment-failed
```

Records a failed runtime payment and recovery attempt.

------------------------------------------------------------------------

## API Documentation

When the backend is running, FastAPI exposes interactive Swagger
documentation at:

``` text
http://127.0.0.1:8001/docs
```

------------------------------------------------------------------------

# 🖥️ Frontend

The frontend is built with:

-   Next.js
-   React
-   TypeScript
-   Tailwind CSS

The UI is intentionally designed around the decision pipeline rather
than hiding the AI behind a generic "Pay" button.

The user can see:

### Recovery Probability

The quantitative ML signal.

### Groq AI Recommendation

The recommended action and confidence.

### AI Reasoning

The contextual explanation.

### Policy Engine Decision

Whether the action is actually permitted.

### Razorpay Execution

The final authorized payment action.

------------------------------------------------------------------------

# 🎬 Demo Flow

The recommended demonstration uses:

``` text
Customer ID:      CUST_000001
Amount:           ₹2,499
Currency:         INR
Payment Method:   UPI
Failure Reason:   insufficient_funds
Attempt Number:   1
```

The live flow is:

``` text
CUST_000001
     ↓
Recovery Analysis
     ↓
XGBoost
     ↓
Groq AI
     ↓
Policy = ALLOWED
     ↓
Recover ₹2,499
     ↓
Razorpay Test Checkout
     ↓
Payment Success
     ↓
Backend Signature Verification
     ↓
Recovery Attempt Completed
```

------------------------------------------------------------------------

# 📁 Project Structure

``` text
RecoveryOS/
│
├── backend/
│   │
│   ├── app/
│   │   ├── core/
│   │   │   └── config.py
│   │   │
│   │   ├── db/
│   │   │   ├── base.py
│   │   │   └── session.py
│   │   │
│   │   ├── ml/
│   │   │   ├── feature_engineering.py
│   │   │   ├── inference_features.py
│   │   │   └── predict.py
│   │   │
│   │   ├── models/
│   │   ├── routers/
│   │   │   └── recovery.py
│   │   │
│   │   ├── schemas/
│   │   │   └── recovery.py
│   │   │
│   │   ├── services/
│   │   │   ├── gemini_agent.py
│   │   │   ├── policy_engine.py
│   │   │   └── razorpay_service.py
│   │   │
│   │   └── main.py
│   │
│   ├── models/
│   │   ├── recovery_model.joblib
│   │   └── recovery_model_metadata.json
│   │
│   ├── scripts/
│   ├── tests/
│   ├── alembic/
│   └── requirements.txt
│
└── frontend/
    ├── app/
    ├── public/
    ├── package.json
    ├── tsconfig.json
    └── ...
```

> **Note:** `gemini_agent.py` is retained as a compatibility-oriented
> filename from the earlier implementation. The active AI provider is
> **Groq**, not Gemini.

------------------------------------------------------------------------

# ⚙️ Tech Stack

## Backend

  Technology   Purpose
  ------------ -----------------------------
  Python       Core backend language
  FastAPI      REST API
  SQLAlchemy   ORM
  MySQL        Persistent storage
  Alembic      Database migrations
  Pydantic     Request/response validation

## Machine Learning

  Technology   Purpose
  ------------ ---------------------------------
  XGBoost      Recovery probability prediction
  Pandas       Data processing
  NumPy        Numerical operations
  Joblib       Model serialization

## AI

  Technology             Purpose
  ---------------------- -----------------------------------
  Groq                   LLM inference
  `openai/gpt-oss-20b`   Recovery reasoning
  Pydantic               Structured AI response validation

## Payments

  Technology                       Purpose
  -------------------------------- --------------------------------
  Razorpay                         Payment execution
  Razorpay Test Mode               Safe demonstration
  HMAC-SHA256 / SDK verification   Payment signature verification

## Frontend

  Technology     Purpose
  -------------- -----------------
  Next.js        Web application
  React          UI
  TypeScript     Type safety
  Tailwind CSS   Styling

## Development

``` text
Git
VS Code
Postman
Docker
Swagger / OpenAPI
```

------------------------------------------------------------------------

# 🔐 Security & Trust Boundaries

RecoveryOS treats payment operations as a controlled backend
responsibility.

### Secrets stay server-side

Sensitive credentials such as:

``` text
GROQ_API_KEY
RAZORPAY_KEY_SECRET
DATABASE_URL
```

are stored in backend environment configuration.

They are never exposed through `NEXT_PUBLIC_*` frontend variables.

### The frontend cannot authorize itself

The frontend does not submit a trusted `allowed=true` decision.

The backend reruns the decision pipeline.

### AI cannot directly execute payments

Groq produces a recommendation.

The policy engine decides whether the recommendation is permitted.

### Payment success is verified

The backend validates the Razorpay signature before recording the
payment as recovered.

------------------------------------------------------------------------

# 🧪 Testing

RecoveryOS includes tests covering the major decision and execution
boundaries.

The test suite covers areas including:

-   Recovery prediction behavior
-   Policy authorization
-   Policy blocking
-   Retry limits
-   Reminder limits
-   Customer opt-out handling
-   Runtime-data isolation
-   Repeated demo stability
-   Razorpay service behavior
-   Payment-result handling
-   Payment-failure handling
-   Idempotency
-   Synthetic-data protection

The backend reached:

``` text
110 / 110 tests passing
```

before the final AI-provider migration.

The Groq integration was then separately validated through a focused
integration test that successfully returned:

``` json
{
  "action": "PAYMENT_RETRY",
  "reason": "...",
  "confidence": 0.95
}
```

------------------------------------------------------------------------

# 🧠 Important Design Decisions

## Why not let the LLM make the final decision?

Because language models are probabilistic systems.

Financial authorization should be deterministic.

Therefore:

``` text
LLM → recommendation
Policy → authorization
```

------------------------------------------------------------------------

## Why not just use XGBoost?

Because a probability does not provide contextual reasoning.

XGBoost can tell us:

``` text
PAYMENT_RETRY = 92.83%
```

But the AI can interpret that together with:

-   customer history
-   failure reason
-   recovery behavior
-   attempt counts
-   alternative actions

The two components solve different problems.

------------------------------------------------------------------------

## Why not just use an LLM?

Because structured historical behavior is better handled by a dedicated
predictive model.

XGBoost provides a reproducible quantitative signal.

The LLM then operates on top of that signal.

This creates a hybrid architecture rather than forcing one model to do
everything.

------------------------------------------------------------------------

## Why have a policy engine?

Because business constraints should not depend on model behavior.

Rules such as:

``` text
Maximum retries
Maximum recovery attempts
Maximum reminders
Customer opt-out
```

must be enforceable regardless of what the AI recommends.

------------------------------------------------------------------------

# 🧱 Failure-Safe Design

RecoveryOS follows a conservative principle:

> **When the system is uncertain about authorization, it should fail
> closed rather than execute an unsafe action.**

Examples:

``` text
Unknown AI action
       ↓
ESCALATE
       ↓
Not executable
```

``` text
Retry limit reached
       ↓
ESCALATE
       ↓
Not executable
```

``` text
Customer opted out
       ↓
STOP
       ↓
Not executable
```

This is intentional.

------------------------------------------------------------------------

# 🔄 Recovery State Model

A recovery action moves through explicit states.

### Payment

``` text
pending
   ↓
captured

or

pending
   ↓
failed
```

### Recovery Attempt

``` text
initiated
   ↓
completed

or

initiated
   ↓
failed
```

This makes execution state explicit rather than relying on UI
assumptions.

------------------------------------------------------------------------

# 📜 Auditability

Important recovery events are recorded through an audit log.

This creates an operational trail that can answer questions such as:

-   What recovery action was attempted?
-   Was it allowed by policy?
-   Did payment execution occur?
-   Did the payment succeed?
-   Was the result verified?
-   What runtime recovery record was associated with it?

The goal is not simply to automate recovery, but to make recovery
**explainable and traceable**.

------------------------------------------------------------------------

# 🧪 Example Decision

Consider:

``` text
Customer: CUST_000001
Amount: ₹2,499
Failure: insufficient_funds
Attempt: 1
```

XGBoost produces a strong retry probability.

The Groq agent receives:

``` text
Payment Context
+
Customer History
+
Recovery History
+
XGBoost Predictions
```

It returns:

``` json
{
  "action": "PAYMENT_RETRY",
  "confidence": 0.95
}
```

The policy engine then checks:

``` text
Opt-out?              No
Recovery limit?       Not reached
Retry limit?          Not reached
Action valid?         Yes
```

Result:

``` text
PAYMENT_RETRY
allowed = true
```

Only now does Razorpay execution become possible.

------------------------------------------------------------------------

# 🌐 Local Development

## 1. Clone the repository

``` bash
git clone <your-repository-url>
cd RecoveryOS
```

------------------------------------------------------------------------

## 2. Backend environment

Create a backend environment file:

``` env
DATABASE_URL=<your_mysql_connection_string>

GROQ_API_KEY=<your_groq_api_key>

RAZORPAY_KEY_ID=<your_razorpay_test_key_id>
RAZORPAY_KEY_SECRET=<your_razorpay_test_secret>
```

Never commit this file.

------------------------------------------------------------------------

## 3. Start the backend

``` powershell
cd backend
.venv\Scripts\activate
uvicorn app.main:app --reload --port 8001
```

Backend:

``` text
http://127.0.0.1:8001
```

Swagger:

``` text
http://127.0.0.1:8001/docs
```

------------------------------------------------------------------------

## 4. Start the frontend

``` powershell
cd frontend
npm install
npm run dev
```

Frontend:

``` text
http://localhost:3000
```

------------------------------------------------------------------------

# 🔑 Environment Variables

  -----------------------------------------------------------------------
  Variable                Used By                 Exposure
  ----------------------- ----------------------- -----------------------
  `DATABASE_URL`          Backend                 Server only

  `GROQ_API_KEY`          Groq AI                 Server only

  `RAZORPAY_KEY_ID`       Razorpay                Backend/frontend
                                                  integration as required

  `RAZORPAY_KEY_SECRET`   Razorpay verification   Server only

  `NEXT_PUBLIC_API_URL`   Frontend                Public
  -----------------------------------------------------------------------

**Never expose:**

``` text
GROQ_API_KEY
RAZORPAY_KEY_SECRET
```

through frontend environment variables.

------------------------------------------------------------------------

# 🚦 What Happens in Each Recovery Action?

## `PAYMENT_RETRY`

``` text
AI recommendation
       ↓
Policy authorization
       ↓
Razorpay order
       ↓
Checkout
       ↓
Backend verification
```

This is the only recovery action that currently triggers payment
execution.

------------------------------------------------------------------------

## `SEND_REMINDER`

The system records the recovery decision and presents it to the
frontend.

No external email/SMS provider is triggered.

This keeps the prototype focused on the recovery decision architecture.

------------------------------------------------------------------------

## `ESCALATE`

The system indicates that automated recovery should not proceed.

The frontend can present this as a manual-review state.

------------------------------------------------------------------------

# 🚧 Current Scope & Limitations

RecoveryOS is intentionally focused on demonstrating the core
intelligent recovery architecture.

### Current implementation

-   Razorpay Test Mode
-   Synthetic historical data
-   Automated payment retry execution
-   AI recommendation
-   Deterministic policy authorization
-   Backend payment verification
-   Audit logging

### Not currently implemented

-   Real-money production payments
-   Automated email/SMS delivery
-   Automatic failure → second AI decision loop
-   Production-scale model retraining pipeline
-   Human-review dashboard
-   Multi-provider payment orchestration

These are extension points rather than hidden assumptions.

------------------------------------------------------------------------

# 🔮 Future Roadmap

## Phase 1 --- More recovery channels

Extend beyond payment retry and reminders:

``` text
Payment Retry
SMS Reminder
Email Reminder
Payment Link
Human Escalation
Alternative Payment Method
```

------------------------------------------------------------------------

## Phase 2 --- Online learning

Continuously incorporate verified recovery outcomes into model
retraining.

``` text
Recovery Outcome
      ↓
Verified Historical Event
      ↓
Feature Store
      ↓
Model Retraining
      ↓
Improved Prediction
```

------------------------------------------------------------------------

## Phase 3 --- Controlled re-evaluation

After a failed recovery attempt, explicitly track runtime state and
perform a new decision cycle under controlled limits.

``` text
Retry Failed
    ↓
Update Runtime State
    ↓
Re-evaluate
    ↓
XGBoost
    ↓
AI
    ↓
Policy
```

------------------------------------------------------------------------

## Phase 4 --- Production-grade observability

Add:

-   Metrics
-   Tracing
-   Model monitoring
-   AI recommendation monitoring
-   Policy-decision analytics
-   Recovery conversion dashboards
-   Drift detection

------------------------------------------------------------------------

# 🏆 What Makes RecoveryOS Different?

RecoveryOS is not simply:

``` text
Payment Failed → Call LLM → Retry
```

It is:

``` text
                 ┌───────────────────┐
                 │ Historical Data   │
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │      XGBoost      │
                 │   Quantitative    │
                 │    Prediction     │
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │     Groq AI       │
                 │ Contextual        │
                 │ Reasoning         │
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │  Policy Engine    │
                 │ Deterministic     │
                 │ Authorization     │
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │    Razorpay       │
                 │    Execution      │
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │ Backend Verify    │
                 │ + Audit + State   │
                 └───────────────────┘
```

The architecture acknowledges that **prediction, reasoning,
authorization, and execution are different responsibilities**.

That separation is what makes the system controllable.

------------------------------------------------------------------------

# 📌 Project Summary

  Category           Implementation
  ------------------ ------------------------------
  Problem            Intelligent payment recovery
  ML                 XGBoost
  AI                 Groq + `openai/gpt-oss-20b`
  Backend            FastAPI
  Frontend           Next.js + React + TypeScript
  Database           MySQL
  ORM                SQLAlchemy
  Payments           Razorpay Test Mode
  Policy             Deterministic rule engine
  Validation         Pydantic
  Migrations         Alembic
  Testing            Pytest
  Deployment focus   Prototype / Buildathon

------------------------------------------------------------------------

# 🎥 Demo

The recommended demo uses:

``` text
Customer ID:      CUST_000001
Amount:           ₹2,499
Currency:         INR
Payment Method:   UPI
Failure Reason:   insufficient_funds
Attempt Number:   1
```

The demo demonstrates the complete pipeline:

> **XGBoost → Groq AI → Policy Engine → Razorpay → Backend
> Verification**

------------------------------------------------------------------------

# 👨‍💻 Built With

**RecoveryOS** was built as an end-to-end exploration of how machine
learning, generative AI, deterministic business rules, and payment
infrastructure can work together without giving any single component
unrestricted control.

The central design philosophy is simple:

> ### **Predict intelligently. Reason contextually. Authorize deterministically. Execute securely.**

------------------------------------------------------------------------

# ⭐ Final Takeaway

Payment recovery should not be a blind automation problem.

It should be a **decision system**.

RecoveryOS demonstrates how to build that system by combining:

``` text
Historical Behavior
        +
Machine Learning
        +
AI Reasoning
        +
Deterministic Policies
        +
Verified Payment Execution
```

into one end-to-end workflow.

> **RecoveryOS --- Intelligent recovery, controlled execution.**
