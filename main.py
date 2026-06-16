import io
import csv
import os
import json
import tempfile
import time
import logging
import hashlib
import urllib.parse
from datetime import datetime, timedelta
import stripe
from sqlalchemy import create_engine, text
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
try:
    from deepgram import DeepgramClient, PrerecordedOptions
except Exception as _deepgram_import_err:  # SDK API drift (e.g. v4 removed PrerecordedOptions) must never crash startup
    import logging as _logging
    _logging.warning(f"Deepgram SDK import failed, transcription disabled: {_deepgram_import_err}")
    DeepgramClient = None
    PrerecordedOptions = None
from fastapi.responses import StreamingResponse, Response, JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from groq import Groq
from cartesia import Cartesia
from dotenv import load_dotenv
import email_service
import uuid as _uuid
import secrets as _secrets
import random as _random
import asyncio
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_START_TIME = time.time()  # process start — used for uptime reporting in /health

load_dotenv()

# ========================
# SENTRY ERROR TRACKING
# ========================
_SENTRY_DSN = os.environ.get("SENTRY_DSN", "")
_SENTRY_ENV = os.environ.get("SENTRY_ENVIRONMENT", os.environ.get("RAILWAY_ENVIRONMENT", "production"))
_API_VERSION = "1.2.0"

_SCRUB_KEYS = frozenset({
    "transcript", "answer", "context", "work_history",
    "preference_text", "session_history", "prior_summaries",
    "session_context", "user_preferences", "audio",
    "history",  # session history arrays
})

def _sentry_before_send(event: dict, hint: dict) -> dict | None:
    """Drop expected 4xx HTTP errors and scrub user-content fields."""
    exc_info = hint.get("exc_info")
    if exc_info:
        exc = exc_info[1]
        # 4xx errors are business logic (bad input, auth, credits) — not bugs
        if isinstance(exc, HTTPException) and exc.status_code < 500:
            return None

    # Scrub any form/body data that could contain user content
    req = event.get("request", {})
    if isinstance(req.get("data"), dict):
        req["data"] = {
            k: ("[redacted]" if k in _SCRUB_KEYS else v)
            for k, v in req["data"].items()
        }

    return event

if _SENTRY_DSN:
    sentry_sdk.init(
        dsn=_SENTRY_DSN,
        environment=_SENTRY_ENV,
        release=f"cerebroecho-backend@{_API_VERSION}",
        integrations=[
            StarletteIntegration(transaction_style="endpoint"),
            FastApiIntegration(transaction_style="endpoint"),
        ],
        traces_sample_rate=0.05,   # 5% of transactions — enough for perf insight
        send_default_pii=False,    # never send IP addresses, cookies, or auth headers
        before_send=_sentry_before_send,
    )
    logger.info(f"✅ Sentry initialized (env={_SENTRY_ENV})")
else:
    logger.info("ℹ️  Sentry disabled — set SENTRY_DSN to enable")

app = FastAPI(title="CerebroEcho API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cerebroecho.com", "https://www.cerebroecho.com", "https://cerebroecho-frontend.vercel.app",
                   "https://hq.cerebroecho.com"],  # hq = standalone owner Command Center (separate app)
    # Allow Vercel PREVIEW deploys of the frontend (hashed URLs) so V2/feature branches are testable
    # without touching production. Scoped to this project's previews only.
    allow_origin_regex=r"https://cerebroecho-frontend-[a-z0-9-]+-cerebroechos-projects\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transcript", "X-Answer", "X-Questions-Used"],
)

from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# ========================
# RATE LIMITING
# ========================
# Uses in-memory store (single-instance safe on Railway).
# NOTE: if Railway ever scales to multiple instances, swap Limiter() for
#       Limiter(storage_uri=os.environ.get("REDIS_URL")) to share state.

def get_client_ip(request: Request) -> str:
    """Resolve real client IP — Railway sits behind a proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return (request.client.host if request.client else None) or "unknown"

# Configurable via Railway env vars — change without redeploying code
RATE_COACH       = os.environ.get("RATE_LIMIT_COACH",       "30/minute")
RATE_TTS         = os.environ.get("RATE_LIMIT_TTS",         "30/minute")
RATE_BRIEF       = os.environ.get("RATE_LIMIT_BRIEF",       "5/minute")
RATE_UPLOAD      = os.environ.get("RATE_LIMIT_UPLOAD",      "10/minute")
RATE_TRANSCRIBE  = os.environ.get("RATE_LIMIT_TRANSCRIBE",  "20/minute")
RATE_SESSION_END = os.environ.get("RATE_LIMIT_SESSION_END", "10/minute")
RATE_CHECKOUT    = os.environ.get("RATE_LIMIT_CHECKOUT",    "10/minute")
RATE_READS        = os.environ.get("RATE_LIMIT_READS",        "60/minute")
RATE_AUTH_REQUEST = os.environ.get("RATE_LIMIT_AUTH_REQUEST", "5/minute")
RATE_AUTH_VERIFY  = os.environ.get("RATE_LIMIT_AUTH_VERIFY",  "10/minute")

limiter = Limiter(key_func=get_client_ip)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(f"⚠️ Rate limit exceeded: {request.url.path} from {get_client_ip(request)} — {exc.detail}")
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": f"Too many requests ({exc.detail}). Please wait before retrying.",
            "retry_after_seconds": 60,
        },
        headers={"Retry-After": "60"},
    )

# Safe Groq client initialization
client = None
try:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("⚠️ GROQ_API_KEY not set")
    else:
        client = Groq(api_key=api_key)
        logger.info("✅ Groq client initialized successfully")
except Exception as e:
    logger.error(f"❌ ERROR creating Groq client: {e}")
    client = None

cartesia_client = None
try:
    cartesia_api_key = os.environ.get("CARTESIA_API_KEY")
    if not cartesia_api_key:
        logger.warning("⚠️ CARTESIA_API_KEY not set")
    else:
        cartesia_client = Cartesia(api_key=cartesia_api_key)
        logger.info("✅ Cartesia client initialized successfully")
except Exception as e:
    logger.error(f"❌ ERROR creating Cartesia client: {e}")
    cartesia_client = None

# Map OpenAI voice names to Cartesia voice IDs
# Update these IDs from https://play.cartesia.ai/voices
CARTESIA_VOICE_MAP = {
    "onyx":   "638efaaa-4d0c-442e-b701-3fae16aad012",  # deep male
    "nova":   "a0e99841-438c-4a64-b679-ae501e7d6091",  # female
    "alloy":  "248be419-c632-4f23-adf1-5324ed7dbf1d",  # neutral
    "echo":   "15a9cd88-84b0-4a8b-95f2-5d583b54c72e",  # male
    "fable":  "79a125e8-cd45-4c13-8a67-188112f4dd22",  # expressive
    "shimmer":"e3827ec5-697a-4b7c-9704-1a23041bbc51",  # warm female
}

deepgram_client = None
try:
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        logger.warning("⚠️ DEEPGRAM_API_KEY not set")
    else:
        deepgram_client = DeepgramClient(deepgram_api_key)
        logger.info("✅ Deepgram client initialized successfully")
except Exception as e:
    logger.error(f"❌ ERROR creating Deepgram client: {e}")
    deepgram_client = None

# ========================
# CREDIT SYSTEM
# ========================
PLAN_CREDITS = {
    "free":        20,       # monthly reset, ~20 text-only Q&As/month (text-only, no audio)
    "echo":        4000,     # Solo $9.99/mo   ← STRIPE_PRICE_ECHO   (now incl. ~1 hr audio)
    "pro":         75000,    # Pro $29.99/mo   ← STRIPE_PRICE_PRO    (~20 hrs audio)
    "command":     150000,   # Power $59.99/mo ← STRIPE_PRICE_COMMAND (~40 hrs audio)
    "operator":    150000,   # Operator (hidden, parity w/ Power) ← STRIPE_PRICE_OPERATOR
    "founding_50": 14000,    # LTD $299 — existing members honored ← STRIPE_PRICE_FOUNDING50
}
# NOTE (Jun 7 2026): values resized for FULL usage-based metering (TTS now charged).
# ~3,600 credits ≈ 1 hr of live audio. Stripe Dashboard prices for pro/command MUST be
# updated to $29.99/$59.99 separately — changing these numbers does NOT change what Stripe bills.

STYLE_COSTS = {
    "Quick":     1,
    "Standard":  2,
    "Full":      4,
    "Nudge":     1,
    "Brief":     2,
    "shorthand": 1,
    "bullet":    2,
    "script":    2,
}

# ── METERED BILLING ───────────────────────────────────────────────────────
# True usage-based billing: measure each action's raw API cost in USD, convert
# to credits at a fixed cost-basis, charge that. STYLE_COSTS above is retained
# ONLY for mapping Quick/Standard/Full → max_tokens, never as the charge.
import math
CREDIT_COST_BASIS_USD = float(os.environ.get("CREDIT_COST_BASIS_USD", "0.000143"))
# Provider unit costs in USD. Env-overridable. VERIFY against live provider pricing.
PROVIDER_RATES = {
    "llama-3.1-8b-instant":     {"in": 0.05e-6, "out": 0.08e-6},   # $/token
    "llama-3.3-70b-versatile":  {"in": 0.59e-6, "out": 0.79e-6},   # $/token
    "whisper":                  {"sec": 0.04/3600},                # $/audio-second ($0.04/hr, whisper-large-v3-turbo — verified Jun 7 2026)
    "cartesia_tts":             {"char": 0.000035},                # $/char (~$35/1M chars, Cartesia Sonic 3 — verified Jun 7 2026)
    "openai_tts":               {"char": 15e-6},                   # $/char (tts-1)
    "jina_fetch":               {"req": 0.0008},                   # $/request ← VERIFY
}


def _llm_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    r = PROVIDER_RATES.get(model, {"in": 0.59e-6, "out": 0.79e-6})
    return (prompt_tokens or 0) * r["in"] + (completion_tokens or 0) * r["out"]


def _cost_to_credits(raw_cost_usd: float) -> int:
    if raw_cost_usd <= 0:
        return 1
    return max(1, math.ceil(raw_cost_usd / CREDIT_COST_BASIS_USD))


def _get_credit_balance(user_id: str) -> dict:
    """Get or create credits record; returns {balance, plan_type, total_used, payg_enabled}."""
    if engine is None:
        return {"balance": 9999, "plan_type": "free", "total_used": 0, "payg_enabled": False}
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO credits (user_id, balance, plan_type, total_used)
                VALUES (:uid, 60, 'free', 0)
                ON CONFLICT (user_id) DO NOTHING
            """), {"uid": user_id})
            conn.commit()
            row = conn.execute(
                text("SELECT balance, plan_type, total_used, payg_enabled FROM credits WHERE user_id=:uid"),
                {"uid": user_id}
            ).fetchone()
        if row:
            return {"balance": row[0], "plan_type": row[1], "total_used": row[2], "payg_enabled": bool(row[3])}
    except Exception as e:
        logger.error(f"_get_credit_balance error: {e}")
        sentry_sdk.capture_exception(e)
    return {"balance": 9999, "plan_type": "free", "total_used": 0, "payg_enabled": False}


def _deduct_credits(user_id: str, cost: int, feature: str = None, idempotency_key: str = None,
                    provider: str = None, input_units: float = None, output_units: float = None,
                    raw_cost_usd: float = None, status: str = "charged") -> int:
    with engine.begin() as conn:
        # Check idempotency — if this key was already processed, return current balance.
        # Short-circuit BEFORE any usage_events write so we never duplicate a metering row.
        if idempotency_key:
            existing = conn.execute(text("""
                SELECT balance_after FROM credit_ledger
                WHERE idempotency_key = :key
            """), {"key": idempotency_key}).fetchone()
            if existing:
                return existing[0]

        # Lock the row — no other request can touch this user's balance until we commit
        result = conn.execute(text("""
            SELECT balance, payg_enabled FROM credits
            WHERE user_id = :uid
            FOR UPDATE
        """), {"uid": user_id}).fetchone()

        if not result:
            return 0

        current_balance = result[0]
        payg_enabled = bool(result[1])

        if cost > 0 and current_balance < cost:
            if not payg_enabled:
                return -1
            # PAYG: allow negative balance at 1.5x cost (30x markup vs 20x)
            actual_cost = round(cost * 1.5)
            new_balance = current_balance - actual_cost
            op_type = "payg_usage"
        else:
            actual_cost = cost
            new_balance = max(0, current_balance - cost)
            op_type = "usage"

        # Update the balance
        conn.execute(text("""
            UPDATE credits
            SET balance = :new_balance,
                total_used = total_used + :cost
            WHERE user_id = :uid
        """), {"new_balance": new_balance, "cost": actual_cost, "uid": user_id})

        # Write immutable ledger entry
        conn.execute(text("""
            INSERT INTO credit_ledger
            (user_id, amount, balance_after, operation_type, feature, idempotency_key)
            VALUES (:uid, :amount, :balance_after, :op_type, :feature, :key)
        """), {
            "uid": user_id,
            "amount": -actual_cost,
            "balance_after": new_balance,
            "op_type": op_type,
            "feature": feature,
            "key": idempotency_key
        })

        # Metered usage_events row (same atomic transaction). Only when a provider
        # is supplied. credits_charged = actual_cost (post-PAYG amount). A failure
        # here must NOT abort the credit deduction — log and continue.
        if provider is not None:
            try:
                conn.execute(text("""
                    INSERT INTO usage_events
                    (user_id, feature, provider, input_units, output_units,
                     raw_cost_usd, credits_charged, status, idempotency_key)
                    VALUES
                    (:uid, :feature, :provider, :in_units, :out_units,
                     :raw_cost, :credits, :status, :key)
                """), {
                    "uid":       user_id,
                    "feature":   feature,
                    "provider":  provider,
                    "in_units":  input_units,
                    "out_units": output_units,
                    "raw_cost":  raw_cost_usd,
                    "credits":   actual_cost,
                    "status":    status,
                    "key":       idempotency_key,
                })
            except Exception as _ue:
                logger.error(f"usage_events insert failed (non-fatal): {_ue}")

        return new_balance


def _track_usage(user_key: str, user_email: str, event_type: str, units: float, estimated_cost_usd: float, metadata: str | None = None) -> None:
    """Insert a cost-tracking row. Non-fatal — never raises to caller."""
    if engine is None:
        return
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO api_usage (user_key, user_email, event_type, units, estimated_cost_usd, metadata)
                VALUES (:uk, :ue, :et, :units, :cost, :meta)
            """), {
                "uk":    user_key or "unknown",
                "ue":    user_email or "",
                "et":    event_type,
                "units": float(units),
                "cost":  float(estimated_cost_usd),
                "meta":  metadata,
            })
            conn.commit()
    except Exception as e:
        logger.warning(f"_track_usage failed ({event_type}): {e}")


def _get_credits_user_id(fallback_user_id: str, account_id: str | None) -> str:
    """Return the canonical user_id for credit operations.

    Authenticated users may have credits stored under a different user_id
    (e.g. from a prior device). Looking up by account_id ensures cross-device
    deductions always hit the same row. Falls back to the composite key for
    anonymous users or when no existing row is found.
    """
    if account_id is None or engine is None:
        return fallback_user_id
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT user_id FROM credits WHERE account_id = :aid ORDER BY balance DESC LIMIT 1"),
                {"aid": account_id}
            ).fetchone()
        if row:
            return row[0]
    except Exception as e:
        logger.error(f"_get_credits_user_id error: {e}")
    return fallback_user_id


def reset_expired_credits():
    """Reset balances for any user whose reset_date has passed (runs at startup)."""
    if engine is None:
        return
    now = datetime.utcnow().isoformat()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT user_id, plan_type FROM credits
                WHERE reset_date IS NOT NULL AND reset_date <= :now
            """), {"now": now}).fetchall()
            for uid, plan in rows:
                new_bal = PLAN_CREDITS.get(plan, PLAN_CREDITS["free"])
                new_reset = (datetime.utcnow() + timedelta(days=30)).isoformat()
                conn.execute(text("""
                    UPDATE credits
                    SET balance = :bal, reset_date = :reset, total_used = 0,
                        audio_seconds_used = 0
                    WHERE user_id = :uid
                """), {"bal": new_bal, "reset": new_reset, "uid": uid})
            if rows:
                conn.commit()
                logger.info(f"🔄 Monthly credits reset for {len(rows)} users")
    except Exception as e:
        logger.error(f"reset_expired_credits error: {e}")
        sentry_sdk.capture_exception(e)


usage_tracker = {}  # kept for non-primary endpoints only


# ========================
# ONBOARDING EMAIL HELPERS
# ========================

def _log_onboarding_email(account_id: str, email_type: str) -> None:
    """Record that an onboarding email was sent or handled. Idempotent via UNIQUE constraint."""
    if engine is None or not account_id:
        return
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO email_log (account_id, email_type, sent_at)
                VALUES (:aid, :etype, :now)
                ON CONFLICT (account_id, email_type) DO NOTHING
            """), {"aid": account_id, "etype": email_type, "now": datetime.utcnow().isoformat()})
            conn.commit()
    except Exception as e:
        logger.warning(f"_log_onboarding_email error: {e}")


def _has_onboarding_email_sent(account_id: str, email_type: str) -> bool:
    if engine is None or not account_id:
        return False
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM email_log WHERE account_id = :aid AND email_type = :etype"),
                {"aid": account_id, "etype": email_type}
            ).fetchone()
        return row is not None
    except Exception as e:
        logger.warning(f"_has_onboarding_email_sent error: {e}")
        return False


def _process_onboarding_emails() -> None:
    """Send pending 24h and 72h onboarding emails. Safe to call repeatedly — email_log prevents duplicates."""
    if engine is None:
        return
    now = datetime.utcnow()
    cutoff_24h = (now - timedelta(hours=24)).isoformat()
    cutoff_72h = (now - timedelta(hours=72)).isoformat()
    try:
        with engine.connect() as conn:
            # Email 2: signed up >24h ago, 0 sessions logged against their account, no 24h email yet
            candidates_24h = conn.execute(text("""
                SELECT a.id, a.email FROM accounts a
                WHERE a.created_at <= :cutoff
                AND NOT EXISTS (
                    SELECT 1 FROM email_log el
                    WHERE el.account_id = a.id AND el.email_type = 'onboarding_24h'
                )
                AND NOT EXISTS (
                    SELECT 1 FROM session_history sh WHERE sh.account_id = a.id
                )
                LIMIT 50
            """), {"cutoff": cutoff_24h}).fetchall()

            # Email 3: signed up >72h ago, no 72h email yet; filter <3 sessions in Python
            candidates_72h = conn.execute(text("""
                SELECT a.id, a.email,
                    (SELECT COUNT(*) FROM session_history sh WHERE sh.account_id = a.id) AS sess_count
                FROM accounts a
                WHERE a.created_at <= :cutoff
                AND NOT EXISTS (
                    SELECT 1 FROM email_log el
                    WHERE el.account_id = a.id AND el.email_type = 'onboarding_72h'
                )
                LIMIT 50
            """), {"cutoff": cutoff_72h}).fetchall()

        for row in candidates_24h:
            account_id, to_email = row[0], row[1]
            email_service.send_onboarding_setup(to_email)
            _log_onboarding_email(account_id, "onboarding_24h")
            _track_usage(account_id, to_email, "resend_email", 1, 0.0001,
                         json.dumps({"type": "onboarding_24h"}))
            logger.info(f"📧 onboarding_24h sent to {to_email}")

        for row in candidates_72h:
            account_id, to_email, sess_count = row[0], row[1], int(row[2])
            if sess_count < 3:
                email_service.send_onboarding_checkin(to_email)
                _track_usage(account_id, to_email, "resend_email", 1, 0.0001,
                             json.dumps({"type": "onboarding_72h"}))
                logger.info(f"📧 onboarding_72h sent to {to_email} ({sess_count} sessions)")
            else:
                logger.info(f"📧 onboarding_72h skipped: {to_email} already has {sess_count} sessions")
            _log_onboarding_email(account_id, "onboarding_72h")

    except Exception as e:
        logger.error(f"_process_onboarding_emails error: {e}")
        sentry_sdk.capture_exception(e)


# ========================
# SESSION CONTEXT CACHE
# Stores uploaded file text or brief text in the preferences table under a
# special user_id key so context survives a page refresh within a session.
# Cleared automatically when /end-session is called.
# ========================

def _save_ctx_cache(user_key: str, content: str) -> None:
    # NOTE: param is named `content`, NOT `text` — `text` is sqlalchemy.text (line 12).
    # Shadowing it here previously made every cache write raise "'str' object is not
    # callable" and fail silently (caught below), so context never persisted.
    if engine is None or not user_key or not content:
        return
    cache_id = f"{user_key}::ctx_cache"
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO preferences (user_id, preference_text, updated_at)
                VALUES (:uid, :txt, :now)
                ON CONFLICT (user_id) DO UPDATE SET
                    preference_text = EXCLUDED.preference_text,
                    updated_at      = EXCLUDED.updated_at
            """), {"uid": cache_id, "txt": content, "now": datetime.utcnow().isoformat()})
            conn.commit()
    except Exception as e:
        logger.warning(f"_save_ctx_cache failed: {e}")


def _get_ctx_cache(user_key: str) -> str:
    if engine is None or not user_key:
        return ""
    cache_id = f"{user_key}::ctx_cache"
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT preference_text FROM preferences WHERE user_id = :uid"),
                {"uid": cache_id}
            ).fetchone()
        return row[0] if row else ""
    except Exception as e:
        logger.warning(f"_get_ctx_cache failed: {e}")
        return ""


def _clear_ctx_cache(user_key: str) -> None:
    if engine is None or not user_key:
        return
    cache_id = f"{user_key}::ctx_cache"
    try:
        with engine.connect() as conn:
            conn.execute(
                text("DELETE FROM preferences WHERE user_id = :uid"),
                {"uid": cache_id}
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"_clear_ctx_cache failed: {e}")

# Keyed by Stripe customer ID → {"plan": str, "status": "active"|"payment_failed"|"revoked"}
customer_plan: dict = {}

# ========================
# AUTH CONFIG
# ========================
MAGIC_LINK_EXPIRY_MINUTES = int(os.environ.get("MAGIC_LINK_EXPIRY_MINUTES", "30"))
OTP_EXPIRY_MINUTES        = int(os.environ.get("OTP_EXPIRY_MINUTES",        "10"))
SESSION_EXPIRY_DAYS       = int(os.environ.get("SESSION_EXPIRY_DAYS",       "30"))
APP_BASE_URL              = os.environ.get("APP_BASE_URL", "https://cerebroecho.com/app")

# ========================
# RETROSPECTIVE QUESTION ASSEMBLY
# ========================
# Keyed by user_key → {"transcript": str, "timestamp": float}
question_buffer: dict = {}
CONTINUATION_TIMEOUT_S = 15.0

FRAGMENT_ENDINGS = [
    r'tell me about$', r'tell us about$', r'what would you$', r'how do you$',
    r'can you$', r'could you$', r'describe$', r'explain$',
    r'walk me$', r'walk us$', r'give me$', r'give us$',
    r'what is$', r'what are$', r'why did$', r'when did$',
    r'how did$', r'have you$', r'do you$', r'did you$',
    r'are you$', r'were you$', r'would you$', r'should you$',
    r'tell me$', r'tell us$',
]

def _looks_like_continuation(prev: str, new: str) -> bool:
    """True if `new` appears to continue an incomplete `prev`."""
    import re
    prev = prev.strip()
    new = new.strip()
    new_words = new.split()

    # Very short new fragment (< 5 words) is almost always a continuation
    if len(new_words) < 5:
        return True

    # Starts with a connector/subordinator — mid-thought continuation
    if re.match(
        r'^(and|but|or|so|because|since|while|when|if|that|which|who|'
        r'specifically|particularly|for example|such as|in terms of|regarding)\b',
        new, re.IGNORECASE
    ):
        return True

    # Previous ended as an incomplete phrase (no terminal punctuation + known fragment ending)
    prev_no_punct = not re.search(r'[.?!]$', prev)
    prev_is_fragment = prev_no_punct and any(
        re.search(p, prev, re.IGNORECASE) for p in FRAGMENT_ENDINGS
    )
    if prev_is_fragment:
        return True

    return False

# ========================
# SYSTEM PROMPT CONSTRUCTION
# ========================

ROLE_PROMPTS = {
    "Interview Coach": (
        "You are a real-time interview coaching assistant. Help the user "
        "translate their experience into sharp, relevant, confident answers. "
        "Flag weak phrasing. Suggest stronger framing. Keep answers concise and story-driven."
    ),
    "Sales": (
        "You are a real-time sales coaching assistant. Help the user handle "
        "objections, clarify value, and move the conversation toward a decision. "
        "Be direct. Focus on outcomes, not features."
    ),
    "Negotiation": (
        "You are a real-time negotiation assistant. Help the user find leverage, "
        "protect their downside, and avoid saying too much. Know when to hold silence. "
        "Never concede without getting something back."
    ),
    "Customer Service": (
        "You are a real-time customer service assistant. Help the user stay calm, "
        "clear, and solution-focused even when the conversation gets tense. "
        "De-escalate when needed. Never match aggression."
    ),
    "Social/Dating": (
        "You are a real-time social coaching assistant. Help the user stay engaging, "
        "genuine, and confident in social or dating conversations. "
        "Keep it light when the moment calls for it. Read the room."
    ),
    "Fact Checker": (
        "You are a real-time fact checking assistant. Flag incorrect or misleading "
        "claims as they appear in conversation. Be precise. Cite your reasoning. "
        "Do not editorialize."
    ),
    "Custom": """
You are operating in Custom Persona mode.
The user has defined their own persona for this session. Apply the user's custom persona description as your primary behavioral directive. Adapt your coaching style, tactics, and responses to serve whatever scenario the user has defined. If no custom description was provided, default to general conversation intelligence — help the user navigate the conversation with clarity and confidence.
""",
}

MODE_MODIFIERS = {
    "Diplomat": (
        "Respond with measured precision. De-escalate where possible. "
        "Choose words that preserve relationships while holding position."
    ),
    "Strategist": (
        "See the full board. Frame every response around tradeoffs, timing, and leverage. "
        "Think three moves ahead."
    ),
    "Closer": (
        "Be direct and outcome-oriented. Every response should move something forward. "
        "Minimal warmth. Maximum clarity."
    ),
    "Coach": (
        "Be calm, supportive, and human. Acknowledge emotion before strategy. "
        "Make the user feel capable."
    ),
    "Analyst": (
        "Lead with evidence. Be skeptical of assumptions. Flag weak logic. "
        "Keep opinions clearly labeled as opinions."
    ),
    "Visionary": (
        "Think expansively. Connect dots others miss. "
        "Reframe every problem around what becomes possible rather than what currently exists. "
        "Lead with the bigger picture before the tactical detail."
    ),
    "Custom": """
The user has defined their own tone for this session. Apply the user's custom tone description to shape how you think, respond, and frame suggestions. Honor the spirit and style of their definition. If no custom description was provided, respond with balanced judgment — direct but not aggressive, clear but not cold.
""",
    "Pirate": """
Respond in the voice of a classic cartoon pirate — bold, theatrical, colorful with nautical language and pirate vocabulary. Despite the colorful delivery the actual advice must still be genuinely useful and tactically sound. Never break character.
""",
}

STYLE_CONFIG = {
    # No MINIMUMS on any length (owner 2026-06-16): a great short answer beats padded junk. These are
    # soft CAPS + a quality-first directive, never a floor. Kept lean to minimise prompt size (latency).
    "Quick":     {"instruction": "LENGTH = QUICK: answer in as few words as possible while still being genuinely useful — a short phrase or one sentence. Hard cap ~25 words. Never pad to fill space.", "max_tokens": 70},
    "Standard":  {"instruction": "LENGTH = STANDARD: a focused reply, at most ~3 sentences (~55 words). Be as brief as the answer allows — if one sentence nails it, stop there. Never pad.", "max_tokens": 160},
    "Full":      {"instruction": "LENGTH = FULL: you MAY develop a thorough answer (up to ~220 words) with specifics and reasoning WHEN depth genuinely helps the user. But if a shorter answer is better, give the shorter one — never add filler to hit a length.", "max_tokens": 550},
    "Nudge":     {"instruction": "Respond in one line or two short bullets maximum. Under 300 characters. Fast and surgical.", "max_tokens": 80},
    "Brief":     {"instruction": "Respond in one short paragraph. Under 800 characters. Balanced and tactical.", "max_tokens": 220},
    "shorthand": {"instruction": "Respond in one line or two short bullets maximum. Under 300 characters.", "max_tokens": 80},
    "bullet":    {"instruction": "Provide 3 tactical bullet points. Max 100 words total.", "max_tokens": 200},
    "script":    {"instruction": "Provide a 2-3 sentence tactical answer. Be concise and confident.", "max_tokens": 150},
}

_DEFAULT_ROLE = (
    "You are a real-time conversation coaching assistant. "
    "Help the user navigate the conversation with clarity and confidence."
)


def build_system_prompt(
    role: str,
    mode: str,
    style: str,
    context: str = "",
    work_history: str = "",
    prior_context: str = "",
    session_context: str = "",
    user_preferences: str = "",
    custom_role_description: str = "",
    custom_mode_description: str = "",
) -> tuple:
    parts = [ROLE_PROMPTS.get(role, _DEFAULT_ROLE)]

    if context and context not in ("", "a professional role"):
        if role == "Custom":
            parts.append(f"User-defined custom identity description: {context}")
        else:
            parts.append(f"Session context: {context}")
    if work_history and work_history not in ("", "no work history provided"):
        parts.append(f"User background: {work_history}")
    if prior_context:
        parts.append(prior_context)
    # FIX 2/7: custom_role_description is the dedicated field for operator custom identity text.
    # When present it is labeled explicitly so Groq knows what it is; session_context
    # is appended alongside it as "Additional context" rather than a separate block.
    if custom_role_description and custom_role_description.strip():
        if session_context and session_context.strip():
            parts.append(
                f"User-defined custom identity: {custom_role_description.strip()}\n\n"
                f"Additional context:\n{session_context.strip()[:8000]}"
            )
        else:
            parts.append(f"User-defined custom identity: {custom_role_description.strip()}")
    elif session_context and session_context.strip():
        parts.append(f"Context provided by user:\n{session_context.strip()[:8000]}")
    mode_mod = MODE_MODIFIERS.get(mode, "")
    # FIX 3: when mode is Custom and the user typed a description, label it
    # explicitly so Groq understands it as the mode directive, not generic context.
    if mode == "Custom" and custom_mode_description and custom_mode_description.strip():
        parts.append(f"User-defined custom tone: {custom_mode_description.strip()}")
    elif mode_mod:
        parts.append(mode_mod)

    style_cfg = STYLE_CONFIG.get(style, STYLE_CONFIG["Standard"])
    parts.append(style_cfg["instruction"])
    parts.append(
        "STAY ON TOPIC: Anchor every suggestion to the CURRENT live conversation and respond directly "
        "to what the other party just said. Do NOT introduce, pitch, or redirect toward a product, company, "
        "brand, app, or agenda drawn from memory or prior context unless the other party has actually raised it "
        "and it is directly relevant to what they are discussing right now. Treat any company/product names in "
        "memory or prior context as background about the user only — never steer the conversation toward them."
    )
    parts.append(
        "Return ONLY the answer the user should say. "
        "No preamble, no labels, no meta-commentary. Start directly with the answer."
    )

    # Length reminder near the END for high recency — small models under-weight a length
    # instruction buried mid-prompt, especially the long-form "Full" mode. Placed before
    # user_preferences so a user's explicit length instruction can still override it.
    if style_cfg.get("reminder"):
        parts.append(style_cfg["reminder"])

    # User preferences placed LAST so the model sees them with highest recency weight.
    if user_preferences and user_preferences.strip():
        parts.append(
            f"CRITICAL USER INSTRUCTIONS — highest priority, always follow these exactly: "
            f"{user_preferences.strip()}"
        )

    return "\n\n".join(parts), style_cfg["max_tokens"]


# ========================
# DATABASE — PostgreSQL via SQLAlchemy
# ========================
_raw_db_url = os.environ.get("DATABASE_URL", "")
# Railway sometimes issues postgres:// — SQLAlchemy 2.x requires postgresql://
if _raw_db_url.startswith("postgres://"):
    _raw_db_url = _raw_db_url.replace("postgres://", "postgresql://", 1)

engine = None
if _raw_db_url:
    try:
        engine = create_engine(_raw_db_url, pool_pre_ping=True, pool_size=5, max_overflow=10)
        logger.info("✅ PostgreSQL engine created")
    except Exception as e:
        logger.error(f"❌ Failed to create DB engine: {e}")
else:
    logger.warning("⚠️ DATABASE_URL not set — session persistence disabled")


def init_db():
    if engine is None:
        return
    ddl = [
        # Existing table — preserved exactly
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id          SERIAL PRIMARY KEY,
            device_id   TEXT NOT NULL,
            user_email  TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            summary     TEXT NOT NULL
        )
        """,
        # Credits / plan tracking per user
        """
        CREATE TABLE IF NOT EXISTS credits (
            id          SERIAL PRIMARY KEY,
            user_id     TEXT NOT NULL UNIQUE,
            balance     INTEGER NOT NULL DEFAULT 0,
            plan_type   TEXT NOT NULL DEFAULT 'free',
            reset_date  TEXT,
            total_used  INTEGER NOT NULL DEFAULT 0
        )
        """,
        # Session history log
        """
        CREATE TABLE IF NOT EXISTS session_history (
            id               SERIAL PRIMARY KEY,
            session_id       TEXT NOT NULL,
            user_id          TEXT NOT NULL,
            role             TEXT,
            persona          TEXT,
            style            TEXT,
            summary          TEXT,
            timestamp        TEXT NOT NULL,
            duration_seconds INTEGER NOT NULL DEFAULT 0
        )
        """,
        # Long-term user preferences for memory
        """
        CREATE TABLE IF NOT EXISTS preferences (
            id               SERIAL PRIMARY KEY,
            user_id          TEXT NOT NULL UNIQUE,
            preference_text  TEXT NOT NULL DEFAULT '',
            updated_at       TEXT NOT NULL
        )
        """,
        # Founding 50 members
        """
        CREATE TABLE IF NOT EXISTS founding_members (
            id                SERIAL PRIMARY KEY,
            user_id           TEXT NOT NULL UNIQUE,
            purchase_date     TEXT NOT NULL,
            stripe_payment_id TEXT NOT NULL
        )
        """,
        # Stripe customer ID → user_id mapping (for subscription cancellation)
        """
        CREATE TABLE IF NOT EXISTS stripe_customers (
            customer_id TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
        """,
        # Unique index on session_id so end-session is idempotent
        "CREATE UNIQUE INDEX IF NOT EXISTS session_history_sid_uniq ON session_history(session_id)",
        # Idempotency log — prevents duplicate Stripe webhook processing on retries
        """
        CREATE TABLE IF NOT EXISTS processed_webhook_events (
            event_id     TEXT PRIMARY KEY,
            event_type   TEXT NOT NULL,
            processed_at TEXT NOT NULL
        )
        """,
        # Subscription health columns — ADD COLUMN IF NOT EXISTS is idempotent in PG 9.6+
        "ALTER TABLE credits ADD COLUMN IF NOT EXISTS payment_failed_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE credits ADD COLUMN IF NOT EXISTS subscription_status TEXT NOT NULL DEFAULT 'active'",
        # ── Auth tables (Phase 1) ─────────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS accounts (
            id          TEXT PRIMARY KEY,
            email       TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            last_login  TEXT,
            CONSTRAINT accounts_email_uniq UNIQUE (email)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS magic_link_tokens (
            token       TEXT PRIMARY KEY,
            account_id  TEXT NOT NULL,
            expires_at  TEXT NOT NULL,
            used_at     TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS auth_sessions (
            token       TEXT PRIMARY KEY,
            account_id  TEXT NOT NULL,
            device_id   TEXT,
            created_at  TEXT NOT NULL,
            expires_at  TEXT NOT NULL,
            last_used   TEXT
        )
        """,
        # Nullable account_id on existing tables — completely additive, zero downtime
        "ALTER TABLE credits          ADD COLUMN IF NOT EXISTS account_id TEXT",
        "ALTER TABLE session_history  ADD COLUMN IF NOT EXISTS account_id TEXT",
        "ALTER TABLE session_history  ADD COLUMN IF NOT EXISTS full_transcript TEXT",
        "ALTER TABLE session_history  ADD COLUMN IF NOT EXISTS session_title TEXT",
        "ALTER TABLE session_history  ADD COLUMN IF NOT EXISTS session_name TEXT",
        "ALTER TABLE session_history  ADD COLUMN IF NOT EXISTS prep_goal TEXT",
        "ALTER TABLE session_history  ADD COLUMN IF NOT EXISTS prep_notes TEXT",
        "ALTER TABLE session_history  ADD COLUMN IF NOT EXISTS briefing_result JSONB",
        "ALTER TABLE session_history  ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'completed'",
        "ALTER TABLE preferences      ADD COLUMN IF NOT EXISTS account_id TEXT",
        "ALTER TABLE stripe_customers ADD COLUMN IF NOT EXISTS account_id TEXT",
        "ALTER TABLE founding_members ADD COLUMN IF NOT EXISTS account_id TEXT",
        # Indexes for fast auth lookups
        "CREATE INDEX IF NOT EXISTS accounts_email_idx    ON accounts(email)",
        "CREATE INDEX IF NOT EXISTS mlt_account_idx        ON magic_link_tokens(account_id)",
        "CREATE INDEX IF NOT EXISTS auth_sessions_acct_idx ON auth_sessions(account_id)",
        # ── Auth (Phase 2) — email-change token support ───────────────────────
        "ALTER TABLE magic_link_tokens ADD COLUMN IF NOT EXISTS token_type TEXT NOT NULL DEFAULT 'sign_in'",
        "ALTER TABLE magic_link_tokens ADD COLUMN IF NOT EXISTS new_email TEXT",
        # ── Auth (Phase 3) — email OTP support ───────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS email_otps (
            id         TEXT PRIMARY KEY,
            code       TEXT NOT NULL,
            account_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used_at    TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS email_otps_acct_idx ON email_otps(account_id)",
        "CREATE INDEX IF NOT EXISTS email_otps_code_idx ON email_otps(code)",
        # ── Per-user API cost tracking ────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS api_usage (
            id                  SERIAL PRIMARY KEY,
            user_key            TEXT NOT NULL,
            user_email          TEXT,
            event_type          TEXT NOT NULL,
            units               REAL NOT NULL,
            estimated_cost_usd  REAL NOT NULL,
            metadata            TEXT,
            created_at          TIMESTAMP DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS api_usage_user_key_idx   ON api_usage(user_key)",
        "CREATE INDEX IF NOT EXISTS api_usage_created_at_idx ON api_usage(created_at)",
        "CREATE INDEX IF NOT EXISTS api_usage_event_type_idx ON api_usage(event_type)",
        # ── Onboarding email log ──────────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS email_log (
            id          SERIAL PRIMARY KEY,
            account_id  TEXT NOT NULL,
            email_type  TEXT NOT NULL,
            sent_at     TEXT NOT NULL,
            CONSTRAINT email_log_acct_type_uniq UNIQUE (account_id, email_type)
        )
        """,
        "CREATE INDEX IF NOT EXISTS email_log_account_idx ON email_log(account_id)",
        # ── Suggestion feedback ────────────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS suggestion_feedback (
            id              SERIAL PRIMARY KEY,
            session_id      TEXT NOT NULL,
            suggestion_text TEXT,
            rating          TEXT NOT NULL,
            reason          TEXT,
            account_id      TEXT,
            created_at      TIMESTAMP DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS suggestion_feedback_session_idx ON suggestion_feedback(session_id)",
        """
        CREATE TABLE IF NOT EXISTS credit_ledger (
            id               BIGSERIAL PRIMARY KEY,
            user_id          TEXT NOT NULL,
            amount           INTEGER NOT NULL,
            balance_after    INTEGER NOT NULL,
            operation_type   TEXT NOT NULL,
            feature          TEXT,
            idempotency_key  TEXT UNIQUE,
            created_at       TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_credit_ledger_user
        ON credit_ledger(user_id, created_at DESC)
        """,
        """
        CREATE TABLE IF NOT EXISTS usage_events (
            id               BIGSERIAL PRIMARY KEY,
            user_id          TEXT NOT NULL,
            feature          TEXT NOT NULL,
            provider         TEXT NOT NULL,
            input_units      NUMERIC,
            output_units     NUMERIC,
            raw_cost_usd     NUMERIC(10,6),
            credits_charged  INTEGER,
            status           TEXT NOT NULL,
            idempotency_key  TEXT,
            created_at       TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_usage_events_user
        ON usage_events(user_id, created_at DESC)
        """,
        # ── Echo plan audio hard-cap tracking ─────────────────────────────────
        "ALTER TABLE credits ADD COLUMN IF NOT EXISTS audio_seconds_used INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE credits ADD COLUMN IF NOT EXISTS audio_seconds_reset_date DATE",
        "ALTER TABLE credits ADD COLUMN IF NOT EXISTS payg_enabled BOOLEAN NOT NULL DEFAULT FALSE",
        # ── Performance indexes for account_id lookups ────────────────────────
        "CREATE INDEX IF NOT EXISTS credits_account_id_idx ON credits(account_id)",
        "CREATE INDEX IF NOT EXISTS session_history_account_id_idx ON session_history(account_id)",
        "CREATE INDEX IF NOT EXISTS preferences_account_id_idx ON preferences(account_id)",
    ]
    try:
        with engine.connect() as conn:
            for stmt in ddl:
                conn.execute(text(stmt))
            conn.commit()
        logger.info("✅ All database tables verified / created")
    except Exception as e:
        logger.error(f"❌ init_db error: {e}")


init_db()
if engine is not None:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ PostgreSQL connected successfully")
    except Exception as e:
        logger.error(f"❌ DATABASE_URL connection failed — session persistence disabled: {e}")
reset_expired_credits()

# ========================
# WEBHOOK HELPER FUNCTIONS
# ========================

def _lookup_user_id(customer_id: str):
    """Return the internal user_id for a Stripe customer_id, or None."""
    if engine is None or not customer_id:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT user_id FROM stripe_customers WHERE customer_id=:cid"),
                {"cid": customer_id}
            ).fetchone()
        return row[0] if row else None
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"_lookup_user_id error: {e}")
        return None


def _is_duplicate_event(event_id: str) -> bool:
    """Return True if this Stripe event_id has already been successfully processed."""
    if engine is None:
        return False
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM processed_webhook_events WHERE event_id=:eid"),
                {"eid": event_id}
            ).fetchone()
        return row is not None
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"_is_duplicate_event error: {e}")
        return False


def _mark_event_processed(event_id: str, event_type: str) -> None:
    """Record that a Stripe event was handled (idempotency journal)."""
    if engine is None:
        return
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO processed_webhook_events (event_id, event_type, processed_at)
                VALUES (:eid, :etype, :now)
                ON CONFLICT (event_id) DO NOTHING
            """), {"eid": event_id, "etype": event_type, "now": datetime.utcnow().isoformat()})
            conn.commit()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"_mark_event_processed error: {e}")


@app.get("/")
async def root():
    return {"status": "CerebroEcho Backend Live", "version": "1.2.0"}

# ── TTS health tracking ──────────────────────────────────────────────────────
# /tts records its last upstream failure here so /health can report TTS as degraded
# WITHOUT making a paid probe call to Cartesia on every health check. A successful
# stream open clears it. A failure is considered "current" for TTS_FAILURE_TTL_S.
# This is what lets /health tell the truth about TTS (the old key-presence check
# reported cartesia=true even when Cartesia returned 402 "credits exhausted").
_tts_last_failure: dict | None = None
TTS_FAILURE_TTL_S = 600  # 10 minutes

def _record_tts_failure(status: int, detail: str) -> None:
    global _tts_last_failure
    _tts_last_failure = {"status": status, "detail": (detail or "")[:200], "at": time.time()}

def _clear_tts_failure() -> None:
    global _tts_last_failure
    _tts_last_failure = None

def _tts_recent_failure() -> dict | None:
    f = _tts_last_failure
    if f and (time.time() - f["at"]) <= TTS_FAILURE_TTL_S:
        return f
    return None


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    """
    Readiness probe — intentionally not rate-limited so monitoring services
    can poll freely. Returns 200 when all critical dependencies are healthy,
    503 when any critical dependency is degraded.

    Critical (affect HTTP status): db, groq, deepgram
    Non-critical (informational only): cartesia, stripe, resend
    Note: `tts` reflects the REAL outcome of recent /tts calls (degrades on a
    Cartesia upstream failure within the last 10 min); `cartesia` is key-presence only.
    """
    checks: dict = {}

    # ── Database (real connectivity check) ────────────────────────────────
    db_ok = False
    if engine is not None:
        try:
            t0 = time.time()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            checks["db_latency_ms"] = round((time.time() - t0) * 1000, 1)
            db_ok = True
        except Exception as e:
            checks["db_error"] = str(e)[:160]
    else:
        checks["db_error"] = "engine not initialised — DATABASE_URL missing?"

    # ── SDK clients (initialised at startup) ─────────────────────────────
    groq_ok     = client is not None
    deepgram_ok = deepgram_client is not None
    # TTS health = key configured AND no recent Cartesia upstream failure. This is
    # the fix for the silent outage: previously tts_ok was key-presence only, so a
    # 402 "credits exhausted" from Cartesia still reported tts=true.
    tts_failure = _tts_recent_failure()
    tts_ok      = (cartesia_client is not None) and (tts_failure is None)

    # ── API key presence (never expose values) ────────────────────────────
    checks.update({
        "db":          db_ok,
        "groq":        groq_ok,
        "deepgram":    deepgram_ok,
        "tts":         tts_ok,                       # real recent /tts outcome
        "cartesia":    cartesia_client is not None,  # key configured (presence only)
        "stripe":      bool(os.getenv("STRIPE_SECRET_KEY")),
        "resend":      bool(os.getenv("RESEND_API_KEY")),
    })
    if tts_failure:
        checks["tts_last_error"] = {
            "upstream_status": tts_failure["status"],
            "detail":          tts_failure["detail"],
            "seconds_ago":     round(time.time() - tts_failure["at"]),
        }

    healthy = db_ok and groq_ok and deepgram_ok
    uptime_s = round(time.time() - _START_TIME)
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status":         "ok" if healthy else "degraded",
            "checks":         checks,
            "version":        "1.2.0",
            "uptime_seconds": uptime_s,
            "ts":             datetime.utcnow().isoformat() + "Z",
        },
    )

@app.get("/founder_spots")
async def get_founder_spots():
    count = 0
    if engine is not None:
        try:
            with engine.connect() as conn:
                row = conn.execute(text("SELECT COUNT(*) FROM founding_members")).fetchone()
                count = row[0] if row else 0
        except Exception as e:
            logger.error(f"founder_spots error: {e}")
    return {"remaining": max(0, 50 - count), "claimed": count}


@app.post("/send-desktop-link")
@limiter.limit("5/minute")
async def send_desktop_link(request: Request):
    """Send a desktop app link email for mobile visitors."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    email = (body.get("email") or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email required")
    html = email_service._html_wrap(
        "Open CerebroEcho on your desktop",
        """<p style="margin:0 0 14px;font-size:15px;color:#B8C4D0;line-height:1.6;">
           You requested a link to open CerebroEcho on your desktop browser.
           It works best in Chrome or Edge.
         </p>
         <p style="margin:0;font-size:13px;color:#596272;">
           CerebroEcho needs a desktop environment to capture audio from Zoom, Meet, or Teams.
         </p>""",
        cta_label="Open CerebroEcho on desktop →",
        cta_url="https://cerebroecho.com/app",
    )
    text = "Open CerebroEcho on your desktop: https://cerebroecho.com/app"
    sent = email_service.send_email(email, "Your CerebroEcho desktop link", html, text)
    return {"ok": sent}


@app.post("/suggestion-feedback")
@limiter.limit("60/minute")
async def post_suggestion_feedback(request: Request):
    """Store thumbs up/down feedback on a suggestion card."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    session_id = (body.get("session_id") or "").strip()
    suggestion_text = (body.get("suggestion_text") or "")[:2000]
    rating = (body.get("rating") or "").strip()
    reason = (body.get("reason") or "")[:200]
    if not session_id or rating not in ("positive", "negative"):
        raise HTTPException(status_code=400, detail="session_id and valid rating required")
    account_id = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        acct, _ = _get_account_from_session(auth_header[7:].strip())
        account_id = acct
    if engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(text(
                    "INSERT INTO suggestion_feedback (session_id, suggestion_text, rating, reason, account_id) "
                    "VALUES (:sid, :txt, :rating, :reason, :acct)"
                ), {"sid": session_id, "txt": suggestion_text, "rating": rating,
                    "reason": reason or None, "acct": account_id})
                conn.commit()
        except Exception as e:
            logger.error(f"suggestion_feedback insert error: {e}")
    return {"ok": True}


@app.post("/transcribe")
@limiter.limit(RATE_TRANSCRIBE)
async def transcribe(request: Request):
    if deepgram_client is None:
        raise HTTPException(status_code=503, detail="Transcription service unavailable: DEEPGRAM_API_KEY not set")
    try:
        audio_bytes = await request.body()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="No audio data provided")

        options = PrerecordedOptions(model="nova-2", smart_format=True)
        response = deepgram_client.listen.prerecorded.v("1").transcribe_file(
            {"buffer": audio_bytes},
            options,
        )
        transcript = response.results.channels[0].alternatives[0].transcript
        return {"transcript": transcript}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Deepgram transcribe error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed. Please try again.")


@app.post("/coach")
@limiter.limit(RATE_COACH)
async def coach(
    request: Request,
    transcript: str = Form(None),
    whisper_seconds_in: float = Form(0.0),   # WS1: client STT duration so the reuse path still bills STT
    audio: UploadFile = File(None),
    file: UploadFile = File(None),
    deviceId: str = Form(...),
    userEmail: str = Form("anonymous"),
    context: str = Form("a professional role"),
    work_history: str = Form(""),
    style: str = Form("Standard"),
    role: str = Form("Interview Coach"),
    mode: str = Form("Diplomat"),
    session_history: str = Form(""),
    prior_summaries: str = Form(""),
    filter_small_talk: str = Form("false"),
    session_context: str = Form(""),
    user_preferences: str = Form(""),
    ghost_mode: str = Form("false"),
    custom_role_description: str = Form(""),
    custom_mode_description: str = Form(""),
):
    if client is None:
        raise HTTPException(status_code=503, detail="AI service unavailable - check server configuration")

    # Resolve authenticated identity — ensures cross-device credit consistency
    raw_user_id, account_id, _ = _resolve_user(request, deviceId, userEmail)
    user_key = _get_credits_user_id(raw_user_id, account_id)

    # ── Credit check + plan metadata ──
    # Metered billing: the real charge is computed AFTER the LLM/STT call from
    # actual token + audio usage. STYLE_COSTS still drives max_tokens via style.
    credits = _get_credit_balance(user_key)
    plan_type = credits["plan_type"]
    tts_allowed = plan_type in TTS_ALLOWED_PLANS

    # Custom persona/style gate (fixed Jun 11 2026). The frontend sends the user's
    # typed custom text AS the role/mode value — it never literally sends "Custom".
    # So: any non-preset value IS a custom persona/style. On CUSTOM_ALLOWED_PLANS it
    # is wired into the dedicated custom_*_description fields (which is the only way
    # build_system_prompt actually injects it — previously typed customs silently
    # fell through to the default prompt for every plan). Below Pro it is downgraded.
    if role not in ROLE_PROMPTS:
        if role.strip() and plan_type in CUSTOM_ALLOWED_PLANS:
            custom_role_description = (custom_role_description or role).strip()
            role = "Custom"
        else:
            role = "Interview Coach"
    elif role in OPERATOR_ONLY_OPTIONS and plan_type not in CUSTOM_ALLOWED_PLANS:
        role = "Interview Coach"
    if mode not in MODE_MODIFIERS:
        if mode.strip() and plan_type in CUSTOM_ALLOWED_PLANS:
            custom_mode_description = (custom_mode_description or mode).strip()
            mode = "Custom"
        else:
            mode = "Diplomat"
    elif mode in OPERATOR_ONLY_OPTIONS and plan_type not in CUSTOM_ALLOWED_PLANS:
        mode = "Diplomat"

    # Free plan: Quick + Standard only (no Full — long responses)
    if plan_type == "free" and style == "Full":
        style = "Standard"

    # Conservative up-front gate: require at least 1 credit (or PAYG enabled).
    # Real metered charge happens post-call so we never block a legitimate request.
    if credits["balance"] < 1 and not credits.get("payg_enabled", False):
        raise HTTPException(status_code=402, detail={
            "code": "INSUFFICIENT_CREDITS",
            "message": "Insufficient credits. Upgrade your plan or purchase an overage pack.",
            "balance": credits["balance"],
            "cost": 1,
        })

    start_time = time.time()
    temp_filename = None
    whisper_seconds = 0.0  # best-effort STT duration for metered billing

    try:
        # --- Transcript path: pre-transcribed text provided by frontend (e.g. Deepgram streaming) ---
        if transcript and transcript.strip():
            transcript = transcript.strip()
            whisper_seconds = max(0.0, min(float(whisper_seconds_in or 0.0), 7200.0))  # WS1: bill reused STT
            logger.info(f"📝 Using pre-transcribed text ({whisper_seconds:.1f}s billed STT): {transcript[:80]}...")
            if plan_type == "echo" and engine is not None and whisper_seconds > 0:
                try:
                    with engine.begin() as _conn:
                        _conn.execute(text("UPDATE credits SET audio_seconds_used = audio_seconds_used + :secs WHERE user_id = :uid"),
                                      {"secs": max(1, round(whisper_seconds)), "uid": user_key})
                except Exception as _e:
                    logger.error(f"audio_seconds increment error (transcript path): {_e}")
        else:
            # --- Audio path: run Groq Whisper STT ---
            actual_file = audio or file
            if not actual_file:
                raise HTTPException(status_code=400, detail="Provide either 'transcript' text or an audio file")

            content = await actual_file.read()
            if len(content) < 1000:
                return {"answer": "Listening...", "transcript": "", "credits_remaining": credits["balance"], "tts_allowed": tts_allowed, "processing_time": round(time.time() - start_time, 3)}

            # ── Echo plan audio hard cap: 3600 seconds (1 hour) per month ──
            if plan_type == "echo" and engine is not None:
                try:
                    with engine.connect() as _conn:
                        _row = _conn.execute(
                            text("SELECT audio_seconds_used FROM credits WHERE user_id=:uid"),
                            {"uid": user_key}
                        ).fetchone()
                    _audio_used = _row[0] if _row else 0
                except Exception as _e:
                    logger.error(f"audio cap read error: {_e}")
                    _audio_used = 0
                if _audio_used >= 3600:
                    return {
                        "answer": "Monthly audio limit reached. Upgrade to Pro for unlimited audio.",
                        "transcript": "",
                        "credits_remaining": credits["balance"],
                        "tts_allowed": tts_allowed,
                        "processing_time": round(time.time() - start_time, 3),
                        "audio_cap_reached": True,
                        "code": "AUDIO_CAP_REACHED",
                        "message": "Monthly audio limit reached. Upgrade to Pro for unlimited audio.",
                    }

            logger.info(f"📝 Transcribing {len(content)} bytes via Whisper...")
            # AUDIO RETENTION: temp file required by Groq file API (cannot stream bytes directly).
            # Written to OS temp dir, read once for transcription, then immediately deleted.
            # Raw audio never reaches the database or any storage layer.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(content)
                temp_filename = tmp.name

            with open(temp_filename, "rb") as af:
                transcription = client.audio.transcriptions.create(
                    file=(temp_filename, af.read(), "audio/webm"),
                    model="whisper-large-v3-turbo",
                    response_format="text",
                )

            # AUDIO RETENTION: temp file deleted immediately after transcription.
            if temp_filename and os.path.exists(temp_filename):
                os.unlink(temp_filename)
                temp_filename = None

            transcript = transcription.strip() if transcription else ""
            # Best-effort audio duration for metered Whisper billing (~32000 bytes/sec).
            whisper_seconds = len(content) / 32000
            _track_usage(user_key, userEmail, "whisper_transcription",
                         len(content), len(content) / 16000 / 2 * 0.0001)

            # ── Echo plan: increment audio seconds consumed ──
            # Count the SAME measure used for billing (whisper_seconds = len/32000)
            # so the 3600s cap equals a real ~1 hour. Previously this used /16000,
            # double-counting the stream so Echo users hit the "1 hour" cap at ~30 min.
            if plan_type == "echo" and engine is not None:
                _audio_secs = max(1, round(whisper_seconds))
                try:
                    with engine.begin() as _conn:
                        _conn.execute(text("""
                            UPDATE credits
                            SET audio_seconds_used = audio_seconds_used + :secs
                            WHERE user_id = :uid
                        """), {"secs": _audio_secs, "uid": user_key})
                except Exception as _e:
                    logger.error(f"audio_seconds increment error: {_e}")
            if not transcript or len(transcript) < 2:
                return {"answer": "Listening...", "transcript": "", "credits_remaining": credits["balance"], "tts_allowed": tts_allowed, "processing_time": round(time.time() - start_time, 3)}

        logger.info(f"✅ Transcript: {transcript[:80]}...")

        # === RETROSPECTIVE QUESTION ASSEMBLY ===
        merged = False
        buf_entry = question_buffer.get(user_key)
        if buf_entry and (time.time() - buf_entry["timestamp"]) <= CONTINUATION_TIMEOUT_S:
            if _looks_like_continuation(buf_entry["transcript"], transcript):
                combined = buf_entry["transcript"].rstrip() + " " + transcript
                logger.info(f"🔗 Merging: '{buf_entry['transcript'][:50]}' + '{transcript[:50]}'")
                transcript = combined
                merged = True

        # Ghost mode: no traces — skip question buffer update
        if ghost_mode.lower() != "true":
            question_buffer[user_key] = {"transcript": transcript, "timestamp": time.time()}

        # === SMALL TALK FILTER ===
        if filter_small_talk.lower() == "true":
            check = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": f"Is this interview small talk or personal conversation rather than a professional interview question? Answer only YES or NO: {transcript}"}],
                temperature=0.0,
                max_tokens=5,
            )
            verdict = check.choices[0].message.content.strip().upper()
            logger.info(f"🤫 Small talk check: '{transcript[:50]}' → {verdict}")
            if verdict.startswith("YES"):
                return {"answer": "", "transcript": transcript, "filtered": True, "credits_remaining": credits["balance"], "tts_allowed": tts_allowed}

        # Parse memory context
        try:
            prior = json.loads(prior_summaries) if prior_summaries else []
        except Exception:
            prior = []
        try:
            history = json.loads(session_history) if session_history else []
        except Exception:
            history = []

        prior_context = ""
        if prior:
            prior_context = "PREVIOUS SESSIONS:\n" + "\n---\n".join(prior)

        # FIX 6: fall back to cached context if the request carries none (e.g. after page refresh)
        effective_session_context = session_context
        if not effective_session_context.strip() and ghost_mode.lower() != "true":
            effective_session_context = _get_ctx_cache(user_key)

        system_prompt, max_tokens = build_system_prompt(
            role=role,
            mode=mode,
            style=style,
            context=context,
            work_history=work_history,
            prior_context=prior_context,
            session_context=effective_session_context,
            user_preferences=user_preferences,
            custom_role_description=custom_role_description,
            custom_mode_description=custom_mode_description,
        )
        logger.info(f"🧠 Role={role} | Mode={mode} | Style={style} | ctx={len(effective_session_context)}c | prefs={len(user_preferences)}c | max_tokens={max_tokens}")
        logger.info(f"📋 SYSTEM PROMPT [{role}/{mode}/{style}]:\n{'─'*60}\n{system_prompt}\n{'─'*60}")

        # Hybrid context: last 10 turns full + older turns as compressed cliff notes
        FULL_TURNS = 10
        recent = history[-FULL_TURNS:]
        older  = history[:-FULL_TURNS] if len(history) > FULL_TURNS else []

        def _clip(text, n):
            t = (text or "").strip()
            return t[:n] + ("…" if len(t) > n else "")

        if older:
            notes = "\n".join(
                f"• {_clip(t.get('question',''), 50)} → {_clip(t.get('answer',''), 80)}"
                for t in older
            )
            system_prompt = system_prompt + f"\n\n[Earlier this session ({len(older)} exchanges):\n{notes}\n]"

        messages = [{"role": "system", "content": system_prompt}]
        for turn in recent:
            messages.append({"role": "user", "content": f"Interview question: {turn['question']}"})
            messages.append({"role": "assistant", "content": turn['answer']})
        messages.append({"role": "user", "content": f"Interview question: {transcript}"})

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.6,
            max_tokens=max_tokens,
            top_p=0.9
        )

        answer = completion.choices[0].message.content.strip()
        logger.info(f"✅ Full answer text ({len(answer)} chars): {answer[:100]}...")
        _track_usage(user_key, userEmail, "groq_llm",
                     len(answer), len(answer) / 4 * 0.0000008)

        # ── Metered billing: real LLM token usage + best-effort Whisper seconds ──
        _usage = getattr(completion, "usage", None)
        pt = getattr(_usage, "prompt_tokens", None) if _usage else None
        ct = getattr(_usage, "completion_tokens", None) if _usage else None
        raw_cost = (
            _llm_cost_usd("llama-3.1-8b-instant", pt, ct)
            + whisper_seconds * PROVIDER_RATES["whisper"]["sec"]
        )
        metered_credits = _cost_to_credits(raw_cost)

        # Hash the FULL transcript (not the first 200 chars): two different short
        # questions that share an opening could otherwise collide in the same 10s
        # bucket, making the second response deduct nothing (delivered free).
        idem_key = hashlib.sha256(
            f"{user_key}:{hashlib.sha256(transcript.encode()).hexdigest()}:{int(time.time() // 10)}".encode()
        ).hexdigest()
        credits_remaining = _deduct_credits(
            user_key, metered_credits, feature=f"coach:{style}", idempotency_key=idem_key,
            provider="groq+whisper", input_units=pt, output_units=ct, raw_cost_usd=raw_cost,
        )
        if credits_remaining < 0:
            credits_remaining = max(0, credits["balance"] - metered_credits)
        processing_time = time.time() - start_time
        logger.info(f"✅ Answer generated in {processing_time:.2f}s | -{metered_credits} credits "
                    f"(raw ${raw_cost:.6f}, pt={pt} ct={ct} stt={whisper_seconds:.1f}s) → {credits_remaining} remaining")

        # Confidence — computed from existing data, no extra API call
        finish_reason = completion.choices[0].finish_reason
        transcript_words = len(transcript.split())
        if finish_reason == "length" or transcript_words < 4:
            confidence = "low"
        elif transcript_words < 8 or merged:
            confidence = "medium"
        else:
            confidence = "high"

        return {"transcript": transcript, "answer": answer, "credits_remaining": credits_remaining, "processing_time": round(processing_time, 3), "merged": merged, "tts_allowed": tts_allowed, "confidence": confidence}

    except HTTPException:
        raise
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"❌ /coach ERROR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if temp_filename and os.path.exists(temp_filename):
            os.unlink(temp_filename)
        raise HTTPException(status_code=500, detail="Processing error. Please try again.")


@app.post("/coach_stream")
@limiter.limit(RATE_COACH)
async def coach_stream(
    request: Request,
    transcript: str = Form(None),
    whisper_seconds_in: float = Form(0.0),
    deviceId: str = Form(...),
    userEmail: str = Form("anonymous"),
    context: str = Form("a professional role"),
    work_history: str = Form(""),
    style: str = Form("Standard"),
    role: str = Form("Interview Coach"),
    mode: str = Form("Diplomat"),
    session_history: str = Form(""),
    prior_summaries: str = Form(""),
    session_context: str = Form(""),
    user_preferences: str = Form(""),
    ghost_mode: str = Form("false"),
    custom_role_description: str = Form(""),
    custom_mode_description: str = Form(""),
):
    """SSE streaming variant of /coach. Streams llama-3.1-8b-instant tokens, then a
    final 'done' event with metered billing. /coach stays as the non-streaming fallback."""
    if client is None:
        raise HTTPException(status_code=503, detail="AI service unavailable")
    if not transcript or not transcript.strip():
        raise HTTPException(status_code=400, detail="coach_stream requires 'transcript'")
    transcript = transcript.strip()

    raw_user_id, account_id, _ = _resolve_user(request, deviceId, userEmail)
    user_key = _get_credits_user_id(raw_user_id, account_id)
    credits = _get_credit_balance(user_key)
    plan_type = credits["plan_type"]
    tts_allowed = plan_type in TTS_ALLOWED_PLANS

    if role in OPERATOR_ONLY_OPTIONS and plan_type not in CUSTOM_ALLOWED_PLANS:
        role = "Interview Coach"
    if mode in OPERATOR_ONLY_OPTIONS and plan_type not in CUSTOM_ALLOWED_PLANS:
        mode = "Diplomat"
    if plan_type == "free" and style == "Full":
        style = "Standard"
    if credits["balance"] < 1 and not credits.get("payg_enabled", False):
        raise HTTPException(status_code=402, detail={
            "code": "INSUFFICIENT_CREDITS",
            "message": "Insufficient credits. Upgrade your plan or purchase an overage pack.",
            "balance": credits["balance"], "cost": 1,
        })

    # WS1/WS2 billing parity: bill the reused STT seconds + echo audio cap.
    whisper_seconds = max(0.0, min(float(whisper_seconds_in or 0.0), 7200.0))
    if plan_type == "echo" and engine is not None and whisper_seconds > 0:
        try:
            with engine.begin() as _conn:
                _conn.execute(text("UPDATE credits SET audio_seconds_used = audio_seconds_used + :s WHERE user_id = :u"),
                              {"s": max(1, round(whisper_seconds)), "u": user_key})
        except Exception as _e:
            logger.error(f"audio_seconds increment (stream): {_e}")

    # question-buffer merge (mirrors /coach)
    merged = False
    buf_entry = question_buffer.get(user_key)
    if buf_entry and (time.time() - buf_entry["timestamp"]) <= CONTINUATION_TIMEOUT_S:
        if _looks_like_continuation(buf_entry["transcript"], transcript):
            transcript = buf_entry["transcript"].rstrip() + " " + transcript
            merged = True
    if ghost_mode.lower() != "true":
        question_buffer[user_key] = {"transcript": transcript, "timestamp": time.time()}

    try:    prior = json.loads(prior_summaries) if prior_summaries else []
    except Exception: prior = []
    try:    history = json.loads(session_history) if session_history else []
    except Exception: history = []
    prior_context = ("PREVIOUS SESSIONS:\n" + "\n---\n".join(prior)) if prior else ""
    effective_session_context = session_context
    if not effective_session_context.strip() and ghost_mode.lower() != "true":
        effective_session_context = _get_ctx_cache(user_key)

    system_prompt, max_tokens = build_system_prompt(
        role=role, mode=mode, style=style, context=context, work_history=work_history,
        prior_context=prior_context, session_context=effective_session_context,
        user_preferences=user_preferences,
        custom_role_description=custom_role_description, custom_mode_description=custom_mode_description,
    )
    FULL_TURNS = 10
    recent = history[-FULL_TURNS:]
    older  = history[:-FULL_TURNS] if len(history) > FULL_TURNS else []
    def _clip(t, n):
        t = (t or "").strip(); return t[:n] + ("…" if len(t) > n else "")
    if older:
        notes = "\n".join(f"• {_clip(t.get('question',''),50)} → {_clip(t.get('answer',''),80)}" for t in older)
        system_prompt += f"\n\n[Earlier this session ({len(older)} exchanges):\n{notes}\n]"
    messages = [{"role": "system", "content": system_prompt}]
    for turn in recent:
        messages.append({"role": "user", "content": f"Interview question: {turn['question']}"})
        messages.append({"role": "assistant", "content": turn['answer']})
    messages.append({"role": "user", "content": f"Interview question: {transcript}"})

    start_time = time.time()
    transcript_words = len(transcript.split())

    async def event_gen():
        def sse(event, data): return f"event: {event}\ndata: {json.dumps(data)}\n\n"
        yield sse("meta", {"transcript": transcript, "tts_allowed": tts_allowed,
                           "merged": merged, "credits_remaining": credits["balance"]})
        full_parts, finish_reason, usage = [], None, None
        try:
            q: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            def _produce():
                try:
                    stream = client.chat.completions.create(
                        model="llama-3.1-8b-instant", messages=messages,
                        temperature=0.6, max_tokens=max_tokens, top_p=0.9,
                        stream=True)
                    for chunk in stream:
                        loop.call_soon_threadsafe(q.put_nowait, ("chunk", chunk))
                    loop.call_soon_threadsafe(q.put_nowait, ("end", None))
                except Exception as e:
                    loop.call_soon_threadsafe(q.put_nowait, ("error", e))
            import threading
            threading.Thread(target=_produce, daemon=True).start()
            while True:
                kind, payload = await q.get()
                if kind == "error": raise payload
                if kind == "end": break
                chunk = payload
                if getattr(chunk, "usage", None): usage = chunk.usage
                if not chunk.choices: continue
                if chunk.choices[0].finish_reason: finish_reason = chunk.choices[0].finish_reason
                tok = getattr(chunk.choices[0].delta, "content", None)
                if tok:
                    full_parts.append(tok)
                    yield sse("token", {"t": tok})
            answer = "".join(full_parts).strip()

            pt = getattr(usage, "prompt_tokens", None) if usage else None
            ct = getattr(usage, "completion_tokens", None) if usage else None
            if pt is None: pt = max(1, sum(len(m.get("content", "")) for m in messages) // 4)  # estimate (no usage chunk)
            if ct is None: ct = max(1, len(answer) // 4)
            raw_cost = (_llm_cost_usd("llama-3.1-8b-instant", pt, ct)
                        + whisper_seconds * PROVIDER_RATES["whisper"]["sec"])
            metered_credits = _cost_to_credits(raw_cost)
            idem_key = hashlib.sha256(
                f"{user_key}:{hashlib.sha256(transcript.encode()).hexdigest()}:{int(time.time()//10)}".encode()
            ).hexdigest()
            credits_remaining = _deduct_credits(
                user_key, metered_credits, feature=f"coach:{style}", idempotency_key=idem_key,
                provider="groq+whisper", input_units=pt, output_units=ct, raw_cost_usd=raw_cost)
            if credits_remaining < 0:
                credits_remaining = max(0, credits["balance"] - metered_credits)
            _track_usage(user_key, userEmail, "groq_llm", len(answer), raw_cost)

            confidence = ("low" if (finish_reason == "length" or transcript_words < 4)
                          else "medium" if (transcript_words < 8 or merged) else "high")
            yield sse("done", {"answer": answer, "transcript": transcript,
                               "credits_remaining": credits_remaining, "merged": merged,
                               "tts_allowed": tts_allowed, "confidence": confidence,
                               "processing_time": round(time.time() - start_time, 3)})
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(f"❌ /coach_stream ERROR: {type(e).__name__}: {e}")
            yield sse("error", {"message": "stream_failed", "detail": f"{type(e).__name__}: {str(e)[:200]}"})

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── WS4: persistent keep-alive httpx client for Cartesia ──
# A fresh AsyncClient per /tts paid a full TCP+TLS handshake to Cartesia every whisper.
# One pooled module-level client removes ~100–200ms/whisper steady-state.
import httpx as _httpx
_CARTESIA_LIMITS = _httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=300.0)
_CARTESIA_TIMEOUT = _httpx.Timeout(connect=4.0, read=30.0, write=10.0, pool=5.0)
_cartesia_client = None

def _get_cartesia_client():
    global _cartesia_client
    if _cartesia_client is None or _cartesia_client.is_closed:
        _cartesia_client = _httpx.AsyncClient(limits=_CARTESIA_LIMITS, timeout=_CARTESIA_TIMEOUT, http2=False)
    return _cartesia_client

@app.on_event("shutdown")
async def _close_cartesia_client():
    global _cartesia_client
    if _cartesia_client is not None and not _cartesia_client.is_closed:
        await _cartesia_client.aclose()

async def _prewarm_cartesia():
    """Warm the pooled connection at session start so the first whisper skips the handshake."""
    if not os.environ.get("CARTESIA_API_KEY"):
        return
    try:
        await _get_cartesia_client().get("https://api.cartesia.ai/", timeout=4.0)
    except Exception as e:
        logger.info(f"Cartesia prewarm skipped: {e}")


@app.post("/tts")
@limiter.limit(RATE_TTS)
async def text_to_speech(request: Request, data: dict):
    text = data.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    speed    = float(data.get("speed", 0.85))
    voice_id = data.get("voice_id", "f9836c6e-a0bd-460e-9d3c-f7299fa60f94")
    _tts_user_id, _tts_account_id, _ = _resolve_user(request, data.get("deviceId", ""), data.get("userEmail", ""))
    _tts_user_key = _get_credits_user_id(_tts_user_id, _tts_account_id)
    plan_info = _get_credit_balance(_tts_user_key)
    if plan_info["plan_type"] not in TTS_ALLOWED_PLANS:
        raise HTTPException(status_code=403, detail={
            "code": "PLAN_REQUIRED",
            "message": "TTS audio requires Pro plan or above.",
            "required_plan": "pro",
        })

    cartesia_api_key = os.environ.get("CARTESIA_API_KEY")
    if cartesia_api_key:
        import httpx
        payload = {
            "model_id": "sonic-3",
            "transcript": text,
            "voice": {"mode": "id", "id": voice_id},
            # 24 kHz 96 kbps — optimal for voice: lighter stream, ~30% faster first byte vs 44.1 kHz 128k
            "output_format": {"container": "mp3", "bit_rate": 96000, "sample_rate": 24000},
            "speed": speed,
        }
        cartesia_headers = {
            "Cartesia-Version": "2025-04-16",
            "X-API-Key": cartesia_api_key,
            "Content-Type": "application/json",
        }
        # ── Metered billing: charge the REAL Cartesia per-character cost ──
        # TTS is the dominant cost of an audio whisper, so it must be metered like
        # everything else (decided Jun 7 2026). Cost is known up front from char count.
        tts_raw_cost = len(text) * PROVIDER_RATES["cartesia_tts"]["char"]
        tts_credits  = _cost_to_credits(tts_raw_cost)
        # Idempotency: dedupe accidental double-fires of the SAME text within a 10s
        # window (client retries, React strict-mode) without blocking a later re-listen.
        tts_idem = hashlib.sha256(
            f"{_tts_user_key}:tts:{hashlib.sha256(text.encode()).hexdigest()}:{int(time.time() // 10)}".encode()
        ).hexdigest()
        logger.info(f"🎙️ Cartesia TTS sonic-3/24kHz voice={voice_id[:8]}… len={len(text)} → {tts_credits} cr (raw ${tts_raw_cost:.6f})")

        # ── Preflight the upstream BEFORE committing to a 200 stream ──
        # StreamingResponse sends the status line the moment the generator starts
        # iterating, so an upstream error discovered *inside* the generator could only
        # ever yield an empty 200 (the silent-TTS bug: Cartesia 402 "credits exhausted"
        # looked like a successful but empty audio response). By sending the request and
        # validating its status HERE, a real failure surfaces to the client as a real
        # error status — and the frontend's `if (!response.ok) throw` falls back to the
        # text answer instead of silently playing nothing.
        http = _get_cartesia_client()  # WS4: shared keep-alive client (do NOT close per-request)
        try:
            req  = http.build_request("POST", "https://api.cartesia.ai/tts/bytes",
                                      headers=cartesia_headers, json=payload)
            resp = await http.send(req, stream=True)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(f"Cartesia connection error: {e}")
            _record_tts_failure(0, str(e)[:160])
            raise HTTPException(status_code=502, detail={
                "code": "TTS_UPSTREAM_UNAVAILABLE",
                "message": "Voice generation is temporarily unavailable. Your response is shown as text.",
            })

        if resp.status_code != 200:
            err_body = ""
            try:
                err_body = (await resp.aread()).decode("utf-8", "replace")[:300]
            except Exception:
                pass
            await resp.aclose()
            logger.error(f"Cartesia TTS upstream {resp.status_code}: {err_body}")
            sentry_sdk.capture_message(f"Cartesia TTS upstream {resp.status_code}: {err_body}")
            _record_tts_failure(resp.status_code, err_body)
            # 402 = Cartesia model-credit limit reached (our account billing); 429 = rate limit.
            # Neither is the user's fault, so report a clean 503 the client can degrade on.
            if resp.status_code in (402, 429):
                raise HTTPException(status_code=503, detail={
                    "code": "TTS_CAPACITY_EXHAUSTED",
                    "message": "Voice generation is temporarily unavailable. Your response is shown as text.",
                    "upstream_status": resp.status_code,
                })
            raise HTTPException(status_code=502, detail={
                "code": "TTS_UPSTREAM_ERROR",
                "message": "Voice generation failed. Your response is shown as text.",
                "upstream_status": resp.status_code,
            })

        logger.info("✅ Cartesia stream open (upstream 200)")
        _clear_tts_failure()

        # Upstream is healthy — stream the bytes to the client. 512-byte chunks deliver
        # the first audio packet ~2× faster than 1024. Charge ONLY for delivered audio.
        async def cartesia_stream():
            streamed_any = False
            try:
                async for chunk in resp.aiter_bytes(chunk_size=512):
                    streamed_any = True
                    yield chunk
            except Exception as e:
                sentry_sdk.capture_exception(e)
                logger.error(f"Cartesia stream error mid-flight: {e}")
            finally:
                await resp.aclose()  # WS4: close the per-request response only; shared client stays open
            if streamed_any:
                try:
                    _deduct_credits(_tts_user_key, tts_credits, feature="tts",
                                    idempotency_key=tts_idem, provider="cartesia",
                                    output_units=len(text), raw_cost_usd=tts_raw_cost)
                except Exception as _e:
                    logger.error(f"TTS credit deduction failed (non-fatal): {_e}")

        _track_usage(_tts_user_key, data.get("userEmail", ""), "cartesia_tts",
                     len(text), tts_raw_cost)
        return StreamingResponse(cartesia_stream(), media_type="audio/mpeg")

    # Cartesia unavailable — return text-only; client handles graceful degradation
    logger.warning("⚠️ TTS unavailable: CARTESIA_API_KEY not set")
    raise HTTPException(status_code=503, detail="TTS service unavailable")

@app.post("/session/start")
@limiter.limit(RATE_READS)
async def session_start(request: Request, data: dict):
    device_id  = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")

    user_id, account_id, _ = _resolve_user(request, device_id, user_email)

    # WS4: warm the pooled Cartesia connection so the first whisper of the session skips the handshake.
    asyncio.create_task(_prewarm_cartesia())

    summaries: list = []
    if engine is not None:
        try:
            with engine.connect() as conn:
                if account_id:
                    rows = conn.execute(text("""
                        SELECT summary FROM session_history
                        WHERE (account_id = :aid OR user_id = :uid)
                          AND summary IS NOT NULL AND summary != ''
                        ORDER BY timestamp DESC LIMIT 3
                    """), {"aid": account_id, "uid": user_id}).fetchall()
                else:
                    rows = conn.execute(text("""
                        SELECT summary FROM session_history
                        WHERE user_id = :uid AND summary IS NOT NULL AND summary != ''
                        ORDER BY timestamp DESC LIMIT 3
                    """), {"uid": user_id}).fetchall()
            summaries = [r[0] for r in rows]
        except Exception as e:
            logger.error(f"session/start DB error: {e}")

    # Whisper a 1-sentence recap of the most recent session so the user has instant continuity
    memory_recap = None
    if summaries and client is not None:
        try:
            recap = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{
                    "role": "user",
                    "content": (
                        "Write a single sentence under 15 words that tells the user what their last session was about. "
                        "Start with 'Last session:' and be specific. Use past tense. No quotes.\n\n"
                        f"Summary:\n{summaries[0]}"
                    ),
                }],
                temperature=0.2,
                max_tokens=40,
            )
            memory_recap = recap.choices[0].message.content.strip().rstrip(".")
        except Exception as e:
            logger.warning(f"memory recap generation skipped: {e}")

    return {"prior_summaries": summaries, "memory_recap": memory_recap}


@app.post("/session/end")
@limiter.limit(RATE_SESSION_END)
async def session_end(request: Request, data: dict):
    device_id = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")
    history = data.get("history", [])

    if not history:
        return {"status": "skipped"}

    if client is not None and len(history) >= 2:
        history_text = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history[-10:]])
        summary_prompt = f"""Summarize this interview session concisely in 150 words or less. Include:
- Key topics discussed
- Names or companies mentioned
- Candidate's stated experience/background
- Interviewer's apparent style and focus areas

Session transcript:
{history_text}"""
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            summary = completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            summary = f"Session with {len(history)} exchanges."
    else:
        summary = f"Session with {len(history)} exchanges."

    try:
        if engine is None:
            raise RuntimeError("No database configured")
        with engine.connect() as conn:
            conn.execute(
                text("INSERT INTO sessions (device_id, user_email, created_at, summary) VALUES (:did, :email, :created_at, :summary)"),
                {"did": device_id, "email": user_email, "created_at": datetime.utcnow().isoformat(), "summary": summary}
            )
            conn.commit()
    except Exception as e:
        logger.error(f"session/end DB write error: {e}")
        return {"status": "error", "detail": str(e)}

    logger.info(f"📝 Session saved for {user_email}: {summary[:60]}...")
    return {"status": "saved", "summary": summary}


@app.post("/quick_transcribe")
@limiter.limit(RATE_TRANSCRIBE)
async def quick_transcribe(
    request: Request,
    audio: UploadFile = File(None),
    file: UploadFile = File(None),
    deviceId: str = Form(""),
    userEmail: str = Form("anonymous"),
):
    """Fast STT-only endpoint for question continuation detection."""
    if client is None:
        raise HTTPException(status_code=503, detail="STT service unavailable")

    # Auth gate — caller must be a KNOWN user (a credits row exists), regardless of
    # balance. Previously this used _deduct_credits(user_key, 0)==0, which (a) wrongly
    # rejected legit paid users sitting at exactly 0 balance and (b) INSERTed a zero
    # credit_ledger row on every call (this endpoint fires frequently for continuation
    # detection). A read-only SELECT avoids both.
    try:
        raw_user_id, account_id, _ = _resolve_user(request, deviceId, userEmail)
        user_key = _get_credits_user_id(raw_user_id, account_id)
        if engine is not None:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT 1 FROM credits WHERE user_id=:uid"),
                    {"uid": user_key}
                ).fetchone()
            if row is None:
                raise HTTPException(status_code=401, detail="Unauthorized")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized")

    actual_file = audio or file
    if not actual_file:
        raise HTTPException(status_code=400, detail="No audio file provided")

    content = await actual_file.read()
    if len(content) < 1000:
        return {"transcript": ""}

    temp_filename = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(content)
            temp_filename = tmp.name
        with open(temp_filename, "rb") as af:
            transcription = client.audio.transcriptions.create(
                file=(temp_filename, af.read(), "audio/webm"),
                model="whisper-large-v3-turbo",
                response_format="text",
            )
        return {"transcript": transcription.strip() if transcription else ""}
    except Exception as e:
        logger.error(f"quick_transcribe error: {e}")
        return {"transcript": ""}
    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.unlink(temp_filename)


# ========================
# CONTEXT UPLOAD
# ========================

@app.post("/upload-context")
@limiter.limit(RATE_UPLOAD)
async def upload_context(
    request: Request,
    file: UploadFile = File(...),
    deviceId: str = Form(""),
    userEmail: str = Form("anonymous"),
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Context uploads are a Solo-and-up feature (landing pricing card).
    _uid, _aid, _ = _resolve_user(request, deviceId, userEmail)
    _plan = _get_credit_balance(_get_credits_user_id(_uid, _aid))["plan_type"]
    if _plan == "free":
        raise HTTPException(status_code=403, detail={
            "code": "PLAN_REQUIRED",
            "message": "Context uploads are available on the Solo plan and above.",
        })

    filename = file.filename.lower()
    content_type = (file.content_type or "").lower()

    if not (filename.endswith(".pdf") or filename.endswith(".txt")
            or "pdf" in content_type or "text" in content_type):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum 5MB.")

    extracted = ""
    if filename.endswith(".txt") or "text" in content_type:
        try:
            extracted = content.decode("utf-8", errors="replace")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read text file: {e}")
    else:
        try:
            from pypdf import PdfReader
            import io as _io
            reader = PdfReader(_io.BytesIO(content))
            pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            extracted = "\n\n".join(pages)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not read PDF: {e}")

    if len(extracted) > 8000:
        extracted = extracted[:8000] + "...[truncated]"

    logger.info(f"📄 upload-context: {file.filename} → {len(extracted)} chars")
    # FIX 6: cache so context survives a page refresh within the session
    if deviceId:
        _save_ctx_cache(f"{deviceId}_{userEmail}", extracted)
    return {"text": extracted, "filename": file.filename, "chars": len(extracted)}


# ========================
# URL BRIEFING (JINA + GROQ)
# Jina+Groq is now primary briefing engine.
# PERPLEXITY_API_KEY reserved for future use.
# ========================

@app.post("/brief")
@limiter.limit(RATE_BRIEF)
async def brief_url(request: Request, data: dict):
    url          = data.get("url", "").strip()
    device_id    = data.get("deviceId", "")
    user_email   = data.get("userEmail", "anonymous")
    role         = (data.get("role", "")   or "General").strip()
    mode         = (data.get("mode", "")   or "Analyst").strip()
    # `tier` is accepted for backward compatibility but no longer branches behavior —
    # there is one standard brief now. Cost is fully METERED (Jina + Groq-70B tokens).
    tier         = "deep_brief"
    goal         = data.get("goal", "").strip()
    context_text = data.get("context_text", "").strip()

    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    if any(blocked in url.lower() for blocked in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254.", "10.", "192.168.", "172.16.")):
        raise HTTPException(status_code=400, detail="That URL is not publicly accessible.")

    # Plan gate: Command+ only — use _resolve_user for cross-device consistency
    user_id, account_id, _ = _resolve_user(request, device_id, user_email)
    user_id = _get_credits_user_id(user_id, account_id)
    plan_info = _get_credit_balance(user_id)
    if plan_info["plan_type"] not in BRIEFING_ALLOWED_PLANS:
        raise HTTPException(status_code=403, detail={
            "code": "PLAN_REQUIRED",
            "message": "URL briefing requires Command plan or above.",
            "required_plan": "command",
        })

    # Step 1 — Fetch page content via Jina Reader (no API key required)
    jina_content = None
    fallback_used = False
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient() as _jina_http:
            jina_resp = await _jina_http.get(
                f"https://r.jina.ai/{url}",
                headers={"Accept": "text/plain"},
                follow_redirects=True,
                timeout=20.0,
            )
        if jina_resp.status_code == 200 and jina_resp.text.strip():
            jina_content = jina_resp.text[:12000]
            _track_usage(user_id, user_email, "jina_fetch",
                         len(jina_content), 0.0, json.dumps({"url": url[:100]}))
    except Exception as e:
        logger.warning(f"Jina fetch failed for {url}: {e}")

    if not jina_content:
        fallback_used = True
        try:
            domain = url.split("//")[-1].split("/")[0]
        except Exception:
            domain = url
        jina_content = (
            f"URL: {url}\nDomain: {domain}\n"
            "(Direct page content unavailable — brief based on domain context only.)"
        )

    # Step 2 — Build the Groq prompt. Single standard brief (the quick_read /
    # war_room tiers were removed Jun 8 2026 — one consistent depth, metered cost).
    sections_instruction = (
        f"Build a structured pre-call intelligence brief with exactly these sections:\n"
        f"SUMMARY (2-3 sentences)\n"
        f"KEY FACTS (5 bullet points most relevant to a {role} conversation)\n"
        f"LIKELY OBJECTIONS (3 objections this person or company might raise)\n"
        f"RECOMMENDED ANGLES (3 tactical approaches given the {mode} mode)\n"
        f"RISKS TO WATCH (2 things that could go wrong)\n"
        f"OPENING LINE (one suggested opening line tailored to {role} and {mode})\n\n"
        f"Keep entire brief under 600 words. Be specific not generic."
    )
    max_tokens = 900

    goal_prefix = (
        f"The user's goal for this session: {goal}\n\n"
        f"Using the above goal and the following material, "
    ) if goal else ""

    extra_context = (
        f"\n\nAdditional context from uploaded file:\n{context_text[:8000]}"
    ) if context_text else ""

    system_content = (
        f"You are a pre-call intelligence analyst.\n"
        f"A user is about to enter a {role} conversation using the {mode} mode.\n"
        f"They have provided this background material:\n\n"
        f"{jina_content}{extra_context}\n\n"
        f"{goal_prefix}{sections_instruction}"
    )

    try:
        groq_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": system_content}],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        brief_text = groq_response.choices[0].message.content.strip()
        brief_pt = getattr(groq_response.usage, "prompt_tokens", 0) or 0
        brief_ct = getattr(groq_response.usage, "completion_tokens", 0) or 0
        _track_usage(user_id, user_email, "groq_llm",
                     len(brief_text), len(brief_text) / 4 * 0.0000008,
                     json.dumps({"tier": tier}))
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Groq brief generation error: {e}")
        raise HTTPException(status_code=500, detail="Brief generation failed. Please try again.")

    if fallback_used:
        brief_text = (
            "Note: Direct page content unavailable. Brief based on domain context only.\n\n"
            + brief_text
        )

    # FIX 6: cache brief text so it survives a page refresh within the session
    _save_ctx_cache(user_id, brief_text)

    # Step 3 — Metered billing: charge the REAL cost of this brief (Jina page fetch +
    # Groq-70B tokens), not a flat tier price. Single source of truth = PROVIDER_RATES.
    brief_raw_cost = _llm_cost_usd("llama-3.3-70b-versatile", brief_pt, brief_ct)
    if not fallback_used:
        brief_raw_cost += PROVIDER_RATES["jina_fetch"]["req"]
    credits_cost = _cost_to_credits(brief_raw_cost)
    idem_key = hashlib.sha256(
        f"{user_id}:{url}:{tier}:{int(time.time() // 10)}".encode()
    ).hexdigest()
    _deduct_credits(user_id, credits_cost, feature=f"briefing:{tier}",
                    idempotency_key=idem_key, provider="groq+jina",
                    input_units=brief_pt, output_units=brief_ct,
                    raw_cost_usd=brief_raw_cost)

    logger.info(f"🔍 {tier} brief for {url} ({len(brief_text)} chars, raw ${brief_raw_cost:.6f}, "
                f"pt={brief_pt} ct={brief_ct}) → {credits_cost} credits deducted")
    return {
        "brief_text":   brief_text,
        "brief":        brief_text,   # backwards compatibility
        "source_url":   url,
        "role_context": role,
        "credits_used": credits_cost,
        "tier":         tier,
        "fallback":     fallback_used,
    }


# ========================
# PREFERENCES
# ========================

@app.get("/preferences")
@limiter.limit(RATE_READS)
async def get_preferences(request: Request, deviceId: str, userEmail: str = "anonymous"):
    if engine is None:
        return {"preference_text": ""}
    user_id, account_id, _ = _resolve_user(request, deviceId, userEmail)
    try:
        with engine.connect() as conn:
            if account_id:
                row = conn.execute(text("""
                    SELECT preference_text FROM preferences
                    WHERE account_id = :aid OR user_id = :uid
                    ORDER BY (CASE WHEN account_id = :aid THEN 0 ELSE 1 END)
                    LIMIT 1
                """), {"aid": account_id, "uid": user_id}).fetchone()
            else:
                row = conn.execute(
                    text("SELECT preference_text FROM preferences WHERE user_id=:uid"),
                    {"uid": user_id}
                ).fetchone()
        return {"preference_text": row[0] if row else ""}
    except Exception as e:
        logger.error(f"get_preferences error: {e}")
        return {"preference_text": ""}


@app.post("/preferences")
@limiter.limit(RATE_READS)
async def save_preferences(request: Request, data: dict):
    device_id = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")
    # Clamp to a sane length — unbounded preference_text is an abuse / row-growth vector.
    preference_text = (data.get("preference_text", "") or "")[:8000]

    if engine is None:
        return {"status": "skipped"}

    user_id, account_id, _ = _resolve_user(request, device_id, user_email)
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO preferences (user_id, preference_text, updated_at, account_id)
                VALUES (:uid, :text, :now, :aid)
                ON CONFLICT (user_id) DO UPDATE SET
                    preference_text = EXCLUDED.preference_text,
                    updated_at = EXCLUDED.updated_at,
                    account_id = COALESCE(preferences.account_id, EXCLUDED.account_id)
            """), {"uid": user_id, "text": preference_text, "now": datetime.utcnow().isoformat(), "aid": account_id})
            conn.commit()
        logger.info(f"⚙️ Preferences saved for {user_id}")
        return {"status": "saved"}
    except Exception as e:
        logger.error(f"save_preferences error: {e}")
        return {"status": "error", "detail": str(e)}


# ========================
# SESSION HISTORY
# ========================

@app.post("/end-session")
@limiter.limit(RATE_SESSION_END)
async def end_session(request: Request, data: dict):
    session_id = data.get("session_id", "")
    device_id  = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")
    role       = data.get("role", "Interview Coach")
    mode       = data.get("mode", "Diplomat")
    style      = data.get("style", "Standard")
    transcript = data.get("transcript", [])
    duration_seconds = int(data.get("duration_seconds", 0))
    ghost_mode = bool(data.get("ghost_mode", False))
    save_memory = bool(data.get("save_memory", False))

    # Ghost mode: wipe in-memory traces and store nothing
    if ghost_mode:
        user_key = f"{device_id}_{user_email}"
        question_buffer.pop(user_key, None)
        _clear_ctx_cache(user_key)
        logger.info(f"👻 Ghost session ended — memory wiped for {user_key[:20]}…")
        return {"status": "ghost", "summary": ""}

    if not transcript or len(transcript) < 2:
        return {"status": "skipped", "summary": ""}

    # Resolve authenticated identity for account-linked session saves
    user_id, account_id, _ = _resolve_user(request, device_id, user_email)
    plan_info = _get_credit_balance(_get_credits_user_id(user_id, account_id))
    can_summarize = plan_info["plan_type"] in SUMMARY_ALLOWED_PLANS

    summary = f"Session with {len(transcript)} exchanges."
    # Always save the transcript so ANY session can be resumed later (ghost mode already returned above).
    full_transcript_json = json.dumps(transcript)

    if client is not None and save_memory:
        history_text = "\n".join(
            [f"Q: {t.get('question','')}\nA: {t.get('answer','')}" for t in transcript]
        )
        prompt = (
            "Summarize this coaching session in 2–4 sentences (under 80 words). "
            "Cover: what topic was practiced, key names or companies mentioned, "
            "what was decided or coached, and how the conversation went. "
            "Be specific and concrete.\n\n" + history_text
        )
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=120,
            )
            summary = completion.choices[0].message.content.strip().rstrip(".")
        except Exception as e:
            logger.error(f"end-session summary error: {e}")
        full_transcript_json = json.dumps(transcript)
        # Deduct 2 credits for memory save
        try:
            credits_uid = _get_credits_user_id(user_id, account_id)
            _deduct_credits(credits_uid, 2, feature="memory_save")
        except Exception as e:
            logger.warning(f"end-session memory_save credit deduct failed: {e}")
    elif client is not None and can_summarize:
        history_text = "\n".join(
            [f"Q: {t.get('question','')}\nA: {t.get('answer','')}" for t in transcript[-10:]]
        )
        prompt = (
            "Summarize this conversation in one sentence of under 20 words. "
            "Focus on what happened and what the outcome was.\n\n" + history_text
        )
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=60,
            )
            summary = completion.choices[0].message.content.strip().rstrip(".")
        except Exception as e:
            logger.error(f"end-session summary error: {e}")

    # Generate a short 3-5 word title from the summary
    session_title = None
    if client is not None and not summary.startswith("Session with "):
        try:
            title_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": f"Generate a 3-5 word title for this conversation. Return only the title, nothing else. No quotes, no punctuation at the end.\n\n{summary}"}],
                temperature=0.3,
                max_tokens=20,
            )
            session_title = title_completion.choices[0].message.content.strip().strip('"').strip("'").rstrip(".")
        except Exception as e:
            logger.error(f"end-session title error: {e}")

    # Persist to session_history with account_id for cross-device memory
    if engine is not None and session_id:
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO session_history
                        (session_id, user_id, account_id, role, persona, style, summary, session_title, timestamp, duration_seconds, full_transcript)
                    VALUES (:sid, :uid, :aid, :role, :mode, :style, :summary, :session_title, :ts, :dur, :ft)
                    ON CONFLICT (session_id) DO UPDATE SET
                        summary         = EXCLUDED.summary,
                        session_title   = COALESCE(EXCLUDED.session_title, session_history.session_title),
                        full_transcript = COALESCE(EXCLUDED.full_transcript, session_history.full_transcript),
                        account_id      = COALESCE(session_history.account_id, EXCLUDED.account_id)
                """), {
                    "sid": session_id,
                    "uid": user_id,
                    "aid": account_id,
                    "role": role,
                    "mode": mode,
                    "style": style,
                    "summary": summary,
                    "session_title": session_title,
                    "ts": datetime.utcnow().isoformat(),
                    "dur": duration_seconds,
                    "ft": full_transcript_json,
                })
                conn.commit()
        except Exception as e:
            logger.error(f"end-session DB error: {e}")

    # FIX 6: clear context cache now that the session is over
    _clear_ctx_cache(f"{device_id}_{user_email}")
    logger.info(f"📝 end-session {session_id[:8]}… | {role}/{mode} | {duration_seconds}s | {summary[:50]}")
    return {"status": "saved", "summary": summary}


@app.get("/session-memory")
@limiter.limit(RATE_READS)
async def get_session_memory(request: Request, deviceId: str, userEmail: str = "anonymous"):
    if engine is None:
        return {"summaries": []}
    user_id, account_id, _ = _resolve_user(request, deviceId, userEmail)
    # Cross-session memory is a Pro-and-up feature (landing pricing card).
    _plan = _get_credit_balance(_get_credits_user_id(user_id, account_id))["plan_type"]
    if _plan not in {"pro", "command", "operator", "founding_50"}:
        return {"summaries": []}
    try:
        with engine.connect() as conn:
            if account_id:
                rows = conn.execute(text("""
                    SELECT summary FROM session_history
                    WHERE (account_id = :aid OR user_id = :uid)
                    AND summary IS NOT NULL AND summary != ''
                    ORDER BY timestamp DESC LIMIT 3
                """), {"aid": account_id, "uid": user_id}).fetchall()
            else:
                rows = conn.execute(text("""
                    SELECT summary FROM session_history
                    WHERE user_id = :uid AND summary IS NOT NULL AND summary != ''
                    ORDER BY timestamp DESC LIMIT 3
                """), {"uid": user_id}).fetchall()
        return {"summaries": [r[0] for r in rows]}
    except Exception as e:
        logger.error(f"session-memory error: {e}")
        return {"summaries": []}


@app.post("/session-memory/clear")
@limiter.limit(RATE_READS)
async def clear_session_memory(request: Request, data: dict):
    """Nullify all session summaries for the user. History rows are kept; only the summary text is wiped."""
    device_id  = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")
    if engine is None:
        return {"status": "skipped", "cleared": 0}
    user_id, account_id, _ = _resolve_user(request, device_id, user_email)
    try:
        with engine.connect() as conn:
            if account_id:
                result = conn.execute(text("""
                    UPDATE session_history SET summary = NULL
                    WHERE (account_id = :aid OR user_id = :uid)
                    AND summary IS NOT NULL
                """), {"aid": account_id, "uid": user_id})
            else:
                result = conn.execute(text("""
                    UPDATE session_history SET summary = NULL
                    WHERE user_id = :uid AND summary IS NOT NULL
                """), {"uid": user_id})
            conn.commit()
            cleared = result.rowcount
        logger.info(f"🗑️  Session memory cleared for {user_id[:24]}… ({cleared} rows)")
        return {"status": "cleared", "cleared": cleared}
    except Exception as e:
        logger.error(f"clear_session_memory error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear session memory.")


@app.get("/session-history")
@limiter.limit(RATE_READS)
async def get_session_history(request: Request, deviceId: str, userEmail: str = "anonymous"):
    if engine is None:
        return {"sessions": []}
    user_id, account_id, _ = _resolve_user(request, deviceId, userEmail)
    try:
        with engine.connect() as conn:
            if account_id:
                rows = conn.execute(text("""
                    SELECT session_id, role, persona, style, summary, timestamp, duration_seconds, full_transcript, session_title
                    FROM session_history
                    WHERE account_id = :aid OR user_id = :uid
                    ORDER BY timestamp DESC LIMIT 50
                """), {"aid": account_id, "uid": user_id}).fetchall()
            else:
                rows = conn.execute(text("""
                    SELECT session_id, role, persona, style, summary, timestamp, duration_seconds, full_transcript, session_title
                    FROM session_history WHERE user_id = :uid
                    ORDER BY timestamp DESC LIMIT 50
                """), {"uid": user_id}).fetchall()
        sessions = [
            {
                "session_id": r[0],
                "role": r[1],
                "mode": r[2],
                "style": r[3],
                "summary": r[4],
                "timestamp": r[5],
                "duration_seconds": r[6],
                "full_transcript": json.loads(r[7]) if r[7] else None,
                "session_title": r[8],
            }
            for r in rows
        ]
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"get_session_history error: {e}")
        return {"sessions": []}


@app.patch("/session/{session_id}/rename")
@limiter.limit(RATE_READS)
async def rename_session(session_id: str, request: Request, data: dict):
    title = (data.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title required")
    device_id  = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")
    user_id, account_id, _ = _resolve_user(request, device_id, user_email)
    if engine is None:
        raise HTTPException(status_code=503, detail="DB unavailable")
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                UPDATE session_history SET session_title = :title
                WHERE session_id = :sid AND (user_id = :uid OR account_id = :aid)
            """), {"title": title, "sid": session_id, "uid": user_id, "aid": account_id or ""})
            conn.commit()
    except Exception as e:
        logger.error(f"rename_session error: {e}")
        raise HTTPException(status_code=500, detail="Rename failed")
    return {"status": "ok"}


@app.delete("/session/{session_id}")
@limiter.limit(RATE_READS)
async def delete_session(session_id: str, request: Request, deviceId: str = "", userEmail: str = "anonymous"):
    user_id, account_id, _ = _resolve_user(request, deviceId, userEmail)
    if engine is None:
        raise HTTPException(status_code=503, detail="DB unavailable")
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                DELETE FROM session_history
                WHERE session_id = :sid AND (user_id = :uid OR account_id = :aid)
            """), {"sid": session_id, "uid": user_id, "aid": account_id or ""})
            conn.commit()
    except Exception as e:
        logger.error(f"delete_session error: {e}")
        raise HTTPException(status_code=500, detail="Delete failed")
    return {"status": "deleted"}


@app.get("/credits")
@limiter.limit(RATE_READS)
async def get_credits(request: Request, deviceId: str, userEmail: str = "anonymous"):
    user_id, account_id, _ = _resolve_user(request, deviceId, userEmail)

    # Account-level lookup when authenticated (cross-device: find the most-credited row)
    credit_data = None
    if account_id and engine is not None:
        try:
            with engine.connect() as conn:
                row = conn.execute(text("""
                    SELECT balance, plan_type, total_used FROM credits
                    WHERE account_id = :aid ORDER BY balance DESC LIMIT 1
                """), {"aid": account_id}).fetchone()
            if row:
                credit_data = {"balance": row[0], "plan_type": row[1], "total_used": row[2]}
        except Exception as e:
            logger.error(f"credits account lookup error: {e}")

    if credit_data is None:
        credit_data = _get_credit_balance(user_id)

    is_founding = False
    if engine is not None:
        try:
            with engine.connect() as conn:
                if account_id:
                    row = conn.execute(
                        text("SELECT 1 FROM founding_members WHERE account_id = :aid OR user_id = :uid LIMIT 1"),
                        {"aid": account_id, "uid": user_id}
                    ).fetchone()
                else:
                    row = conn.execute(
                        text("SELECT 1 FROM founding_members WHERE user_id=:uid"),
                        {"uid": user_id}
                    ).fetchone()
                is_founding = row is not None
        except Exception as e:
            logger.error(f"founding check error: {e}")
    cost_this_month = 0.0
    if engine is not None:
        try:
            lookup_key = _get_credits_user_id(user_id, account_id)
            with engine.connect() as conn:
                r = conn.execute(text("""
                    SELECT COALESCE(SUM(estimated_cost_usd), 0)
                    FROM api_usage
                    WHERE user_key = :uk
                    AND created_at >= DATE_TRUNC('month', NOW())
                """), {"uk": lookup_key}).fetchone()
                if r:
                    cost_this_month = float(r[0])
        except Exception as e:
            logger.error(f"credits: cost_this_month error: {e}")

    return {
        "balance": credit_data["balance"],
        "plan_type": credit_data["plan_type"],
        "total_used": credit_data["total_used"],
        "is_founding_member": is_founding,
        "estimated_cost_this_month": round(cost_this_month, 6),
    }


@app.post("/enable-payg")
@limiter.limit(RATE_READS)
async def enable_payg(request: Request, deviceId: str = Form(...), userEmail: str = Form("anonymous")):
    if engine is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    raw_user_id, account_id, _ = _resolve_user(request, deviceId, userEmail)
    user_key = _get_credits_user_id(raw_user_id, account_id)
    if not user_key:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE credits SET payg_enabled = TRUE WHERE user_id = :uid
            """), {"uid": user_key})
    except Exception as e:
        logger.error(f"enable-payg error: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    logger.info(f"✅ PAYG enabled for {user_key[:12]}…")
    return {"payg_enabled": True, "message": "Pay-as-you-go enabled. Overages will be tracked at 1.5x credit cost."}


@app.post("/disable-payg")
@limiter.limit(RATE_READS)
async def disable_payg(request: Request, deviceId: str = Form(...), userEmail: str = Form("anonymous")):
    if engine is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    raw_user_id, account_id, _ = _resolve_user(request, deviceId, userEmail)
    user_key = _get_credits_user_id(raw_user_id, account_id)
    if not user_key:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE credits SET payg_enabled = FALSE WHERE user_id = :uid
            """), {"uid": user_key})
    except Exception as e:
        logger.error(f"disable-payg error: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    logger.info(f"🔴 PAYG disabled for {user_key[:12]}…")
    return {"payg_enabled": False, "message": "Pay-as-you-go disabled. Service will stop when credits reach zero."}


def _payment_method_label(pm) -> str | None:
    """Human-readable payment method line for billing UI (best-effort)."""
    if not pm:
        return None
    try:
        ptype = pm.get("type") if isinstance(pm, dict) else getattr(pm, "type", None)
        if ptype == "card":
            card = pm.get("card") if isinstance(pm, dict) else pm.card
            if not card:
                return "Card on file"
            brand = (card.get("brand") if isinstance(card, dict) else getattr(card, "brand", None)) or "Card"
            last4 = card.get("last4") if isinstance(card, dict) else getattr(card, "last4", None)
            if last4:
                return f"{str(brand).title()} ···· {last4}"
            return f"{str(brand).title()} card"
    except Exception:
        pass
    return "Payment method on file"


@app.get("/billing-summary")
@limiter.limit(RATE_READS)
async def billing_summary(request: Request, deviceId: str, userEmail: str = "anonymous"):
    """Plan/credits from PostgreSQL plus an optional live Stripe snapshot (subscription dates, PM, recent invoices)."""
    user_id, account_id, _resolved = _resolve_user(request, deviceId, userEmail)
    if engine is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    out = {
        "balance": 0,
        "plan_type": "free",
        "total_used": 0,
        "payg_enabled": False,
        "subscription_status": "active",
        "credits_reset_date": None,
        "included_credits_per_cycle": PLAN_CREDITS["free"],
        "is_founding_member": False,
        "stripe_customer": False,
        "stripe_current_period_end": None,
        "stripe_cancel_at_period_end": None,
        "stripe_subscription_status": None,
        "payment_method_summary": None,
        "invoices": [],
        "stripe_sync_error": None,
    }

    customer_id = None
    row = None
    try:
        with engine.connect() as conn:
            if account_id:
                row = conn.execute(
                    text("""
                        SELECT balance, plan_type, total_used, reset_date, subscription_status, payg_enabled
                        FROM credits WHERE account_id = :aid ORDER BY balance DESC LIMIT 1
                    """),
                    {"aid": account_id},
                ).fetchone()
            if row is None:
                row = conn.execute(
                    text("""
                        SELECT balance, plan_type, total_used, reset_date, subscription_status, payg_enabled
                        FROM credits WHERE user_id = :uid
                    """),
                    {"uid": user_id},
                ).fetchone()
            if row:
                out["balance"] = int(row[0])
                out["plan_type"] = row[1] or "free"
                out["total_used"] = int(row[2] or 0)
                out["credits_reset_date"] = row[3]
                out["subscription_status"] = (row[4] or "active").strip() or "active"
                out["payg_enabled"] = bool(row[5])
                out["included_credits_per_cycle"] = int(PLAN_CREDITS.get(out["plan_type"], PLAN_CREDITS["free"]))

            if account_id:
                r2 = conn.execute(
                    text("""
                        SELECT customer_id FROM stripe_customers
                        WHERE account_id = :aid OR user_id = :uid LIMIT 1
                    """),
                    {"aid": account_id, "uid": user_id},
                ).fetchone()
            else:
                r2 = conn.execute(
                    text("SELECT customer_id FROM stripe_customers WHERE user_id = :uid LIMIT 1"),
                    {"uid": user_id},
                ).fetchone()
            if r2 and r2[0]:
                customer_id = r2[0]

            if account_id:
                fm = conn.execute(
                    text("SELECT 1 FROM founding_members WHERE account_id = :aid OR user_id = :uid LIMIT 1"),
                    {"aid": account_id, "uid": user_id},
                ).fetchone()
            else:
                fm = conn.execute(
                    text("SELECT 1 FROM founding_members WHERE user_id = :uid LIMIT 1"),
                    {"uid": user_id},
                ).fetchone()
            out["is_founding_member"] = fm is not None
    except Exception as e:
        logger.error(f"billing-summary DB error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    if row is None:
        fb = _get_credit_balance(user_id)
        out["balance"] = int(fb.get("balance", 0))
        out["plan_type"] = fb.get("plan_type") or "free"
        out["total_used"] = int(fb.get("total_used") or 0)
        out["included_credits_per_cycle"] = int(PLAN_CREDITS.get(out["plan_type"], PLAN_CREDITS["free"]))
        try:
            with engine.connect() as conn:
                r3 = conn.execute(
                    text("SELECT reset_date, subscription_status FROM credits WHERE user_id = :uid"),
                    {"uid": user_id},
                ).fetchone()
                if r3:
                    out["credits_reset_date"] = r3[0]
                    out["subscription_status"] = (r3[1] or "active").strip() or "active"
        except Exception as e2:
            logger.warning(f"billing-summary credits fallback: {e2}")

    if not customer_id:
        return out

    out["stripe_customer"] = True
    stripe_key = os.environ.get("STRIPE_SECRET_KEY")
    if not stripe_key:
        out["stripe_sync_error"] = "stripe_not_configured"
        return out

    stripe.api_key = stripe_key
    try:
        cust = stripe.Customer.retrieve(customer_id, expand=["invoice_settings.default_payment_method"])
        dpm = None
        try:
            invset = cust.get("invoice_settings") if isinstance(cust, dict) else cust.invoice_settings
            if invset:
                dpm = invset.get("default_payment_method") if isinstance(invset, dict) else invset.default_payment_method
        except Exception:
            dpm = None
        if isinstance(dpm, str) and dpm:
            dpm = stripe.PaymentMethod.retrieve(dpm)
        if dpm:
            out["payment_method_summary"] = _payment_method_label(dpm)

        subs = stripe.Subscription.list(customer=customer_id, limit=10, status="all")
        active = None
        for s in subs.data:
            if s.status in ("active", "trialing", "past_due"):
                active = s
                break
        if active:
            out["stripe_current_period_end"] = int(active.current_period_end)
            out["stripe_cancel_at_period_end"] = bool(getattr(active, "cancel_at_period_end", False))
            out["stripe_subscription_status"] = active.status
            pm_sub = active.get("default_payment_method") if isinstance(active, dict) else getattr(active, "default_payment_method", None)
            if pm_sub:
                if isinstance(pm_sub, str):
                    pm_sub = stripe.PaymentMethod.retrieve(pm_sub)
                out["payment_method_summary"] = out["payment_method_summary"] or _payment_method_label(pm_sub)

        invs = stripe.Invoice.list(customer=customer_id, limit=8)
        lines = []
        for inv in invs.data:
            url = inv.get("hosted_invoice_url") if isinstance(inv, dict) else getattr(inv, "hosted_invoice_url", None)
            if not url:
                continue
            created = inv.get("created") if isinstance(inv, dict) else inv.created
            total = inv.get("total") if isinstance(inv, dict) else inv.total
            cur = (inv.get("currency") if isinstance(inv, dict) else inv.currency) or "usd"
            inv_id = inv.get("id") if isinstance(inv, dict) else inv.id
            status = inv.get("status") if isinstance(inv, dict) else inv.status
            lines.append({
                "id": inv_id,
                "created": int(created) if created is not None else None,
                "total": int(total) if total is not None else None,
                "currency": str(cur).lower(),
                "status": status,
                "hosted_invoice_url": url,
            })
        out["invoices"] = lines
    except Exception as e:
        logger.warning(f"billing-summary Stripe error: {e}")
        out["stripe_sync_error"] = str(e)[:240]

    return out


STRIPE_PRICE_IDS = {
    "echo":        os.environ.get("STRIPE_PRICE_ECHO", ""),
    "pro":         os.environ.get("STRIPE_PRICE_PRO", ""),
    "command":     os.environ.get("STRIPE_PRICE_COMMAND", ""),
    "operator":    os.environ.get("STRIPE_PRICE_OPERATOR", ""),
    "founding_50": os.environ.get("STRIPE_PRICE_FOUNDING50", ""),
}

PLAN_MODES = {
    "echo":        "subscription",
    "pro":         "subscription",
    "command":     "subscription",
    "operator":    "subscription",
    "founding_50": "payment",
}

# Plans that allow TTS audio whisper (Solo/echo added Jun 7 2026 — gets ~1 hr audio)
TTS_ALLOWED_PLANS = {"echo", "pro", "command", "operator", "founding_50"}

# Plans that allow URL briefing (Jina+Groq)
BRIEFING_ALLOWED_PLANS = {"command", "operator", "founding_50"}

# Plans that get AI-generated session summaries
SUMMARY_ALLOWED_PLANS = {"command", "operator", "founding_50"}

# Option names that are plan-gated (the "Custom" free-text persona/style).
OPERATOR_ONLY_OPTIONS = {"Custom"}
# Plans allowed to use the Custom persona/style (Jun 8 2026: opened to Pro+Power;
# was Operator-only, which made Custom dead for every purchasable plan).
CUSTOM_ALLOWED_PLANS = {"pro", "command", "operator", "founding_50"}

@app.post("/billing-portal")
@limiter.limit(RATE_CHECKOUT)
async def billing_portal(request: Request, data: dict):
    """Create a Stripe Customer Portal session for self-serve subscription management."""
    device_id  = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")

    stripe_key = os.environ.get("STRIPE_SECRET_KEY")
    if not stripe_key:
        raise HTTPException(status_code=503, detail="Billing not configured")

    if engine is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    stripe.api_key = stripe_key
    user_id, account_id, _ = _resolve_user(request, device_id, user_email)

    # Look up the Stripe customer ID — prefer account_id match for cross-device access
    try:
        with engine.connect() as conn:
            if account_id:
                row = conn.execute(
                    text("SELECT customer_id FROM stripe_customers WHERE account_id = :aid OR user_id = :uid LIMIT 1"),
                    {"aid": account_id, "uid": user_id}
                ).fetchone()
                # Fallback: pre-migration rows have NULL account_id; join via credits for cross-device match
                if not row:
                    row = conn.execute(
                        text("""
                            SELECT sc.customer_id
                            FROM stripe_customers sc
                            JOIN credits c ON c.user_id = sc.user_id
                            WHERE c.account_id = :aid
                            LIMIT 1
                        """),
                        {"aid": account_id}
                    ).fetchone()
            else:
                row = conn.execute(
                    text("SELECT customer_id FROM stripe_customers WHERE user_id=:uid"),
                    {"uid": user_id}
                ).fetchone()
    except Exception as e:
        logger.error(f"billing-portal DB lookup error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    if not row:
        raise HTTPException(status_code=404, detail={
            "code": "NO_SUBSCRIPTION",
            "message": "No active subscription found for this account. If you recently subscribed and see this message, contact support@cerebroecho.com.",
        })

    customer_id = row[0]
    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url="https://cerebroecho.com/app",
        )
        logger.info(f"🔗 Billing portal session created for user {user_id[:24]}…")
        return {"url": portal_session.url}
    except stripe.StripeError as e:
        logger.error(f"Stripe billing portal error: {e}")
        raise HTTPException(status_code=500, detail="Could not open the billing portal. Please try again.")


RATE_EXPORT  = os.environ.get("RATE_LIMIT_EXPORT",  "5/minute")
RATE_DELETE  = os.environ.get("RATE_LIMIT_DELETE",  "3/hour")

@app.get("/export-data")
@limiter.limit(RATE_EXPORT)
async def export_data(request: Request, deviceId: str, userEmail: str = "anonymous", include: str = ""):
    """Export selected personal data for the requesting user as a downloadable CSV file.

    `include` is an optional comma-separated list of sections to export:
    account, sessions, ledger, events. When omitted or empty, all sections
    are exported (backward compatible with the original one-click export).
    """
    user_id, account_id, resolved_email = _resolve_user(request, deviceId, userEmail)

    # Parse requested sections — default to all when unspecified
    ALL_SECTIONS = {"account", "sessions", "ledger", "events"}
    wanted = {s.strip().lower() for s in include.split(",") if s.strip()}
    wanted &= ALL_SECTIONS
    if not wanted:
        wanted = set(ALL_SECTIONS)

    # Unauthenticated requests still require a registered email to identify records
    if not account_id:
        if not userEmail or userEmail.strip().lower() in ("anonymous", "") or "@" not in userEmail:
            raise HTTPException(
                status_code=400,
                detail="A registered email address is required to export data. Sign in or save your email in the app first.",
            )

    if engine is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    export_email = resolved_email if account_id else userEmail
    account_info: dict = {
        "email": export_email,
        "device_id": deviceId,
        "authenticated": str(account_id is not None),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "plan": "",
        "credits_remaining": "",
        "credits_used_total": "",
        "credits_reset_date": "",
        "subscription_status": "",
        "founding_member_since": "",
        "preferences": "",
    }
    sessions_rows: list = []
    credit_ledger_rows: list = []
    usage_events_rows: list = []

    try:
        with engine.connect() as conn:
            # Account / credits
            if "account" in wanted:
                if account_id:
                    row = conn.execute(text("""
                        SELECT balance, plan_type, total_used, reset_date, subscription_status
                        FROM credits WHERE account_id = :aid OR user_id = :uid LIMIT 1
                    """), {"aid": account_id, "uid": user_id}).fetchone()
                else:
                    row = conn.execute(text("""
                        SELECT balance, plan_type, total_used, reset_date, subscription_status
                        FROM credits WHERE user_id = :uid
                    """), {"uid": user_id}).fetchone()
                if row:
                    account_info["plan"] = row[1]
                    account_info["credits_remaining"] = row[0]
                    account_info["credits_used_total"] = row[2]
                    account_info["credits_reset_date"] = row[3]
                    account_info["subscription_status"] = row[4]

                # Preferences
                if account_id:
                    row = conn.execute(text("""
                        SELECT preference_text FROM preferences
                        WHERE account_id = :aid OR user_id = :uid LIMIT 1
                    """), {"aid": account_id, "uid": user_id}).fetchone()
                else:
                    row = conn.execute(text("""
                        SELECT preference_text FROM preferences WHERE user_id = :uid
                    """), {"uid": user_id}).fetchone()
                if row:
                    account_info["preferences"] = row[0]

            # Session history
            if "sessions" in wanted:
                if account_id:
                    rows = conn.execute(text("""
                        SELECT session_id, role, persona, style, summary, timestamp, duration_seconds
                        FROM session_history WHERE account_id = :aid OR user_id = :uid
                        ORDER BY timestamp DESC
                    """), {"aid": account_id, "uid": user_id}).fetchall()
                else:
                    rows = conn.execute(text("""
                        SELECT session_id, role, persona, style, summary, timestamp, duration_seconds
                        FROM session_history WHERE user_id = :uid
                        ORDER BY timestamp DESC
                    """), {"uid": user_id}).fetchall()
                sessions_rows = [
                    [r[0], r[1], r[2], r[3], r[4], r[5], r[6]]
                    for r in rows
                ]

            # Credit ledger
            if "ledger" in wanted:
                rows = conn.execute(text("""
                    SELECT id, amount, balance_after, operation_type, feature, created_at
                    FROM credit_ledger WHERE user_id = :uid
                    ORDER BY created_at DESC
                """), {"uid": user_id}).fetchall()
                credit_ledger_rows = [
                    [r[0], r[1], r[2], r[3], r[4], r[5]]
                    for r in rows
                ]

            # Usage events
            if "events" in wanted:
                rows = conn.execute(text("""
                    SELECT id, feature, provider, credits_charged, raw_cost_usd, status, created_at
                    FROM usage_events WHERE user_id = :uid
                    ORDER BY created_at DESC
                """), {"uid": user_id}).fetchall()
                usage_events_rows = [
                    [r[0], r[1], r[2], r[3], r[4], r[5], r[6]]
                    for r in rows
                ]

            # Founding member (part of account section)
            if "account" in wanted:
                if account_id:
                    row = conn.execute(text("""
                        SELECT purchase_date FROM founding_members WHERE account_id = :aid OR user_id = :uid LIMIT 1
                    """), {"aid": account_id, "uid": user_id}).fetchone()
                else:
                    row = conn.execute(text("""
                        SELECT purchase_date FROM founding_members WHERE user_id = :uid
                    """), {"uid": user_id}).fetchone()
                if row:
                    account_info["founding_member_since"] = row[0]

    except Exception as e:
        logger.error(f"export-data error for {user_id[:24]}…: {e}")
        raise HTTPException(status_code=500, detail="Export failed. Please try again.")

    # Build multi-section CSV
    buf = io.StringIO()
    writer = csv.writer(buf)

    # Section 1: Account Info
    if "account" in wanted:
        writer.writerow(["# ACCOUNT INFO"])
        writer.writerow(["field", "value"])
        for field, value in account_info.items():
            writer.writerow([field, value])
        writer.writerow([])

    # Section 2: Sessions
    if "sessions" in wanted:
        writer.writerow(["# SESSIONS"])
        writer.writerow(["session_id", "role", "mode", "style", "summary", "timestamp", "duration_seconds"])
        for r in sessions_rows:
            writer.writerow(r)
        writer.writerow([])

    # Section 3: Credit Ledger
    if "ledger" in wanted:
        writer.writerow(["# CREDIT LEDGER"])
        writer.writerow(["id", "amount", "balance_after", "operation_type", "feature", "created_at"])
        for r in credit_ledger_rows:
            writer.writerow(r)
        writer.writerow([])

    # Section 4: Usage Events
    if "events" in wanted:
        writer.writerow(["# USAGE EVENTS"])
        writer.writerow(["id", "feature", "provider", "credits_charged", "raw_cost_usd", "status", "created_at"])
        for r in usage_events_rows:
            writer.writerow(r)

    csv_content = buf.getvalue()

    logger.info(f"📦 Data export for {user_id[:24]}… ({len(sessions_rows)} sessions, {len(credit_ledger_rows)} ledger, {len(usage_events_rows)} events)")
    filename = f"cerebroecho-data-export-{datetime.utcnow().strftime('%Y%m%d')}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/account/delete")
@limiter.limit(RATE_DELETE)
async def delete_account(request: Request, data: dict):
    """
    Self-serve account deletion.

    Deleted:   credits, session_history, preferences, stripe_customers (routing row),
               auth_sessions, magic_link_tokens, accounts record,
               legacy sessions table rows, in-memory usage_tracker entry.
    Retained:  founding_members (financial audit trail for $299 payment),
               processed_webhook_events (Stripe idempotency guard — safe to keep forever).
    Stripe:    active/past_due subscriptions are cancelled immediately before DB deletion.
               The Stripe Customer object itself is NOT deleted (needed for dispute/refund history).
    Auth:      When authenticated via Bearer token, device_id is optional — account_id
               covers all devices. Unauthenticated requests still require device_id.
    """
    device_id     = data.get("deviceId", "")
    user_email    = data.get("userEmail", "")
    confirm_email = data.get("confirmEmail", "")

    user_id, account_id, resolved_email = _resolve_user(request, device_id, user_email)
    check_email = resolved_email if account_id else user_email

    if not check_email or check_email.strip().lower() in ("anonymous", "") or "@" not in check_email:
        raise HTTPException(status_code=400, detail="A registered email address is required to delete an account.")
    if check_email.strip().lower() != confirm_email.strip().lower():
        raise HTTPException(status_code=400, detail="Email confirmation does not match. Please type your email exactly.")
    if not account_id and not device_id:
        raise HTTPException(status_code=400, detail="Device ID is required.")
    if engine is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # Verify the account exists before doing anything destructive
    try:
        with engine.connect() as conn:
            if account_id:
                exists = conn.execute(
                    text("SELECT 1 FROM accounts WHERE id = :aid"),
                    {"aid": account_id}
                ).fetchone()
                if not exists:
                    exists = conn.execute(
                        text("SELECT 1 FROM credits WHERE user_id = :uid"),
                        {"uid": user_id}
                    ).fetchone()
            else:
                exists = conn.execute(
                    text("SELECT 1 FROM credits WHERE user_id = :uid"),
                    {"uid": user_id}
                ).fetchone()
    except Exception as e:
        logger.error(f"delete_account: existence check failed for {user_id[:24]}…: {e}")
        raise HTTPException(status_code=500, detail="Database error. Please try again.")

    if not exists:
        # Treat unknown accounts as already deleted — avoids user enumeration
        logger.warning(f"⚠️ delete_account: no account found for {user_id[:24]}… — treating as already deleted")
        return {"status": "deleted", "stripe_cancelled": False}

    # ── Cancel active Stripe subscriptions ────────────────────────────────
    stripe_cancelled = False
    stripe_key = os.environ.get("STRIPE_SECRET_KEY")
    if stripe_key:
        stripe.api_key = stripe_key
        try:
            with engine.connect() as conn:
                if account_id:
                    sc_row = conn.execute(
                        text("SELECT customer_id FROM stripe_customers WHERE account_id = :aid OR user_id = :uid LIMIT 1"),
                        {"aid": account_id, "uid": user_id}
                    ).fetchone()
                else:
                    sc_row = conn.execute(
                        text("SELECT customer_id FROM stripe_customers WHERE user_id = :uid"),
                        {"uid": user_id}
                    ).fetchone()
            if sc_row:
                customer_id = sc_row[0]
                for sub_status in ("active", "past_due"):
                    subs = stripe.Subscription.list(customer=customer_id, status=sub_status, limit=10)
                    for sub in subs.auto_paging_iter():
                        stripe.Subscription.cancel(sub.id)
                        logger.info(
                            f"🚫 Stripe subscription {sub.id} cancelled "
                            f"(account deletion | user={user_id[:24]}…)"
                        )
                        stripe_cancelled = True
        except stripe.StripeError as e:
            logger.error(f"❌ delete_account: Stripe cancellation error for {user_id[:24]}…: {e}")
            # Non-fatal — DB deletion proceeds; webhook will eventually sync
        except Exception as e:
            logger.error(f"❌ delete_account: unexpected Stripe step error for {user_id[:24]}…: {e}")
    else:
        logger.warning("⚠️ delete_account: STRIPE_SECRET_KEY not set — skipping subscription cancellation")

    # ── Wipe database records ──────────────────────────────────────────────
    try:
        with engine.connect() as conn:
            if account_id:
                n_sessions = conn.execute(
                    text("DELETE FROM session_history WHERE account_id = :aid OR user_id = :uid"),
                    {"aid": account_id, "uid": user_id}
                ).rowcount
                conn.execute(text("DELETE FROM preferences WHERE account_id = :aid OR user_id = :uid"),     {"aid": account_id, "uid": user_id})
                conn.execute(text("DELETE FROM credits WHERE account_id = :aid OR user_id = :uid"),          {"aid": account_id, "uid": user_id})
                conn.execute(text("DELETE FROM stripe_customers WHERE account_id = :aid OR user_id = :uid"), {"aid": account_id, "uid": user_id})
                conn.execute(text("DELETE FROM auth_sessions WHERE account_id = :aid"),                      {"aid": account_id})
                conn.execute(text("DELETE FROM magic_link_tokens WHERE account_id = :aid"),                  {"aid": account_id})
                conn.execute(text("DELETE FROM accounts WHERE id = :aid"),                                   {"aid": account_id})
                if device_id:
                    conn.execute(
                        text("DELETE FROM sessions WHERE device_id = :did AND user_email = :email"),
                        {"did": device_id, "email": check_email}
                    )
            else:
                n_sessions = conn.execute(
                    text("DELETE FROM session_history WHERE user_id = :uid"),
                    {"uid": user_id}
                ).rowcount
                conn.execute(text("DELETE FROM preferences WHERE user_id = :uid"),     {"uid": user_id})
                conn.execute(text("DELETE FROM credits WHERE user_id = :uid"),          {"uid": user_id})
                conn.execute(text("DELETE FROM stripe_customers WHERE user_id = :uid"), {"uid": user_id})
                conn.execute(
                    text("DELETE FROM sessions WHERE device_id = :did AND user_email = :email"),
                    {"did": device_id, "email": user_email}
                )
            conn.commit()
        logger.info(
            f"🗑️ Account deleted | user={user_id[:24]}… | "
            f"sessions_removed={n_sessions} | stripe_cancelled={stripe_cancelled}"
        )
    except Exception as e:
        logger.error(f"❌ delete_account: DB deletion failed for {user_id[:24]}…: {e}")
        raise HTTPException(
            status_code=500,
            detail="Deletion failed. Please contact support@cerebroecho.com if this persists."
        )

    # ── Clear in-memory rate-limiter state ────────────────────────────────
    usage_tracker.pop(user_id, None)
    if device_id:
        usage_tracker.pop(f"{device_id}_anonymous", None)

    return {"status": "deleted", "stripe_cancelled": stripe_cancelled}


@app.post("/create-checkout")
@limiter.limit(RATE_CHECKOUT)
async def create_checkout(request: Request, data: dict):
    plan       = data.get("plan")
    device_id  = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")

    if plan not in STRIPE_PRICE_IDS:
        raise HTTPException(status_code=400, detail=f"Invalid plan. Must be one of: {', '.join(STRIPE_PRICE_IDS.keys())}")

    price_id = STRIPE_PRICE_IDS[plan]
    if not price_id:
        raise HTTPException(status_code=503, detail=f"Stripe price ID not configured for plan '{plan}'. Set STRIPE_PRICE_{plan.upper()} in Railway env vars.")

    stripe_key = os.environ.get("STRIPE_SECRET_KEY")
    if not stripe_key:
        raise HTTPException(status_code=503, detail="Stripe not configured")

    stripe.api_key = stripe_key

    # Founding 50: check available seats before allowing purchase
    if plan == "founding_50" and engine is not None:
        try:
            with engine.connect() as conn:
                row = conn.execute(text("SELECT COUNT(*) FROM founding_members")).fetchone()
                if row and row[0] >= 50:
                    raise HTTPException(status_code=409, detail={
                        "code": "FOUNDING_SEATS_FULL",
                        "message": "All 50 founding seats have been claimed.",
                    })
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"founding seat check error: {e}")

    mode = PLAN_MODES[plan]
    raw_user_id, account_id, _ = _resolve_user(request, device_id, user_email)
    # Use the canonical credits row key so the webhook hits the right row
    user_id = _get_credits_user_id(raw_user_id, account_id)

    try:
        metadata = {"plan": plan, "user_id": user_id}
        if account_id:
            metadata["account_id"] = account_id
        session_kwargs: dict = {
            "mode":         mode,
            "line_items":   [{"price": price_id, "quantity": 1}],
            "metadata":     metadata,
            "success_url":  "https://cerebroecho.com/app?payment=success",
            "cancel_url":   "https://cerebroecho.com/app?payment=cancelled",
        }
        if user_email and user_email != "anonymous" and "@" in user_email:
            session_kwargs["customer_email"] = user_email
        session = stripe.checkout.Session.create(**session_kwargs)
        return {"url": session.url}
    except stripe.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(status_code=500, detail="Could not start checkout. Please try again.")


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload        = await request.body()
    sig_header     = request.headers.get("stripe-signature")
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")

    if not webhook_secret:
        raise HTTPException(status_code=503, detail="Webhook secret not configured")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except stripe.errors.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Parse raw payload as plain dict so all .get() calls work regardless of
    # stripe-sdk version (v10+ returns typed objects, not dicts).
    raw        = json.loads(payload)
    event_id   = raw["id"]
    event_type = raw["type"]
    obj        = raw["data"]["object"]

    # ── Idempotency: Stripe retries on non-2xx; skip already-processed events ──
    if _is_duplicate_event(event_id):
        logger.info(f"🔁 Duplicate webhook {event_id} ({event_type}) — already processed, skipping")
        return {"received": True}

    logger.info(f"📨 Stripe webhook: {event_type} | id={event_id}")

    try:
        # ── checkout.session.completed ────────────────────────────────────────
        if event_type == "checkout.session.completed":
            meta        = obj.get("metadata", {})
            user_id     = meta.get("user_id", "")
            plan        = meta.get("plan", "")
            account_id  = meta.get("account_id") or None
            customer_id = obj.get("customer", "")
            payment_id  = obj.get("payment_intent") or obj.get("id", "")

            # Email fallback: anonymous landing-page checkout → resolve by Stripe email
            if not user_id and engine is not None:
                stripe_email = ((obj.get("customer_details") or {}).get("email") or "").lower().strip()
                if stripe_email:
                    account_id = _create_account_or_get(stripe_email)
                    user_id = _get_credits_user_id(f"account_{account_id}", account_id)
                    if not user_id:
                        user_id = f"account_{account_id}"
                    logger.info(f"📧 checkout: resolved by Stripe email → account={account_id[:8]}…")

            if not user_id or plan not in PLAN_CREDITS:
                logger.warning(f"⚠️ checkout.session.completed: missing user_id or unknown plan='{plan}' — skipping")
            elif engine is None:
                logger.error("❌ checkout.session.completed: no DB engine")
            else:
                new_balance = PLAN_CREDITS[plan]
                reset_date  = (datetime.utcnow() + timedelta(days=30)).isoformat() if plan != "free" else None
                try:
                    with engine.connect() as conn:
                        conn.execute(text("""
                            INSERT INTO credits
                                (user_id, account_id, balance, plan_type, total_used, reset_date,
                                 payment_failed_count, subscription_status)
                            VALUES (:uid, :aid, :bal, :plan, 0, :reset, 0, 'active')
                            ON CONFLICT (user_id) DO UPDATE SET
                                account_id           = COALESCE(credits.account_id, EXCLUDED.account_id),
                                balance              = :bal,
                                plan_type            = :plan,
                                total_used           = 0,
                                reset_date           = :reset,
                                payment_failed_count = 0,
                                subscription_status  = 'active'
                        """), {"uid": user_id, "aid": account_id, "bal": new_balance, "plan": plan, "reset": reset_date})

                        if customer_id:
                            conn.execute(text("""
                                INSERT INTO stripe_customers (customer_id, user_id, account_id, created_at)
                                VALUES (:cid, :uid, :aid, :now)
                                ON CONFLICT (customer_id) DO UPDATE SET
                                    account_id = COALESCE(stripe_customers.account_id, EXCLUDED.account_id)
                            """), {"cid": customer_id, "uid": user_id, "aid": account_id, "now": datetime.utcnow().isoformat()})

                        if plan == "founding_50":
                            conn.execute(text("""
                                INSERT INTO founding_members (user_id, account_id, purchase_date, stripe_payment_id)
                                VALUES (:uid, :aid, :date, :pid)
                                ON CONFLICT (user_id) DO NOTHING
                            """), {"uid": user_id, "aid": account_id, "date": datetime.utcnow().isoformat(), "pid": payment_id})

                        conn.commit()
                    logger.info(f"✅ checkout.session.completed: user={user_id} account={account_id} plan={plan} balance={new_balance}")
                    to_email = user_id.split("_", 1)[1] if "_" in user_id else ""
                    email_service.send_upgrade_email(to_email, plan, new_balance)
                    _track_usage(user_id, to_email, "resend_email", 1, 0.0001, json.dumps({"type": "upgrade", "plan": plan}))
                except Exception as e:
                    logger.error(f"❌ checkout.session.completed DB error: {e}")

        # ── invoice.paid (monthly renewal) ────────────────────────────────────
        # Stripe fires this for every successful invoice, including the initial one.
        # We only reset credits on subscription_cycle to avoid racing checkout.session.completed.
        elif event_type == "invoice.paid":
            customer_id    = obj.get("customer", "")
            billing_reason = obj.get("billing_reason", "")
            amount_paid    = obj.get("amount_paid", 0)

            if billing_reason not in ("subscription_cycle", "subscription_update"):
                logger.info(f"📨 invoice.paid: billing_reason={billing_reason!r} — not a renewal, skipping")
            else:
                uid = _lookup_user_id(customer_id)
                if not uid:
                    logger.warning(f"⚠️ invoice.paid: no user mapping for customer {customer_id}")
                elif engine is not None:
                    creds = _get_credit_balance(uid)
                    plan  = creds["plan_type"]
                    if plan in PLAN_CREDITS and plan != "free":
                        new_balance = PLAN_CREDITS[plan]
                        new_reset   = (datetime.utcnow() + timedelta(days=30)).isoformat()
                        try:
                            with engine.connect() as conn:
                                conn.execute(text("""
                                    UPDATE credits SET
                                        balance              = :bal,
                                        reset_date           = :reset,
                                        total_used           = 0,
                                        payment_failed_count = 0,
                                        subscription_status  = 'active'
                                    WHERE user_id = :uid
                                """), {"bal": new_balance, "reset": new_reset, "uid": uid})
                                conn.commit()
                            logger.info(
                                f"✅ invoice.paid (renewal): user={uid} plan={plan} "
                                f"balance={new_balance} billing_reason={billing_reason} "
                                f"amount_paid={amount_paid}"
                            )
                        except Exception as e:
                            logger.error(f"❌ invoice.paid DB error: {e}")

        # ── invoice.payment_failed ─────────────────────────────────────────────
        elif event_type == "invoice.payment_failed":
            customer_id   = obj.get("customer", "")
            attempt_count = obj.get("attempt_count", 1)
            next_attempt  = obj.get("next_payment_attempt")
            amount_due    = obj.get("amount_due", 0)

            uid = _lookup_user_id(customer_id)
            if not uid:
                logger.warning(f"⚠️ invoice.payment_failed: no user mapping for customer {customer_id}")
            elif engine is not None:
                max_failures = int(os.environ.get("MAX_PAYMENT_FAILURES", "3"))
                try:
                    with engine.connect() as conn:
                        # Increment failure count and set status
                        conn.execute(text("""
                            UPDATE credits SET
                                payment_failed_count = payment_failed_count + 1,
                                subscription_status  = CASE
                                    WHEN payment_failed_count + 1 >= :max THEN 'grace_expired'
                                    ELSE 'past_due'
                                END
                            WHERE user_id = :uid
                        """), {"uid": uid, "max": max_failures})

                        # Check if grace period exceeded → downgrade
                        row = conn.execute(
                            text("SELECT payment_failed_count, plan_type FROM credits WHERE user_id=:uid"),
                            {"uid": uid}
                        ).fetchone()
                        fail_count  = row[0] if row else 0
                        failed_plan = row[1] if row else "unknown"

                        if fail_count >= max_failures:
                            conn.execute(text("""
                                UPDATE credits SET
                                    plan_type = 'free', balance = 0, reset_date = NULL
                                WHERE user_id = :uid
                            """), {"uid": uid})
                            logger.warning(
                                f"🚫 invoice.payment_failed: grace period exceeded — "
                                f"downgraded {uid} (was {failed_plan}) after {fail_count} failures"
                            )
                            to_email_dng = uid.split("_", 1)[1] if "_" in uid else ""
                            email_service.send_downgrade_email(to_email_dng, failed_plan)
                        conn.commit()

                    next_str = (
                        datetime.utcfromtimestamp(next_attempt).isoformat()
                        if next_attempt else "no further retries"
                    )
                    logger.warning(
                        f"⚠️ invoice.payment_failed: customer={customer_id} user={uid} "
                        f"attempt={attempt_count} failures_total={fail_count} "
                        f"next_retry={next_str} amount_due={amount_due}"
                    )
                    to_email = uid.split("_", 1)[1] if "_" in uid else ""
                    email_service.send_payment_failed_email(to_email, attempt_count, next_str, amount_due)
                    _track_usage(uid, to_email, "resend_email", 1, 0.0001, json.dumps({"type": "payment_failed"}))
                except Exception as e:
                    logger.error(f"❌ invoice.payment_failed DB error: {e}")

        # ── customer.subscription.updated ─────────────────────────────────────
        # Fires on: plan change via portal, cancel-at-period-end toggle,
        # status transitions (active → past_due → unpaid → canceled), trial end.
        elif event_type == "customer.subscription.updated":
            customer_id          = obj.get("customer", "")
            status               = obj.get("status", "")
            cancel_at_period_end = obj.get("cancel_at_period_end", False)
            items                = obj.get("items", {}).get("data", [])
            new_price_id         = items[0]["price"]["id"] if items else None

            uid = _lookup_user_id(customer_id)
            if not uid:
                logger.warning(f"⚠️ subscription.updated: no user mapping for customer {customer_id}")
            elif engine is not None:
                # Reverse-map price_id → plan name for portal-initiated plan changes
                price_to_plan = {v: k for k, v in STRIPE_PRICE_IDS.items() if v}
                new_plan = price_to_plan.get(new_price_id) if new_price_id else None

                if status in ("active", "trialing"):
                    try:
                        with engine.connect() as conn:
                            row = conn.execute(
                                text("SELECT plan_type FROM credits WHERE user_id=:uid"),
                                {"uid": uid}
                            ).fetchone()
                            current_plan = row[0] if row else None

                            if new_plan and new_plan in PLAN_CREDITS and current_plan != new_plan:
                                # Plan changed via Customer Portal — update plan_type immediately.
                                # Balance is NOT reset here; invoice.paid handles the next cycle.
                                conn.execute(text("""
                                    UPDATE credits SET
                                        plan_type            = :plan,
                                        payment_failed_count = 0,
                                        subscription_status  = 'active'
                                    WHERE user_id = :uid
                                """), {"plan": new_plan, "uid": uid})
                                logger.info(
                                    f"✅ subscription.updated: plan change "
                                    f"{current_plan} → {new_plan} for {uid}"
                                )
                            else:
                                # Same plan or unknown price — just clear any failure state
                                conn.execute(text("""
                                    UPDATE credits SET
                                        payment_failed_count = 0,
                                        subscription_status  = 'active'
                                    WHERE user_id = :uid
                                """), {"uid": uid})
                                if cancel_at_period_end:
                                    logger.info(f"📨 subscription.updated: {uid} will cancel at period end — no immediate downgrade")
                                else:
                                    logger.info(f"📨 subscription.updated: {uid} status=active, no plan change")
                            conn.commit()
                    except Exception as e:
                        logger.error(f"❌ subscription.updated (active) DB error: {e}")

                elif status == "past_due":
                    try:
                        with engine.connect() as conn:
                            conn.execute(text("""
                                UPDATE credits SET subscription_status = 'past_due'
                                WHERE user_id = :uid
                            """), {"uid": uid})
                            conn.commit()
                        logger.warning(f"⚠️ subscription.updated: {uid} is past_due")
                    except Exception as e:
                        logger.error(f"❌ subscription.updated (past_due) DB error: {e}")

                elif status in ("canceled", "incomplete_expired", "unpaid"):
                    try:
                        with engine.connect() as conn:
                            conn.execute(text("""
                                UPDATE credits SET
                                    plan_type           = 'free',
                                    balance             = 0,
                                    reset_date          = NULL,
                                    subscription_status = 'canceled'
                                WHERE user_id = :uid
                            """), {"uid": uid})
                            conn.commit()
                        logger.info(f"🚫 subscription.updated: status={status} — downgraded {uid} to free")
                    except Exception as e:
                        logger.error(f"❌ subscription.updated (canceled/unpaid) DB error: {e}")

                else:
                    logger.info(f"📨 subscription.updated: unhandled status={status!r} for {uid}")

        # ── customer.subscription.deleted ─────────────────────────────────────
        # Fires when a subscription fully ends (cancel_at_period_end reached,
        # or immediate cancellation). Always means access should end.
        elif event_type == "customer.subscription.deleted":
            customer_id = obj.get("customer", "")
            ended_at    = obj.get("ended_at")

            uid = _lookup_user_id(customer_id)
            if not uid:
                logger.warning(f"⚠️ subscription.deleted: no user mapping for customer {customer_id}")
            elif engine is not None:
                try:
                    with engine.connect() as conn:
                        conn.execute(text("""
                            UPDATE credits SET
                                plan_type            = 'free',
                                balance              = 0,
                                reset_date           = NULL,
                                payment_failed_count = 0,
                                subscription_status  = 'canceled'
                            WHERE user_id = :uid
                        """), {"uid": uid})
                        conn.commit()
                    ended_str = (
                        datetime.utcfromtimestamp(ended_at).isoformat()
                        if ended_at else "unknown"
                    )
                    logger.info(f"🚫 subscription.deleted: downgraded {uid} to free | ended_at={ended_str}")
                    to_email = uid.split("_", 1)[1] if "_" in uid else ""
                    email_service.send_cancellation_email(to_email, ended_str)
                    _track_usage(uid, to_email, "resend_email", 1, 0.0001, json.dumps({"type": "cancellation"}))
                except Exception as e:
                    logger.error(f"❌ subscription.deleted DB error: {e}")

        else:
            logger.info(f"📨 Unhandled Stripe event type: {event_type!r}")

    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"❌ Unhandled webhook exception ({event_type} | {event_id}): {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Still mark processed to prevent infinite retries on bugs.
        # Remove this line if you want Stripe to retry on unexpected exceptions.

    _mark_event_processed(event_id, event_type)
    return {"received": True}


# ========================
# AUTH HELPERS
# ========================

def _create_account_or_get(email: str) -> str:
    """Get or create an account by email. Returns account_id (TEXT UUID)."""
    clean = email.lower().strip()
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT id FROM accounts WHERE email = :email"),
                {"email": clean}
            ).fetchone()
            if row:
                return row[0]
            account_id = str(_uuid.uuid4())
            conn.execute(text("""
                INSERT INTO accounts (id, email, created_at)
                VALUES (:id, :email, :now)
                ON CONFLICT (email) DO NOTHING
            """), {"id": account_id, "email": clean, "now": datetime.utcnow().isoformat()})
            conn.commit()
        # Re-fetch on a fresh connection — handles race where two requests created simultaneously
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT id FROM accounts WHERE email = :email"),
                {"email": clean}
            ).fetchone()
            result = row[0] if row else account_id
        logger.info(f"👤 Account ready: {result[:8]}… ({clean})")
        return result
    except Exception as e:
        logger.error(f"_create_account_or_get error: {e}")
        raise


def _ensure_free_credits(account_id: str) -> None:
    """Provision 20 free-tier credits for a newly verified account if none exist,
    and set a monthly reset_date so the cron refills them every 30 days.

    Idempotent — safe to call on every sign-in; only acts when the account has
    no credits row at all (brand-new registrations with no prior anonymous usage).
    """
    if engine is None:
        return
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM credits WHERE account_id = :aid LIMIT 1"),
                {"aid": account_id}
            ).fetchone()
            if row:
                return
            # Stable synthetic user_id keyed to the account (never collides with device keys)
            uid = f"account_{account_id}"
            reset_date = (datetime.utcnow() + timedelta(days=30)).isoformat()
            conn.execute(text("""
                INSERT INTO credits (user_id, account_id, balance, plan_type, total_used, reset_date)
                VALUES (:uid, :aid, 20, 'free', 0, :reset)
                ON CONFLICT (user_id) DO NOTHING
            """), {"uid": uid, "aid": account_id, "reset": reset_date})
            conn.commit()
        logger.info(f"🎁 Free credits provisioned for new account {account_id[:8]}…")
    except Exception as e:
        logger.error(f"_ensure_free_credits error: {e}")
        sentry_sdk.capture_exception(e)


def _create_magic_link(account_id: str) -> str:
    """Invalidate unused tokens for this account and issue a fresh one."""
    token      = _secrets.token_urlsafe(32)
    expires_at = (datetime.utcnow() + timedelta(minutes=MAGIC_LINK_EXPIRY_MINUTES)).isoformat()
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                DELETE FROM magic_link_tokens WHERE account_id = :aid AND used_at IS NULL
            """), {"aid": account_id})
            conn.execute(text("""
                INSERT INTO magic_link_tokens (token, account_id, expires_at)
                VALUES (:token, :aid, :expires)
            """), {"token": token, "aid": account_id, "expires": expires_at})
            conn.commit()
    except Exception as e:
        logger.error(f"_create_magic_link error: {e}")
        raise
    return token


def _verify_magic_link(token: str):
    """Consume a magic link token. Returns account_id or None."""
    now = datetime.utcnow().isoformat()
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT account_id, expires_at, used_at
                FROM magic_link_tokens WHERE token = :token
            """), {"token": token}).fetchone()
            if not row:
                return None
            account_id, expires_at, used_at = row[0], row[1], row[2]
            if used_at or expires_at < now:
                return None
            conn.execute(text("""
                UPDATE magic_link_tokens SET used_at = :now WHERE token = :token
            """), {"now": now, "token": token})
            conn.commit()
        return account_id
    except Exception as e:
        logger.error(f"_verify_magic_link error: {e}")
        return None


def _create_otp(account_id: str) -> str:
    """Invalidate unused OTPs for this account and issue a fresh 6-digit code."""
    code       = f"{_random.SystemRandom().randint(0, 999999):06d}"
    otp_id     = str(_uuid.uuid4())
    expires_at = (datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)).isoformat()
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                DELETE FROM email_otps WHERE account_id = :aid AND used_at IS NULL
            """), {"aid": account_id})
            conn.execute(text("""
                INSERT INTO email_otps (id, code, account_id, expires_at)
                VALUES (:id, :code, :aid, :expires)
            """), {"id": otp_id, "code": code, "aid": account_id, "expires": expires_at})
            conn.commit()
    except Exception as e:
        logger.error(f"_create_otp error: {e}")
        raise
    return code


def _verify_otp_code(code: str, account_id: str):
    """Consume an OTP. Returns account_id if valid, None otherwise."""
    now = datetime.utcnow().isoformat()
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT id, expires_at, used_at
                FROM email_otps WHERE code = :code AND account_id = :aid
                ORDER BY expires_at DESC LIMIT 1
            """), {"code": code, "aid": account_id}).fetchone()
            if not row:
                return None
            otp_id, expires_at, used_at = row[0], row[1], row[2]
            if used_at or expires_at < now:
                return None
            conn.execute(text("""
                UPDATE email_otps SET used_at = :now WHERE id = :id
            """), {"now": now, "id": otp_id})
            conn.commit()
        return account_id
    except Exception as e:
        logger.error(f"_verify_otp_code error: {e}")
        return None


def _create_auth_session(account_id: str, device_id: str = "") -> str:
    """Create a rolling 30-day session. Returns the session token."""
    token      = _secrets.token_urlsafe(32)
    now        = datetime.utcnow()
    expires_at = (now + timedelta(days=SESSION_EXPIRY_DAYS)).isoformat()
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO auth_sessions
                    (token, account_id, device_id, created_at, expires_at, last_used)
                VALUES (:token, :aid, :did, :now, :expires, :now)
            """), {
                "token": token, "aid": account_id, "did": device_id,
                "now": now.isoformat(), "expires": expires_at,
            })
            conn.execute(text("""
                UPDATE accounts SET last_login = :now WHERE id = :aid
            """), {"now": now.isoformat(), "aid": account_id})
            conn.commit()
    except Exception as e:
        logger.error(f"_create_auth_session error: {e}")
        raise
    return token


def _get_account_from_session(session_token: str):
    """Validate session token. Returns (account_id, email) or (None, None)."""
    if engine is None or not session_token:
        return None, None
    now = datetime.utcnow().isoformat()
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT s.account_id, a.email
                FROM auth_sessions s
                JOIN accounts a ON s.account_id = a.id
                WHERE s.token = :token AND s.expires_at > :now
            """), {"token": session_token, "now": now}).fetchone()
            if not row:
                return None, None
            conn.execute(text("""
                UPDATE auth_sessions SET last_used = :now WHERE token = :token
            """), {"now": now, "token": session_token})
            conn.commit()
        return row[0], row[1]
    except Exception as e:
        logger.error(f"_get_account_from_session error: {e}")
        return None, None


def _backfill_account_id(account_id: str, email: str, device_id: str = "") -> None:
    """Set account_id on existing device-keyed rows for a newly verified account.

    Matches rows by email suffix (covers all past device IDs for this email) and
    optionally by the current device's anonymous key.
    Uses RIGHT() instead of LIKE to avoid _ wildcard issues in email addresses.
    Also renames the current device's anonymous user_id to the email-keyed form
    (credits + preferences only) so composite-key lookups stay consistent.
    """
    if engine is None:
        return
    email_suffix = f"_{email}"
    suffix_len   = len(email_suffix)
    params: dict = {"aid": account_id, "slen": suffix_len, "suffix": email_suffix}
    anon_clause  = ""
    if device_id:
        params["anon_key"] = f"{device_id}_anonymous"
        anon_clause = "OR user_id = :anon_key"
    try:
        with engine.connect() as conn:
            for tbl in ("credits", "session_history", "preferences", "founding_members"):
                n = conn.execute(text(f"""
                    UPDATE {tbl} SET account_id = :aid
                    WHERE account_id IS NULL
                    AND (RIGHT(user_id, :slen) = :suffix {anon_clause})
                """), params).rowcount
                if n:
                    logger.info(f"↗ backfill {tbl}: {n} row(s) → account {account_id[:8]}…")
            n = conn.execute(text(f"""
                UPDATE stripe_customers SET account_id = :aid
                WHERE account_id IS NULL
                AND (RIGHT(user_id, :slen) = :suffix {anon_clause})
            """), params).rowcount
            if n:
                logger.info(f"↗ backfill stripe_customers: {n} row(s) → account {account_id[:8]}…")
            conn.commit()
        # Rename anonymous user_id → email-keyed user_id for this device (best-effort)
        # Only for credits and preferences since those are the PK-keyed lookup tables.
        if device_id:
            anon_key  = f"{device_id}_anonymous"
            email_key = f"{device_id}_{email}"
            try:
                with engine.connect() as conn:
                    for tbl in ("credits", "preferences"):
                        existing = conn.execute(
                            text(f"SELECT 1 FROM {tbl} WHERE user_id = :ek LIMIT 1"),
                            {"ek": email_key}
                        ).fetchone()
                        if not existing:
                            n = conn.execute(text(f"""
                                UPDATE {tbl} SET user_id = :ek WHERE user_id = :ak
                            """), {"ek": email_key, "ak": anon_key}).rowcount
                            if n:
                                logger.info(f"↗ rename {tbl} user_id: anon→email for device {device_id[:8]}…")
                    conn.commit()
            except Exception as rename_err:
                logger.warning(f"_backfill_account_id: user_id rename skipped: {rename_err}")
    except Exception as e:
        logger.error(f"_backfill_account_id error: {e}")
        sentry_sdk.capture_exception(e)


def _resolve_user(request: Request, device_id: str = "", user_email: str = "anonymous"):
    """Dual-mode identity resolution.

    Checks Authorization: Bearer header first. If the session is valid, returns
    the account email (verified) paired with the provided device_id, along with
    the account_id UUID. Falls back to the composite device+email key when no
    valid token is present.

    Returns: (user_id: str, account_id: str | None, email: str)
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            account_id, email = _get_account_from_session(token)
            if account_id and email:
                uid = f"{device_id}_{email}" if device_id else email
                return uid, account_id, email
    return f"{device_id}_{user_email}", None, user_email


# ════════════════════════════════════════════════════════════════════════
# ADMIN DASHBOARD — owner-only metrics (the CEO dashboard)
# Read-only. Gated to ADMIN_EMAILS via the caller's magic-link session.
# Non-admins (and the unauthenticated) get 404 so the endpoint stays hidden.
# Inert if ADMIN_EMAILS is unset → nobody is admin → always 404.
# ════════════════════════════════════════════════════════════════════════
ADMIN_EMAILS = {e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()}
_admin_cache = {"data": None, "ts": 0.0}
_ADMIN_CACHE_TTL_S = 30


def _require_admin(request: Request) -> str:
    """Return the admin email if the Bearer session belongs to an ADMIN_EMAILS account, else 404."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            account_id, email = _get_account_from_session(token)
            if account_id and email and email.strip().lower() in ADMIN_EMAILS:
                return email
    raise HTTPException(status_code=404, detail="Not found")


# Accounts excluded from "real customer" metrics so the dashboard reflects true traction,
# not test/QA/owner logins. Test domains are ALWAYS excluded; INTERNAL_EMAILS (owner's own
# logins) is overridable via the INTERNAL_EMAILS env var (comma-separated).
_METRICS_TEST_DOMAINS = ("example.com", "cerebroecho.test")
INTERNAL_EMAILS = {e.strip().lower() for e in os.environ.get(
    "INTERNAL_EMAILS",
    "mikeaduderstadt@gmail.com,admin@cerebroecho.com,cerebroecho@gmail.com,mikemacmany@msn.com",
).split(",") if e.strip()}


def _real_accounts_filter(alias: str = "a"):
    """SQL WHERE fragment (+bind params) keeping only real external accounts —
    excludes test-domain emails and owner/internal logins. Returns (sql, params)."""
    conds = [f"LOWER({alias}.email) NOT LIKE '%@{d}'" for d in _METRICS_TEST_DOMAINS]
    params: dict = {}
    for i, em in enumerate(sorted(INTERNAL_EMAILS)):
        conds.append(f"LOWER({alias}.email) <> :ie{i}")
        params[f"ie{i}"] = em
    return ("(" + " AND ".join(conds) + ")") if conds else "TRUE", params


def _admin_db_metrics() -> dict:
    """Signups, paying members, and provider-cost aggregates straight from Postgres.
    Customer counts exclude test/QA/owner accounts (see _real_accounts_filter)."""
    out = {
        "signups": {"total": 0, "today": 0, "d7": 0, "d30": 0, "series": []},
        "members": {"total_accounts": 0, "paying": 0, "by_plan": {}, "active_subs": 0, "payg_enabled": 0, "excluded_test": 0},
        "cost":    {"today": 0.0, "d7": 0.0, "d30": 0.0, "all": 0.0, "by_provider": {}, "series": []},
        "credits": {"issued_balance": 0, "total_used": 0},
    }
    if engine is None:
        return out
    today = datetime.utcnow().strftime("%Y-%m-%d")
    d7    = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    d30   = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    try:
        with engine.connect() as conn:
            # ── Signups — REAL external accounts only (exclude test/QA/owner) ──
            # accounts.created_at is ISO text → lexical compare is valid.
            rf, rp = _real_accounts_filter("a")
            out["signups"]["total"] = conn.execute(text(f"SELECT COUNT(*) FROM accounts a WHERE {rf}"), rp).scalar() or 0
            out["signups"]["today"] = conn.execute(text(f"SELECT COUNT(*) FROM accounts a WHERE {rf} AND a.created_at >= :d"), {**rp, "d": today}).scalar() or 0
            out["signups"]["d7"]    = conn.execute(text(f"SELECT COUNT(*) FROM accounts a WHERE {rf} AND a.created_at >= :d"), {**rp, "d": d7}).scalar() or 0
            out["signups"]["d30"]   = conn.execute(text(f"SELECT COUNT(*) FROM accounts a WHERE {rf} AND a.created_at >= :d"), {**rp, "d": d30}).scalar() or 0
            rows = conn.execute(text(f"""
                SELECT SUBSTRING(a.created_at, 1, 10) AS day, COUNT(*)
                FROM accounts a WHERE {rf} AND a.created_at >= :d
                GROUP BY day ORDER BY day
            """), {**rp, "d": d30}).fetchall()
            out["signups"]["series"] = [{"day": r[0], "count": int(r[1])} for r in rows]

            # ── Members / plans — REAL paying customers only ──
            # Join billing→accounts (handles the legacy 'account_<id>' key form). Rows that don't
            # map to a real non-test account (orphan/seed rows, anonymous trials) are dropped.
            _acct_join = "JOIN accounts a ON (c.user_id = a.id OR c.user_id = 'account_' || a.id)"
            raw_total = conn.execute(text("SELECT COUNT(*) FROM accounts")).scalar() or 0
            out["members"]["total_accounts"] = out["signups"]["total"]
            out["members"]["excluded_test"]  = raw_total - out["signups"]["total"]
            out["members"]["paying"]      = conn.execute(text(f"SELECT COUNT(*) FROM credits c {_acct_join} WHERE c.plan_type <> 'free' AND {rf}"), rp).scalar() or 0
            out["members"]["active_subs"] = conn.execute(text(f"SELECT COUNT(*) FROM credits c {_acct_join} WHERE c.plan_type <> 'free' AND c.subscription_status = 'active' AND {rf}"), rp).scalar() or 0
            try:
                out["members"]["payg_enabled"] = conn.execute(text(f"SELECT COUNT(*) FROM credits c {_acct_join} WHERE c.payg_enabled = true AND {rf}"), rp).scalar() or 0
            except Exception:
                pass  # payg_enabled column may not exist on older DBs
            plan_rows = conn.execute(text(f"SELECT c.plan_type, COUNT(*) FROM credits c {_acct_join} WHERE {rf} GROUP BY c.plan_type ORDER BY 2 DESC"), rp).fetchall()
            out["members"]["by_plan"] = {(r[0] or "unknown"): int(r[1]) for r in plan_rows}
            crow = conn.execute(text("SELECT COALESCE(SUM(balance),0), COALESCE(SUM(total_used),0) FROM credits")).fetchone()
            if crow:
                out["credits"]["issued_balance"] = int(crow[0])
                out["credits"]["total_used"]     = int(crow[1])

            # ── Provider cost (usage_events.created_at is TIMESTAMPTZ) ──
            out["cost"]["today"] = float(conn.execute(text("SELECT COALESCE(SUM(raw_cost_usd),0) FROM usage_events WHERE created_at >= DATE_TRUNC('day', NOW())")).scalar() or 0)
            out["cost"]["d7"]    = float(conn.execute(text("SELECT COALESCE(SUM(raw_cost_usd),0) FROM usage_events WHERE created_at >= NOW() - INTERVAL '7 days'")).scalar() or 0)
            out["cost"]["d30"]   = float(conn.execute(text("SELECT COALESCE(SUM(raw_cost_usd),0) FROM usage_events WHERE created_at >= NOW() - INTERVAL '30 days'")).scalar() or 0)
            out["cost"]["all"]   = float(conn.execute(text("SELECT COALESCE(SUM(raw_cost_usd),0) FROM usage_events")).scalar() or 0)
            prov = conn.execute(text("SELECT provider, COALESCE(SUM(raw_cost_usd),0) FROM usage_events WHERE created_at >= NOW() - INTERVAL '30 days' GROUP BY provider ORDER BY 2 DESC")).fetchall()
            out["cost"]["by_provider"] = {(r[0] or "unknown"): round(float(r[1]), 4) for r in prov}
            crows = conn.execute(text("""
                SELECT TO_CHAR(DATE_TRUNC('day', created_at), 'YYYY-MM-DD') AS day, COALESCE(SUM(raw_cost_usd),0)
                FROM usage_events WHERE created_at >= NOW() - INTERVAL '30 days'
                GROUP BY day ORDER BY day
            """)).fetchall()
            out["cost"]["series"] = [{"day": r[0], "cost": round(float(r[1]), 4)} for r in crows]
    except Exception as e:
        logger.error(f"admin db metrics error: {e}")
    return out


def _admin_stripe_metrics() -> dict:
    """Earnings (MRR), recent gross, and refunds from Stripe. Best-effort; never raises."""
    out = {"configured": False, "mrr": 0.0, "active_subscriptions": 0, "gross_30d": 0.0,
           "refunds_30d": 0.0, "refund_count_30d": 0, "currency": "usd", "recent": []}
    stripe_key = os.environ.get("STRIPE_SECRET_KEY")
    if not stripe_key:
        return out
    out["configured"] = True
    try:
        stripe.api_key = stripe_key
        # MRR from active subscriptions, normalized to a monthly figure
        mrr = 0.0
        sub_count = 0
        for s in stripe.Subscription.list(status="active", limit=100).auto_paging_iter():
            sub_count += 1
            for it in s["items"]["data"]:
                price    = it.get("price") or {}
                amt      = (price.get("unit_amount") or 0) / 100.0
                qty      = it.get("quantity", 1) or 1
                interval = (price.get("recurring") or {}).get("interval", "month")
                factor   = {"month": 1, "year": 1 / 12, "week": 52 / 12, "day": 365 / 12}.get(interval, 1)
                mrr += amt * qty * factor
        out["mrr"] = round(mrr, 2)
        out["active_subscriptions"] = sub_count

        since = int((datetime.utcnow() - timedelta(days=30)).timestamp())
        # Gross succeeded charges (30d) + a few recent ones for the activity feed
        gross = 0.0
        recent = []
        for c in stripe.Charge.list(created={"gte": since}, limit=100).auto_paging_iter():
            if c.get("paid") and c.get("status") == "succeeded":
                amt = (c.get("amount") or 0) / 100.0
                gross += amt
                if len(recent) < 12:
                    recent.append({"type": "charge", "amount": round(amt, 2),
                                   "email": (c.get("billing_details") or {}).get("email"),
                                   "created": c.get("created")})
        out["gross_30d"] = round(gross, 2)
        out["recent"] = recent

        # Refunds (30d)
        ref_total = 0.0
        ref_count = 0
        for r in stripe.Refund.list(created={"gte": since}, limit=100).auto_paging_iter():
            ref_total += (r.get("amount") or 0) / 100.0
            ref_count += 1
        out["refunds_30d"] = round(ref_total, 2)
        out["refund_count_30d"] = ref_count
    except Exception as e:
        logger.error(f"admin stripe metrics error: {e}")
    return out


async def _admin_visitor_metrics() -> dict:
    """Plausible site visitors. Inert unless PLAUSIBLE_API_KEY is set."""
    out = {"configured": False, "realtime": None, "visitors_30d": None, "pageviews_30d": None, "site": None}
    key  = os.environ.get("PLAUSIBLE_API_KEY")
    site = os.environ.get("PLAUSIBLE_SITE_ID", "cerebroecho.com")
    if not key:
        return out
    out["configured"] = True
    out["site"] = site
    try:
        import httpx
        headers = {"Authorization": f"Bearer {key}"}
        base = "https://plausible.io/api/v1/stats"
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as h:
            rt = await h.get(f"{base}/realtime/visitors", params={"site_id": site})
            if rt.status_code == 200:
                out["realtime"] = rt.json()
            agg = await h.get(f"{base}/aggregate", params={"site_id": site, "period": "30d", "metrics": "visitors,pageviews"})
            if agg.status_code == 200:
                res = agg.json().get("results", {})
                out["visitors_30d"]  = (res.get("visitors")  or {}).get("value")
                out["pageviews_30d"] = (res.get("pageviews") or {}).get("value")
    except Exception as e:
        logger.error(f"admin visitor metrics error: {e}")
    return out


@app.get("/admin/metrics")
@limiter.limit(RATE_READS)
async def admin_metrics(request: Request):
    admin_email = _require_admin(request)
    now = time.time()
    if _admin_cache["data"] is not None and (now - _admin_cache["ts"]) < _ADMIN_CACHE_TTL_S:
        cached = dict(_admin_cache["data"])
        cached["cached"] = True
        return cached

    db       = _admin_db_metrics()
    stripe_m = _admin_stripe_metrics()
    visitors = await _admin_visitor_metrics()

    # Profit (30d) = Stripe gross − provider cost. Both best-effort.
    profit_30d = round((stripe_m.get("gross_30d") or 0.0) - (db["cost"].get("d30") or 0.0), 2)

    payload = {
        "ok": True,
        "admin_email": admin_email,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "signups":  db["signups"],
        "members":  db["members"],
        "credits":  db["credits"],
        "cost":     db["cost"],
        "revenue":  stripe_m,
        "visitors": visitors,
        "profit_30d": profit_30d,
        "cached": False,
    }
    _admin_cache["data"] = payload
    _admin_cache["ts"]   = now
    return payload


@app.post("/save_email")
async def save_email(data: dict):
    device_id = data.get("deviceId")
    email = data.get("email")
    old_key = f"{device_id}_anonymous"
    new_key = f"{device_id}_{email}"
    
    if old_key in usage_tracker:
        usage_tracker[new_key] = usage_tracker[old_key]
        del usage_tracker[old_key]
    
    logger.info(f"📧 Email saved: {email}")
    email_service.send_welcome_email(email)
    _track_usage(email, email, "resend_email", 1, 0.0001, json.dumps({"type": "welcome"}))
    return {"status": "success", "message": "Trial extended to 10 questions"}

# ========================
# AUTH ENDPOINTS
# ========================

@app.post("/auth/request-link")
@limiter.limit(RATE_AUTH_REQUEST)
async def auth_request_link(request: Request, data: dict):
    """Send a magic sign-in link. Creates account if new. Always returns 200 to avoid email enumeration."""
    email     = (data.get("email") or "").strip().lower()
    device_id = data.get("deviceId", "")

    if not email or "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="A valid email address is required.")
    if engine is None:
        raise HTTPException(status_code=503, detail={
            "code": "SERVICE_UNAVAILABLE",
            "message": "Sign-in is temporarily unavailable. Please try again in a moment.",
        })

    try:
        account_id = _create_account_or_get(email)
        token      = _create_magic_link(account_id)
        magic_url  = f"{APP_BASE_URL}?magic_token={token}"
        email_service.send_magic_link_email(email, magic_url)
        _track_usage(email, email, "resend_email", 1, 0.0001, json.dumps({"type": "magic_link"}))
        logger.info(f"🔗 Magic link sent to {email}")
    except Exception as e:
        logger.error(f"auth_request_link error: {e}")
        raise HTTPException(status_code=500, detail={
            "code": "SEND_FAILED",
            "message": "Failed to send sign-in link. Please try again.",
        })

    return {"status": "sent"}


@app.post("/auth/verify-link")
@limiter.limit(RATE_AUTH_VERIFY)
async def auth_verify_link(request: Request, data: dict):
    """Consume a magic link token and return a session token."""
    token     = (data.get("token") or "").strip()
    device_id = data.get("deviceId", "")

    if not token:
        raise HTTPException(status_code=400, detail="Token required.")
    if engine is None:
        raise HTTPException(status_code=503, detail={
            "code": "SERVICE_UNAVAILABLE",
            "message": "Sign-in is temporarily unavailable. Please try again in a moment.",
        })

    account_id = _verify_magic_link(token)
    if not account_id:
        raise HTTPException(status_code=401, detail={
            "code": "INVALID_OR_EXPIRED_TOKEN",
            "message": "This link has expired or already been used. Request a new one.",
        })

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT email, last_login FROM accounts WHERE id = :id"),
                {"id": account_id}
            ).fetchone()
        email = row[0] if row else ""
        is_first_login = row is not None and row[1] is None
    except Exception as e:
        logger.error(f"auth_verify_link: account lookup error: {e}")
        email = ""
        is_first_login = False

    session_token = _create_auth_session(account_id, device_id)
    _backfill_account_id(account_id, email, device_id)
    _ensure_free_credits(account_id)

    if is_first_login and not _has_onboarding_email_sent(account_id, "onboarding_welcome"):
        try:
            email_service.send_onboarding_welcome(email)
            _log_onboarding_email(account_id, "onboarding_welcome")
            _track_usage(account_id, email, "resend_email", 1, 0.0001,
                         json.dumps({"type": "onboarding_welcome"}))
            logger.info(f"📧 onboarding_welcome sent to {email}")
        except Exception as e:
            logger.error(f"onboarding_welcome send error: {e}")

    logger.info(f"✅ Auth session created: account={account_id[:8]}… device={device_id[:8] if device_id else 'none'}")
    return {
        "session_token": session_token,
        "account_id":    account_id,
        "email":         email,
    }


@app.post("/auth/request-otp")
@limiter.limit(RATE_AUTH_REQUEST)
async def auth_request_otp(request: Request, data: dict):
    """Send a 6-digit OTP to the user's email. Creates account if new."""
    email     = (data.get("email") or "").strip().lower()
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="A valid email address is required.")
    if engine is None:
        raise HTTPException(status_code=503, detail={
            "code": "SERVICE_UNAVAILABLE",
            "message": "Sign-in is temporarily unavailable. Please try again in a moment.",
        })
    try:
        account_id = _create_account_or_get(email)
        code       = _create_otp(account_id)
        email_service.send_otp_email(email, code)
        _track_usage(email, email, "resend_email", 1, 0.0001, json.dumps({"type": "otp"}))
        logger.info(f"🔢 OTP sent to {email}")
    except Exception as e:
        logger.error(f"auth_request_otp error: {e}")
        raise HTTPException(status_code=500, detail={
            "code": "SEND_FAILED",
            "message": "Failed to send sign-in code. Please try again.",
        })
    return {"status": "sent"}


@app.post("/auth/verify-otp")
@limiter.limit(RATE_AUTH_VERIFY)
async def auth_verify_otp(request: Request, data: dict):
    """Verify a 6-digit OTP and return a session token."""
    email     = (data.get("email") or "").strip().lower()
    code      = (data.get("code") or "").strip()
    device_id = data.get("deviceId", "")

    if not email or not code:
        raise HTTPException(status_code=400, detail="Email and code are required.")
    if engine is None:
        raise HTTPException(status_code=503, detail={
            "code": "SERVICE_UNAVAILABLE",
            "message": "Sign-in is temporarily unavailable. Please try again in a moment.",
        })

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT id, last_login FROM accounts WHERE email = :email"),
                {"email": email}
            ).fetchone()
        account_id = row[0] if row else None
        is_first_login = row is not None and row[1] is None
    except Exception as e:
        logger.error(f"auth_verify_otp: account lookup error: {e}")
        account_id = None
        is_first_login = False

    if not account_id or not _verify_otp_code(code, account_id):
        raise HTTPException(status_code=401, detail={
            "code": "INVALID_OR_EXPIRED_CODE",
            "message": "That code is incorrect or has expired. Request a new one.",
        })

    session_token = _create_auth_session(account_id, device_id)
    _backfill_account_id(account_id, email, device_id)
    _ensure_free_credits(account_id)

    if is_first_login and not _has_onboarding_email_sent(account_id, "onboarding_welcome"):
        try:
            email_service.send_onboarding_welcome(email)
            _log_onboarding_email(account_id, "onboarding_welcome")
            _track_usage(account_id, email, "resend_email", 1, 0.0001,
                         json.dumps({"type": "onboarding_welcome"}))
            logger.info(f"📧 onboarding_welcome sent to {email}")
        except Exception as e:
            logger.error(f"onboarding_welcome send error: {e}")

    logger.info(f"✅ OTP auth session created: account={account_id[:8]}… device={device_id[:8] if device_id else 'none'}")
    return {
        "session_token": session_token,
        "account_id":    account_id,
        "email":         email,
    }


@app.get("/auth/session")
@limiter.limit("60/minute")
async def auth_get_session(request: Request):
    """Validate a session token and return account info."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authorization: Bearer <token> required.")
    session_token = auth_header[7:].strip()
    account_id, email = _get_account_from_session(session_token)
    if not account_id:
        raise HTTPException(status_code=401, detail={
            "code": "SESSION_EXPIRED",
            "message": "Session expired. Please sign in again.",
        })
    return {"account_id": account_id, "email": email}


@app.post("/auth/sign-out")
@limiter.limit("30/minute")
async def auth_sign_out(request: Request):
    """Invalidate the current session token."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return {"status": "signed_out"}
    token = auth_header[7:].strip()
    if token and engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(text("DELETE FROM auth_sessions WHERE token = :token"), {"token": token})
                conn.commit()
        except Exception as e:
            logger.error(f"auth_sign_out error: {e}")
    return {"status": "signed_out"}


@app.get("/auth/sessions")
@limiter.limit("30/minute")
async def auth_list_sessions(request: Request):
    """Return active sessions for the authenticated account."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authorization: Bearer <token> required.")
    session_token = auth_header[7:].strip()
    account_id, email = _get_account_from_session(session_token)
    if not account_id:
        raise HTTPException(status_code=401, detail={
            "code": "SESSION_EXPIRED",
            "message": "Session expired. Please sign in again.",
        })
    if engine is None:
        return {"sessions": []}
    now = datetime.utcnow().isoformat()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT device_id, created_at, last_used,
                       CASE WHEN token = :cur THEN true ELSE false END AS is_current
                FROM auth_sessions
                WHERE account_id = :aid AND expires_at > :now
                ORDER BY last_used DESC NULLS LAST
                LIMIT 20
            """), {"aid": account_id, "now": now, "cur": session_token}).fetchall()
        sessions = [
            {
                "device_id":  (r[0] or "")[:12] or None,
                "created_at": r[1],
                "last_used":  r[2],
                "is_current": bool(r[3]),
            }
            for r in rows
        ]
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"auth_list_sessions error: {e}")
        return {"sessions": []}


@app.post("/auth/sign-out-all")
@limiter.limit("10/minute")
async def auth_sign_out_all(request: Request):
    """Revoke all active sessions for the authenticated account."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authorization: Bearer <token> required.")
    token = auth_header[7:].strip()
    account_id, email = _get_account_from_session(token)
    if not account_id:
        raise HTTPException(status_code=401, detail={
            "code": "SESSION_EXPIRED",
            "message": "Session expired. Please sign in again.",
        })
    if engine is None:
        return {"status": "signed_out", "sessions_revoked": 0}
    try:
        with engine.connect() as conn:
            deleted = conn.execute(
                text("DELETE FROM auth_sessions WHERE account_id = :aid"),
                {"aid": account_id}
            ).rowcount
            conn.commit()
        logger.info(f"🔓 sign-out-all: {deleted} session(s) revoked for account {account_id[:8]}…")
        return {"status": "signed_out", "sessions_revoked": deleted}
    except Exception as e:
        logger.error(f"auth_sign_out_all error: {e}")
        raise HTTPException(status_code=500, detail="Failed to sign out. Please try again.")


@app.post("/auth/request-email-change")
@limiter.limit(RATE_AUTH_REQUEST)
async def auth_request_email_change(request: Request, data: dict):
    """Request an email address change. Sends a verification link to the new address."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Must be signed in to change email.")
    token = auth_header[7:].strip()
    account_id, current_email = _get_account_from_session(token)
    if not account_id:
        raise HTTPException(status_code=401, detail={
            "code": "SESSION_EXPIRED",
            "message": "Session expired. Sign in again.",
        })

    new_email = (data.get("new_email") or "").strip().lower()
    if not new_email or "@" not in new_email or "." not in new_email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="A valid email address is required.")
    if new_email == (current_email or "").strip().lower():
        raise HTTPException(status_code=400, detail="New email must be different from your current email.")

    if engine is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        with engine.connect() as conn:
            existing = conn.execute(
                text("SELECT id FROM accounts WHERE email = :email"),
                {"email": new_email}
            ).fetchone()
        if existing and existing[0] != account_id:
            raise HTTPException(status_code=409, detail="That email is already associated with another CerebroEcho account.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"auth_request_email_change: conflict check error: {e}")
        raise HTTPException(status_code=500, detail="Database error. Please try again.")

    change_token = _secrets.token_urlsafe(32)
    expires_at   = (datetime.utcnow() + timedelta(minutes=MAGIC_LINK_EXPIRY_MINUTES)).isoformat()
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                DELETE FROM magic_link_tokens
                WHERE account_id = :aid AND token_type = 'email_change' AND used_at IS NULL
            """), {"aid": account_id})
            conn.execute(text("""
                INSERT INTO magic_link_tokens (token, account_id, expires_at, token_type, new_email)
                VALUES (:token, :aid, :expires, 'email_change', :new_email)
            """), {"token": change_token, "aid": account_id, "expires": expires_at, "new_email": new_email})
            conn.commit()
    except Exception as e:
        logger.error(f"auth_request_email_change: token creation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to send change link. Please try again.")

    change_url = f"{APP_BASE_URL}?email_change_token={change_token}"
    email_service.send_email_change_request(current_email or "", new_email, change_url)
    _track_usage(account_id, current_email or "", "resend_email", 1, 0.0001, json.dumps({"type": "email_change_request"}))
    email_service.send_email_change_notification(current_email or "", new_email)
    _track_usage(account_id, current_email or "", "resend_email", 1, 0.0001, json.dumps({"type": "email_change_notification"}))

    logger.info(f"📧 Email change requested: account {account_id[:8]}… → {new_email}")
    return {"status": "sent", "new_email": new_email}


@app.post("/auth/verify-email-change")
@limiter.limit(RATE_AUTH_VERIFY)
async def auth_verify_email_change(request: Request, data: dict):
    """Consume an email-change token and update the account email."""
    token     = (data.get("token") or "").strip()
    device_id = data.get("deviceId", "")

    if not token:
        raise HTTPException(status_code=400, detail="Token required.")
    if engine is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    now = datetime.utcnow().isoformat()
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT account_id, expires_at, used_at, new_email
                FROM magic_link_tokens
                WHERE token = :token AND token_type = 'email_change'
            """), {"token": token}).fetchone()

        if not row:
            raise HTTPException(status_code=401, detail={
                "code": "INVALID_OR_EXPIRED_TOKEN",
                "message": "This link has expired or already been used.",
            })

        account_id, expires_at, used_at, new_email = row
        if used_at or expires_at < now:
            raise HTTPException(status_code=401, detail={
                "code": "INVALID_OR_EXPIRED_TOKEN",
                "message": "This link has expired or already been used.",
            })
        if not new_email:
            raise HTTPException(status_code=500, detail="Invalid email change token — missing target address.")

        with engine.connect() as conn:
            acc_row = conn.execute(
                text("SELECT email FROM accounts WHERE id = :id"),
                {"id": account_id}
            ).fetchone()
        old_email = acc_row[0] if acc_row else ""

        with engine.connect() as conn:
            conn.execute(
                text("UPDATE magic_link_tokens SET used_at = :now WHERE token = :token"),
                {"now": now, "token": token}
            )
            conn.execute(
                text("UPDATE accounts SET email = :email WHERE id = :aid"),
                {"email": new_email, "aid": account_id}
            )
            conn.commit()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"auth_verify_email_change error: {e}")
        raise HTTPException(status_code=500, detail="Email change failed. Please try again.")

    session_token = _create_auth_session(account_id, device_id)

    try:
        email_service.send_email_changed_confirmation(new_email, old_email)
        _track_usage(account_id, new_email, "resend_email", 1, 0.0001, json.dumps({"type": "email_changed_confirmation"}))
    except Exception:
        pass  # non-fatal

    logger.info(f"✅ Email changed: account {account_id[:8]}… → {new_email}")
    return {
        "status":        "changed",
        "session_token": session_token,
        "email":         new_email,
    }


# ========================
# ADMIN — COST DASHBOARD
# ========================

@app.get("/admin/usage-summary")
async def admin_usage_summary(request: Request):
    """Internal cost dashboard. Protected by APP_SECRET header (x-app-secret)."""
    secret   = request.headers.get("x-app-secret", "")
    expected = os.environ.get("APP_SECRET", "")
    if not expected or secret != expected:
        raise HTTPException(status_code=403, detail="Forbidden")
    if engine is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    now        = datetime.utcnow()
    cutoff_30d = (now - timedelta(days=30)).isoformat()

    try:
        with engine.connect() as conn:
            cost_24h = conn.execute(text("""
                SELECT COALESCE(SUM(estimated_cost_usd), 0)
                FROM api_usage WHERE created_at >= NOW() - INTERVAL '24 hours'
            """)).scalar()

            cost_7d = conn.execute(text("""
                SELECT COALESCE(SUM(estimated_cost_usd), 0)
                FROM api_usage WHERE created_at >= NOW() - INTERVAL '7 days'
            """)).scalar()

            cost_30d = conn.execute(text("""
                SELECT COALESCE(SUM(estimated_cost_usd), 0)
                FROM api_usage WHERE created_at >= NOW() - INTERVAL '30 days'
            """)).scalar()

            top_users = conn.execute(text("""
                SELECT user_key, user_email,
                       SUM(estimated_cost_usd) AS total_cost,
                       COUNT(*) AS event_count
                FROM api_usage
                WHERE created_at >= NOW() - INTERVAL '30 days'
                GROUP BY user_key, user_email
                ORDER BY total_cost DESC
                LIMIT 10
            """)).fetchall()

            by_event = conn.execute(text("""
                SELECT event_type,
                       COUNT(*) AS event_count,
                       SUM(units) AS total_units,
                       SUM(estimated_cost_usd) AS total_cost
                FROM api_usage
                WHERE created_at >= NOW() - INTERVAL '30 days'
                GROUP BY event_type
                ORDER BY total_cost DESC
            """)).fetchall()

            session_count = conn.execute(text("""
                SELECT COUNT(*) FROM session_history
                WHERE timestamp >= :cutoff
            """), {"cutoff": cutoff_30d}).scalar()

        avg_cost = (float(cost_30d) / int(session_count)) if session_count else 0.0

        return {
            "total_cost_usd": {
                "last_24h": round(float(cost_24h), 6),
                "last_7d":  round(float(cost_7d), 6),
                "last_30d": round(float(cost_30d), 6),
            },
            "top_users_last_30d": [
                {
                    "user_key":    r[0],
                    "user_email":  r[1],
                    "total_cost":  round(float(r[2]), 6),
                    "event_count": int(r[3]),
                }
                for r in top_users
            ],
            "by_event_type_last_30d": [
                {
                    "event_type":  r[0],
                    "event_count": int(r[1]),
                    "total_units": round(float(r[2]), 2),
                    "total_cost":  round(float(r[3]), 6),
                }
                for r in by_event
            ],
            "avg_cost_per_session_last_30d": round(avg_cost, 6),
            "session_count_last_30d": int(session_count or 0),
        }
    except Exception as e:
        logger.error(f"admin_usage_summary error: {e}")
        raise HTTPException(status_code=500, detail="Internal error.")


@app.get("/debug/prompts")
async def debug_prompts(request: Request):
    """Return exact system prompts for every role×mode×style combo. Protected by APP_SECRET."""
    secret   = request.headers.get("x-app-secret", "")
    expected = os.environ.get("APP_SECRET", "")
    if not expected or secret != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

    roles  = list(ROLE_PROMPTS.keys())
    modes  = list(MODE_MODIFIERS.keys())
    styles = list(STYLE_CONFIG.keys())

    results = []
    for role in roles:
        for mode in modes:
            for style in styles:
                prompt, max_tok = build_system_prompt(role=role, mode=mode, style=style)
                results.append({
                    "role": role,
                    "mode": mode,
                    "style": style,
                    "max_tokens": max_tok,
                    "system_prompt": prompt,
                })

    return {"count": len(results), "prompts": results}


# ========================
# PRE-CALL SETUP
# ========================

PREP_SESSION_ALLOWED_PLANS = {"pro", "command", "operator", "founding_50"}

@app.post("/prep-session")
@limiter.limit(RATE_BRIEF)
async def prep_session(request: Request, data: dict):
    session_name  = (data.get("session_name", "") or "").strip()
    goal          = (data.get("goal", "") or "").strip()
    url           = (data.get("url", "") or "").strip()
    file_content  = (data.get("file_content", "") or "").strip()
    device_id     = data.get("deviceId", "")
    user_email    = data.get("userEmail", "anonymous")
    role          = (data.get("role", "") or "General").strip()
    mode          = (data.get("mode", "") or "Analyst").strip()
    # `tier` accepted for backward compatibility but no longer branches behavior —
    # one standard depth, cost fully METERED (Jina + Groq-70B tokens) below.
    tier          = "deep_brief"

    user_id, account_id, _ = _resolve_user(request, device_id, user_email)
    user_id = _get_credits_user_id(user_id, account_id)
    plan_info = _get_credit_balance(user_id)

    if plan_info["plan_type"] not in PREP_SESSION_ALLOWED_PLANS:
        raise HTTPException(status_code=403, detail={
            "code": "PLAN_REQUIRED",
            "message": "Pre-Call Setup requires Pro plan or above.",
            "required_plan": "pro",
        })

    if url and plan_info["plan_type"] not in BRIEFING_ALLOWED_PLANS:
        url = ""  # silently drop URL for plans below command

    # Fetch URL content if provided
    jina_content = ""
    fallback_used = False
    if url:
        if not (url.startswith("http://") or url.startswith("https://")):
            url = ""
        elif any(b in url.lower() for b in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254.", "10.", "192.168.", "172.16.")):
            url = ""
        else:
            try:
                import httpx as _httpx
                async with _httpx.AsyncClient() as _jina_http:
                    jina_resp = await _jina_http.get(
                        f"https://r.jina.ai/{url}",
                        headers={"Accept": "text/plain"},
                        follow_redirects=True,
                        timeout=20.0,
                    )
                if jina_resp.status_code == 200 and jina_resp.text.strip():
                    jina_content = jina_resp.text[:12000]
            except Exception as e:
                logger.warning(f"Jina fetch failed for {url}: {e}")
                fallback_used = True

    # Build material from all sources
    material_parts = []
    if jina_content:
        material_parts.append(f"URL Content ({url}):\n{jina_content}")
    elif url and fallback_used:
        material_parts.append(f"URL: {url}\n(Direct page content unavailable — brief based on domain context only.)")
    if file_content:
        material_parts.append(f"Uploaded document:\n{file_content[:8000]}")
    material = "\n\n".join(material_parts) if material_parts else "(No URL or file provided)"

    goal_prefix = f"The user's goal for this session: {goal}\n\n" if goal else ""

    system_content = (
        f"You are a pre-call intelligence analyst helping someone prepare for a {role} conversation.\n"
        f"{goal_prefix}"
        f"Background material:\n{material}\n\n"
        f"Build a structured preparation dossier as valid JSON with exactly these keys:\n"
        f"- summary: 2-3 sentence overview of the situation\n"
        f"- key_findings: array of 4-6 specific bullet points\n"
        f"- goals_and_angles: array of 3-5 tactical approaches tailored to the goal\n"
        f"- plan: paragraph or numbered steps for how to approach this session\n\n"
        f"Return ONLY valid JSON. No markdown code blocks. No extra text. Be specific not generic."
    )

    # Cost is metered (real Jina + Groq-70B tokens), known only after generation.
    # Light pre-gate: reject up front only if the user is already out of credits
    # and not on PAYG. The exact metered amount is deducted after the Groq call.
    if plan_info["balance"] <= 0 and not plan_info.get("payg_enabled", False):
        raise HTTPException(status_code=402, detail={
            "code": "INSUFFICIENT_CREDITS",
            "message": "Insufficient credits for Pre-Call Setup.",
            "balance": plan_info["balance"],
        })

    try:
        groq_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": system_content}],
            max_tokens=1200,
            temperature=0.4,
        )
        raw = groq_response.choices[0].message.content.strip()
        prep_pt = getattr(groq_response.usage, "prompt_tokens", 0) or 0
        prep_ct = getattr(groq_response.usage, "completion_tokens", 0) or 0
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"prep-session Groq error: {e}")
        raise HTTPException(status_code=500, detail="Brief generation failed. Please try again.")

    # Parse JSON; fall back to raw text in all fields if parse fails
    try:
        dossier = json.loads(raw)
        if not isinstance(dossier.get("key_findings"), list):
            dossier["key_findings"] = [dossier.get("key_findings", "")]
        if not isinstance(dossier.get("goals_and_angles"), list):
            dossier["goals_and_angles"] = [dossier.get("goals_and_angles", "")]
    except Exception:
        dossier = {
            "summary": raw[:500],
            "key_findings": [],
            "goals_and_angles": [],
            "plan": raw,
        }

    # Metered billing: charge the REAL cost (Jina page fetch + Groq-70B tokens),
    # not a flat tier price. Single source of truth = PROVIDER_RATES.
    prep_raw_cost = _llm_cost_usd("llama-3.3-70b-versatile", prep_pt, prep_ct)
    if jina_content:
        prep_raw_cost += PROVIDER_RATES["jina_fetch"]["req"]
    credits_cost = _cost_to_credits(prep_raw_cost)
    idem_key = hashlib.sha256(
        f"{user_id}:prep:{int(time.time() // 10)}".encode()
    ).hexdigest()
    _deduct_credits(user_id, credits_cost, feature="prep_session",
                    idempotency_key=idem_key, provider="groq+jina",
                    input_units=prep_pt, output_units=prep_ct,
                    raw_cost_usd=prep_raw_cost)

    # Save to session_history with status='prepped'
    new_session_id = str(_uuid.uuid4())

    if engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO session_history
                        (session_id, user_id, account_id, role, persona, style,
                         summary, session_title, session_name, prep_goal, briefing_result,
                         timestamp, duration_seconds, status)
                    VALUES
                        (:sid, :uid, :aid, :role, :mode, 'Standard',
                         :summary, :title, :sname, :goal, :briefing,
                         :ts, 0, 'prepped')
                    ON CONFLICT (session_id) DO NOTHING
                """), {
                    "sid":      new_session_id,
                    "uid":      user_id,
                    "aid":      account_id,
                    "role":     role,
                    "mode":     mode,
                    "summary":  dossier.get("summary", "")[:500],
                    "title":    session_name or "Untitled Session",
                    "sname":    session_name or None,
                    "goal":     goal or None,
                    "briefing": json.dumps(dossier),
                    "ts":       datetime.utcnow().isoformat(),
                })
                conn.commit()
        except Exception as e:
            logger.error(f"prep-session DB error: {e}")

    # Cache brief so /coach picks it up
    brief_text_for_cache = f"Pre-call preparation:\nGoal: {goal}\n\nSummary: {dossier.get('summary','')}\n\nKey findings:\n" + "\n".join(f"- {f}" for f in dossier.get("key_findings", [])) + "\n\nPlan: " + dossier.get("plan", "")
    _save_ctx_cache(user_id, brief_text_for_cache)

    logger.info(f"⚡ prep-session {new_session_id[:8]}… | {role}/{mode} | {credits_cost} credits")
    return {
        "session_id":   new_session_id,
        "session_name": session_name or "Untitled Session",
        "briefing":     dossier,
        "credits_used": credits_cost,
    }


@app.patch("/prep-session/{session_id}/notes")
@limiter.limit(RATE_READS)
async def update_prep_notes(session_id: str, request: Request):
    data       = await request.json()
    notes      = (data.get("notes", "") or "").strip()
    device_id  = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")

    user_id, account_id, _ = _resolve_user(request, device_id, user_email)
    user_id = _get_credits_user_id(user_id, account_id)

    if engine is None:
        return {"status": "skipped"}
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                UPDATE session_history SET prep_notes = :notes
                WHERE session_id = :sid
                AND (user_id = :uid OR account_id = :aid)
            """), {"notes": notes, "sid": session_id, "uid": user_id, "aid": account_id})
            conn.commit()
    except Exception as e:
        logger.error(f"update_prep_notes error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save notes")
    return {"status": "saved"}


@app.on_event("startup")
async def _startup_onboarding_worker():
    """Start the background task that sends 24h and 72h onboarding emails."""
    async def _worker():
        # Run once immediately on startup to catch any missed sends after a restart
        try:
            _process_onboarding_emails()
        except Exception as e:
            logger.error(f"onboarding worker (startup run): {e}")
        while True:
            await asyncio.sleep(1800)  # check every 30 minutes
            try:
                _process_onboarding_emails()
            except Exception as e:
                logger.error(f"onboarding email worker error: {e}")
    asyncio.create_task(_worker())
    logger.info("✅ Onboarding email background worker started")


# ── Telephony (Twilio Voice) ─────────────────────────────────────────────────
# Inert until `twilio` is installed AND TWILIO_* env vars are set (see TELEPHONY_GOLIVE.md).
# Wrapped so a telephony import/setup error can never stop the backend from booting.
try:
    from telephony import build_telephony_router, init_telephony_tables

    def _telephony_resolve_user_id(request):
        """Map an authed request -> credits user_id ('account_<id>'), or None."""
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return None
        account_id, _email = _get_account_from_session(auth[7:].strip())
        return f"account_{account_id}" if account_id else None

    init_telephony_tables(engine)
    app.include_router(build_telephony_router(
        engine, _deduct_credits, _get_credit_balance, _telephony_resolve_user_id))
    logger.info("✅ Telephony router mounted (inert unless TWILIO_* env vars set)")
except Exception as _tel_e:
    logger.warning(f"Telephony not mounted: {_tel_e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)