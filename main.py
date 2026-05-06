import io
import os
import json
import tempfile
import time
import logging
import urllib.parse
from datetime import datetime, timedelta
import stripe
from sqlalchemy import create_engine, text
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from deepgram import DeepgramClient
from fastapi.responses import StreamingResponse, Response, JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
import openai
from groq import Groq
from cartesia import Cartesia
from dotenv import load_dotenv
import email_service
import uuid as _uuid
import secrets as _secrets
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
    allow_origins=["https://cerebroecho.com", "https://www.cerebroecho.com", "https://cerebroecho-frontend.vercel.app"],
    allow_methods=["*"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
    expose_headers=["X-Transcript", "X-Answer", "X-Questions-Used"],
)

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

openai_client = None
try:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("⚠️ OPENAI_API_KEY not set")
    else:
        openai_client = openai.OpenAI(api_key=openai_api_key)
        logger.info("✅ OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"❌ ERROR creating OpenAI client: {e}")
    openai_client = None

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
    "free":        30,     # one-time, never resets
    "echo":        400,    # monthly
    "pro":         1000,   # monthly
    "command":     2500,   # monthly
    "operator":    6000,   # monthly
    "founding_50": 1000,   # monthly
}

STYLE_COSTS = {
    "Pulse":     1,
    "Signal":    2,
    "Compose":   4,
    # legacy aliases
    "Nudge":     1,
    "Brief":     2,
    "Full":      4,
    "shorthand": 1,
    "bullet":    2,
    "script":    2,
}


def _get_credit_balance(user_id: str) -> dict:
    """Get or create credits record; returns {balance, plan_type, total_used}."""
    if engine is None:
        return {"balance": 9999, "plan_type": "free", "total_used": 0}
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO credits (user_id, balance, plan_type, total_used)
                VALUES (:uid, 30, 'free', 0)
                ON CONFLICT (user_id) DO NOTHING
            """), {"uid": user_id})
            conn.commit()
            row = conn.execute(
                text("SELECT balance, plan_type, total_used FROM credits WHERE user_id=:uid"),
                {"uid": user_id}
            ).fetchone()
        if row:
            return {"balance": row[0], "plan_type": row[1], "total_used": row[2]}
    except Exception as e:
        logger.error(f"_get_credit_balance error: {e}")
        sentry_sdk.capture_exception(e)
    return {"balance": 9999, "plan_type": "free", "total_used": 0}


def _deduct_credits(user_id: str, cost: int) -> int:
    """Deduct credits; returns new balance or -1 on error."""
    if engine is None:
        return -1
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                UPDATE credits
                SET balance   = GREATEST(0, balance - :cost),
                    total_used = total_used + :cost
                WHERE user_id = :uid
                RETURNING balance
            """), {"cost": cost, "uid": user_id}).fetchone()
            conn.commit()
        return result[0] if result else -1
    except Exception as e:
        logger.error(f"_deduct_credits error: {e}")
        sentry_sdk.capture_exception(e)
        return -1


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
                new_bal = PLAN_CREDITS.get(plan, 30)
                new_reset = (datetime.utcnow() + timedelta(days=30)).isoformat()
                conn.execute(text("""
                    UPDATE credits
                    SET balance = :bal, reset_date = :reset, total_used = 0
                    WHERE user_id = :uid
                """), {"bal": new_bal, "reset": new_reset, "uid": uid})
            if rows:
                conn.commit()
                logger.info(f"🔄 Monthly credits reset for {len(rows)} users")
    except Exception as e:
        logger.error(f"reset_expired_credits error: {e}")
        sentry_sdk.capture_exception(e)


usage_tracker = {}  # kept for non-primary endpoints only

# Keyed by Stripe customer ID → {"plan": str, "status": "active"|"payment_failed"|"revoked"}
customer_plan: dict = {}

# ========================
# AUTH CONFIG
# ========================
MAGIC_LINK_EXPIRY_MINUTES = int(os.environ.get("MAGIC_LINK_EXPIRY_MINUTES", "30"))
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
You are CerebroEcho in Custom Role mode.
The user has defined their own role for this session. Apply the user's custom role description as your primary behavioral directive. Adapt your coaching style, tactics, and responses to serve whatever scenario the user has defined. If no custom description was provided, default to general conversation intelligence — help the user navigate the conversation with clarity and confidence.
""",
}

PERSONA_MODIFIERS = {
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
The user has defined their own persona for this session. Apply the user's custom persona description to shape how you think, respond, and frame suggestions. Honor the spirit and style of their definition. If no custom description was provided, respond with balanced judgment — direct but not aggressive, clear but not cold.
""",
    "Pirate": """
Respond in the voice of a classic cartoon pirate — bold, theatrical, colorful with nautical language and pirate vocabulary. Despite the colorful delivery the actual advice must still be genuinely useful and tactically sound. Never break character.
""",
}

STYLE_CONFIG = {
    "Pulse":     {"instruction": "Respond in one line or two short bullets maximum. Under 300 characters. Fast and surgical.", "max_tokens": 80},
    "Signal":    {"instruction": "Respond in one short paragraph. Under 800 characters. Balanced and tactical.", "max_tokens": 220},
    "Compose":   {"instruction": "Respond with complete, polished phrasing. Up to 2000 characters. Use when the moment requires a fully formed response.", "max_tokens": 550},
    # Legacy aliases kept for backward compatibility
    "Nudge":     {"instruction": "Respond in one line or two short bullets maximum. Under 300 characters. Fast and surgical.", "max_tokens": 80},
    "Brief":     {"instruction": "Respond in one short paragraph. Under 800 characters. Balanced and tactical.", "max_tokens": 220},
    "Full":      {"instruction": "Respond with complete, polished phrasing. Up to 2000 characters. Use when the moment requires a fully formed response.", "max_tokens": 550},
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
    persona: str,
    style: str,
    context: str = "",
    work_history: str = "",
    prior_context: str = "",
    session_context: str = "",
    user_preferences: str = "",
) -> tuple:
    parts = [ROLE_PROMPTS.get(role, _DEFAULT_ROLE)]

    if context and context not in ("", "a professional role"):
        parts.append(f"Session context: {context}")
    if work_history and work_history not in ("", "no work history provided"):
        parts.append(f"User background: {work_history}")
    if prior_context:
        parts.append(prior_context)
    if session_context and session_context.strip():
        parts.append(f"Context provided by user:\n{session_context.strip()[:6000]}")
    if user_preferences and user_preferences.strip():
        parts.append(f"User preferences: {user_preferences.strip()}")

    persona_mod = PERSONA_MODIFIERS.get(persona, "")
    if persona_mod:
        parts.append(persona_mod)

    style_cfg = STYLE_CONFIG.get(style, STYLE_CONFIG["Nudge"])
    parts.append(style_cfg["instruction"])
    parts.append(
        "Return ONLY the answer the user should say. "
        "No preamble, no labels, no meta-commentary. Start directly with the answer."
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

@app.get("/health")
async def health():
    """
    Readiness probe — intentionally not rate-limited so monitoring services
    can poll freely. Returns 200 when all critical dependencies are healthy,
    503 when any critical dependency is degraded.

    Critical (affect HTTP status): db, groq, deepgram
    Non-critical (informational only): openai, cartesia, stripe, resend, perplexity
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
    # TTS is available when either Cartesia or OpenAI is initialised
    tts_ok      = cartesia_client is not None or openai_client is not None

    # ── API key presence (never expose values) ────────────────────────────
    checks.update({
        "db":          db_ok,
        "groq":        groq_ok,
        "deepgram":    deepgram_ok,
        "tts":         tts_ok,
        "cartesia":    cartesia_client is not None,
        "openai":      openai_client is not None,
        "stripe":      bool(os.getenv("STRIPE_SECRET_KEY")),
        "resend":      bool(os.getenv("RESEND_API_KEY")),
        "perplexity":  bool(os.getenv("PERPLEXITY_API_KEY")),
    })

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
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")


@app.post("/coach")
@limiter.limit(RATE_COACH)
async def coach(
    request: Request,
    transcript: str = Form(None),
    audio: UploadFile = File(None),
    file: UploadFile = File(None),
    deviceId: str = Form(...),
    userEmail: str = Form("anonymous"),
    context: str = Form("a professional role"),
    work_history: str = Form(""),
    style: str = Form("Nudge"),
    role: str = Form("Interview Coach"),
    persona: str = Form("Diplomat"),
    session_history: str = Form(""),
    prior_summaries: str = Form(""),
    filter_small_talk: str = Form("false"),
    session_context: str = Form(""),
    user_preferences: str = Form(""),
    ghost_mode: str = Form("false"),
):
    if client is None:
        raise HTTPException(status_code=503, detail="AI service unavailable - check server configuration")

    # Resolve authenticated identity — ensures cross-device credit consistency
    raw_user_id, account_id, _ = _resolve_user(request, deviceId, userEmail)
    user_key = _get_credits_user_id(raw_user_id, account_id)

    # ── Credit check + plan metadata ──
    cost = STYLE_COSTS.get(style, 1)
    credits = _get_credit_balance(user_key)
    plan_type = credits["plan_type"]
    tts_allowed = plan_type in TTS_ALLOWED_PLANS

    # Silently downgrade operator-only options for lower plans
    if role in OPERATOR_ONLY_OPTIONS and plan_type != "operator":
        role = "Interview Coach"
    if persona in OPERATOR_ONLY_OPTIONS and plan_type != "operator":
        persona = "Diplomat"

    # Free plan: Pulse + Signal only (no Compose/Full — 4-credit responses)
    if plan_type == "free" and style in ("Full", "Compose"):
        style = "Signal"
        cost = STYLE_COSTS["Signal"]

    if credits["balance"] < cost:
        raise HTTPException(status_code=402, detail={
            "code": "INSUFFICIENT_CREDITS",
            "message": "Insufficient credits. Upgrade your plan or purchase an overage pack.",
            "balance": credits["balance"],
            "cost": cost,
        })

    start_time = time.time()
    temp_filename = None

    try:
        # --- Transcript path: pre-transcribed text provided by frontend (e.g. Deepgram streaming) ---
        if transcript and transcript.strip():
            transcript = transcript.strip()
            logger.info(f"📝 Using pre-transcribed text: {transcript[:80]}...")
        else:
            # --- Audio path: run Groq Whisper STT ---
            actual_file = audio or file
            if not actual_file:
                raise HTTPException(status_code=400, detail="Provide either 'transcript' text or an audio file")

            content = await actual_file.read()
            if len(content) < 1000:
                return {"answer": "Listening...", "transcript": "", "credits_remaining": credits["balance"], "tts_allowed": tts_allowed, "processing_time": round(time.time() - start_time, 3)}

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

        system_prompt, max_tokens = build_system_prompt(
            role=role,
            persona=persona,
            style=style,
            context=context,
            work_history=work_history,
            prior_context=prior_context,
            session_context=session_context,
            user_preferences=user_preferences,
        )
        logger.info(f"🧠 Role={role} | Persona={persona} | Style={style} | ctx={len(session_context)}c | prefs={len(user_preferences)}c | max_tokens={max_tokens}")

        # Build messages with rolling session history as conversation turns
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-5:]:
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
        credits_remaining = _deduct_credits(user_key, cost)
        if credits_remaining < 0:
            credits_remaining = max(0, credits["balance"] - cost)
        processing_time = time.time() - start_time
        logger.info(f"✅ Answer generated in {processing_time:.2f}s | -{cost} credits → {credits_remaining} remaining")

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
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/process_audio")
@limiter.limit(RATE_TRANSCRIBE)
async def process_audio(
    request: Request,
    audio: UploadFile = File(None),
    file: UploadFile = File(None),
    deviceId: str = Form(...),
    userEmail: str = Form("anonymous"),
    context: str = Form("a professional role"),
    work_history: str = Form(""),
    style: str = Form("script")
):
    # Check if Groq is available
    if client is None:
        raise HTTPException(
            status_code=503, 
            detail="AI service unavailable - check server configuration"
        )
    
    # Get the actual file
    actual_file = audio or file
    if not actual_file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Legacy endpoint — credits are the gate now; usage_tracker kept for reference only
    user_key = f"{deviceId}_{userEmail}"
    
    temp_filename = None
    
    try:
        start_time = time.time()
        
        # Read audio content
        content = await actual_file.read()
        
        # Reject tiny files (likely noise or empty clicks)
        if len(content) < 1000:
            logger.warning(f"⚠️ Audio too small: {len(content)} bytes")
            return {
                "answer": "Listening...",
                "transcript": "",
                "questions_used": current_used,
                "processing_time": round(time.time() - start_time, 3)
            }
        
        logger.info(f"📝 Transcribing {len(content)} bytes...")

        # AUDIO RETENTION: temp file exists only for the duration of the Groq API call.
        # Raw audio bytes are never written to database or object storage.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            temp_file.write(content)
            temp_filename = temp_file.name

        with open(temp_filename, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(temp_filename, audio_file.read(), "audio/webm"),
                model="whisper-large-v3-turbo",
                response_format="text",
            )
        
        # Clean up temp file
        if temp_filename and os.path.exists(temp_filename):
            os.unlink(temp_filename)
            temp_filename = None
        
        transcript = transcription.strip() if transcription else ""
        
        if not transcript or len(transcript) < 2:
            return {
                "answer": "Listening...",
                "transcript": "",
                "questions_used": current_used,
                "processing_time": round(time.time() - start_time, 3)
            }
        
        logger.info(f"✅ Transcript: {transcript[:50]}...")
        
        # Generate answer based on style
        if style == "shorthand":
            system_prompt = f"""You are an elite interview coach. The candidate is interviewing for {context}.
Their background: {work_history}
Provide ONE hint or framework name only. Max 10 words."""
            max_tokens = 50
        elif style == "bullet":
            system_prompt = f"""You are an elite interview coach. The candidate is interviewing for {context}.
Their background: {work_history}
Provide 3 tactical bullet points. Max 100 words total."""
            max_tokens = 200
        else:
            system_prompt = f"""You are an elite interview coach. The candidate is interviewing for {context}.
Their background: {work_history}
Provide a 2-3 sentence tactical answer. Be concise and confident."""
            max_tokens = 150
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Interview question: {transcript}"}
            ],
            temperature=0.6,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        answer = completion.choices[0].message.content.strip()
        usage_tracker[user_key] = current_used + 1
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Answer generated in {processing_time:.2f}s")
        logger.info(f"[AUDIO PROCESSED] Device: {deviceId} | Q: {transcript[:50]}... | Time: {processing_time:.3f}s")
        
        return {
            "transcript": transcript,
            "answer": answer,
            "questions_used": usage_tracker[user_key],
            "processing_time": round(processing_time, 3)
        }
        
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"❌ ERROR: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        # Cleanup on error
        if temp_filename and os.path.exists(temp_filename):
            os.unlink(temp_filename)

        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/process_and_speak")
@limiter.limit(RATE_TRANSCRIBE)
async def process_and_speak(
    request: Request,
    audio: UploadFile = File(None),
    file: UploadFile = File(None),
    deviceId: str = Form(...),
    userEmail: str = Form("anonymous"),
    context: str = Form("a professional role"),
    work_history: str = Form(""),
    style: str = Form("script"),
    voice: str = Form("onyx"),
):
    """Unified endpoint: STT + LLM + TTS in one request, streams audio directly."""
    if client is None:
        raise HTTPException(status_code=503, detail="AI service unavailable")
    if cartesia_client is None:
        raise HTTPException(status_code=503, detail="TTS service unavailable")

    actual_file = audio or file
    if not actual_file:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # Legacy endpoint — credits are the gate now
    temp_filename = None
    try:
        start_time = time.time()
        content = await actual_file.read()

        if len(content) < 1000:
            raise HTTPException(status_code=400, detail="Audio too short")

        # AUDIO RETENTION: temp file used only for Groq API call; deleted immediately after.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            temp_file.write(content)
            temp_filename = temp_file.name

        with open(temp_filename, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(temp_filename, audio_file.read(), "audio/webm"),
                model="whisper-large-v3-turbo",
                response_format="text",
            )

        if temp_filename and os.path.exists(temp_filename):
            os.unlink(temp_filename)
            temp_filename = None

        transcript = transcription.strip() if transcription else ""
        if not transcript or len(transcript) < 2:
            raise HTTPException(status_code=400, detail="No speech detected")

        if style == "shorthand":
            system_prompt = f"""You are an elite interview coach. The candidate is interviewing for {context}.
Their background: {work_history}
Provide ONE hint or framework name only. Max 10 words."""
            max_tokens = 50
        elif style == "bullet":
            system_prompt = f"""You are an elite interview coach. The candidate is interviewing for {context}.
Their background: {work_history}
Provide 3 tactical bullet points. Max 100 words total."""
            max_tokens = 200
        else:
            system_prompt = f"""You are an elite interview coach. The candidate is interviewing for {context}.
Their background: {work_history}
Provide a 2-3 sentence tactical answer. Be concise and confident."""
            max_tokens = 150

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Interview question: {transcript}"}
            ],
            temperature=0.6,
            max_tokens=max_tokens,
            top_p=0.9
        )

        answer = completion.choices[0].message.content.strip()
        usage_tracker[user_key] = current_used + 1
        logger.info(f"✅ STT+LLM done in {time.time() - start_time:.2f}s, starting TTS stream")

        voice_id = CARTESIA_VOICE_MAP.get(voice, CARTESIA_VOICE_MAP["onyx"])

        def generate():
            for chunk in cartesia_client.tts.sse(
                model_id="sonic-2",
                transcript=answer[:300],
                voice={"mode": "id", "id": voice_id},
                output_format={"container": "mp3", "bit_rate": 128000, "sample_rate": 44100},
            ):
                yield chunk.audio

        return StreamingResponse(
            generate(),
            media_type="audio/mpeg",
            headers={
                "X-Transcript": urllib.parse.quote(transcript[:200]),
                "X-Answer": urllib.parse.quote(answer[:300]),
                "X-Questions-Used": str(usage_tracker[user_key]),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ process_and_speak ERROR: {e}")
        if temp_filename and os.path.exists(temp_filename):
            os.unlink(temp_filename)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/tts")
@limiter.limit(RATE_TTS)
async def text_to_speech(request: Request, data: dict):
    text = data.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    speed    = float(data.get("speed", 0.85))
    voice_id = data.get("voice_id", "f9836c6e-a0bd-460e-9d3c-f7299fa60f94")

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
        logger.info(f"🎙️ Cartesia TTS sonic-3/24kHz voice={voice_id[:8]}… len={len(text)}")

        # Async generator keeps FastAPI's event loop unblocked during the entire stream.
        # 512-byte chunks deliver the first audio packet to the browser ~2× faster than 1024.
        async def cartesia_stream():
            try:
                async with httpx.AsyncClient(timeout=30.0) as http:
                    async with http.stream(
                        "POST",
                        "https://api.cartesia.ai/tts/bytes",
                        headers=cartesia_headers,
                        json=payload,
                    ) as resp:
                        resp.raise_for_status()
                        logger.info("✅ Cartesia stream open")
                        async for chunk in resp.aiter_bytes(chunk_size=512):
                            yield chunk
            except Exception as e:
                sentry_sdk.capture_exception(e)
                logger.error(f"Cartesia stream error: {e}")
                return  # exhaust generator cleanly; client gets a partial/empty response

        return StreamingResponse(cartesia_stream(), media_type="audio/mpeg")

    # Fallback: OpenAI TTS (sync client — runs in executor to avoid blocking event loop)
    if openai_client is None:
        raise HTTPException(status_code=503, detail="TTS service unavailable: neither CARTESIA_API_KEY nor OPENAI_API_KEY is set")
    voice = data.get("voice", "onyx")
    logger.warning("⚠️ USING OPENAI FALLBACK — CARTESIA_API_KEY not set")
    try:
        def _openai_gen():
            with openai_client.audio.speech.with_streaming_response.create(
                model="tts-1", voice=voice, input=text, speed=speed,
            ) as response:
                yield from response.iter_bytes(chunk_size=4096)

        return StreamingResponse(_openai_gen(), media_type="audio/mpeg")
    except Exception as e:
        sentry_sdk.capture_exception(e)
        import traceback
        logger.error(f"TTS fallback error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

@app.post("/session/start")
@limiter.limit(RATE_READS)
async def session_start(request: Request, data: dict):
    device_id  = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")

    user_id, account_id, _ = _resolve_user(request, device_id, user_email)

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
):
    """Fast STT-only endpoint for question continuation detection."""
    if client is None:
        raise HTTPException(status_code=503, detail="STT service unavailable")
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
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

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
    return {"text": extracted, "filename": file.filename, "chars": len(extracted)}


# ========================
# URL BRIEFING (JINA + GROQ)
# Jina+Groq is now primary briefing engine.
# PERPLEXITY_API_KEY reserved for future use.
# ========================

BRIEFING_TIER_CREDITS = {
    "quick_read": 2,
    "deep_brief": 8,
    "war_room":   20,
}


@app.post("/brief")
@limiter.limit(RATE_BRIEF)
async def brief_url(request: Request, data: dict):
    url        = data.get("url", "").strip()
    device_id  = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")
    role       = (data.get("role", "")    or "General").strip()
    persona    = (data.get("persona", "") or "Analyst").strip()
    tier       = (data.get("tier", "deep_brief") or "deep_brief").strip()

    if tier not in BRIEFING_TIER_CREDITS:
        tier = "deep_brief"

    credits_cost = BRIEFING_TIER_CREDITS[tier]

    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    # Plan gate: Command+ only
    user_id   = f"{device_id}_{user_email}"
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
        jina_resp = _httpx.get(
            f"https://r.jina.ai/{url}",
            headers={"Accept": "text/plain"},
            follow_redirects=True,
            timeout=20.0,
        )
        if jina_resp.status_code == 200 and jina_resp.text.strip():
            jina_content = jina_resp.text[:12000]
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

    # Step 2 — Build tier-appropriate Groq prompt
    if tier == "quick_read":
        sections_instruction = (
            f"Build a brief with exactly these sections:\n"
            f"SUMMARY (2-3 sentences)\n"
            f"KEY FACTS (5 bullet points most relevant to a {role} conversation)\n\n"
            f"Keep entire brief under 300 words. Be specific not generic."
        )
        max_tokens = 500
    elif tier == "war_room":
        sections_instruction = (
            f"Build a structured pre-call intelligence brief with exactly these sections:\n"
            f"SUMMARY (2-3 sentences)\n"
            f"KEY FACTS (5 bullet points most relevant to a {role} conversation)\n"
            f"LIKELY OBJECTIONS (3 objections this person or company might raise)\n"
            f"RECOMMENDED ANGLES (3 tactical approaches given the {persona} persona)\n"
            f"RISKS TO WATCH (2 things that could go wrong)\n"
            f"OPENING LINE (one suggested opening line tailored to {role} and {persona})\n"
            f"COMPETITOR INTELLIGENCE (if detectable from content, otherwise note 'Not detectable')\n"
            f"NEGOTIATION LEVERAGE POINTS (3 specific leverage points)\n"
            f"PSYCHOLOGICAL PROFILE (brief profile of likely counterpart based on content)\n\n"
            f"Keep entire brief under 900 words. Be specific not generic."
        )
        max_tokens = 1400
    else:  # deep_brief
        sections_instruction = (
            f"Build a structured pre-call intelligence brief with exactly these sections:\n"
            f"SUMMARY (2-3 sentences)\n"
            f"KEY FACTS (5 bullet points most relevant to a {role} conversation)\n"
            f"LIKELY OBJECTIONS (3 objections this person or company might raise)\n"
            f"RECOMMENDED ANGLES (3 tactical approaches given the {persona} persona)\n"
            f"RISKS TO WATCH (2 things that could go wrong)\n"
            f"OPENING LINE (one suggested opening line tailored to {role} and {persona})\n\n"
            f"Keep entire brief under 600 words. Be specific not generic."
        )
        max_tokens = 900

    system_content = (
        f"You are a pre-call intelligence analyst.\n"
        f"A user is about to enter a {role} conversation using the {persona} persona.\n"
        f"They have provided this background material:\n\n"
        f"{jina_content}\n\n"
        f"{sections_instruction}"
    )

    try:
        groq_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": system_content}],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        brief_text = groq_response.choices[0].message.content.strip()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Groq brief generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Brief generation failed: {str(e)}")

    if fallback_used:
        brief_text = (
            "Note: Direct page content unavailable. Brief based on domain context only.\n\n"
            + brief_text
        )

    # Step 3 — Deduct credits
    if engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO credits (user_id, balance, plan_type, total_used)
                    VALUES (:uid, :neg_cost, 'free', :cost)
                    ON CONFLICT (user_id) DO UPDATE SET
                        balance = credits.balance - :cost,
                        total_used = credits.total_used + :cost
                """), {"uid": user_id, "cost": credits_cost, "neg_cost": -credits_cost})
                conn.commit()
        except Exception as e:
            logger.error(f"Credits deduction error: {e}")

    logger.info(f"🔍 {tier} brief for {url} ({len(brief_text)} chars), {credits_cost} credits deducted")
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
async def save_preferences(request: Request, data: dict):
    device_id = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")
    preference_text = data.get("preference_text", "")

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
    persona    = data.get("persona", "Diplomat")
    style      = data.get("style", "Nudge")
    transcript = data.get("transcript", [])
    duration_seconds = int(data.get("duration_seconds", 0))
    ghost_mode = bool(data.get("ghost_mode", False))

    # Ghost mode: wipe in-memory traces and store nothing
    if ghost_mode:
        user_key = f"{device_id}_{user_email}"
        question_buffer.pop(user_key, None)
        logger.info(f"👻 Ghost session ended — memory wiped for {user_key[:20]}…")
        return {"status": "ghost", "summary": ""}

    if not transcript or len(transcript) < 2:
        return {"status": "skipped", "summary": ""}

    # Resolve authenticated identity for account-linked session saves
    user_id, account_id, _ = _resolve_user(request, device_id, user_email)
    plan_info = _get_credit_balance(_get_credits_user_id(user_id, account_id))
    can_summarize = plan_info["plan_type"] in SUMMARY_ALLOWED_PLANS

    summary = f"Session with {len(transcript)} exchanges."
    if client is not None and can_summarize:
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

    # Persist to session_history with account_id for cross-device memory
    if engine is not None and session_id:
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO session_history
                        (session_id, user_id, account_id, role, persona, style, summary, timestamp, duration_seconds)
                    VALUES (:sid, :uid, :aid, :role, :persona, :style, :summary, :ts, :dur)
                    ON CONFLICT (session_id) DO UPDATE SET
                        summary    = EXCLUDED.summary,
                        account_id = COALESCE(session_history.account_id, EXCLUDED.account_id)
                """), {
                    "sid": session_id,
                    "uid": user_id,
                    "aid": account_id,
                    "role": role,
                    "persona": persona,
                    "style": style,
                    "summary": summary,
                    "ts": datetime.utcnow().isoformat(),
                    "dur": duration_seconds,
                })
                conn.commit()
        except Exception as e:
            logger.error(f"end-session DB error: {e}")

    logger.info(f"📝 end-session {session_id[:8]}… | {role}/{persona} | {duration_seconds}s | {summary[:50]}")
    return {"status": "saved", "summary": summary}


@app.get("/session-memory")
@limiter.limit(RATE_READS)
async def get_session_memory(request: Request, deviceId: str, userEmail: str = "anonymous"):
    if engine is None:
        return {"summaries": []}
    user_id, account_id, _ = _resolve_user(request, deviceId, userEmail)
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
                    SELECT session_id, role, persona, style, summary, timestamp, duration_seconds
                    FROM session_history
                    WHERE account_id = :aid OR user_id = :uid
                    ORDER BY timestamp DESC LIMIT 50
                """), {"aid": account_id, "uid": user_id}).fetchall()
            else:
                rows = conn.execute(text("""
                    SELECT session_id, role, persona, style, summary, timestamp, duration_seconds
                    FROM session_history WHERE user_id = :uid
                    ORDER BY timestamp DESC LIMIT 50
                """), {"uid": user_id}).fetchall()
        sessions = [
            {
                "session_id": r[0],
                "role": r[1],
                "persona": r[2],
                "style": r[3],
                "summary": r[4],
                "timestamp": r[5],
                "duration_seconds": r[6],
            }
            for r in rows
        ]
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"get_session_history error: {e}")
        return {"sessions": []}


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
    return {
        "balance": credit_data["balance"],
        "plan_type": credit_data["plan_type"],
        "total_used": credit_data["total_used"],
        "is_founding_member": is_founding,
    }


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
                        SELECT balance, plan_type, total_used, reset_date, subscription_status
                        FROM credits WHERE account_id = :aid ORDER BY balance DESC LIMIT 1
                    """),
                    {"aid": account_id},
                ).fetchone()
            if row is None:
                row = conn.execute(
                    text("""
                        SELECT balance, plan_type, total_used, reset_date, subscription_status
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

# Plans that allow TTS audio whisper
TTS_ALLOWED_PLANS = {"pro", "command", "operator", "founding_50"}

# Plans that allow URL briefing (Jina+Groq)
BRIEFING_ALLOWED_PLANS = {"command", "operator", "founding_50"}

# Plans that get AI-generated session summaries
SUMMARY_ALLOWED_PLANS = {"command", "operator", "founding_50"}

# Custom role/persona requires Operator
OPERATOR_ONLY_OPTIONS = {"Custom"}

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
        raise HTTPException(status_code=500, detail=f"Billing portal error: {str(e)}")


RATE_EXPORT  = os.environ.get("RATE_LIMIT_EXPORT",  "5/minute")
RATE_DELETE  = os.environ.get("RATE_LIMIT_DELETE",  "3/hour")

@app.get("/export-data")
@limiter.limit(RATE_EXPORT)
async def export_data(request: Request, deviceId: str, userEmail: str = "anonymous"):
    """Export all personal data for the requesting user as a downloadable JSON file."""
    user_id, account_id, resolved_email = _resolve_user(request, deviceId, userEmail)

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
    payload: dict = {
        "export_metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "app": "CerebroEcho",
            "schema_version": "2",
        },
        "identity": {
            "email": export_email,
            "device_id": deviceId,
            "authenticated": account_id is not None,
        },
        "account": None,
        "preferences": None,
        "sessions": [],
        "founding_member": None,
    }

    try:
        with engine.connect() as conn:
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
                payload["account"] = {
                    "plan": row[1],
                    "credits_remaining": row[0],
                    "credits_used_total": row[2],
                    "credits_reset_date": row[3],
                    "subscription_status": row[4],
                }

            if account_id:
                row = conn.execute(text("""
                    SELECT preference_text, updated_at FROM preferences
                    WHERE account_id = :aid OR user_id = :uid LIMIT 1
                """), {"aid": account_id, "uid": user_id}).fetchone()
            else:
                row = conn.execute(text("""
                    SELECT preference_text, updated_at FROM preferences WHERE user_id = :uid
                """), {"uid": user_id}).fetchone()
            if row:
                payload["preferences"] = {
                    "text": row[0],
                    "last_updated": row[1],
                }

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
            payload["sessions"] = [
                {
                    "session_id": r[0],
                    "role": r[1],
                    "persona": r[2],
                    "style": r[3],
                    "summary": r[4],
                    "timestamp": r[5],
                    "duration_seconds": r[6],
                }
                for r in rows
            ]

            if account_id:
                row = conn.execute(text("""
                    SELECT purchase_date FROM founding_members WHERE account_id = :aid OR user_id = :uid LIMIT 1
                """), {"aid": account_id, "uid": user_id}).fetchone()
            else:
                row = conn.execute(text("""
                    SELECT purchase_date FROM founding_members WHERE user_id = :uid
                """), {"uid": user_id}).fetchone()
            if row:
                payload["founding_member"] = {"purchase_date": row[0]}

    except Exception as e:
        logger.error(f"export-data error for {user_id[:24]}…: {e}")
        raise HTTPException(status_code=500, detail="Export failed. Please try again.")

    logger.info(f"📦 Data export for {user_id[:24]}… ({len(payload['sessions'])} sessions)")
    filename = f"cerebroecho-export-{datetime.utcnow().strftime('%Y%m%d')}.json"
    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
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
        raise HTTPException(status_code=500, detail=str(e))


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

    event_id   = event["id"]
    event_type = event["type"]
    obj        = event["data"]["object"]

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
    """Provision 30 free-tier credits for a newly verified account if none exist.

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
            conn.execute(text("""
                INSERT INTO credits (user_id, account_id, balance, plan_type, total_used)
                VALUES (:uid, :aid, 30, 'free', 0)
                ON CONFLICT (user_id) DO NOTHING
            """), {"uid": uid, "aid": account_id})
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
                text("SELECT email FROM accounts WHERE id = :id"),
                {"id": account_id}
            ).fetchone()
        email = row[0] if row else ""
    except Exception as e:
        logger.error(f"auth_verify_link: email lookup error: {e}")
        email = ""

    session_token = _create_auth_session(account_id, device_id)
    _backfill_account_id(account_id, email, device_id)
    _ensure_free_credits(account_id)

    logger.info(f"✅ Auth session created: account={account_id[:8]}… device={device_id[:8] if device_id else 'none'}")
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
    email_service.send_email_change_notification(current_email or "", new_email)

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
    except Exception:
        pass  # non-fatal

    logger.info(f"✅ Email changed: account {account_id[:8]}… → {new_email}")
    return {
        "status":        "changed",
        "session_token": session_token,
        "email":         new_email,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)