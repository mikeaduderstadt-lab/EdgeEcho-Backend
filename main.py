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
from fastapi.responses import StreamingResponse, Response
import openai
from groq import Groq
from cartesia import Cartesia
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="CerebroEcho API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cerebroecho.com", "https://www.cerebroecho.com", "https://cerebroecho-frontend.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transcript", "X-Answer", "X-Questions-Used"],
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
        return -1


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


usage_tracker = {}  # kept for non-primary endpoints only

# Keyed by Stripe customer ID → {"plan": str, "status": "active"|"payment_failed"|"revoked"}
customer_plan: dict = {}

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
reset_expired_credits()

@app.get("/")
async def root():
    return {"status": "CerebroEcho Backend Live", "version": "1.2.0"}

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "groq_configured": client is not None,
        "api_key_present": bool(os.getenv("GROQ_API_KEY"))
    }

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
async def coach(
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

    user_key = f"{deviceId}_{userEmail}"
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

        return {"transcript": transcript, "answer": answer, "credits_remaining": credits_remaining, "processing_time": round(processing_time, 3), "merged": merged, "tts_allowed": tts_allowed}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ /coach ERROR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if temp_filename and os.path.exists(temp_filename):
            os.unlink(temp_filename)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/process_audio")
async def process_audio(
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
        logger.error(f"❌ ERROR: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Cleanup on error
        if temp_filename and os.path.exists(temp_filename):
            os.unlink(temp_filename)
        
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/process_and_speak")
async def process_and_speak(
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
async def text_to_speech(data: dict):
    text = data.get("text", "").strip()
    logger.info(f"📢 TTS received ({len(text)} chars): {text[:100]}...")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    speed = float(data.get("speed", 0.85))
    voice_id = data.get("voice_id", "f9836c6e-a0bd-460e-9d3c-f7299fa60f94")

    cartesia_api_key = os.environ.get("CARTESIA_API_KEY")
    if cartesia_api_key:
        try:
            import httpx
            payload = {
                "model_id": "sonic-3",
                "transcript": text,
                "voice": {"mode": "id", "id": voice_id},
                "output_format": {"container": "mp3", "bit_rate": 128000, "sample_rate": 44100},
                "speed": speed,
                "generation_config": {"speed": 1, "volume": 1},
            }
            logger.info(f"🎙️ Cartesia request body: {json.dumps(payload)}")
            cartesia_headers = {
                "Cartesia-Version": "2025-04-16",
                "X-API-Key": cartesia_api_key,
                "Content-Type": "application/json",
            }

            def cartesia_stream():
                with httpx.Client() as client:
                    with client.stream(
                        "POST",
                        "https://api.cartesia.ai/tts/bytes",
                        headers=cartesia_headers,
                        json=payload,
                        timeout=30.0,
                    ) as resp:
                        resp.raise_for_status()
                        logger.info("✅ Cartesia streaming started")
                        for chunk in resp.iter_bytes(chunk_size=1024):
                            yield chunk

            return StreamingResponse(cartesia_stream(), media_type="audio/mpeg")
        except Exception as e:
            import traceback
            logger.error(f"Cartesia failed: {traceback.format_exc()}")

    # Fallback: OpenAI TTS
    if openai_client is None:
        raise HTTPException(status_code=503, detail="TTS service unavailable: neither CARTESIA_API_KEY nor OPENAI_API_KEY is set")
    voice = data.get("voice", "onyx")
    try:
        logger.warning("⚠️ USING OPENAI FALLBACK — Cartesia failed")
        logger.info(f"TTS streaming started (OpenAI tts-1, voice={voice}, speed={speed}) for: {text[:50]}...")

        def generate():
            with openai_client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice=voice,
                input=text,
                speed=speed,
            ) as response:
                for chunk in response.iter_bytes(chunk_size=4096):
                    yield chunk

        return StreamingResponse(generate(), media_type="audio/mpeg")
    except Exception as e:
        import traceback
        logger.error(f"TTS error: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"TTS error: {type(e).__name__}: {str(e)}")

@app.post("/session/start")
async def session_start(data: dict):
    device_id = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")
    try:
        if engine is None:
            raise RuntimeError("No database configured")
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT summary FROM sessions WHERE device_id=:did AND user_email=:email ORDER BY created_at DESC LIMIT 3"),
                {"did": device_id, "email": user_email}
            ).fetchall()
        summaries = [r[0] for r in rows]
    except Exception as e:
        logger.error(f"session/start DB error: {e}")
        summaries = []
    return {"prior_summaries": summaries}


@app.post("/session/end")
async def session_end(data: dict):
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
async def quick_transcribe(
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
async def upload_context(
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
# URL BRIEFING (PERPLEXITY)
# ========================

@app.post("/brief")
async def brief_url(data: dict):
    url = data.get("url", "").strip()
    device_id = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")

    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    # Plan gate: Command+ only
    user_id = f"{device_id}_{user_email}"
    plan_info = _get_credit_balance(user_id)
    if plan_info["plan_type"] not in BRIEFING_ALLOWED_PLANS:
        raise HTTPException(status_code=403, detail={
            "code": "PLAN_REQUIRED",
            "message": "URL briefing requires Command plan or above.",
            "required_plan": "command",
        })

    perplexity_key = os.environ.get("PERPLEXITY_API_KEY")
    if not perplexity_key:
        raise HTTPException(status_code=503, detail="Briefing service unavailable: PERPLEXITY_API_KEY not set")

    prompt = (
        f"Research the following URL and provide a concise brief covering: "
        f"key facts, likely topics of conversation, potential objections or questions "
        f"that may arise, and any relevant background information. URL: {url}"
    )

    try:
        import httpx as _httpx
        resp = _httpx.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {perplexity_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        brief_text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Perplexity API error: {e}")
        raise HTTPException(status_code=500, detail=f"Briefing failed: {str(e)}")

    # Deduct 5 credits
    if engine is not None:
        user_id = f"{device_id}_{user_email}"
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO credits (user_id, balance, plan_type, total_used)
                    VALUES (:uid, -5, 'free', 5)
                    ON CONFLICT (user_id) DO UPDATE SET
                        balance = credits.balance - 5,
                        total_used = credits.total_used + 5
                """), {"uid": user_id})
                conn.commit()
        except Exception as e:
            logger.error(f"Credits deduction error: {e}")

    logger.info(f"🔍 Brief generated for {url} ({len(brief_text)} chars), 5 credits deducted")
    return {"brief": brief_text, "url": url, "credits_used": 5}


# ========================
# PREFERENCES
# ========================

@app.get("/preferences")
async def get_preferences(deviceId: str, userEmail: str = "anonymous"):
    if engine is None:
        return {"preference_text": ""}
    user_id = f"{deviceId}_{userEmail}"
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT preference_text FROM preferences WHERE user_id=:uid"),
                {"uid": user_id}
            ).fetchone()
        return {"preference_text": row[0] if row else ""}
    except Exception as e:
        logger.error(f"get_preferences error: {e}")
        return {"preference_text": ""}


@app.post("/preferences")
async def save_preferences(data: dict):
    device_id = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")
    preference_text = data.get("preference_text", "")

    if engine is None:
        return {"status": "skipped"}

    user_id = f"{device_id}_{user_email}"
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO preferences (user_id, preference_text, updated_at)
                VALUES (:uid, :text, :now)
                ON CONFLICT (user_id) DO UPDATE SET
                    preference_text = EXCLUDED.preference_text,
                    updated_at = EXCLUDED.updated_at
            """), {"uid": user_id, "text": preference_text, "now": datetime.utcnow().isoformat()})
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
async def end_session(data: dict):
    session_id = data.get("session_id", "")
    device_id  = data.get("deviceId", "")
    user_email = data.get("userEmail", "anonymous")
    role       = data.get("role", "Interview Coach")
    persona    = data.get("persona", "Diplomat")
    style      = data.get("style", "Nudge")
    transcript = data.get("transcript", [])
    duration_seconds = int(data.get("duration_seconds", 0))
    ghost_mode = bool(data.get("ghost_mode", False))

    # Ghost mode: process nothing, store nothing, leave no trace
    if ghost_mode:
        logger.info(f"👻 Ghost session ended — no data stored")
        return {"status": "ghost", "summary": ""}

    if not transcript or len(transcript) < 2:
        return {"status": "skipped", "summary": ""}

    # One-sentence summary via Groq (Command+ only)
    user_id = f"{device_id}_{user_email}"
    plan_info = _get_credit_balance(user_id)
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

    # Persist to session_history
    user_id = f"{device_id}_{user_email}"
    if engine is not None and session_id:
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO session_history
                        (session_id, user_id, role, persona, style, summary, timestamp, duration_seconds)
                    VALUES (:sid, :uid, :role, :persona, :style, :summary, :ts, :dur)
                    ON CONFLICT (session_id) DO UPDATE SET summary = EXCLUDED.summary
                """), {
                    "sid": session_id,
                    "uid": user_id,
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


@app.get("/session-history")
async def get_session_history(deviceId: str, userEmail: str = "anonymous"):
    if engine is None:
        return {"sessions": []}
    user_id = f"{deviceId}_{userEmail}"
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT session_id, role, persona, style, summary, timestamp, duration_seconds
                FROM session_history
                WHERE user_id = :uid
                ORDER BY timestamp DESC
                LIMIT 50
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
async def get_credits(deviceId: str, userEmail: str = "anonymous"):
    user_id = f"{deviceId}_{userEmail}"
    data = _get_credit_balance(user_id)
    is_founding = False
    if engine is not None:
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT 1 FROM founding_members WHERE user_id=:uid"),
                    {"uid": user_id}
                ).fetchone()
                is_founding = row is not None
        except Exception as e:
            logger.error(f"founding check error: {e}")
    return {
        "balance": data["balance"],
        "plan_type": data["plan_type"],
        "total_used": data["total_used"],
        "is_founding_member": is_founding,
    }


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

# Plans that allow URL briefing (Perplexity)
BRIEFING_ALLOWED_PLANS = {"command", "operator", "founding_50"}

# Plans that get AI-generated session summaries
SUMMARY_ALLOWED_PLANS = {"command", "operator", "founding_50"}

# Custom role/persona requires Operator
OPERATOR_ONLY_OPTIONS = {"Custom"}

@app.post("/create-checkout")
async def create_checkout(data: dict):
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

    mode    = PLAN_MODES[plan]
    user_id = f"{device_id}_{user_email}"

    try:
        session = stripe.checkout.Session.create(
            mode=mode,
            line_items=[{"price": price_id, "quantity": 1}],
            metadata={"plan": plan, "user_id": user_id},
            success_url="https://cerebroecho.com/app?payment=success",
            cancel_url="https://cerebroecho.com/app?payment=cancelled",
        )
        return {"url": session.url}
    except stripe.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload    = await request.body()
    sig_header = request.headers.get("stripe-signature")
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")

    if not webhook_secret:
        raise HTTPException(status_code=503, detail="Webhook secret not configured")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except stripe.errors.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type   = event["type"]
    session_data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        user_id    = session_data.get("metadata", {}).get("user_id", "")
        plan       = session_data.get("metadata", {}).get("plan", "unknown")
        customer_id = session_data.get("customer", "")
        payment_id  = session_data.get("payment_intent") or session_data.get("id", "")

        if user_id and plan in PLAN_CREDITS and engine is not None:
            new_balance = PLAN_CREDITS[plan]
            reset_date  = (datetime.utcnow() + timedelta(days=30)).isoformat() if plan != "free" else None
            try:
                with engine.connect() as conn:
                    # Update credits / plan
                    conn.execute(text("""
                        INSERT INTO credits (user_id, balance, plan_type, total_used, reset_date)
                        VALUES (:uid, :bal, :plan, 0, :reset)
                        ON CONFLICT (user_id) DO UPDATE SET
                            balance    = :bal,
                            plan_type  = :plan,
                            total_used = 0,
                            reset_date = :reset
                    """), {"uid": user_id, "bal": new_balance, "plan": plan, "reset": reset_date})

                    # Store customer_id → user_id mapping for subscription events
                    if customer_id:
                        conn.execute(text("""
                            INSERT INTO stripe_customers (customer_id, user_id, created_at)
                            VALUES (:cid, :uid, :now)
                            ON CONFLICT (customer_id) DO NOTHING
                        """), {"cid": customer_id, "uid": user_id, "now": datetime.utcnow().isoformat()})

                    # Record founding member
                    if plan == "founding_50":
                        conn.execute(text("""
                            INSERT INTO founding_members (user_id, purchase_date, stripe_payment_id)
                            VALUES (:uid, :date, :pid)
                            ON CONFLICT (user_id) DO NOTHING
                        """), {"uid": user_id, "date": datetime.utcnow().isoformat(), "pid": payment_id})

                    conn.commit()
                logger.info(f"✅ checkout.session.completed: user={user_id} plan={plan} balance={new_balance}")
            except Exception as e:
                logger.error(f"webhook DB error: {e}")

    elif event_type == "customer.subscription.deleted":
        # Downgrade to free when subscription is cancelled
        customer_id = session_data.get("customer", "")
        if customer_id and engine is not None:
            try:
                with engine.connect() as conn:
                    row = conn.execute(
                        text("SELECT user_id FROM stripe_customers WHERE customer_id=:cid"),
                        {"cid": customer_id}
                    ).fetchone()
                    if row:
                        uid = row[0]
                        conn.execute(text("""
                            UPDATE credits
                            SET plan_type='free', balance=0, reset_date=NULL
                            WHERE user_id=:uid
                        """), {"uid": uid})
                        conn.commit()
                        logger.info(f"🚫 Subscription cancelled — downgraded {uid} to free")
                    else:
                        logger.warning(f"🚫 Subscription cancelled for unknown customer {customer_id}")
            except Exception as e:
                logger.error(f"subscription.deleted DB error: {e}")

    elif event_type == "invoice.payment_failed":
        customer_id = session_data.get("customer", "")
        logger.warning(f"⚠️ Payment failed for customer {customer_id}")

    return {"received": True}


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
    return {"status": "success", "message": "Trial extended to 10 questions"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)