"""
CerebroEcho — Telephony module (Twilio Voice).  STATUS: foundation, NOT yet wired/deployed.

This file is INERT until:
  1. `twilio` is added to requirements.txt and installed, AND
  2. TWILIO_* env vars are set on Railway, AND
  3. main.py includes the router (one line — see TELEPHONY_GOLIVE.md).

Until then it imports cleanly (no top-level third-party imports) and does nothing, so it
cannot affect the live backend. Build it once, flip it on after the account is funded.

What it provides (hybrid model — see PHONE_CALLING_SPEC.md):
  GET  /voice/token      -> mint a Twilio Voice access token for the in-app softphone (managed acct)
  POST /voice/outbound   -> TwiML: bridge the user's softphone to a dialed PSTN number (with credit guard)
  POST /voice/inbound    -> TwiML: consent announcement, then ring the user's softphone (<Client>)
  POST /voice/status     -> Twilio status callback: on call completion, meter minutes -> credits

Billing reuses the SAME path as Whisper/Cartesia: raw provider cost (USD) -> credits via
CREDIT_COST_BASIS_USD, deducted with row-lock + idempotency (CallSid as the key) so a retried
status callback never double-charges. This automatically applies the existing margin.
"""
from __future__ import annotations

import os
import math
import logging
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import Response, JSONResponse

logger = logging.getLogger("telephony")

# Must match main.py's basis exactly (1 credit ≈ $0.000143 of raw API cost).
CREDIT_COST_BASIS_USD = 0.000143

# --- Env (managed CerebroEcho business account) ---------------------------------
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_API_KEY_SID = os.environ.get("TWILIO_API_KEY_SID", "")
TWILIO_API_KEY_SECRET = os.environ.get("TWILIO_API_KEY_SECRET", "")
TWILIO_TWIML_APP_SID = os.environ.get("TWILIO_TWIML_APP_SID", "")
# Default caller ID for outbound (a managed CerebroEcho number); per-user numbers override this.
TWILIO_CALLER_ID = os.environ.get("TWILIO_CALLER_ID", "")
# Play "this call may be recorded" on inbound (decided: ON). Set to "0" to disable.
TELEPHONY_CONSENT_ANNOUNCE = os.environ.get("TELEPHONY_CONSENT_ANNOUNCE", "1") != "0"
CONSENT_TEXT = os.environ.get(
    "TELEPHONY_CONSENT_TEXT",
    "This call may be recorded and transcribed for coaching.",
)
# Minimum credits required to start a call (≈ a minute of headroom). Tunable.
MIN_CALL_CREDITS = int(os.environ.get("TELEPHONY_MIN_CALL_CREDITS", "200"))

TELEPHONY_ENABLED = bool(
    TWILIO_ACCOUNT_SID and TWILIO_API_KEY_SID and TWILIO_API_KEY_SECRET and TWILIO_TWIML_APP_SID
)


def minutes_cost_to_credits(raw_cost_usd: float) -> int:
    """Same conversion as the rest of the app — applies the existing margin."""
    if raw_cost_usd <= 0:
        return 1
    return max(1, math.ceil(raw_cost_usd / CREDIT_COST_BASIS_USD))


def init_telephony_tables(engine) -> None:
    """Create the phone_numbers table. Safe to call repeatedly (IF NOT EXISTS)."""
    if engine is None:
        return
    from sqlalchemy import text
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS phone_numbers (
                    id           SERIAL PRIMARY KEY,
                    user_id      TEXT NOT NULL,
                    e164         TEXT NOT NULL,                 -- +15551234567
                    twilio_sid   TEXT,                          -- PNxxxx (managed) or NULL (BYO/ported)
                    account_mode TEXT NOT NULL DEFAULT 'managed', -- 'managed' | 'byo'
                    source       TEXT NOT NULL DEFAULT 'web',     -- 'web' (new) | 'ported'
                    status       TEXT NOT NULL DEFAULT 'active',   -- 'active'|'porting'|'released'
                    byo_sid      TEXT,                          -- BYO: their Account SID (creds in vault, not here)
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (e164)
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_phone_numbers_user
                ON phone_numbers(user_id, status)
            """))
            # Port-in (Option B) tracking. NOTE: we deliberately DO NOT persist the
            # carrier PIN or account number — those are pass-through to Twilio only.
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS port_requests (
                    id              SERIAL PRIMARY KEY,
                    user_id         TEXT NOT NULL,
                    e164            TEXT NOT NULL,                  -- number being ported in
                    status          TEXT NOT NULL DEFAULT 'submitted', -- submitted|pending|completed|failed|canceled
                    twilio_port_sid TEXT,                           -- Twilio PortIn SID once submitted
                    holder_name     TEXT,                           -- account holder (non-sensitive)
                    note            TEXT,                           -- last status detail / error
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_port_requests_user
                ON port_requests(user_id, status)
            """))
        logger.info("phone_numbers + port_requests tables ready")
    except Exception as e:
        logger.error(f"init_telephony_tables failed: {e}")


def _twiml(xml_body: str) -> Response:
    return Response(content=f'<?xml version="1.0" encoding="UTF-8"?><Response>{xml_body}</Response>',
                    media_type="application/xml")


def build_telephony_router(engine, deduct_credits, get_credit_balance, resolve_user_id):
    """
    Factory so we avoid circular imports — main.py passes in its own helpers:
      deduct_credits(user_id, cost, feature, idempotency_key, provider, raw_cost_usd, status) -> int
      get_credit_balance(user_id) -> {"balance": int, ...}
      resolve_user_id(request) -> str | None   (maps an authed request to its user_id)
    """
    router = APIRouter(prefix="/voice", tags=["telephony"])

    @router.get("/status")
    async def telephony_status():
        """Cheap readiness probe for the feature (not the Twilio status callback)."""
        return {"enabled": TELEPHONY_ENABLED, "consent_announce": TELEPHONY_CONSENT_ANNOUNCE}

    @router.get("/token")
    async def voice_token(request: Request):
        """Mint a short-lived Twilio Voice access token for the in-app softphone."""
        if not TELEPHONY_ENABLED:
            raise HTTPException(503, "TELEPHONY_DISABLED")
        user_id = resolve_user_id(request)
        if not user_id:
            raise HTTPException(401, "AUTH_REQUIRED")
        bal = get_credit_balance(user_id)
        if bal.get("balance", 0) < MIN_CALL_CREDITS and not bal.get("payg_enabled"):
            raise HTTPException(402, "INSUFFICIENT_CREDITS")
        try:
            from twilio.jwt.access_token import AccessToken
            from twilio.jwt.access_token.grants import VoiceGrant
        except Exception:
            raise HTTPException(503, "TWILIO_SDK_MISSING")  # add `twilio` to requirements at go-live
        identity = user_id  # softphone identity == our user_id, so inbound <Client> can target it
        token = AccessToken(TWILIO_ACCOUNT_SID, TWILIO_API_KEY_SID, TWILIO_API_KEY_SECRET,
                            identity=identity, ttl=3600)
        token.add_grant(VoiceGrant(outgoing_application_sid=TWILIO_TWIML_APP_SID,
                                   incoming_allow=True))
        return JSONResponse({"token": token.to_jwt(), "identity": identity})

    @router.post("/outbound")
    async def voice_outbound(request: Request, To: str = Form(""), From: str = Form("")):
        """TwiML for an outbound call placed from the softphone. Guards credits, then dials PSTN."""
        if not TELEPHONY_ENABLED:
            return _twiml("<Say>Calling is not available right now.</Say>")
        # The softphone identity arrives as From="client:<user_id>".
        user_id = From.split("client:", 1)[1] if From.startswith("client:") else resolve_user_id(request)
        if user_id:
            bal = get_credit_balance(user_id)
            if bal.get("balance", 0) < MIN_CALL_CREDITS and not bal.get("payg_enabled"):
                return _twiml("<Say>You don't have enough credits for this call. "
                              "Please top up and try again.</Say>")
        if not To:
            return _twiml("<Say>No number was dialed.</Say>")
        caller_id = TWILIO_CALLER_ID  # TODO at go-live: look up this user's own number for caller ID
        # statusCallback fires the metering hook when the call ends.
        return _twiml(
            f'<Dial callerId="{caller_id}" answerOnBridge="true" '
            f'  record="do-not-record">'
            f'<Number statusCallbackEvent="completed" '
            f'        statusCallback="/voice/status" statusCallbackMethod="POST">{To}</Number>'
            f'</Dial>'
        )

    @router.post("/inbound")
    async def voice_inbound(To: str = Form(""), From: str = Form("")):
        """TwiML for an inbound call to a user's number: announce consent, ring their softphone."""
        if not TELEPHONY_ENABLED:
            return _twiml("<Say>This number is not available right now.</Say>")
        # Map the dialed number (To) -> owning user_id -> ring their <Client>.
        user_id = None
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                row = conn.execute(text(
                    "SELECT user_id FROM phone_numbers WHERE e164=:e AND status='active'"),
                    {"e": To}).fetchone()
                user_id = row[0] if row else None
        except Exception as e:
            logger.error(f"inbound lookup failed: {e}")
        if not user_id:
            return _twiml("<Say>This number is not in service.</Say>")
        announce = f"<Say>{CONSENT_TEXT}</Say>" if TELEPHONY_CONSENT_ANNOUNCE else ""
        return _twiml(
            f'{announce}'
            f'<Dial answerOnBridge="true">'
            f'<Client statusCallbackEvent="completed" '
            f'        statusCallback="/voice/status" statusCallbackMethod="POST">{user_id}</Client>'
            f'</Dial>'
        )

    @router.post("/status")
    async def voice_status_callback(
        CallSid: str = Form(""), CallStatus: str = Form(""),
        CallDuration: str = Form("0"), Price: str = Form(""), From: str = Form(""), To: str = Form(""),
    ):
        """Twilio status callback. On completion, meter the call into credits (idempotent on CallSid)."""
        if CallStatus != "completed" or not CallSid:
            return Response(status_code=204)
        # Resolve which user pays: prefer the client leg.
        payer = None
        for cand in (From, To):
            if cand.startswith("client:"):
                payer = cand.split("client:", 1)[1]
                break
        if not payer:
            try:
                from sqlalchemy import text
                with engine.connect() as conn:
                    row = conn.execute(text(
                        "SELECT user_id FROM phone_numbers WHERE e164 IN (:f,:t) AND status='active' LIMIT 1"),
                        {"f": From, "t": To}).fetchone()
                    payer = row[0] if row else None
            except Exception as e:
                logger.error(f"status payer lookup failed: {e}")
        if not payer:
            logger.warning(f"call {CallSid} completed but no payer resolved; not charged")
            return Response(status_code=204)
        # Twilio Price is a negative string in account currency (e.g. "-0.0085"); may be empty if not ready.
        try:
            raw_usd = abs(float(Price)) if Price else 0.0
        except ValueError:
            raw_usd = 0.0
        if raw_usd == 0.0:  # fallback estimate if Price not yet populated
            mins = math.ceil(int(CallDuration or "0") / 60) or 1
            raw_usd = mins * 0.022  # conservative US outbound estimate
        credits = minutes_cost_to_credits(raw_usd)
        try:
            deduct_credits(payer, credits, feature="call", idempotency_key=f"twilio:{CallSid}",
                           provider="twilio", raw_cost_usd=raw_usd, status="charged")
            logger.info(f"call {CallSid}: {CallDuration}s ${raw_usd:.4f} -> {credits} cr from {payer[:16]}…")
        except Exception as e:
            logger.error(f"call {CallSid} metering failed: {e}")
        return Response(status_code=204)

    # ── Porting (Option B): port an existing number into the managed account ──────
    def _update_port(port_id, status=None, sid=None, note=None):
        from sqlalchemy import text
        sets, p = ["updated_at = NOW()"], {"id": port_id}
        if status is not None: sets.append("status = :s"); p["s"] = status
        if sid is not None:    sets.append("twilio_port_sid = :sid"); p["sid"] = sid
        if note is not None:   sets.append("note = :n"); p["n"] = note[:480]
        try:
            with engine.begin() as conn:
                conn.execute(text(f"UPDATE port_requests SET {', '.join(sets)} WHERE id = :id"), p)
        except Exception as e:
            logger.error(f"_update_port failed: {e}")

    @router.post("/port/request")
    async def port_request(
        request: Request,
        phone_number: str = Form(""),
        carrier_account_number: str = Form(""),
        carrier_pin: str = Form(""),
        holder_name: str = Form(""),
        service_address: str = Form(""),
        loa_authorized: str = Form(""),  # "1" => user attests they're authorized to port this number
    ):
        """Submit a port-in. Carrier PIN/account# are pass-through to Twilio — never stored."""
        if not TELEPHONY_ENABLED:
            raise HTTPException(503, "TELEPHONY_DISABLED")
        user_id = resolve_user_id(request)
        if not user_id:
            raise HTTPException(401, "AUTH_REQUIRED")
        e164 = phone_number.strip()
        if not (e164.startswith("+") and e164[1:].isdigit() and 8 <= len(e164) <= 16):
            raise HTTPException(422, "INVALID_NUMBER")          # expect E.164 e.g. +15551234567
        if loa_authorized != "1":
            raise HTTPException(422, "LOA_NOT_AUTHORIZED")      # user must attest authorization
        if not (carrier_account_number and carrier_pin):
            raise HTTPException(422, "MISSING_CARRIER_INFO")

        from sqlalchemy import text
        with engine.begin() as conn:                            # track without PII (no PIN/acct#)
            port_id = conn.execute(text("""
                INSERT INTO port_requests (user_id, e164, status, holder_name, note)
                VALUES (:u, :e, 'submitted', :h, 'created') RETURNING id
            """), {"u": user_id, "e": e164, "h": holder_name or None}).fetchone()[0]

        # Submit to Twilio's Porting API.
        # ⚠️ LIVE-TEST TODO: a real port-in also needs a signed LOA + recent bill uploaded as
        # documents, whose SIDs go in "documents". That step requires the funded account to
        # validate — left as the integration point below. Wrapped so failures don't crash.
        twilio_port_sid = None
        try:
            import httpx, base64
            auth = base64.b64encode(f"{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}".encode()).decode()
            payload = {
                "phone_numbers": [e164],
                "losing_carrier_information": {
                    "account_number": carrier_account_number,
                    "pin": carrier_pin,
                    "account_holder_name": holder_name,
                    "service_address": service_address,
                },
                # "documents": [<LOA_SID>, <BILL_SID>],  # TODO: upload + reference at go-live
            }
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    "https://numbers.twilio.com/v1/Porting/PortIn",
                    headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
                    json=payload,
                )
            if resp.status_code in (200, 201):
                twilio_port_sid = (resp.json() or {}).get("port_in_request_sid")
                _update_port(port_id, status="pending", sid=twilio_port_sid, note="submitted to Twilio")
            else:
                _update_port(port_id, status="failed", note=f"twilio {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            _update_port(port_id, status="failed", note=f"submit error: {e}")
            logger.error(f"port submit failed: {e}")

        return JSONResponse({
            "port_id": port_id,
            "twilio_port_sid": twilio_port_sid,
            "status": "pending" if twilio_port_sid else "submitted",
            "note": "Porting typically completes in 3–5 business days.",
        })

    @router.get("/port/status")
    async def port_status(request: Request):
        """List this user's port requests + latest status."""
        if not TELEPHONY_ENABLED:
            raise HTTPException(503, "TELEPHONY_DISABLED")
        user_id = resolve_user_id(request)
        if not user_id:
            raise HTTPException(401, "AUTH_REQUIRED")
        from sqlalchemy import text
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT e164, status, twilio_port_sid, note, created_at, updated_at
                FROM port_requests WHERE user_id = :u ORDER BY created_at DESC LIMIT 20
            """), {"u": user_id}).fetchall()
        return JSONResponse({"ports": [
            {"e164": r[0], "status": r[1], "twilio_port_sid": r[2], "note": r[3],
             "created_at": str(r[4]), "updated_at": str(r[5])} for r in rows]})

    return router
