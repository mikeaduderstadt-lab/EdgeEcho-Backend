# CerebroEcho Backend — Context for Claude Code

FastAPI backend for CerebroEcho (real-time AI conversation intelligence). All endpoints, plan logic,
metering, and Stripe handling live in `main.py`. Frontend/shared context is in the parent repo's
`../CLAUDE.md`; dated session history is in `../HANDOFF_LOG.md`.

## This is a SEPARATE git repo
- Remote: `EdgeEcho-Backend` (GitHub, owner mikeaduderstadt-lab). Branch `master`.
- It sits inside the frontend working tree but is its own repo — frontend pushes do NOT include it.
- **Deploy = push to master.** `git push origin master` auto-triggers the Railway deploy (no manual
  redeploy needed; wait for SUCCESS). Only push when you actually edited backend code.
- Railway: project id `a704747a-4770-49f3-b482-e89f5e939f58`, env `production`, services `web` + `Postgres`.
  CLI: `railway variables` (list env), `railway redeploy` (after an env-var change). `railway run <cmd>`
  runs against the linked service (needs Postgres linked for DB scripts; uses `DATABASE_PUBLIC_URL`).
- `DATABASE_URL` must be confirmed set before any session involving auth or credits.

## Files
- `main.py` — FastAPI app, all endpoints, plan logic, metering, Stripe webhooks.
- `email_service.py` — Resend transactional emails (magic link, downgrade, etc.).
- `requirements.txt` · `Procfile`. (`.env` is gitignored.)

## Stack touchpoints
STT Groq Whisper (`whisper-large-v3-turbo`) · LLM Groq LLaMA (`llama-3.1-8b-instant` for /coach;
`llama-3.3-70b` for /brief) · TTS Cartesia Sonic-3 (OpenAI fallback) · Research Jina AI · DB Postgres ·
Payments Stripe · Email Resend.

## What I (Claude) can do for you — automation & integration (Jun 14, 2026)

**Installed & wired at user scope (all your projects):**
- **Playwright MCP** — I can drive a real browser to access provider dashboards (Groq, Cartesia, Stripe, Railway, Resend).
- **1Password CLI (`op`)** — I read secrets from your 1Password vault via `op read "op://..."` (refs never leak into git).
- **GitHub CLI (`gh`)** — I can manage the `EdgeEcho-Backend` GitHub repo (pushes, PRs, issues).
- **Google Cloud SDK (`gcloud`)** & **Railway CLI** — I can manage cloud infrastructure and Railway env vars.

**What this means for CerebroEcho Backend:**
- **API key updates:** I mint/rotate Groq, Cartesia, Stripe, Resend, Jina keys via browser or CLI, store in 1Password.
- **Railway env vars:** I set/update `GROQ_API_KEY`, `CARTESIA_API_KEY`, `STRIPE_*` keys, `DATABASE_URL`, etc. via `railway variables` without manual entry.
- **Secrets stay encrypted:** All keys live in 1Password vault; `.env` is gitignored; no plaintext secrets in git.
- **Safe deployments:** After env var changes, I can run `railway redeploy` to push updates live.

**In future sessions, assume I will:**
- Open provider dashboards and drive the clicks to mint/rotate keys
- Update Railway env vars directly via CLI
- Store credentials in 1Password and reference them
- Test the backend post-deploy (health checks, endpoint verification)
- Never ask you to paste raw secrets into chat

**Critical:** After restart and first auth, tell me "ready" — I'll orchestrate the backend provider setup (Groq, Cartesia, Stripe, Resend, Jina) and wire Railway env vars. No manual entry needed.

---

## Protected — do not modify without owner sign-off
- `PLAN_CREDITS` dict
- `TTS_ALLOWED_PLANS`
- Stripe webhook handling code

## Plan structure (`PLAN_CREDITS`) — owner-locked values
Resized Jun 7, 2026 for full usage-based metering (~3,600 credits ≈ 1 hr audio). **Verify against these
locked values before editing — never infer from old code.**

| Internal key | Display | Price | Credits/mo | Notes |
|---|---|---|---|---|
| `free` | Free | — | 20 (signed-in, monthly reset) | Anonymous no-account device = 60 one-time, never resets. **Text only.** |
| `echo` | Solo | $9.99 | 4,000 | Entry paid. ~1 hr audio (3,600s/mo hard cap applies to echo only). |
| `pro` | Pro | $29.99 | 75,000 | ~20 hrs audio, no audio cap. |
| `command` | Power | $59.99 | 150,000 | ~40 hrs audio, no audio cap. |
| `operator` | (hidden) | — | 150,000 | Parity with Power. |
| `founding_50` | LTD | $299 one-time | 14,000 | Existing members honored; removed from public surfaces. |

- Stripe price env keys: `STRIPE_PRICE_ECHO` / `STRIPE_PRICE_PRO` / `STRIPE_PRICE_COMMAND` /
  `STRIPE_PRICE_OPERATOR` / `STRIPE_PRICE_FOUNDING50`.
- **`PLAN_CREDITS` keys MUST exactly match the `STRIPE_PRICE_IDS` keys.** A mismatch makes
  `checkout.session.completed` silently skip credit provisioning — subscribers land with 0 credits, no error.
- `PLAN_CREDITS` values are NOT the marketing/display numbers — they're the real allowances above.
- Free email accounts get `reset_date` on first sign-in via `_ensure_free_credits` (monthly cron refills
  to the free allowance). Anonymous users have NULL `reset_date` (one-time, never resets). Free→paid
  checkout webhook uses `ON CONFLICT DO UPDATE SET balance=:bal` (replaces old free balance).

## Feature gates (verified Jun 11, 2026)
- `TTS_ALLOWED_PLANS` = echo+ (Free is text-only; Solo/echo IS included → gets audio).
- `CUSTOM_ALLOWED_PLANS` = pro+ · PREP_SESSION (Pre-Call dossier) = pro+ ·
  BRIEFING (`/brief`) = command+ · SUMMARY (auto after-call) = command+.
- Cross-session **recall** (`GET /session-memory` — injects prior-session summaries into the prompt) is
  **Pro+** only (`{pro, command, operator, founding_50}`; returns `[]` otherwise). This matches the Landing
  "Pro plan and up" copy and the Solo pricing card ("your current call" only). Session history views and Save
  Memory transcripts remain all-plans. **(Source of truth = Pro+ as of 2026-07-01 audit — reconciled the prior
  "all plans" claim that contradicted the code gate.)**
- Audio hard cap (3,600 s/month) applies only to echo (Solo).
- Never advertise persona marketplace / priority speed / expanded uploads — no backend, removed Jun 11.

---

## Billing model — how credits are charged (source of truth)
**Credits are 100% METERED on real provider API cost.** No fixed per-action prices, no output length
caps. The only cap is the monthly balance; hitting it shows an upgrade prompt. Never surface fixed
"X credits per action" anywhere.

- **Conversion:** `credits = max(1, ceil(raw_provider_cost_USD / CREDIT_COST_BASIS_USD))`,
  `CREDIT_COST_BASIS_USD = $0.000143` (1 credit ≈ $0.000143 of raw API cost).
- **`/coach`:** metered = LLaMA tokens (`llama-3.1-8b-instant`, $0.05/M in, $0.08/M out) + Whisper STT
  seconds ($0.04/hr ≈ 0.078 cr/audio-sec). `max_tokens` by Length: Quick 80 / Standard 220 / Full 550 —
  a TOKEN BUDGET, not a price.
- **`STYLE_COSTS {Quick:1,Standard:2,Full:4}` is LEGACY** — maps length→max_tokens only. NOT a charge.
  Never surface these numbers as costs.
- **TTS (`/tts`):** metered & charged (since Jun 7). ~$35/1M chars ≈ 0.24 cr/char. TTS DOMINATES audio
  cost (~30/85/175 cr by length). Gated to echo+. Charged post-stream (only audio actually delivered),
  10s idempotency window.
- **`/brief`:** metered & charged (since Jun 7). `llama-3.3-70b` + Jina fetch ≈ 10–35 cr.
  `BRIEFING_TIER_CREDITS` is now only a tier-name validator, NOT the charge. Gated to command+.
- **After-call summary / save memory:** metered 8B call ≈ 1–2 cr.
- **PAYG (opt-in):** when balance is insufficient, charge at **1.5×** and allow negative balance so a user
  is never cut off mid-call. Row locking (`FOR UPDATE`) + idempotency keys on `/coach` and `/brief`;
  logged in `credit_ledger`.
- Per-action reference: text-only response ≈ 1 cr · audio whisper ≈ 37/98/295 cr (Quick/Standard/Full) ·
  1 hr live audio ≈ 3,600 cr central · briefing 10–15 cr · summary 1–2 cr.
- If `PROVIDER_RATES` / `CREDIT_COST_BASIS_USD` change, re-run this math and update `PLAN_CREDITS` + the
  frontend pricing copy together.

## State ↔ field mapping (frontend pills invert the API)
`/coach` form fields: `role` = **Persona**, `mode` = **Style** (DB col `mode`/`persona`, legacy alias),
`style` = **Length** (Quick/Standard/Full). Empty `style`/`mode` default to "Standard" /
Interview-Coach-style defaults (`STYLE_CONFIG.get(style,"Standard")`, `Form("Standard")`). Custom persona/
style = the free text is sent AS the role/mode value (not the literal "Custom").

---

## TTS failure handling (`/tts` + `/health`) — shipped Jun 12, 2026
`StreamingResponse` sends the 200 before the generator runs, so an upstream error caught *inside* the
generator could only yield an empty 200 (silent failure). The fix preflights the Cartesia call and
validates status BEFORE returning `StreamingResponse`:
- `/tts`: sends via `http.send(req, stream=True)`, checks `resp.status_code`; non-200 raises a real
  `HTTPException` (503 `TTS_CAPACITY_EXHAUSTED` for 402/429, 502 otherwise) including the upstream status;
  only streams + charges on a true 200. Frontend does `if(!response.ok) throw` → degrades to text.
- `/health`: `tts` = key present AND no Cartesia upstream failure in last 10 min (was key-presence only);
  adds a `tts_last_error` block when degraded. `cartesia` key unchanged (presence only) so StatusPage keys
  don't break. In-memory tracker → reads `true` right after a restart until the first /tts failure.
- Helpers near `/health`: `_record_tts_failure` / `_clear_tts_failure` / `_tts_recent_failure`,
  global `_tts_last_failure`, `TTS_FAILURE_TTL_S=600`.
- ⚠️ The code fix does NOT restore audio — the live outage is a **Cartesia model-credit exhaustion**
  (HTTP 402). Owner must top up at https://play.cartesia.ai/subscription.

---

## Architecture decisions / lessons (backend)
- `StatusPage` `SERVICE_CHECKS` must only list keys `/health` actually returns (stale keys = dead dots).
- Security headers middleware (X-Content-Type-Options, X-Frame-Options, Referrer-Policy,
  Permissions-Policy) applied AFTER CORS.
- `/brief` validates URL — rejects non-http(s) and private/localhost addresses.
- All emails: `reply_to: support@cerebroecho.com` + CAN-SPAM/GDPR unsubscribe footer.
- `send_downgrade_email` fires on payment-failure grace expiry.
- Stripe prices are IMMUTABLE: to change an amount, create a NEW Price object, repoint the
  `STRIPE_PRICE_*` env var, redeploy, then archive the old price. Changing `PLAN_CREDITS` numbers does
  NOT change what Stripe bills.
- After updating any critical env var, **redeploy** the service to pick it up.
- DB-driver note (Windows dev box): needs `pip install psycopg2-binary sqlalchemy python-dotenv`.

## Telephony (Twilio Voice) — STAGED, NOT DEPLOYED (2026-06-15)
- `telephony.py` (committed `30670e3` on `master`, **not pushed**) is mounted in `main.py` via
  `build_telephony_router(...)`, wrapped so it can't block boot. **Inert** unless `TWILIO_*` env vars are set.
- Provides `/voice/token|outbound|inbound|status` (calls + per-minute credit metering via `_deduct_credits`,
  idempotent on CallSid) and `/voice/port/request|status` (port-in; carrier PIN/acct# pass-through, NOT stored).
  Tables `phone_numbers` + `port_requests` auto-create on startup.
- `twilio` added to requirements. Go-live steps: `../TELEPHONY_GOLIVE.md`. Creds: `../.secrets` (Twilio section).
- Do NOT claim calling works until the runbook is done + a real call tested.

## Pending backend items
- ✅ **DONE (verified 2026-06-15):** Railway `STRIPE_PRICE_PRO` = `price_1ThJAkEl32Rmtak4GUv2jcb7` ($29.99),
  confirmed live via `railway variables`. Do NOT re-flag as pending.
- Tear down test account `fable-test@cerebroecho.test` when QA done
  (`railway run python ../scripts/fable-test/delete_test_pass.py`).
