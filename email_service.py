"""
Transactional email service for CerebroEcho.

Provider: Resend (https://resend.com)
To swap provider: replace _send_via_resend() only.
send_email() contract stays identical — no other file needs to change.

Required env vars:
  RESEND_API_KEY        — Resend API key (re_...)
  EMAIL_FROM_ADDRESS    — Sender identity, e.g. "CerebroEcho <hello@cerebroecho.com>"
  EMAIL_ENABLED         — Set to "false" to suppress all sends (dev/test)
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_APP_URL  = "https://cerebroecho.com/app"
_FROM_DEFAULT = "CerebroEcho <hello@cerebroecho.com>"


# ═══════════════════════════════════════════════════════════════════════════════
# BASE HTML WRAPPER
# Table-based layout for broad email client compatibility.
# ═══════════════════════════════════════════════════════════════════════════════

def _html_wrap(heading: str, body_html: str, cta_label: str = "", cta_url: str = "") -> str:
    cta_block = ""
    if cta_label and cta_url:
        cta_block = f"""
            <tr><td style="padding-top:28px;">
              <a href="{cta_url}"
                 style="display:inline-block;background:#7DE7FF;color:#08090b;
                        font-weight:700;font-size:14px;padding:13px 28px;
                        border-radius:8px;text-decoration:none;letter-spacing:0.01em;">
                {cta_label}
              </a>
            </td></tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>{heading}</title>
</head>
<body style="margin:0;padding:0;background:#08090b;
             font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
             -webkit-font-smoothing:antialiased;">
  <table width="100%" cellpadding="0" cellspacing="0" role="presentation"
         style="background:#08090b;padding:48px 20px;">
    <tr><td align="center">
      <table width="100%" cellpadding="0" cellspacing="0" role="presentation"
             style="max-width:520px;">

        <!-- Wordmark -->
        <tr><td style="padding-bottom:28px;text-align:left;">
          <span style="font-size:20px;font-weight:800;color:#E8EDF2;letter-spacing:-0.03em;">Cerebro</span><span
                style="font-size:20px;font-weight:800;color:#7DE7FF;letter-spacing:-0.03em;">Echo</span>
        </td></tr>

        <!-- Content card -->
        <tr><td style="background:#0d1014;border:1px solid #1e2530;border-radius:14px;padding:36px;">
          <table width="100%" cellpadding="0" cellspacing="0" role="presentation">
            <tr><td style="padding-bottom:18px;">
              <h1 style="margin:0;font-size:22px;font-weight:700;color:#E8EDF2;line-height:1.3;">
                {heading}
              </h1>
            </td></tr>
            <tr><td>
              {body_html}
            </td></tr>
            {cta_block}
          </table>
        </td></tr>

        <!-- Footer -->
        <tr><td style="padding-top:28px;text-align:center;">
          <p style="margin:0;font-size:11px;color:#596272;line-height:1.7;">
            CerebroEcho &nbsp;&middot;&nbsp;
            <a href="https://cerebroecho.com" style="color:#596272;text-decoration:underline;">cerebroecho.com</a>
          </p>
          <p style="margin:5px 0 0;font-size:11px;color:#596272;">
            You're receiving this because you have an account at CerebroEcho.
          </p>
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER: RESEND
# Replace this function to switch providers. Nothing else needs to change.
# ═══════════════════════════════════════════════════════════════════════════════

def _send_via_resend(to: str, subject: str, html: str, text: str) -> bool:
    api_key = os.environ.get("RESEND_API_KEY", "")
    if not api_key:
        logger.warning("⚠️ RESEND_API_KEY not set — email suppressed")
        return False
    try:
        import resend  # lazy import: missing package only fails on first send attempt
        resend.api_key = api_key
        result = resend.Emails.send({
            "from":    os.environ.get("EMAIL_FROM_ADDRESS", _FROM_DEFAULT),
            "to":      [to],
            "subject": subject,
            "html":    html,
            "text":    text,
        })
        email_id = result.get("id", "unknown") if isinstance(result, dict) else getattr(result, "id", "unknown")
        logger.info(f"✅ Email sent via Resend: id={email_id} to={to!r} subject={subject!r}")
        return True
    except ImportError:
        logger.error("❌ resend package not installed — run: pip install 'resend>=2.0.0'")
        return False
    except Exception as e:
        logger.error(f"❌ Resend error (to={to!r} subject={subject!r}): {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def send_email(to: str, subject: str, html: str, text: str) -> bool:
    """
    Send a transactional email. Never raises — logs all failures.
    Returns True on success, False on suppression or delivery failure.
    """
    enabled = os.environ.get("EMAIL_ENABLED", "true").strip().lower()
    if enabled not in ("1", "true", "yes"):
        logger.info(f"📧 Email suppressed (EMAIL_ENABLED={enabled!r}): {subject!r} → {to!r}")
        return False

    if not to or to.strip().lower() == "anonymous" or "@" not in to:
        logger.info(f"📧 Email skipped — no valid address ({to!r}): {subject!r}")
        return False

    return _send_via_resend(to, subject, html, text)


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

def send_welcome_email(to: str) -> bool:
    """
    Sent when a user first saves their email (free account created).
    Trigger: POST /save_email
    Status: FULLY WIRED
    """
    subject = "Welcome to CerebroEcho"
    body_html = """
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        You're in. CerebroEcho is your real-time conversation intelligence layer —
        built to help you think faster, respond sharper, and stay a step ahead in
        interviews, sales calls, negotiations, and high-stakes conversations.
      </p>
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        Your free account includes
        <span style="color:#E8EDF2;font-weight:600;">30 credits</span> to get started.
        Pick a role, choose a mode, and start your first session.
      </p>
      <p style="margin:0;font-size:13px;line-height:1.6;color:#596272;">
        Need more credits or audio whisper? Upgrade anytime from inside the app.
      </p>
    """
    text = (
        "Welcome to CerebroEcho.\n\n"
        "Your free account includes 30 credits. Pick a role, choose a mode, "
        "and start your first session at cerebroecho.com/app.\n\n"
        "Need more credits or audio whisper? Upgrade anytime from inside the app.\n\n"
        "— CerebroEcho"
    )
    return send_email(to, subject, _html_wrap(subject, body_html, "Open the App", _APP_URL), text)


def send_upgrade_email(to: str, plan: str, balance: int) -> bool:
    """
    Sent when a subscription checkout completes.
    Trigger: checkout.session.completed webhook
    Status: FULLY WIRED
    """
    plan_display = plan.replace("_", " ").title()
    subject = f"You're on CerebroEcho {plan_display}"
    body_html = f"""
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        Your <span style="color:#7DE7FF;font-weight:600;">{plan_display}</span> subscription
        is active. You have
        <span style="color:#E8EDF2;font-weight:600;">{balance:,} credits</span>
        ready to use this month.
      </p>
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        Credits reset at the start of each billing cycle. Use them for live sessions,
        URL briefings, and audio whisper.
      </p>
      <p style="margin:0;font-size:13px;line-height:1.6;color:#596272;">
        Manage your subscription anytime — open the app sidebar and click
        <strong style="color:#9ba4b0;">Manage plan</strong>.
      </p>
    """
    text = (
        f"Your {plan_display} subscription is active.\n\n"
        f"You have {balance:,} credits ready to use this month. "
        "Credits reset at the start of each billing cycle.\n\n"
        "Manage your subscription at cerebroecho.com/app (sidebar → Manage plan).\n\n"
        "— CerebroEcho"
    )
    return send_email(to, subject, _html_wrap(subject, body_html, "Go to App", _APP_URL), text)


def send_payment_failed_email(to: str, attempt: int, next_retry: str, amount_due: int) -> bool:
    """
    Sent on each failed subscription renewal payment. Escalates on attempt 2+.
    Trigger: invoice.payment_failed webhook
    Status: FULLY WIRED
    """
    is_urgent   = attempt > 1
    subject     = "Urgent — payment still failing" if is_urgent else "Action required — payment failed"
    amount_str  = f"${amount_due / 100:.2f}" if amount_due else "your subscription amount"
    has_retry   = next_retry and next_retry != "no further retries"
    retry_line  = f"Stripe will retry automatically on {next_retry}." if has_retry else \
                  "There are no further automatic retries scheduled."

    urgency_block = ""
    if is_urgent:
        urgency_block = (
            f'<p style="margin:0 0 16px;font-size:14px;line-height:1.6;'
            f'color:#f87171;font-weight:600;">'
            f'This is attempt {attempt}. Your access will be suspended '
            f'if payment continues to fail.</p>'
        )

    body_html = f"""
      {urgency_block}
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        We couldn't collect
        <span style="color:#E8EDF2;font-weight:600;">{amount_str}</span>
        for your CerebroEcho subscription. {retry_line}
      </p>
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        Update your payment method to keep uninterrupted access to your plan.
      </p>
      <p style="margin:0;font-size:13px;line-height:1.6;color:#596272;">
        Questions? Reply to this email or contact
        <a href="mailto:support@cerebroecho.com" style="color:#596272;">support@cerebroecho.com</a>.
      </p>
    """
    text = (
        f"Action required — we couldn't collect {amount_str} for your CerebroEcho subscription.\n\n"
        f"{retry_line}\n\n"
        "Update your payment method at cerebroecho.com/app (sidebar → Manage plan).\n\n"
        "Questions? Contact support@cerebroecho.com\n\n"
        "— CerebroEcho"
    )
    return send_email(to, subject, _html_wrap(subject, body_html, "Update Payment Method", _APP_URL), text)


def send_magic_link_email(to: str, magic_url: str) -> bool:
    """
    Sent when a user requests a magic sign-in link.
    Trigger: POST /auth/request-link
    Status: FULLY WIRED
    """
    subject = "Sign in to CerebroEcho"
    body_html = f"""
      <p style="margin:0 0 16px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        Click the button below to sign in. This link expires in 30 minutes and
        can only be used once.
      </p>
      <p style="margin:0 0 16px;font-size:13px;line-height:1.6;color:#596272;">
        If you didn't request this, you can safely ignore this email.
      </p>
    """
    text = (
        "Sign in to CerebroEcho.\n\n"
        f"Click here to sign in: {magic_url}\n\n"
        "This link expires in 30 minutes and can only be used once.\n\n"
        "If you didn't request this, ignore this email.\n\n"
        "— CerebroEcho"
    )
    return send_email(to, subject, _html_wrap(subject, body_html, "Sign in", magic_url), text)


def send_otp_email(to: str, code: str) -> bool:
    """
    Sent when a user requests an OTP sign-in code.
    Trigger: POST /auth/request-otp
    Status: FULLY WIRED
    """
    subject = f"{code} — your CerebroEcho sign-in code"
    body_html = f"""
      <p style="margin:0 0 8px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        Your sign-in code is:
      </p>
      <p style="margin:0 0 24px;font-size:40px;font-weight:700;letter-spacing:0.15em;color:#e2e8f0;font-family:monospace;">
        {code}
      </p>
      <p style="margin:0 0 16px;font-size:13px;line-height:1.6;color:#596272;">
        Enter this code in the app. It expires in 10 minutes and can only be used once.
      </p>
      <p style="margin:0 0 16px;font-size:13px;line-height:1.6;color:#596272;">
        If you didn't request this, you can safely ignore this email.
      </p>
    """
    text = (
        f"Your CerebroEcho sign-in code is: {code}\n\n"
        "Enter it in the app within 10 minutes.\n\n"
        "If you didn't request this, ignore this email.\n\n"
        "— CerebroEcho"
    )
    return send_email(to, subject, _html_wrap(subject, body_html), text)


def send_email_change_request(old_email: str, new_email: str, verify_url: str) -> bool:
    """
    Sent to NEW email asking the user to confirm the address change.
    Trigger: POST /auth/request-email-change
    Status: FULLY WIRED
    """
    subject = "Verify your new CerebroEcho email"
    body_html = f"""
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        A request was made to change the email address on a CerebroEcho account from
        <span style="color:#E8EDF2;font-weight:600;">{old_email}</span>
        to this address.
      </p>
      <p style="margin:0 0 16px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        Click the button below to confirm this new email. The link expires in 30&nbsp;minutes.
      </p>
      <p style="margin:0;font-size:13px;line-height:1.6;color:#596272;">
        If you didn&apos;t request this, you can safely ignore this email.
      </p>
    """
    text = (
        f"A request was made to change a CerebroEcho account email from {old_email} to {new_email}.\n\n"
        f"Confirm here: {verify_url}\n\n"
        "Link expires in 30 minutes.\n\n"
        "If you didn't request this, ignore this email.\n\n"
        "— CerebroEcho"
    )
    return send_email(new_email, subject, _html_wrap(subject, body_html, "Confirm new email", verify_url), text)


def send_email_change_notification(old_email: str, new_email: str) -> bool:
    """
    Security alert sent to the OLD email when a change is requested.
    Trigger: POST /auth/request-email-change
    Status: FULLY WIRED
    """
    subject = "CerebroEcho email change requested"
    body_html = f"""
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        A request was made to change the email address on your CerebroEcho account to
        <span style="color:#E8EDF2;font-weight:600;">{new_email}</span>.
      </p>
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        A verification link has been sent to that new address. Your current email
        remains active until the change is confirmed.
      </p>
      <p style="margin:0;font-size:13px;line-height:1.6;color:#596272;">
        If you did <strong style="color:#f87171;">not</strong> request this, contact
        <a href="mailto:support@cerebroecho.com" style="color:#596272;">support@cerebroecho.com</a> immediately.
      </p>
    """
    text = (
        f"A request was made to change your CerebroEcho email to {new_email}.\n\n"
        "Check your new inbox for a verification link. Your current email stays active "
        "until the change is confirmed.\n\n"
        "If you didn't request this, contact support@cerebroecho.com immediately.\n\n"
        "— CerebroEcho"
    )
    return send_email(old_email, subject, _html_wrap(subject, body_html), text)


def send_email_changed_confirmation(new_email: str, old_email: str) -> bool:
    """
    Confirmation sent to NEW email after a successful email change.
    Trigger: POST /auth/verify-email-change
    Status: FULLY WIRED
    """
    subject = "CerebroEcho email updated"
    body_html = f"""
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        Your CerebroEcho email has been updated to
        <span style="color:#7DE7FF;font-weight:600;">{new_email}</span>.
      </p>
      <p style="margin:0;font-size:13px;line-height:1.6;color:#596272;">
        The previous address ({old_email}) is no longer associated with your account.
        Questions? Contact
        <a href="mailto:support@cerebroecho.com" style="color:#596272;">support@cerebroecho.com</a>.
      </p>
    """
    text = (
        f"Your CerebroEcho email has been updated to {new_email}.\n\n"
        f"The previous address ({old_email}) is no longer associated with your account.\n\n"
        "Questions? Contact support@cerebroecho.com\n\n"
        "— CerebroEcho"
    )
    return send_email(new_email, subject, _html_wrap(subject, body_html, "Open the App", _APP_URL), text)


def send_onboarding_welcome(to: str) -> bool:
    """
    Email 1 of 3 in the onboarding sequence.
    Sent immediately after first magic link or OTP verification.
    Trigger: auth_verify_link / auth_verify_otp (first login only)
    Status: FULLY WIRED
    """
    subject = "You're in. Here's how to hear it work."
    body_html = """
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        Welcome to CerebroEcho.
      </p>
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        The fastest way to understand what this does is to hear it.
      </p>
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        Open the app, pick a scenario, share your meeting tab when prompted,
        and start a real session on your next call.
      </p>
      <p style="margin:0 0 20px;font-size:15px;line-height:1.7;color:#E8EDF2;font-weight:600;">
        Your first session is on us.
      </p>
      <p style="margin:0 0 14px;font-size:14px;line-height:1.7;color:#596272;">
        One thing worth knowing: the suggestion appears in about 2&nbsp;seconds.
        That is not a coincidence. That is the point.
      </p>
      <p style="margin:0;font-size:13px;line-height:1.6;color:#596272;">
        — The CerebroEcho team
      </p>
    """
    text = (
        "Welcome to CerebroEcho.\n\n"
        "The fastest way to understand what this does is to hear it.\n\n"
        "Open the app, pick a scenario, share your meeting tab when prompted, "
        "and start a real session on your next call.\n\n"
        "Your first session is on us.\n\n"
        "cerebroecho.com/app\n\n"
        "One thing worth knowing: the suggestion appears in about 2 seconds. "
        "That is not a coincidence. That is the point.\n\n"
        "— The CerebroEcho team"
    )
    return send_email(to, subject, _html_wrap(subject, body_html, "Open CerebroEcho →", _APP_URL), text)


def send_onboarding_setup(to: str) -> bool:
    """
    Email 2 of 3 in the onboarding sequence.
    Sent 24 hours after signup if session count = 0.
    Trigger: _process_onboarding_emails() background worker
    Status: FULLY WIRED
    """
    subject = "Haven't tried it yet? Here's the 3-minute setup."
    body_html = """
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        Setting up CerebroEcho takes about 3 minutes.
        Here&#39;s exactly what to do:
      </p>
      <ol style="margin:0 0 20px;padding-left:20px;font-size:15px;line-height:2.2;color:#9ba4b0;">
        <li>Open Chrome, Edge, or Firefox on your desktop</li>
        <li>Go to <span style="color:#7DE7FF;">cerebroecho.com/app</span></li>
        <li>Click <span style="color:#E8EDF2;font-weight:600;">Start</span></li>
        <li>When prompted, click <span style="color:#E8EDF2;font-weight:600;">Share Tab</span>
            and select your meeting tab</li>
        <li>That&#39;s it — suggestions appear as the conversation happens</li>
      </ol>
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        No downloads. No installs.
        Just a tab share at the start of each call.
      </p>
      <p style="margin:0;font-size:13px;line-height:1.6;color:#596272;">
        If you run into anything reply to this email.
      </p>
    """
    text = (
        "Setting up CerebroEcho takes about 3 minutes. Here's exactly what to do:\n\n"
        "1. Open Chrome, Edge, or Firefox on your desktop\n"
        "2. Go to cerebroecho.com/app\n"
        "3. Click Start\n"
        "4. When prompted, click Share Tab and select your meeting tab\n"
        "5. That's it — suggestions appear as the conversation happens\n\n"
        "No downloads. No installs. Just a tab share at the start of each call.\n\n"
        "cerebroecho.com/app\n\n"
        "If you run into anything reply to this email.\n\n"
        "— CerebroEcho"
    )
    return send_email(to, subject, _html_wrap(subject, body_html, "Set up in 3 minutes →", _APP_URL), text)


def send_onboarding_checkin(to: str) -> bool:
    """
    Email 3 of 3 in the onboarding sequence.
    Sent 72 hours after signup if session count < 3.
    Trigger: _process_onboarding_emails() background worker
    Status: FULLY WIRED
    """
    subject = "Did it land when you needed it?"
    body_html = """
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        The best way to know if CerebroEcho works for you is one real call.
      </p>
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        Not a test. Not a demo. A call where something is actually at stake.
      </p>
      <p style="margin:0 0 20px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        If you have one coming up — try it.<br>
        If you haven&#39;t set it up yet — it takes 3 minutes.
      </p>
      <p style="margin:0;font-size:13px;line-height:1.6;color:#596272;">
        If something didn&#39;t work or felt off, reply here.
        We read every one.
      </p>
    """
    text = (
        "The best way to know if CerebroEcho works for you is one real call.\n\n"
        "Not a test. Not a demo. A call where something is actually at stake.\n\n"
        "If you have one coming up — try it.\n"
        "If you haven't set it up yet — it takes 3 minutes.\n\n"
        "cerebroecho.com/app\n\n"
        "If something didn't work or felt off, reply here. We read every one.\n\n"
        "— CerebroEcho"
    )
    return send_email(to, subject, _html_wrap(subject, body_html, "cerebroecho.com/app →", _APP_URL), text)


def send_cancellation_email(to: str, ended_at: str) -> bool:
    """
    Sent when a subscription is fully cancelled (period ended or immediate cancel).
    Trigger: customer.subscription.deleted webhook
    Status: FULLY WIRED
    """
    # Format the ISO timestamp into a human-readable date
    try:
        dt = datetime.fromisoformat(ended_at)
        ended_display = dt.strftime("%B %d, %Y")
    except Exception:
        ended_display = ended_at

    subject = "Your CerebroEcho subscription has ended"
    body_html = f"""
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        Your CerebroEcho subscription ended on
        <span style="color:#E8EDF2;font-weight:600;">{ended_display}</span>.
        Your account has been moved to the free plan.
      </p>
      <p style="margin:0 0 14px;font-size:15px;line-height:1.7;color:#9ba4b0;">
        Your session history and preferences are still saved.
        You can re-subscribe anytime and pick up right where you left off.
      </p>
      <p style="margin:0;font-size:13px;line-height:1.6;color:#596272;">
        We're sorry to see you go. If there's anything we could have done better,
        just reply to this email — we read every response.
      </p>
    """
    text = (
        f"Your CerebroEcho subscription ended on {ended_display}. "
        "Your account has been moved to the free plan.\n\n"
        "Your session history and preferences are still saved. "
        "Re-subscribe anytime at cerebroecho.com/app.\n\n"
        "We're sorry to see you go. Reply to this email with any feedback.\n\n"
        "— CerebroEcho"
    )
    return send_email(to, subject, _html_wrap(subject, body_html, "Re-subscribe", _APP_URL), text)
