# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (dev)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run directly
python main.py
```

There are no tests or linters configured in this project.

## Required Environment Variable

```
GROQ_API_KEY=<your-groq-api-key>
```

Load via a `.env` file (python-dotenv is already wired in). The app starts without it but returns `503` on any AI call.

## Architecture

The entire backend is a single file: `main.py`. There are no modules, packages, or subdirectories.

**Stack:** FastAPI + Uvicorn (ASGI), Groq SDK for both speech-to-text and LLM inference.

**Deployment:** Heroku via `Procfile` (`web: uvicorn main:app --host 0.0.0.0 --port $PORT`). Python version pinned in `runtime.txt` (3.11).

## Groq Integration

Two Groq capabilities used:

| Capability | Model |
|---|---|
| Audio transcription | `whisper-large-v3-turbo` |
| Chat completion | `llama-3.1-8b-instant` |

The `client` is initialized once at module load (lines 26–36). If `GROQ_API_KEY` is missing or init fails, `client` is `None` and all `/process_audio` calls return `503`.

**Audio upload quirk:** The transcription call passes a 3-tuple `(filename, bytes, "audio/webm")` to the Groq SDK — the explicit MIME type is required for correct transcription. Audio files smaller than 1000 bytes are silently rejected (treated as noise/empty clicks) and return `{"answer": "Listening..."}` without consuming a quota slot.

The `/process_audio` endpoint accepts the audio upload under either the `audio` or `file` form field name (both are checked, `audio` takes precedence).

## Usage Tracking

`usage_tracker` is an in-memory dict keyed by `"{deviceId}_{userEmail}"`. Limits:

- `anonymous` users: 5 questions
- Registered (email provided) users: 10 questions

**Important:** This state is lost on every server restart — there is no persistent storage.

`/save_email` migrates the anonymous key to the email key when a user registers, preserving their existing count.

## Response Styles

`/process_audio` accepts a `style` form field that controls the LLM prompt and token budget:

| `style` | Prompt instruction | `max_tokens` |
|---|---|---|
| `shorthand` | One hint/framework name, ≤10 words | 50 |
| `bullet` | 3 tactical bullet points, ≤100 words | 200 |
| `script` (default) | 2–3 sentence tactical answer | 150 |

## API Surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Liveness check |
| GET | `/health` | Returns Groq config status |
| GET | `/founder_spots` | Returns random int 12–47 (mock data) |
| POST | `/process_audio` | Transcribe audio + generate coaching answer |
| POST | `/save_email` | Register email and migrate usage quota |

CORS is open (`allow_origins=["*"]`).
