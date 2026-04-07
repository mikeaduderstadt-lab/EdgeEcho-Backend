import io
import os
import tempfile
import time
import logging
import urllib.parse
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from deepgram import DeepgramClient, PrerecordedOptions
from fastapi.responses import StreamingResponse
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
    allow_origins=["*"],
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

usage_tracker = {}

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
    import random
    return {"remaining": random.randint(12, 47)}

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
    
    # Usage tracking
    user_key = f"{deviceId}_{userEmail}"
    current_used = usage_tracker.get(user_key, 0)
    limit = 999  # TODO: restore to (10 if userEmail != "anonymous" else 5) before launch
    
    if current_used >= limit:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "TRIAL_LIMIT_REACHED",
                "message": f"Trial limit reached ({limit} questions)"
            }
        )
    
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
        
        # Save to temp file with .webm extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            temp_file.write(content)
            temp_filename = temp_file.name
        
        # THE FIX: Add "audio/webm" as the 3rd parameter in the tuple
        with open(temp_filename, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(temp_filename, audio_file.read(), "audio/webm"),  # ✅ MIME type added
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

    user_key = f"{deviceId}_{userEmail}"
    current_used = usage_tracker.get(user_key, 0)
    limit = 999
    if current_used >= limit:
        raise HTTPException(status_code=403, detail={"code": "TRIAL_LIMIT_REACHED", "message": f"Trial limit reached ({limit} questions)"})

    temp_filename = None
    try:
        start_time = time.time()
        content = await actual_file.read()

        if len(content) < 1000:
            raise HTTPException(status_code=400, detail="Audio too short")

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
    voice = data.get("voice", "onyx")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    if openai_client is None:
        raise HTTPException(status_code=503, detail="TTS service unavailable: OPENAI_API_KEY not set")
    try:
        logger.info(f"TTS streaming started (OpenAI tts-1, voice={voice}) for: {text[:50]}...")

        def generate():
            with openai_client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice=voice,
                input=text[:300],
            ) as response:
                for chunk in response.iter_bytes(chunk_size=4096):
                    yield chunk

        return StreamingResponse(generate(), media_type="audio/mpeg")
    except Exception as e:
        import traceback
        logger.error(f"TTS error: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"TTS error: {type(e).__name__}: {str(e)}")

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