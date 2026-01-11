import os
import tempfile
import time
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
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
    limit = 10 if userEmail != "anonymous" else 5
    
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
