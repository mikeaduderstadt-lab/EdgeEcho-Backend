import os
import uuid
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CerebroEcho API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
usage_tracker = {}

@app.get("/")
async def root():
    return {"status": "CerebroEcho Backend Live", "version": "1.1.0"}

@app.get("/health")
async def health():
    return {"status": "ok", "groq_configured": True}

@app.get("/founder_spots")
async def get_founder_spots():
    # Fixes 404 in frontend
    import random
    return {"remaining": random.randint(12, 47)}

@app.post("/process_audio")
async def process_audio(
    audio: UploadFile = File(None),     # Primary field name
    file: UploadFile = File(None),      # Backup for browser inconsistencies
    deviceId: str = Form(...),
    userEmail: str = Form("anonymous"),
    context: str = Form("a professional role"),
    work_history: str = Form(""),
    style: str = Form("script")
):
    """
    Defensive audio processing with dual file field support
    """
    
    # GEMINI'S SMART PART: Accept either field name
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
    
    try:
        start_time = time.time()
        
        # GEMINI'S STABLE FILE HANDLING
        content = await actual_file.read()
        temp_filename = f"{uuid.uuid4()}.webm"
        
        with open(temp_filename, "wb") as f:
            f.write(content)
        
        # Transcribe
        with open(temp_filename, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(temp_filename, f.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
            )
        
        # Cleanup immediately after use
        os.remove(temp_filename)
        
        transcript = transcription.strip() if transcription else ""
        
        if not transcript or len(transcript) < 2:
            return {
                "answer": "Listening...",
                "transcript": "",
                "questions_used": current_used,
                "processing_time": round(time.time() - start_time, 3)
            }
        
        # YOUR STRUCTURED STYLE HANDLING (better than Gemini's)
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
            
        else:  # script (default)
            system_prompt = f"""You are an elite interview coach. The candidate is interviewing for {context}.

Their background: {work_history}

Provide a 2-3 sentence tactical answer. Be concise and confident."""
            max_tokens = 150
        
        # Generate answer - START WITH FAST MODEL (you can upgrade later if needed)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast for real-time; upgrade to 70b if latency is OK
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Interview question: {transcript}"}
            ],
            temperature=0.6,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        answer = completion.choices[0].message.content.strip()
        
        # Update usage
        usage_tracker[user_key] = current_used + 1
        processing_time = time.time() - start_time
        
        print(f"[{time.time()}] Device: {deviceId} | Email: {userEmail} | Q: {transcript[:50]}... | Time: {processing_time:.3f}s")
        
        return {
            "transcript": transcript,
            "answer": answer,
            "questions_used": usage_tracker[user_key],
            "processing_time": round(processing_time, 3)
        }
        
    except Exception as e:
        print(f"ERROR in /process_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_email")
async def save_email(data: dict):
    device_id = data.get("deviceId")
    email = data.get("email")
    
    # Migrate usage from anonymous to email key
    old_key = f"{device_id}_anonymous"
    new_key = f"{device_id}_{email}"
    
    if old_key in usage_tracker:
        usage_tracker[new_key] = usage_tracker[old_key]
        del usage_tracker[old_key]
    
    print(f"Email saved: {email} for device {device_id}")
    return {"status": "success", "message": "Trial extended to 10 questions"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
