# main.py - Interview Whisperer Backend v1.0
import os
import time
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
APP_SECRET = os.getenv("APP_SECRET", "default-secret-change-this")

# Validate required env vars
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY environment variable is not set")

# Initialize FastAPI
app = FastAPI(
    title="Interview Whisperer API",
    description="Real-time AI interview coaching powered by Groq",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    user_id: str = "anonymous"
    mode: str = "fast"  # fast or practice

class AnswerResponse(BaseModel):
    answer: str
    model: str
    processing_time: float
    timestamp: float

# ================== ENDPOINTS ==================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "Interview Whisperer API",
        "status": "online",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "API is online and ready for production",
        "groq_configured": True
    }

@app.post("/api/answer", response_model=AnswerResponse)
async def get_answer(
    request: QuestionRequest,
    x_api_key: str = Header(None)
):
    """
    Main endpoint: Receives interview question, returns strategic answer
    
    Headers required:
        x-api-key: Your secret API key
    
    Body:
        question: The interview question (string)
        user_id: User identifier (optional, default: "anonymous")
        mode: "fast" or "practice" (optional, default: "fast")
    """
    
    # Authentication check
    if x_api_key != APP_SECRET:
        raise HTTPException(
            status_code=403,
            detail="Unauthorized: Invalid API key"
        )
    
    # Validate input
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        start_time = time.time()
        
        # Configure prompt based on mode
        if request.mode == "practice":
            system_prompt = """You are a detailed interview coach in PRACTICE mode.

Provide a comprehensive answer with:
1. Framework or strategic approach
2. Key points (2-3 bullets)
3. Brief example if relevant

Be thorough but structured. Max 150 words."""
            max_tokens = 300
            
        else:  # fast mode (default)
            system_prompt = """You are an elite interview coach for McKinsey, Goldman Sachs, and Google candidates.

Provide a CONCISE, strategic answer in 1-2 sentences.
Focus on frameworks, key insights, or memory triggers - NOT full speeches.
The candidate needs to remember this quickly under pressure.
Be confident and direct."""
            max_tokens = 100
        
        # Call Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Interview question: {request.question}"
                }
            ],
            # 💥 THE FINAL, STABLE MODEL FIX IS HERE 💥
            model="llama-3.1-8b-instant",  # The currently stable and fast model
            temperature=0.6,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        # Extract answer
        answer_text = chat_completion.choices[0].message.content.strip()
        processing_time = time.time() - start_time
        
        # Log for debugging (visible in Railway logs)
        print(f"[{time.time()}] User: {request.user_id} | Q: {request.question[:50]}... | Time: {processing_time:.3f}s")
        
        # Return response
        return AnswerResponse(
            answer=answer_text,
            model="groq-llama-3.1-8b",
            processing_time=round(processing_time, 3),
            timestamp=time.time()
        )
        
    except Exception as e:
        # Log error
        print(f"ERROR in /api/answer: {str(e)}")
        
        # Return 500 error
        raise HTTPException(
            status_code=500,
            detail=f"AI processing failed: {str(e)}"
        )

# Optional: Local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)