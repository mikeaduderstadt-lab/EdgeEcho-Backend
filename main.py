from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import time

app = FastAPI(title="EdgeEcho API") 

# --- 1. CORS Setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Groq Initialization ---
# Railway will automatically load the GROQ_API_KEY from the Variables tab.
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
groq_client = Groq(api_key=GROQ_API_KEY)

# This secret key protects your endpoint (Make sure this matches your client code!)
APP_SECRET = "your-secret-key-2024" 

# --- 3. Data Models ---
class QuestionRequest(BaseModel):
    question: str
    user_id: str = "beta"
    
class AnswerResponse(BaseModel):
    answer: str
    model: str
    processing_time: float
    source: str = "Groq"

# --- 4. Main AI Endpoint: /api/answer ---
@app.post("/api/answer", response_model=AnswerResponse)
async def get_answer(
    request: QuestionRequest,
    x_api_key: str = Header(None)
):
    # Security check: Ensure the local client has the correct secret
    if x_api_key != APP_SECRET:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    try:
        start_time = time.time()
        
        # Call Groq with ELITE speed settings (llama3-70b-8192 is the fastest elite model)
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192", 
            messages=[
                {
                    "role": "system",
                    "content": """You are an elite interview coach for high-stakes candidates (McKinsey, Google, etc.).
                    
Provide a concise, strategic answer in 1-2 sentences. Be confident and direct.
Focus on frameworks, key insights, or memory triggers - NOT full speeches.
The candidate needs to remember this quickly under pressure."""
                },
                {
                    "role": "user",
                    "content": f"Interview question: {request.question}"
                }
            ],
            temperature=0.6,
            max_tokens=100, 
            top_p=0.9
        )
        
        processing_time = time.time() - start_time
        answer_text = response.choices[0].message.content
        
        return AnswerResponse(
            answer=answer_text,
            model="groq-llama3-70b",
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"AI processing failed: {str(e)}"
        )