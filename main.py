# main.py

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# 1. Load Environment Variables
# This loads GROQ_API_KEY and APP_SECRET from the .env file (locally)
# Railway will use the environment variables set in the dashboard instead.
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2. Initialize FastAPI App
app = FastAPI(
    title="Interview Whisperer Backend API",
    description="Real-time AI Interview Feedback powered by Groq",
    version="1.0.0",
)

# 3. Configure CORS (Critical for allowing your local client script to connect)
# In production, you would restrict this to your specific client origin.
origins = [
    "*", # Allow all origins for initial testing (replace with specific domain later)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Initialize Groq Client
if not GROQ_API_KEY:
    # If key is missing, crash immediately with an error that is visible in logs
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# This is the line that required the dependency update, now fixed!
groq_client = Groq(api_key=GROQ_API_KEY)


# --- API Endpoints ---

# 5. Health Check Endpoint (The fix for the 404 error)
@app.get("/health")
def health_check():
    """Returns a simple status to confirm the API is running."""
    return {"status": "ok", "message": "API is online and ready for production"}


# Data model for the incoming request from the local client script
class WhisperRequest(BaseModel):
    transcript_chunk: str

# 6. Main Logic Endpoint (Placeholder for your Groq logic)
@app.post("/api/whisper")
async def process_transcript(data: WhisperRequest):
    """
    Receives a chunk of transcribed text and sends it to Groq for analysis.
    """
    if not data.transcript_chunk:
        raise HTTPException(status_code=400, detail="Transcript chunk is empty")
    
    # --- Groq Integration Logic Goes Here ---
    
    # Example: Send the chunk to Groq for analysis
    # This is placeholder logic; the actual implementation might involve streaming or state management.
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a real-time interview coach. Analyze the user's latest spoken sentence for clarity and content. Keep your response very short, no more than 10 words.",
                },
                {
                    "role": "user",
                    "content": data.transcript_chunk,
                }
            ],
            model="mixtral-8x7b-32768", # Fast model choice
        )
        
        feedback = chat_completion.choices[0].message.content
        
        return {"feedback": feedback, "chunk_received": data.transcript_chunk}

    except Exception as e:
        print(f"Groq API Error: {e}")
        # Return a 500 error but keep the server running
        raise HTTPException(status_code=500, detail="Internal Groq API Error")

# Optional: Run the app locally for testing (only runs if executed directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)