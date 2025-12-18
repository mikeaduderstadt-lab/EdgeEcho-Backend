import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # This allows your frontend to talk to this backend

# 1. THE "HOME" ROUTE - Stops the "Not Found" error
@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "EdgeEcho Backend is Live",
        "version": "1.0.1"
    })

# 2. THE HEALTH CHECK - Confirms your API Key is working
@app.route('/health')
def health():
    api_key = os.environ.get("GROQ_API_KEY")
    return jsonify({
        "status": "ok",
        "groq_configured": bool(api_key),
        "runtime": "Flask on Railway"
    })

# 3. THE TACTICAL ENGINE - Processes your AI questions
@app.route('/process', methods=['POST'])
def process_question():
    data = request.json
    user_query = data.get("question")
    
    if not user_query:
        return jsonify({"error": "No question provided"}), 400

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "API Key missing on server"}), 500

    # Talking to Groq Llama 3.1
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a tactical advisor. Provide concise, actionable strategy."},
            {"role": "user", "content": user_query}
        ]
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 4. THE RAILWAY PORT FIX - Stops the crash loop
if __name__ == "__main__":
    # Railway assigns a port dynamically via environment variables
    port = int(os.environ.get("PORT", 8080))
    # Must bind to 0.0.0.0 to be accessible externally
    app.run(host="0.0.0.0", port=port)
