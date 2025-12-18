import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"status": "online", "message": "EdgeEcho Backend is Live"})

@app.route('/health')
def health():
    api_key = os.environ.get("GROQ_API_KEY")
    return jsonify({
        "status": "ok",
        "groq_configured": bool(api_key),
        "runtime": "Gunicorn on Railway"
    })

@app.route('/process', methods=['POST'])
def process_question():
    data = request.json
    user_query = data.get("question")
    if not user_query:
        return jsonify({"error": "No question provided"}), 400
    
    api_key = os.environ.get("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "Tactical advisor mode active."},
            {"role": "user", "content": user_query}
        ]
    }
    response = requests.post(url, json=payload, headers=headers)
    return jsonify(response.json())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
