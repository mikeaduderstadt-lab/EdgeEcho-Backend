import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# The Bouncer's Memory - Stores usage counts
usage_counts = {}

@app.route('/health')
def health():
    return jsonify({"status": "ok", "runtime": "Gunicorn on Railway"})

@app.route('/process', methods=['POST'])
def process_question():
    data = request.json
    user_query = data.get("question")
    visitor_id = data.get("visitorId", "anonymous")

    # Limit to 5 free questions per user
    count = usage_counts.get(visitor_id, 0)
    if count >= 5:
        return jsonify({
            "error": "Limit Reached",
            "message": "You've used your free tactical credits.",
            "action": "upgrade_required"
        }), 429

    api_key = os.environ.get("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a tactical advisor. Sharp and elite."},
            {"role": "user", "content": user_query}
        ]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        usage_counts[visitor_id] = count + 1
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
