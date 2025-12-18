from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import requests

app = Flask(__name__)
CORS(app) 

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

@app.route('/process', methods=['POST'])
def get_answer():
    start_time = time.time()
    data = request.json
    user_question = data.get("question", "")

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": "Expert coach. 2 sentences max. Tactical."},
                    {"role": "user", "content": f"Interview question: {user_question}"}
                ],
                "temperature": 0.6,
                "max_tokens": 150,
                "top_p": 0.9
            }
        )
        
        res_json = response.json()
        answer_text = res_json['choices'][0]['message']['content']
        processing_time = round(time.time() - start_time, 3)

        return jsonify({
            "answer": answer_text,
            "model": "groq-llama-3.1-8b",
            "processing_time": processing_time,
            "timestamp": time.time()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)