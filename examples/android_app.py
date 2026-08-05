import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates")

state = {
    "model": "google/gemma-4-26B-A4B-it",
    "layers": 4,
    "ram": 4.0,
    "chat_history": []
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        data = request.json or {}
        state["model"] = data.get("model", state["model"])
        state["layers"] = data.get("layers", state["layers"])
        state["ram"] = data.get("ram", state["ram"])
        return jsonify({"status": "ok", "state": state})
    return jsonify(state)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_prompt = data.get("prompt", "")
    if not user_prompt:
        return jsonify({"reply": "Por favor escribe un prompt válido."})

    state["chat_history"].append({"role": "user", "content": user_prompt})
    
    # Mock inference response simulation for mobile app preview
    reply = f"Respuesta simulada para '{user_prompt}' ejecutando {state['layers']} capas activas de {state['model']} (100% Zero-Disk RAM Ingest)."
    state["chat_history"].append({"role": "assistant", "content": reply})
    
    return jsonify({
        "reply": reply,
        "tps": 42.5,
        "latency_ms": 23.5,
        "streamed_mb": 64.5
    })

if __name__ == "__main__":
    print("🚀 Starting ZeroCopy-Infer Android Mobile Web App on http://127.0.0.1:5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
