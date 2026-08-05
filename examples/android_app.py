import os
from flask import Flask, request, jsonify, render_template_string
import threading
import time

app = Flask(__name__)

# Mock backend state
backend_state = {
    "model": "Kimi K3",
    "active_layers": 10,
    "ram_cache_limit": 4096,
    "status": "idle"
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZeroCopy-Infer Mobile Dashboard</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --surface-color: #1e293b;
            --primary-color: #38bdf8;
            --text-color: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --success-color: #10b981;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }

        header {
            background-color: var(--surface-color);
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        h1 {
            margin: 0;
            font-size: 1.25rem;
            color: var(--primary-color);
        }

        .status-badge {
            background-color: rgba(16, 185, 129, 0.2);
            color: var(--success-color);
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }

        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            padding: 1rem;
            gap: 1rem;
        }

        .controls-panel {
            background-color: var(--surface-color);
            border-radius: 0.75rem;
            padding: 1rem;
            border: 1px solid var(--border-color);
        }

        .control-group {
            margin-bottom: 1rem;
        }

        .control-group:last-child {
            margin-bottom: 0;
        }

        label {
            display: block;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
            color: var(--text-muted);
        }

        select, input[type="range"] {
            width: 100%;
            padding: 0.5rem;
            background-color: var(--bg-color);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            border-radius: 0.5rem;
            outline: none;
        }
        
        select:focus {
            border-color: var(--primary-color);
        }

        .slider-header {
            display: flex;
            justify-content: space-between;
        }

        .chat-container {
            flex: 1;
            background-color: var(--surface-color);
            border-radius: 0.75rem;
            border: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-messages {
            flex: 1;
            padding: 1rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .message {
            max-width: 85%;
            padding: 0.75rem 1rem;
            border-radius: 1rem;
            font-size: 0.9375rem;
            line-height: 1.4;
        }

        .message.user {
            background-color: var(--primary-color);
            color: #000;
            align-self: flex-end;
            border-bottom-right-radius: 0.25rem;
        }

        .message.bot {
            background-color: var(--bg-color);
            border: 1px solid var(--border-color);
            align-self: flex-start;
            border-bottom-left-radius: 0.25rem;
        }

        .chat-input {
            padding: 1rem;
            border-top: 1px solid var(--border-color);
            display: flex;
            gap: 0.5rem;
        }

        .chat-input input[type="text"] {
            flex: 1;
            padding: 0.75rem 1rem;
            border-radius: 9999px;
            border: 1px solid var(--border-color);
            background-color: var(--bg-color);
            color: var(--text-color);
            outline: none;
        }

        .chat-input input[type="text"]:focus {
            border-color: var(--primary-color);
        }

        .chat-input button {
            background-color: var(--primary-color);
            color: #000;
            border: none;
            border-radius: 9999px;
            padding: 0 1.25rem;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }

        .chat-input button:hover {
            opacity: 0.9;
        }
        
        .chat-input button:disabled {
            background-color: var(--border-color);
            color: var(--text-muted);
            cursor: not-allowed;
        }
        
        /* Loading animation */
        .typing-indicator {
            display: none;
            align-items: center;
            gap: 4px;
            padding: 0.5rem 1rem;
            background-color: var(--bg-color);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            align-self: flex-start;
            width: fit-content;
        }
        
        .typing-indicator span {
            width: 6px;
            height: 6px;
            background-color: var(--text-muted);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

    </style>
</head>
<body>
    <header>
        <h1>ZeroCopy-Infer</h1>
        <div class="status-badge" id="connection-status">Online</div>
    </header>

    <main>
        <!-- Configuration Panel -->
        <div class="controls-panel">
            <div class="control-group">
                <label for="model-select">Model Selection</label>
                <select id="model-select" onchange="updateSettings()">
                    <option value="Gemma 4">Gemma 4</option>
                    <option value="Kimi K3" selected>Kimi K3</option>
                    <option value="MiMo V2.5">MiMo V2.5</option>
                    <option value="DeepSeek V3">DeepSeek V3</option>
                    <option value="Qwen 2.5">Qwen 2.5</option>
                </select>
            </div>

            <div class="control-group">
                <div class="slider-header">
                    <label for="layers-slider">Active Layers</label>
                    <span id="layers-value" class="text-sm text-primary">10</span>
                </div>
                <input type="range" id="layers-slider" min="1" max="64" value="10" oninput="document.getElementById('layers-value').textContent = this.value; updateSettings();">
            </div>

            <div class="control-group">
                <div class="slider-header">
                    <label for="ram-slider">RAM Cache Limit (MB)</label>
                    <span id="ram-value" class="text-sm text-primary">4096</span>
                </div>
                <input type="range" id="ram-slider" min="512" max="16384" step="512" value="4096" oninput="document.getElementById('ram-value').textContent = this.value; updateSettings();">
            </div>
        </div>

        <!-- Chat Interface -->
        <div class="chat-container">
            <div class="chat-messages" id="chat-messages">
                <div class="message bot">Welcome to ZeroCopy-Infer mobile dashboard. How can I help you?</div>
                <div class="typing-indicator" id="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
            <div class="chat-input">
                <input type="text" id="prompt-input" placeholder="Enter prompt..." onkeypress="if(event.key === 'Enter') sendMessage()">
                <button id="send-button" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </main>

    <script>
        function updateSettings() {
            const model = document.getElementById('model-select').value;
            const layers = document.getElementById('layers-slider').value;
            const ram = document.getElementById('ram-slider').value;

            fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: model,
                    active_layers: parseInt(layers),
                    ram_cache_limit: parseInt(ram)
                })
            }).then(response => response.json())
              .then(data => console.log('Settings updated:', data));
        }

        function appendMessage(text, sender) {
            const messagesContainer = document.getElementById('chat-messages');
            const typingIndicator = document.getElementById('typing-indicator');
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            messageDiv.textContent = text;
            
            messagesContainer.insertBefore(messageDiv, typingIndicator);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function setTyping(isTyping) {
            const typingIndicator = document.getElementById('typing-indicator');
            const sendButton = document.getElementById('send-button');
            const promptInput = document.getElementById('prompt-input');
            
            typingIndicator.style.display = isTyping ? 'flex' : 'none';
            sendButton.disabled = isTyping;
            promptInput.disabled = isTyping;
            
            if (isTyping) {
                const messagesContainer = document.getElementById('chat-messages');
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        }

        function sendMessage() {
            const input = document.getElementById('prompt-input');
            const text = input.value.trim();
            if (!text) return;

            appendMessage(text, 'user');
            input.value = '';
            setTyping(true);

            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ prompt: text })
            })
            .then(response => response.json())
            .then(data => {
                setTyping(false);
                appendMessage(data.response, 'bot');
            })
            .catch(err => {
                setTyping(false);
                appendMessage("Error communicating with backend.", 'bot');
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    global backend_state
    data = request.json
    if data:
        backend_state.update(data)
        # Here you would typically interface with the ZeroCopy-Infer engine
        # e.g., engine.set_model(backend_state['model'])
        # engine.set_active_layers(backend_state['active_layers'])
        # engine.set_ram_limit(backend_state['ram_cache_limit'])
        return jsonify({"status": "success", "state": backend_state})
    return jsonify({"status": "error"}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt', '')
    
    # Simulate processing time for the mock response
    time.sleep(1.5)
    
    # Mock inference response based on the current model
    model = backend_state.get('model', 'Unknown Model')
    response_text = f"[{model}] Response to: {prompt}\n\nRunning on {backend_state['active_layers']} active layers with {backend_state['ram_cache_limit']}MB RAM cache."
    
    return jsonify({"response": response_text})

if __name__ == '__main__':
    print("Starting ZeroCopy-Infer Android Mobile Dashboard (Termux compatible)...")
    print("Access the dashboard at http://127.0.0.1:5000 in your mobile browser.")
    app.run(host='0.0.0.0', port=5000, debug=True)
