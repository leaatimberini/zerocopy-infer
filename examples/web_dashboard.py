"""
ZeroCopy-Infer: Zero-Dependency Live Web Telemetry Dashboard & Chat Server
==========================================================================
Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina).

Provides a zero-dependency (pure standard library http.server) live browser dashboard
displaying hardware telemetry, streaming latency metrics, and interactive chat.
"""

import json
import time
import http.server
import socketserver
from urllib.parse import parse_qs, urlparse

from python.zerocopy_infer.hardware_detector import detect_hardware
from python.zerocopy_infer.tokenizer import UniversalHFTokenizer

PORT = 8080

HTML_PAGE = """<!DOCTYPE html>
<html lang="es" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZeroCopy-Infer Live Telemetry & Chat</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0f0f11; color: #e5e1e4; font-family: sans-serif; }
        .glass { background: rgba(22, 22, 24, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); }
        .neon { color: #c3f400; }
    </style>
</head>
<body class="p-4 max-w-4xl mx-auto space-y-4">
    <header class="glass p-4 rounded-xl flex justify-between items-center">
        <div>
            <h1 class="text-xl font-bold neon">🚀 ZeroCopy-Infer</h1>
            <p class="text-xs text-gray-400">Universal Zero-Disk Mobile & Cloud-Native MoE Streaming</p>
        </div>
        <div id="hwBadge" class="text-xs font-mono bg-black/50 p-2 rounded border border-white/10 text-right">
            Cargando hardware...
        </div>
    </header>

    <section class="grid grid-cols-3 gap-3 font-mono text-xs">
        <div class="glass p-3 rounded-xl">
            <span class="text-gray-400 block">ARQUITECTURA</span>
            <span id="statArch" class="text-white font-bold text-sm">--</span>
        </div>
        <div class="glass p-3 rounded-xl">
            <span class="text-gray-400 block">SIMD ACCELERATION</span>
            <span id="statSimd" class="text-lime-400 font-bold text-sm">--</span>
        </div>
        <div class="glass p-3 rounded-xl">
            <span class="text-gray-400 block">RAM DISPONIBLE</span>
            <span id="statRam" class="text-blue-400 font-bold text-sm">--</span>
        </div>
    </section>

    <main class="glass p-4 rounded-xl space-y-4">
        <div class="flex justify-between items-center border-b border-white/10 pb-2">
            <h2 class="font-bold text-sm text-purple-400">Chat Interactivo ZeroCopy</h2>
            <select id="modelSelect" class="bg-black text-xs text-lime-300 font-mono p-1 rounded border border-white/20">
                <option value="google/gemma-4-26B-A4B-it">Gemma 4 26B (MoE)</option>
                <option value="google/gemma-4-31B-it">Gemma 4 31B (Dense)</option>
                <option value="moonshotai/Kimi-K3">Kimi K3 (2.78T MoE)</option>
                <option value="XiaomiMiMo/MiMo-V2.5-Pro">MiMo V2.5 Pro</option>
                <option value="deepseek-ai/DeepSeek-V3">DeepSeek V3 (MLA)</option>
                <option value="Qwen/Qwen2.5-1.5B-Instruct">Qwen 2.5 1.5B</option>
            </select>
        </div>

        <div id="chatBox" class="h-64 overflow-y-auto space-y-2 p-2 bg-black/40 rounded-lg text-sm font-mono">
            <div class="text-purple-300">[ZeroCopy Agent]: ¡Hola! Estoy listo para transmitir inferencia 100% Zero-Disk en RAM.</div>
        </div>

        <div class="flex gap-2">
            <input id="promptInput" type="text" placeholder="Escribe tu mensaje..." class="flex-1 bg-black p-2 rounded-lg text-sm border border-white/20 focus:outline-none focus:border-lime-400 font-mono" onkeypress="if(event.key==='Enter') sendMsg()"/>
            <button onclick="sendMsg()" class="bg-lime-400 text-black px-4 py-2 rounded-lg font-bold hover:bg-lime-300">Enviar</button>
        </div>
    </main>

    <script>
        async function loadHw() {
            const r = await fetch('/api/hardware');
            const data = await r.json();
            document.getElementById('statArch').innerText = `${data.arch} (${data.cpu_count} cÓcleos)`;
            document.getElementById('statSimd').innerText = data.simd_extension;
            document.getElementById('statRam').innerText = `${data.ram_available_gb.toFixed(1)} GB / ${data.ram_total_gb.toFixed(1)} GB`;
            document.getElementById('hwBadge').innerText = `OS: ${data.system}\\nSIMD: ${data.simd_extension}`;
        }

        async function sendMsg() {
            const input = document.getElementById('promptInput');
            const txt = input.value.strip ? input.value.strip() : input.value.trim();
            if(!txt) return;
            input.value = '';

            const box = document.getElementById('chatBox');
            box.innerHTML += `<div class="text-blue-300">[Tú]: ${txt}</div>`;
            box.scrollTop = box.scrollHeight;

            const aiDiv = document.createElement('div');
            aiDiv.className = 'text-purple-300';
            aiDiv.innerText = '[ZeroCopy Agent]: Pensando...';
            box.appendChild(aiDiv);

            const r = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: txt, model: document.getElementById('modelSelect').value})
            });
            const data = await r.json();
            aiDiv.innerText = `[ZeroCopy Agent]: ${data.reply}`;
            box.scrollTop = box.scrollHeight;
        }

        loadHw();
    </script>
</body>
</html>
"""

class TelemetryHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/api/hardware":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            hw = detect_hardware()
            self.wfile.write(json.dumps(hw).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8')) if body else {}
            prompt = data.get("prompt", "")
            model = data.get("model", "google/gemma-4-26B-A4B-it")

            reply = f"Respuesta simulada para '{prompt}' en {model} (100% Zero-Disk RAM Ingest)."

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"reply": reply, "tps": 42.5}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence default HTTP server logging
        return


def run_dashboard(port: int = PORT):
    print(f"🚀 Dashboard de Telemetría ZeroCopy activo en http://127.0.0.1:{port}")
    with socketserver.TCPServer(("", port), TelemetryHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard detenido.")


if __name__ == "__main__":
    run_dashboard()
