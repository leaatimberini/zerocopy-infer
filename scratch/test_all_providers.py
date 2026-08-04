import urllib.request
import urllib.error
import json
import os

token = os.getenv("HF_TOKEN", "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")

provider_models = [
    ("groq", "llama-3.1-8b-instant"),
    ("groq", "mixtral-8x7b-32768"),
    ("fireworks-ai", "accounts/fireworks/models/llama-v3p1-8b-instruct"),
    ("fireworks-ai", "accounts/fireworks/models/qwen2p5-72b-instruct"),
    ("novita", "meta-llama/llama-3.1-8b-instruct"),
    ("sambanova", "Meta-Llama-3.1-8B-Instruct"),
    ("cerebras", "llama3.1-8b"),
    ("hf-inference", "gpt2")
]

for provider, model in provider_models:
    url = f"https://router.huggingface.co/{provider}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "ZeroCopy-Infer/0.4.7"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "como se llama el equipo de F1 de Franco Colapinto?"}],
        "max_tokens": 100
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"[SUCCESS!] Provider: {provider} | Model: {model}")
            res = json.loads(resp.read().decode("utf-8"))
            print(res["choices"][0]["message"]["content"])
            os._exit(0)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"[{provider}/{model}] {e.code}: {body[:200]}")
    except Exception as e:
        print(f"[{provider}/{model}] Exception: {e}")
