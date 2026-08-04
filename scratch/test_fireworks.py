import urllib.request
import urllib.error
import json
import os

token = os.getenv("HF_TOKEN", "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")

fw_models = [
    "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "accounts/fireworks/models/deepseek-r1",
    "accounts/fireworks/models/deepseek-v3",
    "accounts/fireworks/models/qwen2p5-coder-32b-instruct",
    "accounts/fireworks/models/f1-mini"
]

url = "https://router.huggingface.co/fireworks-ai/v1/chat/completions"

for model in fw_models:
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
            print(f"[SUCCESS!] Model: {model}")
            res = json.loads(resp.read().decode("utf-8"))
            print(res["choices"][0]["message"]["content"])
            os._exit(0)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"[{model}] {e.code}: {body[:200]}")
    except Exception as e:
        print(f"[{model}] Exception: {e}")
