import urllib.request
import urllib.error
import json
import os

token = os.getenv("HF_TOKEN", "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")

url = "https://router.huggingface.co/v1/chat/completions"

models = [
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

for model in models:
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
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[SUCCESS] Model: {model}")
            res = json.loads(resp.read().decode("utf-8"))
            print(res["choices"][0]["message"]["content"])
            os._exit(0)
    except urllib.error.HTTPError as e:
        print(f"[{model}] HTTPError {e.code}: {e.read().decode('utf-8')[:200]}")
    except Exception as e:
        print(f"[{model}] Exception: {e}")
