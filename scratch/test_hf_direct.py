import urllib.request
import urllib.error
import json
import os

token = os.getenv("HF_TOKEN", "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")

models = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2"
]

for model in models:
    url = f"https://api-inference.huggingface.co/models/{model}"
    print(f"\n--- Testing Direct HF: {url} ---")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "ZeroCopy-Infer/0.4.7"
    }
    payload = {
        "inputs": "El equipo de F1 de Franco Colapinto se llama",
        "parameters": {"max_new_tokens": 50}
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print("Status 200 OK:")
            print(resp.read().decode("utf-8"))
            os._exit(0)
    except urllib.error.HTTPError as e:
        print(f"HTTPError {e.code}: {e.read().decode('utf-8')[:300]}")
    except Exception as e:
        print(f"Exception: {e}")
