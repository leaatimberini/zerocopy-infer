import urllib.request
import urllib.error
import json
import os

token = os.getenv("HF_TOKEN", "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")

together_models = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
]

url = "https://router.huggingface.co/together/v1/chat/completions"

for model in together_models:
    print(f"\n--- Testing Together Model: {model} ---")
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
            res = json.loads(resp.read().decode("utf-8"))
            content = res["choices"][0]["message"]["content"]
            print(f"[SUCCESS] Model: {model}")
            print(f"Response: {content}")
            os._exit(0)
    except urllib.error.HTTPError as e:
        print(f"HTTPError {e.code}: {e.read().decode('utf-8')[:300]}")
    except Exception as e:
        print(f"Exception: {e}")
