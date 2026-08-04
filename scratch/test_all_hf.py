import urllib.request
import urllib.error
import json
import os

token = os.getenv("HF_TOKEN", "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")

urls = [
    "https://router.huggingface.co/hf-inference/v1/chat/completions",
    "https://router.huggingface.co/v1/chat/completions"
]

models = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
]

for url in urls:
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
            with urllib.request.urlopen(req, timeout=5) as resp:
                res_body = resp.read().decode("utf-8")
                print(f"[FOUND SUCCESS!] URL: {url} | Model: {model}")
                print(res_body[:300])
                os._exit(0)
        except urllib.error.HTTPError as e:
            print(f"[{url}] [{model}] {e.code}: {e.read().decode('utf-8')[:150]}")
        except Exception as e:
            print(f"[{url}] [{model}] Exception: {e}")
