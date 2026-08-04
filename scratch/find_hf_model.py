import urllib.request
import urllib.error
import json
import os

token = os.getenv("HF_TOKEN", "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")

candidate_urls = [
    ("https://router.huggingface.co/hf-inference/v1/chat/completions", "Qwen/Qwen2.5-72B-Instruct"),
    ("https://router.huggingface.co/hf-inference/v1/chat/completions", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
    ("https://router.huggingface.co/hf-inference/v1/chat/completions", "meta-llama/Llama-3.3-70B-Instruct"),
    ("https://router.huggingface.co/hf-inference/v1/chat/completions", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("https://router.huggingface.co/hf-inference/v1/chat/completions", "mistralai/Mistral-7B-Instruct-v0.2"),
    ("https://router.huggingface.co/hf-inference/v1/chat/completions", "google/gemma-2-9b-it"),
    ("https://router.huggingface.co/hf-inference/v1/chat/completions", "HuggingFaceH4/zephyr-7b-beta"),
    ("https://router.huggingface.co/hf-inference/v1/chat/completions", "tiiuae/falcon-7b-instruct"),
]

for url, model in candidate_urls:
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
            print(f"[FOUND SUCCESS!] Model: {model}")
            res = json.loads(resp.read().decode("utf-8"))
            print(res["choices"][0]["message"]["content"])
            break
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"[{model}] {e.code}: {body[:200]}")
    except Exception as e:
        print(f"[{model}] Exception: {e}")
