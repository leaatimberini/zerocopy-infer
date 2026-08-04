import urllib.request
import urllib.error
import json
import os

token = os.getenv("HF_TOKEN", "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")

providers = ["together", "nebius", "fireworks-ai", "novita", "hyperbolic", "sambanova", "groq", "cerebras"]
models = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

for provider in providers:
    for model in models:
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
                res = json.loads(resp.read().decode("utf-8"))
                content = res["choices"][0]["message"]["content"]
                print(f"[SUCCESS] Provider: {provider} | Model: {model}")
                print(f"Response: {content}")
                os._exit(0)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            if "not supported" not in err_body:
                print(f"[{provider}/{model}] {e.code}: {err_body[:200]}")
        except Exception:
            pass
print("No matching HF provider endpoint.")
