import urllib.request
import urllib.error
import json
import os

token = os.getenv("HF_TOKEN", "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")
url = "https://router.huggingface.co/v1/chat/completions"

models = ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"]

for model in models:
    print(f"\n--- Testing Streaming for {model} ---")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "ZeroCopy-Infer/0.4.7"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "como se llama el equipo de F1 de Franco Colapinto?"}],
        "stream": True,
        "max_tokens": 150
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            full_text = ""
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        full_text += delta
                    except Exception:
                        pass
            print(f"[SUCCESS] Streaming Response ({model}):")
            print(full_text)
            os._exit(0)
    except urllib.error.HTTPError as e:
        print(f"[{model}] HTTPError {e.code}: {e.read().decode('utf-8')[:200]}")
    except Exception as e:
        print(f"[{model}] Exception: {e}")
