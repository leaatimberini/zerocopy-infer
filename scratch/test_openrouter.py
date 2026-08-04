import urllib.request
import urllib.error
import json

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "User-Agent": "ZeroCopy-Infer/0.4.7"
}
payload = {
    "model": "deepseek/deepseek-r1:free",
    "messages": [{"role": "user", "content": "como se llama el equipo de F1 de Franco Colapinto?"}],
    "max_tokens": 100
}
req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print("[SUCCESS OpenRouter Free!]")
        res = json.loads(resp.read().decode("utf-8"))
        print(res["choices"][0]["message"]["content"])
except urllib.error.HTTPError as e:
    print(f"HTTPError {e.code}: {e.read().decode('utf-8')[:300]}")
except Exception as e:
    print(f"Exception: {e}")
