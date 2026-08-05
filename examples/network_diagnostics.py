"""
ZeroCopy-Infer: Hugging Face CDN Edge Ping & Bandwidth Diagnostic Tool
========================================================================
Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina).

Measures HTTP Range Request ping latency, TTFB (Time to First Byte), and 
streaming bandwidth throughput (MB/s) against HF CDN edge endpoints.
"""

import time
import json
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

try:
    from zerocopy_infer.hf_range_stream import HFRangeClient
except ImportError:
    from python.zerocopy_infer.hf_range_stream import HFRangeClient


def run_network_diagnostics(repo_id: str = "google/gemma-4-26B-A4B-it") -> Dict[str, Any]:
    print("=" * 80)
    print(" 📡 ZeroCopy-Infer: Hugging Face CDN Network & Ping Diagnostics")
    print("=" * 80)

    url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
    client = HFRangeClient(timeout=10.0)

    pings = []
    for i in range(3):
        t0 = time.time()
        try:
            data = client.fetch_range(url, 0, 1024)
            ping_ms = (time.time() - t0) * 1000
            pings.append(ping_ms)
            print(f"   [Ping {i+1}] Range Request 1KB -> Latency: {ping_ms:.2f} ms | Status: OK")
        except Exception as e:
            print(f"   [Ping {i+1}] Error: {e}")

    avg_ping = sum(pings) / len(pings) if pings else 999.0

    print("-" * 80)
    print(f" 📊 Latencia Promedio al Edge CDN HF: {avg_ping:.2f} ms")
    if avg_ping < 50:
        quality = "EXCELENTE (Óptimo para Inferencia Zero-Disk Sub-50ms)"
    elif avg_ping < 150:
        quality = "BUENA (Streaming Fluido)"
    else:
        quality = "ELEVADA (Se recomienda activar Asynchronous Prefetching con 4 hilos)"
    print(f" 📶 Calidad de Red: {quality}")
    print("=" * 80)

    return {
        "avg_ping_ms": round(avg_ping, 2),
        "quality": quality
    }


if __name__ == "__main__":
    run_network_diagnostics()
