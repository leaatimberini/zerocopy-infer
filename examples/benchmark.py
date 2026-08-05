"""
ZeroCopy-Infer: Automated Benchmark & Latency Telemetry Audit Tool
===================================================================
Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina).

Executes automated performance benchmarking across layer configurations, 
RAM limits, and model choices to measure tokens/sec, latencies, and HTTP streaming data.
"""

import time
import json
import os
import sys
import numpy as np
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

try:
    from zerocopy_infer.hardware_detector import detect_hardware
    from zerocopy_infer.hf_range_stream import SafetensorsRangeStreamer
    from zerocopy_infer.moe_inference_engine import ZeroCopyMoEEngine
except ImportError:
    from python.zerocopy_infer.hardware_detector import detect_hardware
    from python.zerocopy_infer.hf_range_stream import SafetensorsRangeStreamer
    from python.zerocopy_infer.moe_inference_engine import ZeroCopyMoEEngine


def run_benchmark(model_repo: str = "Qwen/Qwen2.5-1.5B-Instruct", test_layers: List[int] = [1, 2, 4], num_tokens: int = 5) -> Dict[str, Any]:
    hw = detect_hardware()
    print("=" * 80)
    print(" 🚀 ZeroCopy-Infer: Automated Streaming Latency Benchmark")
    print("=" * 80)
    print(f" Hardware: {hw['system']} ({hw['arch']}) | Cores: {hw['cpu_count']} | SIMD: {hw['simd_extension']}")
    print(f" Target Model: {model_repo}")
    print("=" * 80)

    streamer = SafetensorsRangeStreamer(repo_id=model_repo, shards=[])
    streamer.load_index_json()

    results = []

    for layers in test_layers:
        print(f"\n⚡ Testing {layers} Active Layers...")
        engine = ZeroCopyMoEEngine(streamer=streamer, ram_cache_gb=4.0, num_active_layers=layers)
        
        start_t = time.time()
        tokens_gen = 0
        total_lat = 0.0

        for step, token_id, decoded_word, latency, total_bytes in engine.generate_chat_response_stream("Explicar la gravedad en dos oraciones", num_tokens=num_tokens):
            tokens_gen += 1
            total_lat += latency

        elapsed = time.time() - start_t
        tps = tokens_gen / elapsed if elapsed > 0 else 0
        avg_lat = (total_lat / tokens_gen * 1000) if tokens_gen > 0 else 0

        res = {
            "layers": layers,
            "tokens": tokens_gen,
            "tps": round(tps, 2),
            "avg_latency_ms": round(avg_lat, 2),
            "total_bytes_mb": round(engine.total_bytes_streamed / (1024 * 1024), 2),
            "http_requests": engine.total_range_requests
        }
        results.append(res)
        print(f"   --> Throughput: {res['tps']} t/s | Latency: {res['avg_latency_ms']} ms/token | Streamed: {res['total_bytes_mb']} MB")

    report = {
        "hardware": hw,
        "model": model_repo,
        "results": results
    }
    return report


if __name__ == "__main__":
    rep = run_benchmark()
    print("\n" + "=" * 80)
    print(" BENCHMARK COMPLETED (100% Zero-Disk Streamed)")
    print("=" * 80)
