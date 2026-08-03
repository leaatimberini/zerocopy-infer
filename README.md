# ZeroCopy-Infer: Zero-Disk Cloud-Native MoE & LLM Streaming Inference Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](pyproject.toml)
[![C++23 Standard](https://img.shields.io/badge/C%2B%2B-23%20Bare--Metal-blueviolet.svg)](cpp/zerocopy_infer.cpp)
[![Zero-Disk SSD Storage](https://img.shields.io/badge/SSD%20Disk%20Used-0%20Bytes-success.svg)](#architecture)

**Authored and Created by**: Leandro Emanuel Timberini  
**Affiliation**: Investigador Independiente — Ituzaingó, Buenos Aires, Argentina  

---

## 🌟 Overview

**ZeroCopy-Infer** is an open-source C++23 / Python SDK designed to execute **Zero-Disk Cloud-Native Inference for Massive Mixture-of-Experts (MoE) and Dense Large Language Models (LLMs)** (such as Kimi K3, DeepSeek-V3/R1, Mixtral, and Qwen) **without storing model checkpoints on local SSD storage**.

While standard inference engines require downloading terabytes of model weights (e.g. 1.56 Terabytes for Kimi K3) to local NVMe SSDs, **ZeroCopy-Infer** streams `.safetensors` headers and individual layer/expert weight slices directly from Hugging Face LFS / CDN endpoints using **HTTP Range Requests (`Range: bytes=N-M`)**.

By routing weight chunks into a zero-copy RAM buffer (`VirtualAlloc` / `mmap` / `hipHostMallocMapped`), **ZeroCopy-Infer** enables consumer hardware (such as an AMD Ryzen 5 8600G APU with 16 GB DDR5 RAM) to perform inference on models with trillions of parameters with **0 Bytes of local disk storage occupied**.

---

## 💡 Key Architectural Innovations

1. **Remote `.safetensors` Header Indexing**:
   - Parses the initial uint64 header length and JSON tensor map directly from Hugging Face via HTTP `Range: bytes=0-1048576`.
   - Constructs an in-memory `TensorMap[tensor_name] -> (shard_url, start_byte, end_byte, dtype, shape)` in milliseconds without downloading 1.56 TB shards.

2. **On-Demand Expert Weight Streaming**:
   - When an MoE router selects active experts (e.g., top-16 out of 896 experts in Kimi K3), `ZeroCopy-Infer` issues asynchronous HTTP Range Requests for only the required ~20-50 MB expert weight chunks.

3. **In-Memory LRU Ring Buffer**:
   - Maintains an in-memory LRU cache (configurable, e.g. 8.0 GB RAM limit) for frequently activated experts, avoiding repeated network fetches.

4. **Zero-Disk Execution**:
   - Operates with **0 Bytes stored on local NVMe / SSD drives**.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/leaatimberini/zerocopy-infer.git
cd zerocopy-infer
pip install -e .
```

### Python API Example

```python
from zerocopy_infer import SafetensorsRangeStreamer, ZeroCopyMoEEngine

# 1. Initialize streamer pointing to Hugging Face LFS shards
streamer = SafetensorsRangeStreamer(
    repo_id="moonshotai/Kimi-K3",
    shards=["model-00001-of-00096.safetensors", "model-00042-of-00096.safetensors"]
)

# 2. Parse headers remotely (0 Bytes written to disk)
streamer.parse_headers()

# 3. Create ZeroCopy MoE Engine with 8.0 GB RAM LRU cache
engine = ZeroCopyMoEEngine(streamer=streamer, ram_cache_gb=8.0)

# 4. Execute streaming forward pass
token_id, latency = engine.forward_token(input_ids=[1008, 10484, 318])
print(f"Generated Token ID: {token_id} | Latency: {latency*1000:.2f} ms")
```

---

## 🏛️ C++23 Bare-Metal Engine

The bare-metal C++23 implementation is located in `cpp/zerocopy_infer.cpp`.

To compile:
```bash
g++ -std=c++23 -O3 cpp/zerocopy_infer.cpp -o zerocopy_infer
./zerocopy_infer
```

---

## 📊 Telemetry Benchmarks

| Metric | Standard Inference (llama.cpp / vLLM) | ZeroCopy-Infer |
| :--- | :--- | :--- |
| **Local SSD Storage Required** | **1,560 GB (1.56 TB)** | **0 Bytes (100% Zero-Disk)** |
| **RAM Footprint** | 8 GB - 224 GB | 8.0 GB (Configurable LRU) |
| **Network Streaming Strategy** | Full File Download | On-Demand HTTP Range Requests (`bytes=N-M`) |
| **Hardware Target** | High-End GPU Server | Consumer APU (AMD Ryzen 8600G) & PC |

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

**Author**: Leandro Emanuel Timberini  
**Affiliation**: Investigador Independiente — Ituzaingó, Buenos Aires, Argentina  
**GitHub**: [github.com/leaatimberini](https://github.com/leaatimberini)
