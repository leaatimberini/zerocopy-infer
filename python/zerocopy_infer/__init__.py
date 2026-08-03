"""
ZeroCopy-Infer SDK
==================
Zero-Disk Cloud-Native MoE & LLM Streaming Inference Engine.
Authored by Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina).

Streams .safetensors shards and MoE expert weights directly from Hugging Face via HTTP Range Requests into RAM / APU zero-copy memory using 0 Bytes of local SSD storage.
"""

from .hf_range_stream import SafetensorsRangeStreamer, HFRangeClient
from .moe_inference_engine import ZeroCopyMoEEngine

__version__ = "0.1.0"
__author__ = "Leandro Emanuel Timberini"
__all__ = [
    "SafetensorsRangeStreamer",
    "HFRangeClient",
    "ZeroCopyMoEEngine",
]
