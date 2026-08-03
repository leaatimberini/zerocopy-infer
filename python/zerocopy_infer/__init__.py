"""
ZeroCopy-Infer Package
======================
Authored by Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina).
"""

from .hf_range_stream import SafetensorsRangeStreamer
from .tokenizer import ZeroCopyKimiTokenizer, ZeroCopyKimiTokenizer as ZeroCopyTokenizer
from .moe_inference_engine import ZeroCopyMoEEngine

__version__ = "0.1.5"
__author__ = "Leandro Emanuel Timberini"
__all__ = [
    "SafetensorsRangeStreamer",
    "ZeroCopyKimiTokenizer",
    "ZeroCopyTokenizer",
    "ZeroCopyMoEEngine",
]
