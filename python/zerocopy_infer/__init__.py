"""
ZeroCopy-Infer Package
======================
Authored by Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina).
"""

from .hf_range_stream import SafetensorsRangeStreamer
from .tokenizer import ZeroCopyKimiTokenizer, ZeroCopyKimiTokenizer as ZeroCopyTokenizer
from .moe_inference_engine import ZeroCopyMoEEngine, KimiK3Config
from .media_utils import navit_resize_image, navit_patchify, normalize_image
from .modeling_kimi_k3 import KimiK3ForConditionalGeneration, PatchMergerMLPV2

__version__ = "0.7.0"
__author__ = "Leandro Emanuel Timberini"
__all__ = [
    "SafetensorsRangeStreamer",
    "ZeroCopyKimiTokenizer",
    "ZeroCopyTokenizer",
    "ZeroCopyMoEEngine",
    "KimiK3Config",
    "KimiK3ForConditionalGeneration",
    "PatchMergerMLPV2",
    "navit_resize_image",
    "navit_patchify",
    "normalize_image",
]
