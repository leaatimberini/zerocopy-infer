"""
ZeroCopy-Infer: High-Speed BF16 / FP16 Vector Dequantization Utilities
========================================================================
Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina).

Provides optimized C-level bit manipulation for converting bfloat16 and float16
binary safetensors buffers directly into IEEE 754 float32 arrays in RAM.
"""

import numpy as np


def dequantize_bf16_to_fp32(bf16_bytes: bytes) -> np.ndarray:
    """
    Converts raw uint16 bfloat16 buffer into float32 NumPy array.
    bfloat16 has 1 sign bit, 8 exponent bits, 7 mantissa bits (same exponent as FP32).
    Bit shift left by 16 bits converts uint16 to uint32 bit pattern of float32.
    """
    u16 = np.frombuffer(bf16_bytes, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def dequantize_fp16_to_fp32(fp16_bytes: bytes) -> np.ndarray:
    """
    Converts raw uint16 float16 buffer into float32 NumPy array via NumPy fast view.
    """
    f16 = np.frombuffer(fp16_bytes, dtype=np.float16)
    return f16.astype(np.float32)
