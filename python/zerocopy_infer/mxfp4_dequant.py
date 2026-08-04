"""
ZeroCopy-Infer: Official Kimi-K3 MXFP4 / MXFP8 Microscaling Dequantizer
=======================================================================
Authored by Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina).

Implements official OCP MXFP4 (Microscaling FP4 E2M1 + E8M0 Exponent Scale) Dequantization:
- Unpacks 4-bit uint8 nibbles (2 FP4 elements per byte)
- Applies E8M0 shared exponent scaling (2**(scale - 127)) per 32-element block
- Returns exact float32 weight matrices for GEMM / MoE forward passes
"""

import numpy as np
from typing import Tuple

# Official OCP Microscaling FP4 (E2M1) lookup table (16 values)
FP4_E2M1_TABLE = np.array([
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=np.float32)

def dequantize_mxfp4(weight_packed: np.ndarray, weight_scale: np.ndarray, block_size: int = 32) -> np.ndarray:
    """
    Dequantizes MXFP4 packed weight tensor into full precision float32 matrix.
    
    Parameters:
    - weight_packed: uint8 array of shape (N, K // 2) containing 4-bit nibbles.
    - weight_scale: uint8 array of shape (N, K // 32) containing E8M0 exponent scales.
    - block_size: 32 (default block size for OCP Microscaling).
    
    Returns:
    - float32 matrix of shape (N, K).
    """
    if weight_packed.dtype != np.uint8:
        weight_packed = weight_packed.view(np.uint8)
        
    low_nibble = weight_packed & 0x0F
    high_nibble = (weight_packed >> 4) & 0x0F
    
    # Interleave low and high nibbles: (N, K // 2, 2) -> (N, K)
    shape_N = weight_packed.shape[0]
    fp4_indices = np.stack([low_nibble, high_nibble], axis=-1).reshape(shape_N, -1)
    
    # Map 4-bit indices to E2M1 float values
    unscaled_weights = FP4_E2M1_TABLE[fp4_indices]
    
    # Compute E8M0 exponent multipliers: 2^(scale - 127)
    scale_floats = weight_scale.astype(np.float32)
    scales = np.power(2.0, scale_floats - 127.0)
    
    # Repeat scale across 32 elements along the K dimension
    scale_expanded = np.repeat(scales, block_size, axis=-1)[:, :unscaled_weights.shape[1]]
    
    return unscaled_weights * scale_expanded
