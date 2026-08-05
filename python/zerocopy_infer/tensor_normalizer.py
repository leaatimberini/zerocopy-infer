"""
ZeroCopy-Infer: Dynamic Tensor Outlier Normalizer & Safety Guards
==================================================================
Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina).

Provides numerical stability guards, NaN/Inf replacement, and outlier clipping
for dynamically streamed weights and activations during zero-disk matrix operations.
"""

import numpy as np
from typing import Optional


class TensorNormalizer:
    """
    Utility class for numerical stability and outlier normalization.
    """
    @staticmethod
    def sanitize_tensor(tensor: np.ndarray, clip_val: float = 1e4) -> np.ndarray:
        """
        Replaces NaN and Infinity values with zeros and clips extreme outliers.
        """
        clean = np.nan_to_num(tensor, nan=0.0, posinf=clip_val, neginf=-clip_val)
        return np.clip(clean, -clip_val, clip_val)

    @staticmethod
    def scale_layer_activation(x: np.ndarray, target_norm: float = 1.0) -> np.ndarray:
        """
        Rescales activation vector to prevent exploding / vanishing gradients.
        """
        norm = np.linalg.norm(x)
        if norm > 1e-6 and not np.isnan(norm):
            return x * (target_norm / norm)
        return x
