"""
ZeroCopy-Infer: Official Kimi-K3 NaViT Vision & Media Preprocessing Utilities
==============================================================================
Authored by Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina).

Implements exact NaViT (Native Resolution Vision Transformer) image resizing, patchification & grid calculations from media_utils.py:
- navit_resize_image
- navit_patchify
- fill_transparent_bg_with
- normalize
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, Optional

def navit_resize_image(
    width: int,
    height: int,
    patch_size: int = 14,
    merge_kernel_size: int = 2,
    in_patch_limit: int = 4096,
    patch_limit_on_one_side: int = 100,
    fixed_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calculates exact NaViT patch grid dimensions and padding for Kimi-K3.
    """
    s1 = math.sqrt(
        in_patch_limit /
        (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w = min(new_w, patch_limit_on_one_side * patch_size)
    new_h = min(new_h, patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor
        num_tokens = token_height * token_width

    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "sampled_nframes": 1,
    }

def navit_patchify(pixel_values: np.ndarray, patch_size: int = 14) -> Dict[str, np.ndarray]:
    """
    Reshapes pixel_values (T, H, W, C) into NaViT patches and grid_thw tuple.
    """
    T, H, W, C = pixel_values.shape
    assert C == 3, "pixel_values must have 3 channels (RGB)"

    patches = pixel_values.reshape(
        T, H // patch_size, patch_size,
        W // patch_size, patch_size, C
    )
    patches = patches.transpose(0, 1, 3, 5, 2, 4)
    patches = patches.reshape(-1, C, patch_size, patch_size)
    grid_thw = np.array([T, H // patch_size, W // patch_size], dtype=np.int64)
    
    return {"pixel_values": patches, "grid_thw": grid_thw}

def normalize_image(
    pixel_values: np.ndarray,
    mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
    std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
) -> np.ndarray:
    """
    Normalizes image array to float32 [0, 1] range minus mean divided by std.
    """
    x = (pixel_values / 255.0).astype(np.float32)
    x -= np.array(mean, dtype=np.float32)
    x *= (1.0 / np.array(std, dtype=np.float32))
    return x
