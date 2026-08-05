"""
ZeroCopy-Infer: Rolling KV-Cache & Sliding Window Context Manager
===================================================================
Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina).

Provides lightweight, in-RAM Key-Value state management across multi-turn 
interactive chat turns to accelerate autoregressive streaming and support 
Gemma 4 / Mistral sliding window attention patterns without extra memory footprint.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List


class ZeroCopyContextManager:
    """
    Manages rolling Key and Value vectors per transformer layer in RAM.
    """
    def __init__(self, max_context_length: int = 2048, sliding_window: Optional[int] = 512):
        self.max_context_length = max_context_length
        self.sliding_window = sliding_window
        # layer_idx -> {"k": np.ndarray, "v": np.ndarray}
        self._kv_cache: Dict[int, Dict[str, np.ndarray]] = {}

    def update_kv_cache(self, layer_idx: int, k_new: np.ndarray, v_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Appends new Key and Value vectors to layer cache and returns updated (K_seq, V_seq).
        Applies sliding window trimming if sliding_window is configured.
        """
        if layer_idx not in self._kv_cache:
            self._kv_cache[layer_idx] = {
                "k": k_new.reshape(1, -1) if k_new.ndim == 1 else k_new,
                "v": v_new.reshape(1, -1) if v_new.ndim == 1 else v_new
            }
        else:
            k_curr = self._kv_cache[layer_idx]["k"]
            v_curr = self._kv_cache[layer_idx]["v"]
            
            k_add = k_new.reshape(1, -1) if k_new.ndim == 1 else k_new
            v_add = v_new.reshape(1, -1) if v_new.ndim == 1 else v_new
            
            k_updated = np.vstack([k_curr, k_add])
            v_updated = np.vstack([v_curr, v_add])
            
            # Apply sliding window or max context length limits
            limit = self.sliding_window or self.max_context_length
            if k_updated.shape[0] > limit:
                k_updated = k_updated[-limit:]
                v_updated = v_updated[-limit:]
                
            self._kv_cache[layer_idx]["k"] = k_updated
            self._kv_cache[layer_idx]["v"] = v_updated

        return self._kv_cache[layer_idx]["k"], self._kv_cache[layer_idx]["v"]

    def clear(self):
        """
        Clears the KV cache across all layers.
        """
        self._kv_cache.clear()

    @property
    def cached_layers_count(self) -> int:
        return len(self._kv_cache)
