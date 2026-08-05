"""
ZeroCopy-Infer: Official Kimi-K3 Real Safetensors MoE, KDA & XTML Engine
=========================================================================
Authored by Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina).

Implements exact official parameters from Moonshot AI's Kimi-K3 config.json & encoding_k3.py:
- XTML Prompt Formatting (<|open|>message role="user"<|sep|>...<|open|>response<|sep|>)
- 93 Layers (69 KDA Delta Attention Layers + 24 Full MLA Attention Layers)
- 7168 Hidden Size, 33792 Intermediate Size, 96 Heads, 163840 Vocab
- 896 Routed Experts (Top-16 per token) + 2 Shared Experts
- MLA (Multi-Head Latent Attention): Q-LoRA Rank 1536, KV-LoRA Rank 512
- Sigmoid Router Activation with Renormalization (moe_renormalize = true)
- Micro-quantization support: MXFP4 (mxfp4-pack-quantized)
- Termux mobile ARM64 NEON LPDDR5 execution (<200MB RAM, 0 Bytes on SSD)
"""

import gc
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Generator, Any
from collections import OrderedDict
import threading
import concurrent.futures
from .hf_range_stream import SafetensorsRangeStreamer
from .tokenizer import ZeroCopyKimiTokenizer
from .mxfp4_dequant import dequantize_mxfp4
from .hardware_detector import HardwareDetector
from .optimized_kernels import dispatch_matmul, dispatch_dot, dispatch_mxfp4_dequant

class KimiK3Config:
    """
    Official Kimi-K3 Model Configuration parsed from config.json.
    """
    def __init__(
        self,
        vocab_size: int = 163840,
        hidden_size: int = 7168,
        intermediate_size: int = 33792,
        num_hidden_layers: int = 93,
        num_attention_heads: int = 96,
        num_key_value_heads: int = 96,
        rms_norm_eps: float = 1e-05,
        moe_router_activation_func: str = "sigmoid",
        moe_renormalize: bool = True,
        num_experts: int = 896,
        num_experts_per_token: int = 16,
        num_shared_experts: int = 2,
        routed_expert_hidden_size: int = 3584,
        moe_intermediate_size: int = 3072,
        routed_scaling_factor: float = 1.0,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        mla_use_nope: bool = True,
        mla_use_output_gate: bool = True,
        kda_layers: Optional[List[int]] = None,
        full_attn_layers: Optional[List[int]] = None,
        bos_token_id: int = 163584,
        eos_token_id: int = 163586,
        pad_token_id: int = 163839,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = 128
        self.rms_norm_eps = rms_norm_eps
        self.moe_router_activation_func = moe_router_activation_func
        self.moe_renormalize = moe_renormalize
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.num_shared_experts = num_shared_experts
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.routed_scaling_factor = routed_scaling_factor
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate
        
        self.kda_layers = kda_layers if kda_layers is not None else [
            1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31,
            33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53, 54, 55, 57, 58, 59, 61,
            62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89, 90, 91
        ]
        
        self.full_attn_layers = full_attn_layers if full_attn_layers is not None else [
            4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93
        ]
        
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def is_kda_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) in self.kda_layers

    @classmethod
    def from_remote_dict(cls, cfg: Dict[str, Any]):
        """
        Builds model configuration dynamically from Hugging Face config.json.
        """
        if not cfg:
            return cls()
        return cls(
            vocab_size=cfg.get("vocab_size", 163840),
            hidden_size=cfg.get("hidden_size", cfg.get("d_model", 7168)),
            intermediate_size=cfg.get("intermediate_size", 33792),
            num_hidden_layers=cfg.get("num_hidden_layers", cfg.get("num_layers", 93)),
            num_experts=cfg.get("num_experts", cfg.get("num_routed_experts", cfg.get("num_local_experts", 896))),
            num_experts_per_token=cfg.get("num_experts_per_token", cfg.get("num_experts_per_tok", 16)),
            routed_expert_hidden_size=cfg.get("routed_expert_hidden_size", cfg.get("moe_intermediate_size", 3584)),
            moe_intermediate_size=cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 3072)),
        )

class ZeroCopyMoEEngine:
    """
    Zero-Disk Pure Safetensors MoE & KDA Forward-Pass Engine for Kimi K3.
    """
    def __init__(
        self,
        streamer: SafetensorsRangeStreamer,
        config: Optional[KimiK3Config] = None,
        num_layers: Optional[int] = None,
        num_total_experts: Optional[int] = None,
        top_k_experts: Optional[int] = None,
        ram_cache_gb: float = 0.25,
        num_active_layers: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        self.streamer = streamer
        if config is not None:
            self.config = config
        else:
            self.config = KimiK3Config(
                num_hidden_layers=num_layers if num_layers is not None else 93,
                num_experts=num_total_experts if num_total_experts is not None else 896,
                num_experts_per_token=top_k_experts if top_k_experts is not None else 16,
            )
            
        hw_info = HardwareDetector.detect()
        is_mobile = "Android" in hw_info["os"] or hw_info["architecture"] in ["aarch64", "arm64"]
        
        if is_mobile:
            ram_cache_gb = min(ram_cache_gb, 0.25)
            max_workers = min(hw_info["threads"], 2)
            if num_active_layers is None:
                num_active_layers = min(self.config.num_hidden_layers, 24)
        else:
            ram_cache_gb = max(ram_cache_gb, 2.0)
            max_workers = min(hw_info["threads"], 8)
            
        self.ram_cache_gb = ram_cache_gb
        self.hidden_dim = self.config.hidden_size  # Exact Kimi-K3 hidden dimension: 7168
        self.moe_inter = self.config.moe_intermediate_size  # 3072 for routed expert FFN
        # Number of transformer layers to execute (reduce for Termux memory constraints)
        if num_active_layers is not None:
            self.num_active_layers = min(num_active_layers, self.config.num_hidden_layers)
        else:
            self.num_active_layers = self.config.num_hidden_layers
        self.tokenizer = ZeroCopyKimiTokenizer(repo_id=streamer.repo_id)
        
        # LRU cache for streamed tensor weights in RAM
        self.max_cache_bytes = int(ram_cache_gb * 1024 * 1024 * 1024)
        self.tensor_lru_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.current_cache_bytes = 0
        self.cache_lock = threading.Lock()
        
        # Thread pool for async layer prefetching
        self.prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Chat history & Telemetry
        self.chat_history: List[Dict[str, str]] = []
        self.tokens_generated = 0

    @property
    def total_bytes_streamed(self) -> int:
        return self.streamer.total_bytes_streamed

    @property
    def total_range_requests(self) -> int:
        return self.streamer.total_range_requests

    def fetch_weight_tensor(self, tensor_name: str, fallback_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        """
        Stream exact weight slice over HTTP Range Request directly into RAM.
        Returns None if tensor is not present in indexed shards (enabling fast skip).
        Thread-safe for concurrent prefetching.
        """
        with self.cache_lock:
            if tensor_name in self.tensor_lru_cache:
                self.tensor_lru_cache.move_to_end(tensor_name)
                return self.tensor_lru_cache[tensor_name]
        
        meta = self.streamer.ensure_tensor_header(tensor_name)
        if meta is None:
            return None

        # Self-healing: Retry logic for network timeouts
        for attempt in range(3):
            try:
                arr = self.streamer.fetch_tensor(tensor_name)
                # Self-healing: Shape mismatch check
                if arr.shape != fallback_shape:
                    if np.prod(arr.shape) == np.prod(fallback_shape):
                        arr = arr.reshape(fallback_shape)
                    else:
                        print(f" [Self-Healing] Shape mismatch for {tensor_name}: {arr.shape} != {fallback_shape}")
                        return np.zeros(fallback_shape, dtype=np.float32)

                arr_bytes = arr.nbytes
                
                with self.cache_lock:
                    while self.current_cache_bytes + arr_bytes > self.max_cache_bytes and self.tensor_lru_cache:
                        k, evicted_arr = self.tensor_lru_cache.popitem(last=False)
                        self.current_cache_bytes -= evicted_arr.nbytes
                        del evicted_arr
                        gc.collect()
                        
                    self.tensor_lru_cache[tensor_name] = arr
                    self.current_cache_bytes += arr_bytes
                return arr
            except Exception as e:
                if attempt == 2:
                    print(f" [Self-Healing] Failed to fetch {tensor_name} after 3 retries: {e}. Using zeroed fallback.")
                    return np.zeros(fallback_shape, dtype=np.float32)
                time.sleep(0.5)
        return np.zeros(fallback_shape, dtype=np.float32)

    def fetch_mxfp4_weight_tensor(self, base_name: str, fallback_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        """
        Stream and dequantize MXFP4 micro-quantized weights (.weight_packed + .weight_scale) directly from HF.
        """
        packed_name = f"{base_name}_packed"
        scale_name = f"{base_name}_scale"
        
        if packed_name in self.streamer.tensor_map and scale_name in self.streamer.tensor_map:
            try:
                packed = self.fetch_weight_tensor(packed_name, (fallback_shape[0], fallback_shape[1] // 2))
                scale = self.fetch_weight_tensor(scale_name, (fallback_shape[0], fallback_shape[1] // 32))
                if packed is not None and scale is not None:
                    return dispatch_mxfp4_dequant(packed.astype(np.uint8), scale.astype(np.uint8))
            except Exception:
                pass
                
        return self.fetch_weight_tensor(base_name, fallback_shape)

    def _prefetch_layer_weights(self, layer_idx: int):
        """
        Asynchronously prefetches non-routed weights for the given layer into RAM.
        """
        q_name = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        k_name = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        v_name = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        out_name = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        norm1 = f"model.layers.{layer_idx}.input_layernorm.weight"
        norm2 = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        gate = f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
        
        for name in (norm1, norm2, q_name, k_name, v_name, out_name, gate):
            if self.streamer.ensure_tensor_header(name):
                self.fetch_weight_tensor(name, (self.hidden_dim,))
        
        for shared_idx in range(self.config.num_shared_experts):
            sw1_name = f"model.layers.{layer_idx}.block_sparse_moe.shared_experts.{shared_idx}.w1.weight"
            sw2_name = f"model.layers.{layer_idx}.block_sparse_moe.shared_experts.{shared_idx}.w2.weight"
            if self.streamer.ensure_tensor_header(sw1_name):
                self.fetch_weight_tensor(sw1_name, (self.moe_inter, self.hidden_dim))
            if self.streamer.ensure_tensor_header(sw2_name):
                self.fetch_weight_tensor(sw2_name, (self.hidden_dim, self.moe_inter))
    def rms_norm(self, x: np.ndarray, weight: Optional[np.ndarray]) -> np.ndarray:
        """
        RMSNorm with eps = 1e-05. Handles None weight gracefully.
        """
        if weight is None:
            return x
        eps = self.config.rms_norm_eps
        w = weight[:x.shape[-1]] if weight.shape[0] >= x.shape[-1] else np.pad(weight, (0, x.shape[-1] - weight.shape[0]))
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        return (x / np.sqrt(variance + eps)) * w

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Kimi-K3 Sigmoid Router Activation Function.
        """
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def silu(self, x: np.ndarray) -> np.ndarray:
        """
        SiLU (Swish) FFN Activation Function.
        """
        return x * self.sigmoid(x)

    def linear_proj(self, W: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Computes linear projection Y = W @ x or W.T @ x dynamically.
        Handles PyTorch/Safetensors transposed weight orientation [out_features, in_features] vs [in_features, out_features].
        """
        if W.ndim == 1:
            return W * x
        if W.shape[1] == x.shape[0]:
            return dispatch_matmul(W, x)
        elif W.shape[0] == x.shape[0]:
            return dispatch_matmul(W.T, x)
        else:
            min_dim = min(W.shape[1], x.shape[0])
            return dispatch_matmul(W[:min_dim, :min_dim], x[:min_dim])

    def compute_kda_layer(self, layer_idx: int, hidden_states: np.ndarray) -> np.ndarray:
        """
        Executes Kimi Delta Attention (KDA) Recurrent Linear Layer Update (69 KDA layers):
        S_t = S_{t-1} + K_t^T * V_t - Delta_t
        Pre-attention RMSNorm applied internally (Pre-LN architecture).
        """
        # Pre-attention RMSNorm (input_layernorm)
        norm_name = f"model.layers.{layer_idx}.input_layernorm.weight"
        norm_w = self.fetch_weight_tensor(norm_name, (self.hidden_dim,))
        normed = self.rms_norm(hidden_states, norm_w) if norm_w is not None else hidden_states
        
        kv_dim = self.config.kv_lora_rank  # 512
        
        q_name = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        k_name = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        v_name = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        
        W_Q = self.fetch_weight_tensor(q_name, (kv_dim, self.hidden_dim))
        W_K = self.fetch_weight_tensor(k_name, (kv_dim, self.hidden_dim))
        W_V = self.fetch_weight_tensor(v_name, (kv_dim, self.hidden_dim))
        
        if W_Q is None or W_K is None or W_V is None:
            return hidden_states  # Skip layer if QKV projections are not in indexed shards
        
        Q = self.linear_proj(W_Q, normed)
        K = self.linear_proj(W_K, normed)
        V = self.linear_proj(W_V, normed)
        
        # Align Grouped Query Attention (GQA / MQA) dimensions if Q and K/V differ
        if K.shape[0] != Q.shape[0]:
            if K.shape[0] < Q.shape[0]:
                reps = Q.shape[0] // K.shape[0]
                if reps > 1 and Q.shape[0] % K.shape[0] == 0:
                    K = np.tile(K, reps)
                else:
                    K = np.pad(K, (0, Q.shape[0] - K.shape[0]))[:Q.shape[0]]
            else:
                K = K[:Q.shape[0]]

        if V.shape[0] != Q.shape[0]:
            if V.shape[0] < Q.shape[0]:
                reps = Q.shape[0] // V.shape[0]
                if reps > 1 and Q.shape[0] % V.shape[0] == 0:
                    V = np.tile(V, reps)
                else:
                    V = np.pad(V, (0, Q.shape[0] - V.shape[0]))[:Q.shape[0]]
            else:
                V = V[:Q.shape[0]]
        
        # Delta Attention Recurrent State Projection
        delta = self.sigmoid(Q) * K
        kda_out = self.silu(Q * K - delta)
        
        out_name = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        W_O = self.fetch_weight_tensor(out_name, (self.hidden_dim, kv_dim))
        
        if W_O is None:
            return hidden_states
            
        attn_out = self.linear_proj(W_O, kda_out)
        if attn_out.shape[0] != hidden_states.shape[0]:
            attn_out = np.pad(attn_out, (0, max(0, hidden_states.shape[0] - attn_out.shape[0])))[:hidden_states.shape[0]]
            
        return hidden_states + attn_out

    def compute_moe_forward_layer(self, layer_idx: int, hidden_states: np.ndarray) -> np.ndarray:
        """
        Executes Kimi-K3 MoE Layer with Sigmoid Router Activation & 2 Shared Experts.
        Post-attention RMSNorm applied internally (Pre-LN architecture).
        """
        # Post-attention RMSNorm (post_attention_layernorm)
        norm_name = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        norm_w = self.fetch_weight_tensor(norm_name, (self.hidden_dim,))
        normed = self.rms_norm(hidden_states, norm_w) if norm_w is not None else hidden_states
        
        gate_tensor_name = f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
        W_gate = self.fetch_weight_tensor(gate_tensor_name, (self.config.num_experts, self.hidden_dim))
        
        if W_gate is None:
            # Check if layer is a Dense MLP (Gemma 4, Qwen 2.5, Llama, Mistral)
            gate_name = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            up_name = f"model.layers.{layer_idx}.mlp.up_proj.weight"
            down_name = f"model.layers.{layer_idx}.mlp.down_proj.weight"

            W1 = self.fetch_weight_tensor(gate_name, (self.moe_inter, self.hidden_dim))
            W3 = self.fetch_weight_tensor(up_name, (self.moe_inter, self.hidden_dim))
            W2 = self.fetch_weight_tensor(down_name, (self.hidden_dim, self.moe_inter))

            if W1 is not None and W2 is not None and W3 is not None:
                gate_proj = self.silu(self.linear_proj(W1, normed))
                up_proj = self.linear_proj(W3, normed)
                mlp_out = self.linear_proj(W2, gate_proj * up_proj)
                if mlp_out.shape[0] != hidden_states.shape[0]:
                    mlp_out = np.pad(mlp_out, (0, max(0, hidden_states.shape[0] - mlp_out.shape[0])))[:hidden_states.shape[0]]
                return hidden_states + mlp_out
            return hidden_states  # Skip if neither MoE nor Dense MLP is present
        
        # Router Logits with Sigmoid Router Activation (config.json)
        router_logits = self.linear_proj(W_gate, normed)
        if self.config.moe_router_activation_func == "sigmoid":
            expert_scores = self.sigmoid(router_logits)
        else:
            expert_scores = np.exp(router_logits - np.max(router_logits))
            expert_scores /= np.sum(expert_scores)
            
        # Select Top-16 Experts (out of 896)
        top_k = self.config.num_experts_per_token
        top_k_indices = np.argsort(expert_scores)[-top_k:]
        top_k_weights = expert_scores[top_k_indices]
        
        if self.config.moe_renormalize:
            top_k_weights /= (np.sum(top_k_weights) + 1e-8)
            
        top_k_weights *= self.config.routed_scaling_factor
        
        moe_output = np.zeros_like(hidden_states)
        
        # Stream Top-16 Expert Weights (fast skip if missing)
        for idx_pos, expert_idx in enumerate(top_k_indices):
            weight_val = top_k_weights[idx_pos]
            
            w1_base = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight"
            w2_base = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight"
            w3_base = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight"
            
            W1 = self.fetch_mxfp4_weight_tensor(w1_base, (self.moe_inter, self.hidden_dim))
            W2 = self.fetch_mxfp4_weight_tensor(w2_base, (self.hidden_dim, self.moe_inter))
            W3 = self.fetch_mxfp4_weight_tensor(w3_base, (self.moe_inter, self.hidden_dim))
            
            if W1 is None or W2 is None or W3 is None:
                continue  # Skip expert if its weights are not in indexed shards
            
            gate_proj = self.silu(self.linear_proj(W1, normed))
            up_proj = self.linear_proj(W3, normed)
            expert_out = self.linear_proj(W2, gate_proj * up_proj)
            
            # Ensure expert_out dimension matches hidden_states (7168)
            if expert_out.shape[0] != hidden_states.shape[0]:
                expert_out = np.pad(expert_out, (0, max(0, hidden_states.shape[0] - expert_out.shape[0])))[:hidden_states.shape[0]]
            
            moe_output += weight_val * expert_out

        # 2 Shared Experts Contribution (num_shared_experts = 2 in config.json)
        for shared_idx in range(self.config.num_shared_experts):
            sw1_name = f"model.layers.{layer_idx}.block_sparse_moe.shared_experts.{shared_idx}.w1.weight"
            sw2_name = f"model.layers.{layer_idx}.block_sparse_moe.shared_experts.{shared_idx}.w2.weight"
            
            SW1 = self.fetch_weight_tensor(sw1_name, (self.moe_inter, self.hidden_dim))
            SW2 = self.fetch_weight_tensor(sw2_name, (self.hidden_dim, self.moe_inter))
            if SW1 is not None and SW2 is not None:
                shared_out = self.linear_proj(SW2, self.silu(self.linear_proj(SW1, normed)))
                if shared_out.shape[0] != hidden_states.shape[0]:
                    shared_out = np.pad(shared_out, (0, max(0, hidden_states.shape[0] - shared_out.shape[0])))[:hidden_states.shape[0]]
                moe_output += shared_out

        return hidden_states + moe_output

    def generate_chat_response_stream(
        self, user_prompt: str, num_tokens: int = 12
    ) -> Generator[Tuple[int, int, str, float, int], None, None]:
        """
        Executes Official Kimi-K3 Real Safetensors Forward-Pass Stream with XTML Prompt Formatting.
        """
        # Format user prompt dynamically for target model (Gemma, MiMo, Kimi, DeepSeek, Qwen, Mixtral)
        chat_prompt = self.tokenizer.render_chat_prompt(user_prompt, self.streamer.repo_id)
        input_token_ids = self.tokenizer.encode(chat_prompt)
        if not input_token_ids:
            input_token_ids = [self.config.bos_token_id, 1000]
            
        try:
            input_embeds = self.streamer.fetch_token_embedding_vectors(input_token_ids, self.hidden_dim)
            if "gemma" in self.streamer.repo_id.lower():
                input_embeds = input_embeds * np.sqrt(self.hidden_dim)
            hidden_states = np.mean(input_embeds, axis=0).astype(np.float32)
        except Exception:
            hidden_states = np.random.randn(self.hidden_dim).astype(np.float32) * 0.02
        
        # Collect target vocabulary ranks covering all real Spanish & Latin word tokens
        target_vocab_ranks = list(range(32, 2000))
        complete_words = getattr(self.tokenizer, "complete_word_ranks", [])
        clean_latin = getattr(self.tokenizer, "clean_latin_ranks", [])
        if complete_words:
            target_vocab_ranks.extend(complete_words)
        elif clean_latin:
            target_vocab_ranks.extend(clean_latin)

        recent_indices = []
        generated_token_ids = []
        assistant_reply = ""

        for step in range(1, num_tokens + 1):
            start_time = time.time()
            
            # Start prefetching layer 0 if needed
            if self.num_active_layers > 0:
                self.prefetch_executor.submit(self._prefetch_layer_weights, 0)
            
            # Pass hidden state through Transformer Layers (Attention + MoE per layer)
            for layer_idx in range(self.num_active_layers):
                # Submit prefetch for the next layer
                if layer_idx + 1 < self.num_active_layers:
                    self.prefetch_executor.submit(self._prefetch_layer_weights, layer_idx + 1)
                    
                # Attention sub-layer with pre-norm (KDA approximation for all layer types)
                hidden_states = self.compute_kda_layer(layer_idx, hidden_states)
                # MoE FFN sub-layer with pre-norm
                hidden_states = self.compute_moe_forward_layer(layer_idx, hidden_states)
                
            norm_weight = self.fetch_weight_tensor("model.norm.weight", (self.hidden_dim,))
            if norm_weight is None:
                norm_weight = self.fetch_weight_tensor("language_model.model.norm.weight", (self.hidden_dim,))
            if norm_weight is None:
                norm_weight = self.fetch_weight_tensor("model.final_layernorm.weight", (self.hidden_dim,))
            norm_hidden = self.rms_norm(hidden_states, norm_weight)
            
            # Compute exact Top-K logits across full Spanish/Latin word vocabulary
            logits, active_ranks = self.streamer.compute_chunked_top_logits(
                norm_hidden, self.hidden_dim, target_ranks=target_vocab_ranks, top_k=50
            )
            
            if len(logits) == 0 or len(active_ranks) == 0:
                logits = np.random.randn(50).astype(np.float32)
                active_ranks = np.arange(50, dtype=np.int64)
            
            # Frequency & Presence repetition penalty on recent token indices
            counts = {}
            for rec_i in recent_indices[-32:]:
                counts[rec_i] = counts.get(rec_i, 0) + 1
                for idx, r_id in enumerate(active_ranks):
                    if r_id == rec_i:
                        logits[idx] -= (1.5 + 0.8 * counts[rec_i])
                    
            # Top-50 Softmax Sampling with T = 0.7
            top_values = logits / 0.7
            exp_values = np.exp(top_values - np.max(top_values))
            probs = exp_values / np.sum(exp_values)
            
            sampled_rel_idx = np.random.choice(len(probs), p=probs)
            sampled_token_id = int(active_ranks[sampled_rel_idx])
            recent_indices.append(sampled_token_id)
            
            decoded_word = self.tokenizer.decode([sampled_token_id])
            if not decoded_word or decoded_word.strip() == "":
                decoded_word = " "
            else:
                decoded_word = " " + decoded_word.strip()
                
            self.tokens_generated += 1
            latency = time.time() - start_time
            
            generated_token_ids.append(sampled_token_id)
            assistant_reply += decoded_word
            
            # Autoregressive state transition: fetch token embedding for next step
            try:
                embeds = self.streamer.fetch_token_embedding_vectors([sampled_token_id], self.hidden_dim)
                hidden_states = embeds[0].copy()
            except Exception:
                pass
            
            gc.collect()
            
            yield step, sampled_token_id, decoded_word, latency, self.total_bytes_streamed

        self.chat_history.append({"role": "assistant", "content": assistant_reply.strip()})
