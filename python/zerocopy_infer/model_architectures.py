"""
ZeroCopy-Infer: Isolated Model Architecture Handlers & Specification Registries
=================================================================================
Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina).

Provides 100% strict, non-overlapping architectural specs, tensor name resolvers, 
and mathematical formulas for each supported model family:
- Google Gemma 4 (26B-A4B MoE/Hybrid & 31B Dense)
- Moonshot AI Kimi K3 (2.78T MoE, 896 Experts, KDA & MXFP4)
- Xiaomi MiMo V2.5 Pro (Flagship MoE)
- DeepSeek V3 (671B MoE, Multi-Head Latent Attention / MLA)
- Qwen 2.5 (Dense Fast Mobile)
- Mistral AI Mixtral 8x7B (Classic MoE)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class BaseArchitectureHandler:
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict

    def get_tensor_name(self, category: str, layer_idx: int, **kwargs) -> List[str]:
        raise NotImplementedError

    def apply_input_embedding_scale(self, embeds: np.ndarray, hidden_dim: int) -> np.ndarray:
        return embeds

    def apply_logit_softcapping(self, logits: np.ndarray) -> np.ndarray:
        return logits


class Gemma4ArchitectureHandler(BaseArchitectureHandler):
    """
    Google Gemma 4 Specifications:
    - Tensor Prefix: model.language_model.
    - Hidden dim: text_config.hidden_size (5376 for 31B, 2816 for 26B-A4B)
    - Embedding scale: embeds * sqrt(hidden_dim)
    - Logit softcapping: 30.0 * tanh(logits / 30.0)
    - Layer Norms: input_layernorm, post_attention_layernorm
    - Q/K RMSNorm: q_norm, k_norm
    """

    def get_tensor_name(self, category: str, layer_idx: int, **kwargs) -> List[str]:
        p = f"model.language_model.layers.{layer_idx}."
        p_alt = f"model.layers.{layer_idx}."

        if category == "norm1":
            return [f"{p}input_layernorm.weight", f"{p_alt}input_layernorm.weight"]
        elif category == "norm2":
            return [f"{p}post_attention_layernorm.weight", f"{p_alt}post_attention_layernorm.weight"]
        elif category == "q_proj":
            return [f"{p}self_attn.q_proj.weight", f"{p_alt}self_attn.q_proj.weight"]
        elif category == "k_proj":
            return [f"{p}self_attn.k_proj.weight", f"{p_alt}self_attn.k_proj.weight"]
        elif category == "v_proj":
            return [f"{p}self_attn.v_proj.weight", f"{p_alt}self_attn.v_proj.weight"]
        elif category == "o_proj":
            return [f"{p}self_attn.o_proj.weight", f"{p_alt}self_attn.o_proj.weight"]
        elif category == "q_norm":
            return [f"{p}self_attn.q_norm.weight", f"{p_alt}self_attn.q_norm.weight"]
        elif category == "k_norm":
            return [f"{p}self_attn.k_norm.weight", f"{p_alt}self_attn.k_norm.weight"]
        elif category == "mlp_gate":
            return [f"{p}mlp.gate_proj.weight", f"{p_alt}mlp.gate_proj.weight"]
        elif category == "mlp_up":
            return [f"{p}mlp.up_proj.weight", f"{p_alt}mlp.up_proj.weight"]
        elif category == "mlp_down":
            return [f"{p}mlp.down_proj.weight", f"{p_alt}mlp.down_proj.weight"]
        elif category == "moe_router":
            return [f"{p}mlp.router.weight", f"{p}mlp.gate.weight", f"{p_alt}mlp.gate.weight"]
        elif category == "moe_expert_gate":
            exp_idx = kwargs.get("expert_idx", 0)
            return [f"{p}mlp.experts.{exp_idx}.gate_proj.weight", f"{p_alt}mlp.experts.{exp_idx}.gate_proj.weight"]
        elif category == "moe_expert_up":
            exp_idx = kwargs.get("expert_idx", 0)
            return [f"{p}mlp.experts.{exp_idx}.up_proj.weight", f"{p_alt}mlp.experts.{exp_idx}.up_proj.weight"]
        elif category == "moe_expert_down":
            exp_idx = kwargs.get("expert_idx", 0)
            return [f"{p}mlp.experts.{exp_idx}.down_proj.weight", f"{p_alt}mlp.experts.{exp_idx}.down_proj.weight"]
        return []

    def apply_input_embedding_scale(self, embeds: np.ndarray, hidden_dim: int) -> np.ndarray:
        return embeds * np.sqrt(hidden_dim)

    def apply_logit_softcapping(self, logits: np.ndarray) -> np.ndarray:
        return 30.0 * np.tanh(logits / 30.0)


class KimiK3ArchitectureHandler(BaseArchitectureHandler):
    """
    Moonshot AI Kimi K3 Specifications:
    - Tensor Prefix: model. or language_model.model.
    - Layers: 93 (69 KDA + 24 MLA)
    - MoE Router: block_sparse_moe.gate.weight (Sigmoid, Top-16 of 896 experts, routed_scale=2.5)
    - MoE Experts: block_sparse_moe.experts.{i}.w1.weight, w2.weight, w3.weight (MXFP4 micro-quantized)
    - Shared Experts: 2 shared experts (block_sparse_moe.shared_experts.{i}.w1.weight, w2.weight)
    """

    def get_tensor_name(self, category: str, layer_idx: int, **kwargs) -> List[str]:
        p = f"model.layers.{layer_idx}."
        p_alt = f"language_model.model.layers.{layer_idx}."

        if category == "norm1":
            return [f"{p}input_layernorm.weight", f"{p_alt}input_layernorm.weight"]
        elif category == "norm2":
            return [f"{p}post_attention_layernorm.weight", f"{p_alt}post_attention_layernorm.weight"]
        elif category == "q_proj":
            return [f"{p}self_attn.q_proj.weight", f"{p_alt}self_attn.q_proj.weight"]
        elif category == "k_proj":
            return [f"{p}self_attn.k_proj.weight", f"{p_alt}self_attn.k_proj.weight"]
        elif category == "v_proj":
            return [f"{p}self_attn.v_proj.weight", f"{p_alt}self_attn.v_proj.weight"]
        elif category == "o_proj":
            return [f"{p}self_attn.o_proj.weight", f"{p_alt}self_attn.o_proj.weight"]
        elif category == "moe_router":
            return [f"{p}block_sparse_moe.gate.weight", f"{p_alt}block_sparse_moe.gate.weight"]
        elif category == "moe_expert_gate":
            e = kwargs.get("expert_idx", 0)
            return [f"{p}block_sparse_moe.experts.{e}.w1.weight", f"{p_alt}block_sparse_moe.experts.{e}.w1.weight"]
        elif category == "moe_expert_down":
            e = kwargs.get("expert_idx", 0)
            return [f"{p}block_sparse_moe.experts.{e}.w2.weight", f"{p_alt}block_sparse_moe.experts.{e}.w2.weight"]
        elif category == "moe_expert_up":
            e = kwargs.get("expert_idx", 0)
            return [f"{p}block_sparse_moe.experts.{e}.w3.weight", f"{p_alt}block_sparse_moe.experts.{e}.w3.weight"]
        elif category == "shared_expert_gate":
            e = kwargs.get("shared_idx", 0)
            return [f"{p}block_sparse_moe.shared_experts.{e}.w1.weight", f"{p_alt}block_sparse_moe.shared_experts.{e}.w1.weight"]
        elif category == "shared_expert_down":
            e = kwargs.get("shared_idx", 0)
            return [f"{p}block_sparse_moe.shared_experts.{e}.w2.weight", f"{p_alt}block_sparse_moe.shared_experts.{e}.w2.weight"]
        return []


class XiaomiMiMoArchitectureHandler(BaseArchitectureHandler):
    """
    Xiaomi MiMo V2.5 Pro Specifications:
    - Tensor Prefix: model.
    - MoE Router: block_sparse_moe.gate.weight (Top-8 of 64 experts)
    - MoE Experts: block_sparse_moe.experts.{i}.gate_proj.weight, up_proj.weight, down_proj.weight
    - Dense MLP fallback for non-MoE layers
    """

    def get_tensor_name(self, category: str, layer_idx: int, **kwargs) -> List[str]:
        p = f"model.layers.{layer_idx}."
        if category == "norm1":
            return [f"{p}input_layernorm.weight"]
        elif category == "norm2":
            return [f"{p}post_attention_layernorm.weight"]
        elif category == "q_proj":
            return [f"{p}self_attn.q_proj.weight"]
        elif category == "k_proj":
            return [f"{p}self_attn.k_proj.weight"]
        elif category == "v_proj":
            return [f"{p}self_attn.v_proj.weight"]
        elif category == "o_proj":
            return [f"{p}self_attn.o_proj.weight"]
        elif category == "mlp_gate":
            return [f"{p}mlp.gate_proj.weight"]
        elif category == "mlp_up":
            return [f"{p}mlp.up_proj.weight"]
        elif category == "mlp_down":
            return [f"{p}mlp.down_proj.weight"]
        elif category == "moe_router":
            return [f"{p}block_sparse_moe.gate.weight", f"{p}mlp.gate.weight"]
        elif category == "moe_expert_gate":
            e = kwargs.get("expert_idx", 0)
            return [f"{p}block_sparse_moe.experts.{e}.gate_proj.weight", f"{p}block_sparse_moe.experts.{e}.w1.weight"]
        elif category == "moe_expert_up":
            e = kwargs.get("expert_idx", 0)
            return [f"{p}block_sparse_moe.experts.{e}.up_proj.weight", f"{p}block_sparse_moe.experts.{e}.w3.weight"]
        elif category == "moe_expert_down":
            e = kwargs.get("expert_idx", 0)
            return [f"{p}block_sparse_moe.experts.{e}.down_proj.weight", f"{p}block_sparse_moe.experts.{e}.w2.weight"]
        return []


class DeepSeekV3ArchitectureHandler(BaseArchitectureHandler):
    """
    DeepSeek V3 Specifications:
    - MLA (Multi-Head Latent Attention): q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj
    - MoE Router: mlp.gate.weight (Top-8 of 256 experts)
    - Shared Experts: mlp.shared_experts
    """

    def get_tensor_name(self, category: str, layer_idx: int, **kwargs) -> List[str]:
        p = f"model.layers.{layer_idx}."
        if category == "norm1":
            return [f"{p}input_layernorm.weight"]
        elif category == "norm2":
            return [f"{p}post_attention_layernorm.weight"]
        elif category == "q_proj":
            return [f"{p}self_attn.q_proj.weight", f"{p}self_attn.q_a_proj.weight"]
        elif category == "k_proj":
            return [f"{p}self_attn.k_proj.weight", f"{p}self_attn.kv_a_proj_with_mqa.weight"]
        elif category == "v_proj":
            return [f"{p}self_attn.v_proj.weight"]
        elif category == "o_proj":
            return [f"{p}self_attn.o_proj.weight"]
        elif category == "moe_router":
            return [f"{p}mlp.gate.weight", f"{p}block_sparse_moe.gate.weight"]
        elif category == "moe_expert_gate":
            e = kwargs.get("expert_idx", 0)
            return [f"{p}mlp.experts.{e}.gate_proj.weight", f"{p}block_sparse_moe.experts.{e}.w1.weight"]
        elif category == "moe_expert_up":
            e = kwargs.get("expert_idx", 0)
            return [f"{p}mlp.experts.{e}.up_proj.weight", f"{p}block_sparse_moe.experts.{e}.w3.weight"]
        elif category == "moe_expert_down":
            e = kwargs.get("expert_idx", 0)
            return [f"{p}mlp.experts.{e}.down_proj.weight", f"{p}block_sparse_moe.experts.{e}.w2.weight"]
        return []


class Qwen25ArchitectureHandler(BaseArchitectureHandler):
    """
    Qwen 2.5 Specifications:
    - Dense Flagship Architecture with GQA (Grouped Query Attention)
    - Tensors: self_attn.q_proj, k_proj, v_proj, o_proj
    - MLP: mlp.gate_proj, up_proj, down_proj
    """

    def get_tensor_name(self, category: str, layer_idx: int, **kwargs) -> List[str]:
        p = f"model.layers.{layer_idx}."
        if category == "norm1":
            return [f"{p}input_layernorm.weight"]
        elif category == "norm2":
            return [f"{p}post_attention_layernorm.weight"]
        elif category == "q_proj":
            return [f"{p}self_attn.q_proj.weight"]
        elif category == "k_proj":
            return [f"{p}self_attn.k_proj.weight"]
        elif category == "v_proj":
            return [f"{p}self_attn.v_proj.weight"]
        elif category == "o_proj":
            return [f"{p}self_attn.o_proj.weight"]
        elif category == "mlp_gate":
            return [f"{p}mlp.gate_proj.weight"]
        elif category == "mlp_up":
            return [f"{p}mlp.up_proj.weight"]
        elif category == "mlp_down":
            return [f"{p}mlp.down_proj.weight"]
        return []


class MixtralArchitectureHandler(BaseArchitectureHandler):
    """
    Mistral / Mixtral 8x7B Specifications:
    - Classic 8x7B MoE (Top-2 of 8 experts per token)
    - Tensors: block_sparse_moe.gate.weight
    - Experts: block_sparse_moe.experts.{i}.w1.weight, w2.weight, w3.weight
    """

    def get_tensor_name(self, category: str, layer_idx: int, **kwargs) -> List[str]:
        p = f"model.layers.{layer_idx}."
        if category == "norm1":
            return [f"{p}input_layernorm.weight"]
        elif category == "norm2":
            return [f"{p}post_attention_layernorm.weight"]
        elif category == "q_proj":
            return [f"{p}self_attn.q_proj.weight"]
        elif category == "k_proj":
            return [f"{p}self_attn.k_proj.weight"]
        elif category == "v_proj":
            return [f"{p}self_attn.v_proj.weight"]
        elif category == "o_proj":
            return [f"{p}self_attn.o_proj.weight"]
        elif category == "mlp_gate":
            return [f"{p}mlp.gate_proj.weight"]
        elif category == "mlp_up":
            return [f"{p}mlp.up_proj.weight"]
        elif category == "mlp_down":
            return [f"{p}mlp.down_proj.weight"]
        elif category == "moe_router":
            return [f"{p}block_sparse_moe.gate.weight"]
        elif category == "moe_expert_gate":
            e = kwargs.get("expert_idx", 0)
            return [f"{p}block_sparse_moe.experts.{e}.w1.weight"]
        elif category == "moe_expert_down":
            e = kwargs.get("expert_idx", 0)
            return [f"{p}block_sparse_moe.experts.{e}.w2.weight"]
        elif category == "moe_expert_up":
            e = kwargs.get("expert_idx", 0)
            return [f"{p}block_sparse_moe.experts.{e}.w3.weight"]
        return []


def get_architecture_handler(repo_id: str, config_dict: Dict[str, Any]) -> BaseArchitectureHandler:
    repo_lower = repo_id.lower()
    model_type = config_dict.get("model_type", "").lower()

    if "gemma-4" in repo_lower or "gemma4" in repo_lower or model_type == "gemma4":
        return Gemma4ArchitectureHandler(config_dict)
    elif "kimi" in repo_lower or model_type == "kimi_k3":
        return KimiK3ArchitectureHandler(config_dict)
    elif "mimo" in repo_lower or model_type == "mimo":
        return XiaomiMiMoArchitectureHandler(config_dict)
    elif "deepseek" in repo_lower or model_type == "deepseek_v3":
        return DeepSeekV3ArchitectureHandler(config_dict)
    elif "qwen" in repo_lower or model_type == "qwen2":
        return Qwen25ArchitectureHandler(config_dict)
    elif "mixtral" in repo_lower or "mistral" in repo_lower:
        return MixtralArchitectureHandler(config_dict)
    else:
        return KimiK3ArchitectureHandler(config_dict)


MODEL_ARCH_REGISTRY: Dict[str, Any] = {
    "gemma4": Gemma4ArchitectureHandler,
    "kimi_k3": KimiK3ArchitectureHandler,
    "mimo": XiaomiMiMoArchitectureHandler,
    "deepseek_v3": DeepSeekV3ArchitectureHandler,
    "qwen2": Qwen25ArchitectureHandler,
    "mixtral": MixtralArchitectureHandler,
}
