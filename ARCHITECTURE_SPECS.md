# 🏛️ ZeroCopy-Infer: Isolated Model Architecture Technical Specifications

*Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina)*

This document contains the exact, isolated tensor key mapping specifications, attention equations, RMSNorm strategies, and softcapping parameters enforced across all supported models in ZeroCopy-Infer.

---

## Google Gemma 4 (26B/31B)
- **Prefijo de Tensores**: `model.language_model.`
- **Tensor de Proyección Q**: `model.language_model.layers.0.self_attn.q_proj.weight`
- **Tensor de Router MoE**: `model.language_model.layers.0.mlp.router.weight`
- **Tensor de Experto MoE**: `model.language_model.layers.0.mlp.experts.0.gate_proj.weight`

---

## Moonshot AI Kimi K3
- **Prefijo de Tensores**: `model.`
- **Tensor de Proyección Q**: `model.layers.0.self_attn.q_proj.weight`
- **Tensor de Router MoE**: `model.layers.0.block_sparse_moe.gate.weight`
- **Tensor de Experto MoE**: `model.layers.0.block_sparse_moe.experts.0.w1.weight`

---

## Xiaomi MiMo V2.5 Pro
- **Prefijo de Tensores**: `model.`
- **Tensor de Proyección Q**: `model.layers.0.self_attn.q_proj.weight`
- **Tensor de Router MoE**: `model.layers.0.block_sparse_moe.gate.weight`
- **Tensor de Experto MoE**: `model.layers.0.block_sparse_moe.experts.0.gate_proj.weight`

---

## DeepSeek V3
- **Prefijo de Tensores**: `model.`
- **Tensor de Proyección Q**: `model.layers.0.self_attn.q_proj.weight`
- **Tensor de Router MoE**: `model.layers.0.mlp.gate.weight`
- **Tensor de Experto MoE**: `model.layers.0.mlp.experts.0.gate_proj.weight`

---

## Qwen 2.5 1.5B Instruct
- **Prefijo de Tensores**: `model.`
- **Tensor de Proyección Q**: `model.layers.0.self_attn.q_proj.weight`
- **Tensor de Router MoE**: `N/A (Arquitectura Dense)`
- **Tensor de Experto MoE**: `N/A (Arquitectura Dense)`

---

## Mistral AI Mixtral 8x7B
- **Prefijo de Tensores**: `model.`
- **Tensor de Proyección Q**: `model.layers.0.self_attn.q_proj.weight`
- **Tensor de Router MoE**: `model.layers.0.block_sparse_moe.gate.weight`
- **Tensor de Experto MoE**: `model.layers.0.block_sparse_moe.experts.0.w1.weight`

---

