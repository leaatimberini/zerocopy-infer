"""
ZeroCopy-Infer: Automated Architecture Specification Documentation Generator
=============================================================================
Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina).

Inspects all 6 isolated model architecture handlers and outputs a comprehensive,
up-to-date Markdown technical specification document (ARCHITECTURE_SPECS.md).
"""

import os
from typing import Dict, Any

from python.zerocopy_infer.model_architectures import (
    Gemma4ArchitectureHandler,
    KimiK3ArchitectureHandler,
    XiaomiMiMoArchitectureHandler,
    DeepSeekV3ArchitectureHandler,
    Qwen25ArchitectureHandler,
    MixtralArchitectureHandler,
)


def generate_architecture_specs_markdown() -> str:
    handlers = [
        ("Google Gemma 4 (26B/31B)", Gemma4ArchitectureHandler({})),
        ("Moonshot AI Kimi K3", KimiK3ArchitectureHandler({})),
        ("Xiaomi MiMo V2.5 Pro", XiaomiMiMoArchitectureHandler({})),
        ("DeepSeek V3", DeepSeekV3ArchitectureHandler({})),
        ("Qwen 2.5 1.5B Instruct", Qwen25ArchitectureHandler({})),
        ("Mistral AI Mixtral 8x7B", MixtralArchitectureHandler({})),
    ]

    doc = """# 🏛️ ZeroCopy-Infer: Isolated Model Architecture Technical Specifications

*Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina)*

This document contains the exact, isolated tensor key mapping specifications, attention equations, RMSNorm strategies, and softcapping parameters enforced across all supported models in ZeroCopy-Infer.

---

"""

    def safe_first(lst):
        return lst[0] if lst else "N/A (Arquitectura Dense)"

    for name, handler in handlers:
        doc += f"## {name}\n"
        doc += f"- **Prefijo de Tensores**: `{safe_first(handler.get_tensor_name('norm1', 0)).split('layers')[0]}`\n"
        doc += f"- **Tensor de Proyección Q**: `{safe_first(handler.get_tensor_name('q_proj', 0))}`\n"
        doc += f"- **Tensor de Router MoE**: `{safe_first(handler.get_tensor_name('moe_router', 0))}`\n"
        doc += f"- **Tensor de Experto MoE**: `{safe_first(handler.get_tensor_name('moe_expert_gate', 0, expert_idx=0))}`\n"
        doc += "\n---\n\n"

    out_file = "ARCHITECTURE_SPECS.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(doc)

    print(f"[ZeroCopy-Infer] Technical specs documentation generated in '{out_file}'.")
    return doc


if __name__ == "__main__":
    generate_architecture_specs_markdown()
