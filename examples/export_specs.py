"""
ZeroCopy-Infer: System Specs & Telemetry JSON Exporter
======================================================
Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina).

Exports device hardware capabilities, SIMD vector support, and network diagnostic 
metrics into a structured JSON file (system_specs.json).
"""

import json
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

try:
    from zerocopy_infer.hardware_detector import detect_hardware
    from zerocopy_infer.model_architectures import MODEL_ARCH_REGISTRY
except ImportError:
    from python.zerocopy_infer.hardware_detector import detect_hardware
    from python.zerocopy_infer.model_architectures import MODEL_ARCH_REGISTRY


def export_system_specs(out_file: str = "system_specs.json") -> Dict[str, Any]:
    hw = detect_hardware()
    models_supported = list(MODEL_ARCH_REGISTRY.keys())

    specs = {
        "engine": "ZeroCopy-Infer 1.0.0",
        "author": "Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina)",
        "hardware": hw,
        "supported_models_count": len(models_supported),
        "supported_model_keys": models_supported
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2)

    print(f"[ZeroCopy-Infer] System specs exported to '{out_file}'.")
    return specs


if __name__ == "__main__":
    export_system_specs()
