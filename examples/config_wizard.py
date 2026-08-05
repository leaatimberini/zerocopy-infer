"""
ZeroCopy-Infer: Hardware Auto-Tuning Setup Wizard & Profile Generator
======================================================================
Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina).

Auto-detects device hardware capabilities (RAM, CPU, SIMD) and generates 
an optimal zero-disk streaming configuration profile.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

try:
    from zerocopy_infer.hardware_detector import detect_hardware
except ImportError:
    from python.zerocopy_infer.hardware_detector import detect_hardware


def run_config_wizard() -> Dict[str, Any]:
    hw = detect_hardware()
    print("=" * 80)
    print(" [ZeroCopy-Infer] Hardware Auto-Tuning Configuration Wizard")
    print("=" * 80)
    print(f" OS / Arch: {hw['system']} ({hw['arch']})")
    print(f" CPU Cores: {hw['cpu_count']}")
    print(f" SIMD Vector Extension: {hw['simd_extension']}")
    print(f" Available RAM: {hw['ram_available_gb']:.2f} GB / Total: {hw['ram_total_gb']:.2f} GB")
    print("=" * 80)

    # Determine recommended profile
    ram_gb = hw["ram_total_gb"]
    if ram_gb <= 4.0:
        rec_model = "Qwen/Qwen2.5-1.5B-Instruct"
        rec_layers = 2
        rec_cache = 1.5
    elif ram_gb <= 8.0:
        rec_model = "google/gemma-4-26B-A4B-it"
        rec_layers = 4
        rec_cache = 3.5
    elif ram_gb <= 16.0:
        rec_model = "google/gemma-4-26B-A4B-it"
        rec_layers = 8
        rec_cache = 6.0
    else:
        rec_model = "google/gemma-4-31B-it"
        rec_layers = 16
        rec_cache = 12.0

    print("\n RECOMENDACIÓN AUTOMÁTICA SEGÚN TU HARDWARE:")
    print(f"  * Modelo Recomendado: {rec_model}")
    print(f"  * Capas Activas Recomendadas: {rec_layers}")
    print(f"  * Límite de Caché RAM: {rec_cache} GB")
    print("=" * 80)

    profile = {
        "hardware": hw,
        "recommended_model": rec_model,
        "recommended_layers": rec_layers,
        "recommended_cache_gb": rec_cache
    }

    out_path = "config_preset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"[ZeroCopy-Infer] Perfil guardado en '{out_path}'.")
    return profile


if __name__ == "__main__":
    run_config_wizard()
