"""
ZeroCopy-Infer: Universal Interactive Terminal Model Selector & Inference Launcher
==================================================================================
Authored by Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina).
"""

import sys
import os
import argparse
from typing import Dict

from .hf_range_stream import SafetensorsRangeStreamer
from .moe_inference_engine import ZeroCopyMoEEngine, KimiK3Config
import platform
import time

PRESET_MODELS: Dict[int, Dict[str, str]] = {
    1: {
        "name": "Google — Gemma 4 26B A4B Instruct (MoE & Hybrid Attention)",
        "repo_id": "google/gemma-4-26B-A4B-it",
        "description": "Google's State-of-the-Art Gemma 4 26B Architecture",
    },
    2: {
        "name": "Google — Gemma 4 31B Instruct (Dense Flagship Model)",
        "repo_id": "google/gemma-4-31B-it",
        "description": "Google's Gemma 4 31B High-Precision Reasoning LLM",
    },
    3: {
        "name": "Moonshot AI — Kimi K3 (2.78-Trillion Parameters, 896 Experts)",
        "repo_id": "moonshotai/Kimi-K3",
        "description": "Massive MoE with KDA Delta Attention & MXFP4 Microscaling",
    },
    4: {
        "name": "Xiaomi — MiMo V2.5 Pro (State-of-the-Art MoE)",
        "repo_id": "XiaomiMiMo/MiMo-V2.5-Pro",
        "description": "Xiaomi's Flagship MoE Language & Reasoning Model",
    },
    5: {
        "name": "DeepSeek — DeepSeek V3 (671-Billion Parameters, 256 Experts)",
        "repo_id": "deepseek-ai/DeepSeek-V3",
        "description": "DeepSeek Multi-Head Latent Attention & Sparse MoE",
    },
    6: {
        "name": "Qwen — Qwen2.5 1.5B Instruct (Dense Fast Mobile)",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "Compact Alibaba Cloud LLM for ultra-fast mobile testing",
    },
    7: {
        "name": "Mistral AI — Mixtral 8x7B Instruct (Sparse MoE)",
        "repo_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "description": "Classic 8x7B Sparse Mixture-of-Experts",
    },
}

def print_banner():
    print("================================================================================")
    print(" 🚀 ZeroCopy-Infer: Universal Zero-Disk Cloud-Native MoE & LLM Streaming CLI")
    print(" Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina)")
    print("================================================================================")

def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print_banner()

    print("\n📋 Selecciona el modelo que deseas ejecutar en Zero-Disk RAM Ingest:")
    print("--------------------------------------------------------------------------------")
    for key, info in PRESET_MODELS.items():
        print(f" [{key}] {info['name']}")
        print(f"     HF Repo: {info['repo_id']} | {info['description']}")
    print(" [8] ✏️  Ingresar repositorio personalizado de Hugging Face (Custom Repo ID)")
    print("--------------------------------------------------------------------------------")

    choice_raw = input("\nIngresa tu opción [1-8] (por defecto 1): ").strip()
    if not choice_raw:
        choice = 1
    else:
        try:
            choice = int(choice_raw)
        except ValueError:
            choice = 1

    if choice in PRESET_MODELS:
        selected_repo = PRESET_MODELS[choice]["repo_id"]
        selected_name = PRESET_MODELS[choice]["name"]
    elif choice == 8:
        selected_repo = input("Ingresa el ID del repositorio en Hugging Face (ej. org/model): ").strip()
        if not selected_repo:
            selected_repo = "google/gemma-4-26B-A4B-it"
        selected_name = f"Custom: {selected_repo}"
    else:
        selected_repo = PRESET_MODELS[1]["repo_id"]
        selected_name = PRESET_MODELS[1]["name"]

    from .hardware_detector import detect_hardware
    hw = detect_hardware()
    is_mobile = hw["system"] == "Android" or hw["arch"] in ("aarch64", "arm64")

    default_layers = 1 if is_mobile else 4
    default_cache = 0.75 if is_mobile else 3.0

    print(f"\n✅ Modelo seleccionado: {selected_name} ({selected_repo})")

    layers_raw = input(f"Número de capas a ejecutar (por defecto: {default_layers} para {'celular' if is_mobile else 'PC'}): ").strip()
    layers = int(layers_raw) if layers_raw.isdigit() else default_layers

    cache_raw = input(f"Límite de Cache RAM en GB (por defecto: {default_cache} GB para celular): ").strip()
    try:
        cache_gb = float(cache_raw) if cache_raw else default_cache
    except ValueError:
        cache_gb = default_cache

    if is_mobile and cache_gb > 1.5:
        print(f"\033[93m[Aviso de Seguridad Móvil] Ajustando caché RAM a 1.2 GB para evitar cierre de Termux por el SO Android.\033[0m")
        cache_gb = 1.2

    tokens_raw = input("Número de tokens a generar (por defecto: 10): ").strip()
    tokens = int(tokens_raw) if tokens_raw.isdigit() else 10

    print("\n--------------------------------------------------------------------------------")
    print(f"🚀 Iniciando Streaming Inferencia Zero-Disk en {selected_repo}...")
    
    print("\n\033[96m[Hardware Status Dashboard]\033[0m")
    print(f" 💻 Architecture: {hw['arch']} | CPU Cores: {hw['cpu_count']} ({hw['system']})")
    print(f" ⚡ SIMD Acceleration: {hw['simd_extension']}")
    print(f" 💾 Available RAM: {hw['ram_available_gb']:.2f} GB / Total: {hw['ram_total_gb']:.2f} GB (Safe Limit: {cache_gb:.2f} GB)")
    print(f" 📚 Active Layers: {layers}")
    print("--------------------------------------------------------------------------------\n")

    streamer = SafetensorsRangeStreamer(repo_id=selected_repo, shards=[])
    
    print("[1/3] Step 1: Cargando configuración y mapa de índices desde Hugging Face...")
    remote_cfg = streamer.fetch_remote_config()
    try:
        streamer.parse_headers()
    except Exception as e:
        print(f"[Info] Header parse notice: {e}")

    model_config = KimiK3Config.from_remote_dict(remote_cfg)

    print(f"\n[2/3] Step 2: Inicializando Motor MoE/LLM ZeroCopy...")
    engine = ZeroCopyMoEEngine(
        streamer=streamer,
        config=model_config,
        ram_cache_gb=cache_gb,
        num_active_layers=layers,
    )

    # Attach MemoryPressureGuard for proactive RAM safety
    try:
        from .memory_guard import MemoryPressureGuard
        guard = MemoryPressureGuard(target_max_ram_ratio=0.85, purge_callback=engine.clear_lru_cache)
        guard.enforce_safety()
    except Exception:
        pass

    print(f"\n[3/3] Step 3: Iniciando Chat Interactivo Agentic (escribe 'exit' o 'quit' para salir)")
    print("--------------------------------------------------------------------------------")

    from .tokenizer import UniversalHFTokenizer
    tokenizer = UniversalHFTokenizer(repo_id=selected_repo)
    
    chat_history = []

    while True:
        try:
            user_prompt = input("\n\033[94m[Tú]\033[0m: ").strip()
        except EOFError:
            break
            
        if user_prompt.lower() in ["exit", "quit"]:
            break
        if not user_prompt:
            continue
            
        chat_history.append({"role": "user", "content": user_prompt})
        
        full_prompt = tokenizer.render_chat_prompt(chat_history, repo_id=selected_repo)

        print("\033[95m[ZeroCopy Agent]\033[0m: ", end="", flush=True)
        output_text = ""
        
        start_time = time.time()
        tokens_gen = 0
        total_latency = 0.0
        
        # Generación streaming fluida con self-healing
        try:
            for step, token_id, decoded_word, latency, total_bytes in engine.generate_chat_response_stream(full_prompt, num_tokens=tokens):
                print(f"\033[92m{decoded_word}\033[0m", end="", flush=True)
                output_text += decoded_word
                tokens_gen += 1
                total_latency += latency
        except Exception as e:
            print(f"\n\033[93m[Self-Healing] Recovered from generation error: {e}\033[0m")

        end_time = time.time()
        elapsed = end_time - start_time
        tps = tokens_gen / elapsed if elapsed > 0 else 0
        avg_lat = (total_latency / tokens_gen * 1000) if tokens_gen > 0 else 0

        print("\n")
        chat_history.append({"role": "assistant", "content": output_text.strip()})

        print("\033[90m--------------------------------------------------------------------------------")
        print(f" ⚡ Throughput: {tps:.2f} tokens/sec | Latency: {avg_lat:.1f} ms/token | Streamed: {engine.total_bytes_streamed / (1024*1024):.2f} MB")
        print(f" 🌐 HTTP Range Requests Sent: {engine.total_range_requests}")
        print("--------------------------------------------------------------------------------\033[0m")

if __name__ == "__main__":
    main()
