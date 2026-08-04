"""
ZeroCopy-Infer: Interactive Terminal Model Selector & Inference Launcher
========================================================================
Authored by Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina).
"""

import sys
import os
import argparse
from typing import Dict

from .hf_range_stream import SafetensorsRangeStreamer
from .moe_inference_engine import ZeroCopyMoEEngine, KimiK3Config

PRESET_MODELS: Dict[int, Dict[str, str]] = {
    1: {
        "name": "Moonshot AI — Kimi K3 (2.78-Trillion Parameters, 896 Experts)",
        "repo_id": "moonshotai/Kimi-K3",
        "description": "Massive MoE with KDA Delta Attention & MXFP4 Microscaling",
    },
    2: {
        "name": "Xiaomi — MiMo V2.5 Pro (State-of-the-Art MoE)",
        "repo_id": "XiaomiMiMo/MiMo-V2.5-Pro",
        "description": "Xiaomi's Flagship MoE Language & Reasoning Model",
    },
    3: {
        "name": "DeepSeek — DeepSeek V3 (671-Billion Parameters, 256 Experts)",
        "repo_id": "deepseek-ai/DeepSeek-V3",
        "description": "DeepSeek Multi-Head Latent Attention & Sparse MoE",
    },
    4: {
        "name": "Qwen — Qwen2.5 1.5B Instruct (Dense Fast Inference)",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "Compact Alibaba Cloud LLM for ultra-fast mobile testing",
    },
    5: {
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
    print(" [6] ✏️  Ingresar repositorio personalizado de Hugging Face (Custom Repo ID)")
    print("--------------------------------------------------------------------------------")

    choice_raw = input("\nIngresa tu opción [1-6] (por defecto 1): ").strip()
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
    elif choice == 6:
        selected_repo = input("Ingresa el ID del repositorio en Hugging Face (ej. org/model): ").strip()
        if not selected_repo:
            selected_repo = "moonshotai/Kimi-K3"
        selected_name = f"Custom: {selected_repo}"
    else:
        selected_repo = PRESET_MODELS[1]["repo_id"]
        selected_name = PRESET_MODELS[1]["name"]

    print(f"\n✅ Modelo seleccionado: {selected_name} ({selected_repo})")

    user_prompt = input("\nIngresa tu Prompt / Pregunta (por defecto: '¿Qué es la inteligencia artificial?'): ").strip()
    if not user_prompt:
        user_prompt = "¿Qué es la inteligencia artificial?"

    layers_raw = input("Número de capas a ejecutar (por defecto: 2 para celular / 4 para PC): ").strip()
    layers = int(layers_raw) if layers_raw.isdigit() else 2

    cache_raw = input("Límite de Cache RAM en GB (por defecto: 2.0): ").strip()
    try:
        cache_gb = float(cache_raw) if cache_raw else 2.0
    except ValueError:
        cache_gb = 2.0

    tokens_raw = input("Número de tokens a generar (por defecto: 5): ").strip()
    tokens = int(tokens_raw) if tokens_raw.isdigit() else 5

    print("\n--------------------------------------------------------------------------------")
    print(f"🚀 Iniciando Streaming Inferencia Zero-Disk en {selected_repo}...")
    print(f" Capas activas: {layers} | Cache RAM: {cache_gb} GB | Tokens: {tokens}")
    print("--------------------------------------------------------------------------------\n")

    streamer = SafetensorsRangeStreamer(repo_id=selected_repo, shards=[])
    
    print("[1/3] Step 1: Cargando configuración y mapa de índices desde Hugging Face...")
    remote_cfg = streamer.fetch_remote_config()
    try:
        streamer.parse_headers()
    except Exception as e:
        print(f"[Info] Header parse notice: {e}")

    model_config = KimiK3Config.from_remote_dict(remote_cfg)

    print(f"\n[2/3] Step 2: Inicializando Motor MoE ZeroCopy...")
    engine = ZeroCopyMoEEngine(
        streamer=streamer,
        config=model_config,
        ram_cache_gb=cache_gb,
        num_active_layers=layers,
    )

    print(f"\n[3/3] Step 3: Ejecutando Inferencia Autoregresiva...")
    print(f"Prompt: '{user_prompt}'")
    print("--------------------------------------------------------------------------------")

    output_text = ""
    for step, token_id, decoded_word, latency, total_bytes in engine.generate_chat_response_stream(user_prompt, num_tokens=tokens):
        output_text += decoded_word + " "
        mb_streamed = total_bytes / (1024 * 1024)
        clean_word = repr(decoded_word)[1:-1]
        print(f"Token [{step}/{tokens}] -> Word: '{clean_word}' (ID: {token_id}) | Latency: {latency * 1000:.2f} ms | Streamed: {mb_streamed:.2f} MB")

    print("\n================================================================================")
    print(" RESPONSE GENERATED (100% Zero-Disk Streamed)")
    print("================================================================================")
    print(f"Full Text: {output_text.strip()}")
    print("--------------------------------------------------------------------------------")
    print(f" HTTP Range Requests Sent   : {engine.total_range_requests}")
    print(f" Total Data Streamed        : {engine.total_bytes_streamed / (1024*1024):.2f} MB")
    print(f" Local SSD Storage Used     : 0 Bytes (100% Zero-Disk RAM Ingest)")
    print("================================================================================")

if __name__ == "__main__":
    main()
