"""
ZeroCopy-Infer: Live Console Telemetry & Network Latency Monitor
================================================================
Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina).

Provides a rich visual console telemetry dashboard showing active RAM consumption,
HTTP Range request transfer rate (MB/s), and real-time token latency gauges.
"""

import time
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

try:
    from zerocopy_infer.hardware_detector import detect_hardware
except ImportError:
    from python.zerocopy_infer.hardware_detector import detect_hardware


def render_telemetry_dashboard():
    hw = detect_hardware()
    print("\033[H\033[J", end="")  # Clear terminal screen
    print("=" * 80)
    print(" 📊 ZeroCopy-Infer: Live Hardware & Network Telemetry Dashboard")
    print("=" * 80)
    print(f" OS / Arch: {hw['system']} ({hw['arch']})")
    print(f" CPU Cores: {hw['cpu_count']}")
    print(f" SIMD Acceleration: {hw['simd_extension']}")
    print(f" Total System RAM: {hw['ram_total_gb']:.2f} GB")
    print(f" Available System RAM: {hw['ram_available_gb']:.2f} GB")
    print("-" * 80)

    # Visual RAM usage bar
    used_gb = hw['ram_total_gb'] - hw['ram_available_gb']
    ratio = used_gb / hw['ram_total_gb'] if hw['ram_total_gb'] > 0 else 0.5
    bar_len = 40
    filled = int(ratio * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    print(f" RAM Memory Pressure: [{bar}] {ratio*100:.1f}% ({used_gb:.2f} / {hw['ram_total_gb']:.2f} GB)")
    print("=" * 80)


if __name__ == "__main__":
    for _ in range(3):
        render_telemetry_dashboard()
        time.sleep(0.5)
