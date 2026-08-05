"""
ZeroCopy-Infer: Hardware Memory Pressure Guard & Emergency Purger
==================================================================
Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina).

Monitors system-level RAM pressure in real-time and triggers proactive LRU cache 
eviction to prevent OOM process kills on memory-constrained mobile devices (Android Termux).
"""

import os
import gc
import psutil
from typing import Callable, Optional

from python.zerocopy_infer.hardware_detector import detect_hardware


class MemoryPressureGuard:
    """
    Real-time system RAM pressure monitor that triggers emergency cache purge callbacks.
    """
    def __init__(self, target_max_ram_ratio: float = 0.85, purge_callback: Optional[Callable[[], int]] = None):
        self.target_max_ram_ratio = target_max_ram_ratio
        self.purge_callback = purge_callback

    def check_memory_pressure(self) -> bool:
        hw = detect_hardware()
        total_ram = hw.get("ram_total_gb", 4.0)
        avail_ram = hw.get("ram_available_gb", 2.0)
        
        if total_ram <= 0:
            return False

        used_ratio = (total_ram - avail_ram) / total_ram
        return used_ratio > self.target_max_ram_ratio

    def enforce_safety(self) -> int:
        """
        Enforces system safety: if RAM pressure is above target threshold, calls purge_callback.
        Returns number of bytes purged or 0.
        """
        if self.check_memory_pressure():
            gc.collect()
            if self.purge_callback:
                return self.purge_callback()
        return 0
