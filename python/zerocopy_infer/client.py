"""
ZeroCopy-Infer: High-Level Python Client SDK
=============================================
Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina).

Provides a simple, elegant Python SDK for developers to run ZeroCopy-Infer 
in Python scripts, Jupyter notebooks, or backend services with minimal setup.
"""

from typing import Dict, Any, Optional, Generator, List, Callable

from python.zerocopy_infer.hf_range_stream import SafetensorsRangeStreamer
from python.zerocopy_infer.moe_inference_engine import ZeroCopyMoEEngine
from python.zerocopy_infer.hardware_detector import detect_hardware


class ZeroCopyClient:
    """
    High-level Python SDK client for ZeroCopy-Infer zero-disk streaming.
    """
    def __init__(
        self,
        model: str = "google/gemma-4-26B-A4B-it",
        layers: int = 4,
        ram_limit_gb: float = 4.0,
        token: Optional[str] = None
    ):
        self.model = model
        self.layers = layers
        self.ram_limit_gb = ram_limit_gb

        self.streamer = SafetensorsRangeStreamer(repo_id=model, token=token)
        self.streamer.load_index_json()
        
        self.engine = ZeroCopyMoEEngine(
            streamer=self.streamer,
            ram_cache_gb=ram_limit_gb,
            num_active_layers=layers
        )

    def chat(self, prompt: str, num_tokens: int = 20, stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Executes zero-disk streaming chat inference and returns completed response string.
        Optionally invokes stream_callback(word) on each generated token.
        """
        full_text = ""
        for step, token_id, word, latency, total_bytes in self.engine.generate_chat_response_stream(prompt, num_tokens=num_tokens):
            full_text += word
            if stream_callback:
                stream_callback(word)
        return full_text.strip()

    def get_hardware_status(self) -> Dict[str, Any]:
        """
        Returns detected hardware capabilities.
        """
        return detect_hardware()
