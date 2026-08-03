"""
ZeroCopy-Infer: Hugging Face HTTP Range Request Streamer
=========================================================
Authored by Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina).

Provides zero-disk streaming of .safetensors headers and tensor slices
directly from Hugging Face LFS / CDN endpoints into RAM.
"""

import json
import struct
import urllib.request
import urllib.error
import numpy as np
from typing import Dict, Any, Tuple, Optional

class HFRangeClient:
    """
    HTTP Client that executes byte-level Range Requests against Hugging Face URLs
    without saving any data to local disk.
    """
    def __init__(self, token: Optional[str] = None, timeout: float = 3.0):
        self.token = token
        self.timeout = timeout
        self.headers = {
            "User-Agent": "ZeroCopy-Infer/0.1.0 (Leandro Timberini AGI Engine)"
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def fetch_range(self, url: str, start_byte: int, end_byte: int) -> bytes:
        """
        Fetch a specific range [start_byte, end_byte] from HF CDN into RAM.
        Returns raw bytes.
        """
        req = urllib.request.Request(url, headers=self.headers)
        req.add_header("Range", f"bytes={start_byte}-{end_byte}")
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status in (200, 206):
                    return resp.read()
                else:
                    raise IOError(f"HTTP Range request failed with status {resp.status}")
        except Exception as e:
            raise IOError(f"Failed to fetch bytes {start_byte}-{end_byte} from {url}: {e}")

class SafetensorsRangeStreamer:
    """
    Remote .safetensors header parser and tensor range streamer.
    Parses headers from Hugging Face shards without downloading full files.
    """
    def __init__(self, repo_id: str, shards: list, token: Optional[str] = None):
        self.repo_id = repo_id
        self.shards = shards
        self.client = HFRangeClient(token=token)
        self.tensor_map: Dict[str, Dict[str, Any]] = {}
        self.base_url = f"https://huggingface.co/{repo_id}/resolve/main"

    def parse_headers(self) -> Dict[str, Dict[str, Any]]:
        """
        Parse header metadata from each shard by reading the first 8 bytes
        (header size) plus the JSON header length.
        """
        print(f"[ZeroCopy-Infer] Parsing remote .safetensors headers for repo: {self.repo_id}...")
        total_tensors = 0
        
        for shard_filename in self.shards:
            shard_url = f"{self.base_url}/{shard_filename}"
            # Fetch first 8 bytes to get uint64 header size
            header_size_bytes = self.client.fetch_range(shard_url, 0, 7)
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            
            # Fetch JSON header bytes [8, 8 + header_size - 1]
            json_bytes = self.client.fetch_range(shard_url, 8, 8 + header_size - 1)
            header_json = json.loads(json_bytes.decode("utf-8"))
            
            # Store tensor locations
            for tensor_name, info in header_json.items():
                if tensor_name == "__metadata__":
                    continue
                data_offsets = info["data_offsets"]
                data_start = 8 + header_size + data_offsets[0]
                data_end = 8 + header_size + data_offsets[1] - 1
                length = data_end - data_start + 1
                
                self.tensor_map[tensor_name] = {
                    "shard_url": shard_url,
                    "shard_file": shard_filename,
                    "dtype": info["dtype"],
                    "shape": info["shape"],
                    "start_byte": data_start,
                    "end_byte": data_end,
                    "length": length,
                }
                total_tensors += 1

        print(f"[ZeroCopy-Infer] Successfully indexed {total_tensors} tensors across {len(self.shards)} shards (0 Bytes written to SSD).")
        return self.tensor_map

    def fetch_tensor(self, tensor_name: str) -> np.ndarray:
        """
        Fetch a specific tensor by name directly from Hugging Face into a NumPy array in RAM.
        """
        if tensor_name not in self.tensor_map:
            raise KeyError(f"Tensor '{tensor_name}' not found in remote tensor map.")
        
        meta = self.tensor_map[tensor_name]
        raw_bytes = self.client.fetch_range(meta["shard_url"], meta["start_byte"], meta["end_byte"])
        
        dtype_map = {
            "F32": np.float32,
            "F16": np.float16,
            "BF16": np.uint16,
            "I64": np.int64,
            "I32": np.int32,
            "I8": np.int8,
            "U8": np.uint8,
        }
        np_dtype = dtype_map.get(meta["dtype"], np.uint8)
        arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(meta["shape"])
        return arr
