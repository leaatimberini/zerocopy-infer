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
from typing import Dict, Any, Tuple, Optional, List

class HFRangeClient:
    """
    HTTP Client that executes byte-level Range Requests against Hugging Face URLs
    without saving any data to local disk.
    """
    def __init__(self, token: Optional[str] = None, timeout: float = 10.0):
        self.token = token
        self.timeout = timeout
        self.headers = {
            "User-Agent": "ZeroCopy-Infer/0.8.5 (Leandro Timberini AGI Engine)"
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
    def __init__(self, repo_id: str, shards: Optional[List[str]] = None, token: Optional[str] = None):
        self.repo_id = repo_id
        if shards is None:
            shards = ["model-00094-of-000096.safetensors"]
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
            try:
                header_size_bytes = self.client.fetch_range(shard_url, 0, 7)
                header_size = struct.unpack("<Q", header_size_bytes)[0]
                
                json_bytes = self.client.fetch_range(shard_url, 8, 8 + header_size - 1)
                header_json = json.loads(json_bytes.decode("utf-8"))
                
                for tensor_name, info in header_json.items():
                    if tensor_name == "__metadata__":
                        continue
                    data_offsets = info["data_offsets"]
                    data_start = 8 + header_size + data_offsets[0]
                    data_end = 8 + header_size + data_offsets[1] - 1
                    length = data_end - data_start + 1
                    
                    entry = {
                        "shard_url": shard_url,
                        "shard_file": shard_filename,
                        "dtype": info["dtype"],
                        "shape": info["shape"],
                        "start_byte": data_start,
                        "end_byte": data_end,
                        "length": length,
                    }
                    self.tensor_map[tensor_name] = entry
                    
                    # Store stripped prefix aliases
                    if tensor_name.startswith("language_model.model."):
                        short_key = tensor_name.replace("language_model.", "")
                        self.tensor_map[short_key] = entry
                    elif tensor_name.startswith("language_model."):
                        short_key = tensor_name.replace("language_model.", "")
                        self.tensor_map[short_key] = entry

                    total_tensors += 1
            except Exception as e:
                print(f"[ZeroCopy-Infer] Shard {shard_filename} notice: {e}")

        print(f"[ZeroCopy-Infer] Successfully indexed {total_tensors} tensors across {len(self.shards)} shards (0 Bytes written to SSD).")
        return self.tensor_map

    def fetch_tensor(self, tensor_name: str) -> np.ndarray:
        """
        Fetch a specific tensor by name directly from Hugging Face into a NumPy array in RAM.
        Converts BF16 automatically to FP32.
        """
        target_key = tensor_name
        if target_key not in self.tensor_map:
            # Fallback alias search
            alt_key = f"language_model.{tensor_name}"
            if alt_key in self.tensor_map:
                target_key = alt_key
            else:
                alt_key2 = f"language_model.model.{tensor_name.replace('model.', '')}"
                if alt_key2 in self.tensor_map:
                    target_key = alt_key2
                else:
                    raise KeyError(f"Tensor '{tensor_name}' not found in remote tensor map.")

        meta = self.tensor_map[target_key]
        raw_bytes = self.client.fetch_range(meta["shard_url"], meta["start_byte"], meta["end_byte"])
        
        if meta["dtype"] == "BF16":
            u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            return u32.view(np.float32).reshape(meta["shape"])
        
        dtype_map = {
            "F32": np.float32,
            "F16": np.float16,
            "I64": np.int64,
            "I32": np.int32,
            "I8": np.int8,
            "U8": np.uint8,
        }
        np_dtype = dtype_map.get(meta["dtype"], np.uint8)
        arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(meta["shape"])
        return arr.astype(np.float32)

    def fetch_token_embedding_vectors(self, token_ids: List[int], hidden_dim: int = 7168) -> np.ndarray:
        """
        Streams exact bfloat16 embedding vectors for a list of token IDs directly from HF.
        """
        embed_key = "language_model.model.embed_tokens.weight"
        if embed_key not in self.tensor_map:
            embed_key = "model.embed_tokens.weight"
            
        meta = self.tensor_map[embed_key]
        h_start = meta["start_byte"]
        
        vectors = []
        for tid in token_ids:
            safe_tid = tid % meta["shape"][0]
            byte_offset = h_start + safe_tid * hidden_dim * 2
            raw = self.client.fetch_range(meta["shard_url"], byte_offset, byte_offset + hidden_dim * 2 - 1)
            u16 = np.frombuffer(raw, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            f32 = u32.view(np.float32)
            vectors.append(f32)
            
        return np.array(vectors, dtype=np.float32)
