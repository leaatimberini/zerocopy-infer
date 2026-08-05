"""
ZeroCopy-Infer: Universal Hugging Face Tokenizer & Chat Template Renderer
========================================================================
Authored by Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina).

Supports native remote streaming of tokenizers directly from Hugging Face LFS / CDN into RAM:
- Gemma 4 (Google) -> SentencePiece / Gemma Chat (<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n)
- MiMo V2.5 Pro (Xiaomi) -> ChatML (<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n)
- Kimi K3 (Moonshot AI) -> TikToken BPE + XTML (<|open|>message role="user"<|sep|>)
- DeepSeek V3 (DeepSeek) -> DeepSeek Chat (<｜User｜>...<｜Assistant｜>)
- Qwen 2.5 (Alibaba) -> ChatML (<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n)
- Mixtral / Mistral -> Mistral ([INST] ... [/INST])
"""

import base64
import json
import urllib.request
from typing import Dict, List, Optional, Tuple, Any

OPEN_TOKEN = "<|open|>"
CLOSE_TOKEN = "<|close|>"
SEP_TOKEN = "<|sep|>"
END_OF_MSG_TOKEN = "<|end_of_msg|>"

MEDIA_BEGIN_TOKEN = "<|media_begin|>"
MEDIA_CONTENT_TOKEN = "<|media_content|>"
MEDIA_PAD_TOKEN = "<|media_pad|>"
MEDIA_END_TOKEN = "<|media_end|>"

class UniversalHFTokenizer:
    """
    Universal Hugging Face Zero-Disk Tokenizer & Multi-Model Chat Template Renderer.
    Streams tokenizer.json / tiktoken.model directly from HF into RAM.
    """
    def __init__(self, repo_id: str = "moonshotai/Kimi-K3", token: Optional[str] = None):
        self.repo_id = repo_id
        self.token = token
        self.encoder: Dict[bytes, int] = {}
        self.decoder: Dict[int, bytes] = {}
        self.clean_latin_ranks: List[int] = []
        self.complete_word_ranks: List[int] = []
        self.special_tokens: Dict[str, int] = {}
        self.is_loaded = False
        
        self.base_url = f"https://huggingface.co/{repo_id}/resolve/main"
        self._load_tokenizer()

    def _is_clean_latin(self, token_str: str) -> bool:
        if not token_str:
            return False
        for char in token_str:
            code = ord(char)
            if (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or (0x3040 <= code <= 0x30FF) or (0x0400 <= code <= 0x04FF):
                return False
        return True

    def _load_tokenizer(self):
        """
        Attempts to load tokenizer.json or tiktoken.model from HF repository into RAM.
        """
        headers = {"User-Agent": "ZeroCopy-Infer/0.9.5 (Leandro Timberini AGI Engine)"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        # 1. Try fetching tokenizer.json (Standard for Gemma, Qwen, DeepSeek, Xiaomi MiMo)
        tokenizer_json_url = f"{self.base_url}/tokenizer.json"
        try:
            print(f"[UniversalTokenizer] Attempting to load tokenizer.json from {self.repo_id}...")
            req = urllib.request.Request(tokenizer_json_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                vocab = {}
                model_sec = data.get("model", {})
                if "vocab" in model_sec:
                    vocab = model_sec["vocab"]
                elif "vocab" in data:
                    vocab = data["vocab"]

                for token_str, tid in vocab.items():
                    b_token = token_str.encode("utf-8")
                    self.encoder[b_token] = tid
                    self.decoder[tid] = b_token
                    if self._is_clean_latin(token_str):
                        self.clean_latin_ranks.append(tid)
                        stripped = token_str.strip()
                        if len(stripped) >= 2 and stripped.isalnum():
                            self.complete_word_ranks.append(tid)

                added_tokens = data.get("added_tokens", [])
                for item in added_tokens:
                    if isinstance(item, dict):
                        t_content = item.get("content", "")
                        t_id = item.get("id")
                        if t_content and t_id is not None:
                            self.special_tokens[t_content] = t_id

                self.is_loaded = True
                print(f"[UniversalTokenizer] Loaded {len(self.encoder)} tokens from tokenizer.json ({len(self.complete_word_ranks)} Latin words) into RAM!")
                return
        except Exception as e:
            print(f"[UniversalTokenizer] tokenizer.json notice ({e}). Checking alternative formats...")

        # 2. Try fetching tiktoken.model (For Moonshot Kimi K3)
        tiktoken_url = f"{self.base_url}/tiktoken.model"
        try:
            print(f"[UniversalTokenizer] Attempting to load tiktoken.model from {self.repo_id}...")
            req = urllib.request.Request(tiktoken_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
                for line in content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ")
                    if len(parts) == 2:
                        b64_token, rank_str = parts[0], parts[1]
                        try:
                            token_bytes = base64.b64decode(b64_token)
                            rank = int(rank_str)
                            token_str = token_bytes.decode("utf-8", errors="ignore")
                            self.encoder[token_bytes] = rank
                            self.decoder[rank] = token_bytes
                            if self._is_clean_latin(token_str):
                                self.clean_latin_ranks.append(rank)
                                stripped = token_str.strip()
                                if len(stripped) >= 2 and stripped.isalnum() and rank >= 500:
                                    self.complete_word_ranks.append(rank)
                        except Exception:
                            continue
                self.is_loaded = True
                print(f"[UniversalTokenizer] Loaded {len(self.encoder)} BPE tokens from tiktoken.model into RAM!")
                return
        except Exception as e:
            print(f"[UniversalTokenizer] tiktoken.model notice ({e}). Populating fallback vocabulary...")

        # 3. Fallback core BPE vocabulary
        self._populate_core_bpe_vocab()

    def _populate_core_bpe_vocab(self):
        for i in range(32, 127):
            b = bytes([i])
            self.encoder[b] = i
            self.decoder[i] = b
            self.clean_latin_ranks.append(i)
            
        common_words = [
            "el", "la", "los", "las", "un", "una", "es", "son", "de", "en", "por", "para",
            "con", "sin", "sobre", "entre", "que", "se", "su", "sus", "al", "del", "este", "esta",
            "inteligencia", "artificial", "motor", "sistema", "modelo", "datos", "RAM", "inferencia",
            "respuesta", "proceso", "información", "tiempo", "real", "Gemma", "Kimi", "MiMo", "DeepSeek"
        ]
        
        for idx, w in enumerate(common_words):
            b = w.encode("utf-8")
            rank = idx + 1000
            self.encoder[b] = rank
            self.decoder[rank] = b
            self.clean_latin_ranks.append(rank)
            self.complete_word_ranks.append(rank)

        self.is_loaded = True

    def render_chat_prompt(self, user_prompt: str, repo_id: Optional[str] = None) -> str:
        """
        Renders exact Chat Template for the target model family:
        - Gemma 4 (Google): <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
        - Xiaomi MiMo / Qwen: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
        - DeepSeek: <｜User｜>...<｜Assistant｜>
        - Mistral: [INST] ... [/INST]
        - Kimi K3: XTML format
        """
        target_repo = (repo_id or self.repo_id).lower()
        
        if "gemma-4" in target_repo or "gemma4" in target_repo:
            return f"<bos><|turn>user\n{user_prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        elif "gemma" in target_repo:
            return f"<bos><start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
        elif "qwen" in target_repo or "mimo" in target_repo:
            return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif "deepseek" in target_repo:
            return f"<｜User｜>{user_prompt}<｜Assistant｜>"
        elif "mixtral" in target_repo or "mistral" in target_repo:
            return f"[INST] {user_prompt} [/INST]"
        else:
            # Default XTML format for Kimi-K3
            return self.render_xtml_chat_prompt(user_prompt, thinking=False)

    def render_xtml_chat_prompt(self, user_prompt: str, image_size: Optional[Tuple[int, int]] = None, thinking: bool = False) -> str:
        user_msg = f"{OPEN_TOKEN}message role=\"user\"{SEP_TOKEN}{user_prompt}{CLOSE_TOKEN}message{SEP_TOKEN}{END_OF_MSG_TOKEN}\n"
        if thinking:
            assistant_gen = f"{OPEN_TOKEN}message role=\"assistant\"{SEP_TOKEN}{OPEN_TOKEN}think{SEP_TOKEN}"
        else:
            assistant_gen = f"{OPEN_TOKEN}message role=\"assistant\"{SEP_TOKEN}{OPEN_TOKEN}response{SEP_TOKEN}"
        return user_msg + assistant_gen

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        text_bytes = text.encode("utf-8")
        tokens = []
        i = 0
        n = len(text_bytes)
        while i < n:
            matched = False
            for length in range(min(32, n - i), 0, -1):
                sub = text_bytes[i:i + length]
                if sub in self.encoder:
                    tokens.append(self.encoder[sub])
                    i += length
                    matched = True
                    break
            if not matched:
                b = bytes([text_bytes[i]])
                tokens.append(self.encoder.get(b, text_bytes[i]))
                i += 1
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        res_bytes = bytearray()
        for tid in token_ids:
            if tid in self.decoder:
                res_bytes.extend(self.decoder[tid])
            elif tid in self.special_tokens.values():
                continue
            else:
                res_bytes.extend(f"[{tid}]".encode("utf-8"))
        return res_bytes.decode("utf-8", errors="replace")

    def decode_token(self, token_id: int) -> str:
        return self.decode([token_id])

# Alias for backward compatibility
ZeroCopyKimiTokenizer = UniversalHFTokenizer
ZeroCopyTokenizer = UniversalHFTokenizer
