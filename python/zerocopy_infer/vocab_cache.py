"""
ZeroCopy-Infer: High-Speed Tokenizer Vocabulary Index Cache
============================================================
Authored by Leandro Emanuel Timberini (Ituzaingó, Buenos Aires, Argentina).

Provides fast binary serialization and in-RAM caching for HF tokenizer.json 
structures to eliminate parsing latency on mobile CPUs (Termux aarch64).
"""

import json
import os
import struct
from typing import Dict, Any, List, Optional


class VocabIndexCache:
    """
    In-memory and compact binary cache for tokenizer vocabulary ranks.
    """
    def __init__(self):
        self._cache: Dict[str, List[int]] = {}

    def get_latin_word_ranks(self, vocab: Dict[str, int]) -> List[int]:
        """
        Extracts ranks for common Spanish & Latin word tokens.
        """
        latin_ranks = []
        for word, idx in vocab.items():
            clean = word.replace(" ", "").replace("Ġ", "").replace(" ", "")
            if clean and clean.isalpha() and clean.isascii():
                latin_ranks.append(idx)
        return latin_ranks
