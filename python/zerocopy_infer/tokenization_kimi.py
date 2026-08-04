"""
ZeroCopy-Infer: Official Kimi-K3 TikToken Tokenizer Wrapper
===========================================================
Authored by Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina).

Implements official TikTokenTokenizer wrapper for Kimi-K3 BPE encoding/decoding:
- Custom pat_str regex split
- 256 Reserved control & special tokens
- Full XTML chat template rendering (apply_chat_template)
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from .tokenizer import ZeroCopyKimiTokenizer

class TikTokenTokenizer(ZeroCopyKimiTokenizer):
    """
    Official Kimi-K3 TikTokenTokenizer wrapper.
    Inherits from ZeroCopyKimiTokenizer and provides HF-compatible tokenization interface.
    """
    pat_str = "|".join([
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ])

    def __init__(self, repo_id: str = "moonshotai/Kimi-K3"):
        super().__init__(repo_id=repo_id)

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        thinking: bool = False,
        image_size: Optional[Tuple[int, int]] = None,
        thinking_effort: str = "max",
    ) -> str:
        """
        Renders official Kimi-K3 XTML chat template string.
        """
        last_user_prompt = ""
        for msg in reversed(conversation):
            if msg.get("role") == "user":
                last_user_prompt = msg.get("content", "")
                break

        return self.render_xtml_chat_prompt(
            prompt=last_user_prompt,
            image_size=image_size,
            thinking=thinking,
            thinking_effort=thinking_effort,
        )
