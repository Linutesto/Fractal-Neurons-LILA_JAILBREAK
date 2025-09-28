from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Sequence


@dataclass
class TokenizerConfig:
    type: str = "bytes"   # bytes | hf
    name_or_path: Optional[str] = None  # For HF tokenizer
    truncation_side: str = "right"      # left | right
    pad_to_multiple_of: Optional[int] = None


class BaseTokenizer:
    def __init__(self):
        self.vocab_size: int = 0
        self.mask_id: Optional[int] = None
        self.pad_id: Optional[int] = None

    def encode(self, text: str, max_len: int) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError


class ByteTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self.vocab_size = 257  # 256 bytes + [MASK]
        self.mask_id = 256
        self.pad_id = 0  # Using byte 0 as pad

    def encode(self, text: str, max_len: int) -> List[int]:
        b = text.encode("utf-8", errors="replace")
        ids = list(b[:max_len])
        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        buf = bytes(int(i) % 256 for i in ids)
        return buf.decode("utf-8", errors="replace")


class HFTokenizer(BaseTokenizer):
    def __init__(self, name_or_path: str, truncation_side: str = "right"):
        super().__init__()
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError("Install transformers to use HF tokenizer: pip install transformers sentencepiece") from e
        tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        if tok.pad_token is None:
            # Try to set pad token to eos if available
            if getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({"pad_token": "<|pad|>"})
        self.tok = tok
        self.tok.truncation_side = truncation_side
        self.vocab_size = int(tok.vocab_size)
        self.pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else None
        self.mask_id = int(getattr(tok, "mask_token_id", None)) if getattr(tok, "mask_token_id", None) is not None else None

    def encode(self, text: str, max_len: int) -> List[int]:
        out = self.tok(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            return_tensors=None,
        )
        return list(out["input_ids"]) if isinstance(out["input_ids"], list) else list(out["input_ids"][0])

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(
            list(int(i) for i in ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )


def build_tokenizer(cfg: TokenizerConfig) -> BaseTokenizer:
    t = (cfg.type or "bytes").lower()
    if t == "bytes":
        return ByteTokenizer()
    if t == "hf":
        if not cfg.name_or_path:
            raise RuntimeError("Tokenizer type 'hf' requires name_or_path (local dir or model name)")
        return HFTokenizer(cfg.name_or_path, truncation_side=cfg.truncation_side)
    raise ValueError(f"Unknown tokenizer type: {cfg.type}")
