"""Utility to train a ByteLevel BPE tokenizer for Fractal Neurons."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors


def build_bpe_tokenizer(
    corpus_files: Sequence[str],
    out_dir: Path,
    vocab_size: int = 65536,
    lowercase: bool = False,
    special_tokens: Sequence[str] | None = None,
) -> None:
    if not corpus_files:
        raise ValueError("At least one corpus file is required to build a tokenizer")

    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens or ["<pad>", "<bos>", "<eos>", "<unk>"]),
        lowercase=lowercase,
    )
    tokenizer.train([str(Path(p).expanduser()) for p in corpus_files], trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    try:
        from transformers import PreTrainedTokenizerFast  # type: ignore
    except Exception as exc:  # pragma: no cover - defer runtime dependency errors
        raise RuntimeError(
            "Install transformers to export the tokenizer: pip install transformers"
        ) from exc

    special = list(special_tokens or ["<pad>", "<bos>", "<eos>", "<unk>"])
    pad_token, bos_token, eos_token, unk_token = (special + [None] * 4)[:4]

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        unk_token=unk_token,
    )
    hf_tokenizer.save_pretrained(str(out_dir))

    meta = {
        "pad_id": int(hf_tokenizer.pad_token_id) if hf_tokenizer.pad_token_id is not None else 0,
        "bos_id": int(hf_tokenizer.bos_token_id) if hf_tokenizer.bos_token_id is not None else 1,
        "eos_id": int(hf_tokenizer.eos_token_id) if hf_tokenizer.eos_token_id is not None else 2,
        "unk_id": int(hf_tokenizer.unk_token_id) if hf_tokenizer.unk_token_id is not None else 3,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] tokenizer saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a ByteLevel BPE tokenizer")
    parser.add_argument("--corpus", nargs="+", help="Corpus file(s) to ingest", required=True)
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--save-special-tokens", action="store_true", help="No-op placeholder for compatibility")
    args = parser.parse_args()

    build_bpe_tokenizer(
        corpus_files=args.corpus,
        out_dir=Path(args.out).expanduser(),
        vocab_size=args.vocab_size,
        lowercase=bool(args.lowercase),
    )


if __name__ == "__main__":
    main()
