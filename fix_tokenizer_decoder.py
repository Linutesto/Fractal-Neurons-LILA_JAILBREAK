#!/usr/bin/env python3
"""Ensure a saved tokenizer decodes byte-level tokens without Ġ/Ċ artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders


def ensure_special_tokens_map(directory: Path) -> None:
    special_map = directory / "special_tokens_map.json"
    if special_map.exists():
        return
    default_map = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    special_map.write_text(json.dumps(default_map, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach a ByteLevel decoder to a tokenizer.json")
    parser.add_argument(
        "--tokenizer-dir",
        default="runs/fsi_en_v1/tokenizer",
        help="Directory containing tokenizer.json (default: runs/fsi_en_v1/tokenizer)",
    )
    args = parser.parse_args()

    tok_dir = Path(args.tokenizer_dir).expanduser()
    tok_json = tok_dir / "tokenizer.json"
    if not tok_json.exists():
        raise FileNotFoundError(f"tokenizer.json not found at {tok_json}")

    tokenizer = Tokenizer.from_file(str(tok_json))
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(tok_json))

    ensure_special_tokens_map(tok_dir)
    print(f"[done] ByteLevel decoder attached to {tok_json}")


if __name__ == "__main__":
    main()
