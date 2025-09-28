"""Finetuning pipeline for English datasets using existing checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def run(cmd: str) -> None:
    print(f"\n▶ {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def build_config(
    corpus_dir: Path,
    base_ckpt: str,
    steps: int,
    lr: float,
    batch_size: int,
    seq_len: int,
    num_experts: int,
    expert_top_k: int,
    ema_decay: float,
    warmup: int,
    grad_clip: float,
    tokenizer_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tokenizer = tokenizer_cfg.copy() if tokenizer_cfg else {}
    if not tokenizer:
        tokenizer = {
            "type": "hf",
            "name_or_path": "bert-base-uncased",
            "truncation_side": "right",
            "pad_to_multiple_of": 128,
        }
    tokenizer.setdefault("type", "hf")
    if tokenizer["type"].lower() != "hf":
        tokenizer.update({"type": "hf", "name_or_path": "bert-base-uncased"})
        tokenizer.setdefault("truncation_side", "right")
        tokenizer.setdefault("pad_to_multiple_of", 128)

    return {
        "data": {
            "source": "textdir",
            "text_root": str(corpus_dir),
            "seq_len": seq_len,
            "text_tokenize_in_workers": True,
            "text_shuffle_buffer": 4096,
            "text_globs": [
                "*.txt",
                "*.md",
                "*.jsonl",
                "*.json",
                "*.log",
                "*.py",
                "**/*.txt",
                "**/*.md",
                "**/*.jsonl",
                "**/*.json",
                "**/*.log",
                "**/*.py",
            ],
        },
        "model": {
            "num_experts": num_experts,
            "expert_top_k": expert_top_k,
        },
        "train": {
            "steps": steps,
            "lr": lr,
            "warmup": warmup,
            "ema_decay": ema_decay,
            "grad_clip": grad_clip,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "resume": False,
            "device": "cuda",
        },
        "tokenizer": tokenizer,
        "init": {
            "from_checkpoint": base_ckpt,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Finetune on a local English corpus")
    ap.add_argument("--corpus", required=True, help="Directory of text files")
    ap.add_argument("--ckpt", required=True, help="Base checkpoint to finetune")
    ap.add_argument("--out", default="runs/finetune_english.yaml")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--num_experts", type=int, default=4)
    ap.add_argument("--expert_top_k", type=int, default=2)
    ap.add_argument("--ema_decay", type=float, default=0.9995)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--tokenizer_type", default="hf")
    ap.add_argument("--tokenizer_name", default="bert-base-uncased")
    ap.add_argument("--tokenizer_truncation_side", default="right")
    ap.add_argument("--tokenizer_pad_multiple", type=int, default=128)
    args = ap.parse_args()

    corpus_dir = Path(args.corpus).expanduser()
    if not corpus_dir.exists():
        raise SystemExit(f"Corpus directory not found: {corpus_dir}")

    tokenizer_cfg = {
        "type": args.tokenizer_type,
        "name_or_path": args.tokenizer_name,
        "truncation_side": args.tokenizer_truncation_side,
    }
    if args.tokenizer_pad_multiple is not None:
        tokenizer_cfg["pad_to_multiple_of"] = args.tokenizer_pad_multiple

    cfg = build_config(
        corpus_dir,
        args.ckpt,
        args.steps,
        args.lr,
        args.batch_size,
        args.seq_len,
        args.num_experts,
        args.expert_top_k,
        args.ema_decay,
        args.warmup,
        args.grad_clip,
        tokenizer_cfg=tokenizer_cfg,
    )
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote finetune config to {out_path}")
    run(f"python -m fractal_neurons.train --config {out_path}")


if __name__ == "__main__":
    main()
