"""Self-distillation pipeline: sample prompts, generate responses, filter, and save."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List

from .generate import load_generation_model, generate_text, log_generation

DEFAULT_PROMPTS = [
    "Summarize the benefits of modular neural networks.",
    "Explain the concept of fractal intelligence in simple terms.",
    "Write three bullet points about efficient training tricks.",
    "Describe a future application of MoE models in robotics.",
]


def load_prompts(path: Path | None) -> List[str]:
    if path is None:
        return DEFAULT_PROMPTS
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(DEFAULT_PROMPTS), encoding="utf-8")
    with path.open("r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts or DEFAULT_PROMPTS


def passes_filters(text: str, min_len: int, max_len: int, accept_all: bool = False) -> bool:
    if accept_all:
        return True
    if not text:
        return False
    if len(text) < min_len:
        return False
    if max_len > 0 and len(text) > max_len:
        return False
    bad_tokens = {"<unk>", "####"}
    if any(tok in text for tok in bad_tokens):
        return False
    return True


def distill(
    ckpt: str,
    prompts: Iterable[str],
    output_path: Path,
    limit: int,
    temperature: float,
    top_k: int,
    seq_len: int,
    max_new: int,
    steps: int,
    device: str | None,
    log_path: str,
    min_len: int,
    max_len: int,
    accept_all: bool,
) -> None:
    model, tokenizer, _ = load_generation_model(ckpt, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    with output_path.open("a", encoding="utf-8") as out:
        for idx, prompt in enumerate(prompts):
            if limit and accepted >= limit:
                break
            text, stats = generate_text(
                ckpt,
                prompt,
                seq_len=seq_len,
                max_new_tokens=max_new,
                steps=steps,
                temperature=temperature,
                top_k=top_k,
                device=device,
                cached=(model, tokenizer),
                return_stats=True,
            )
            stats["mode"] = "distill"
            if passes_filters(text, min_len, max_len, accept_all=accept_all):
                record = {
                    "prompt": prompt,
                    "completion": text,
                    "metadata": stats,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted += 1
                if log_path:
                    log_generation(log_path, stats)
    print(f"Distillation complete. Accepted {accepted} samples -> {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Self-distillation sampler")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt_file", "--prompt-file", dest="prompt_file")
    ap.add_argument("--output", default="distilled.jsonl")
    ap.add_argument("--limit", "--n-samples", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", "--top-k", type=int, default=20)
    ap.add_argument("--seq_len", "--seq-len", type=int, default=768)
    ap.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min_len", "--min-len", type=int, default=64)
    ap.add_argument("--max_len", "--max-len", type=int, default=1024)
    ap.add_argument("--accept_all", "--accept-all", action="store_true", help="Disable filtering heuristics and accept every sample")
    args = ap.parse_args()

    random.seed(args.seed)
    prompt_path = Path(args.prompt_file).expanduser() if args.prompt_file else None
    prompts = load_prompts(prompt_path)
    random.shuffle(prompts)
    distill(
        args.ckpt,
        prompts,
        Path(args.output).expanduser(),
        args.limit,
        args.temperature,
        args.top_k,
        args.seq_len,
        args.max_new_tokens,
        args.steps,
        args.device,
        args.log,
        args.min_len,
        args.max_len,
        args.accept_all,
    )


if __name__ == "__main__":
    main()
