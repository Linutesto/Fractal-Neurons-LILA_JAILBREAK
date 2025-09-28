import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import torch
from torch import nn

from .infer import load_model, resolve_ckpt_path
from .tokenizer import ByteTokenizer


@torch.no_grad()
def iterative_fill_sample(
    model: nn.Module,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 8,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Iteratively fills masked positions by sampling from model predictions."""
    device = next(model.parameters()).device
    x = tokens.to(device).clone()
    mask = mask.to(device)
    mask_id = getattr(model, "mask_id", None)
    if mask_id is None:
        cfg = getattr(model, "cfg", None)
        vocab_default = getattr(cfg, "vocab_size", 257) if cfg is not None else 257
        mask_id = vocab_default - 1
    mask_id = int(mask_id)
    moe_records: List[Dict[str, Any]] = []
    for _ in range(max(1, steps)):
        x[mask] = mask_id
        logits, _ = model(x.unsqueeze(0), loss_mask=mask.unsqueeze(0), targets=x.unsqueeze(0))
        logits = logits[0][mask]

        if repetition_penalty != 1.0:
            # Penalize repetition of tokens that are not masked
            for token_id in torch.unique(x[~mask]):
                logits[:, token_id] /= repetition_penalty

        if temperature != 1.0:
            logits = logits / max(1e-5, temperature)
        
        if top_k > 0:
            top_vals, top_idx = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            probs = torch.softmax(top_vals, dim=-1)
            sampled = []
            for i in range(top_idx.size(0)):
                idx = torch.multinomial(probs[i], 1)
                sampled.append(top_idx[i, idx].item())
            sampled = torch.tensor(sampled, device=device)
        elif top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)

        x[mask] = sampled
        moe_info = getattr(model, "last_moe_info", None)
        aux = getattr(model, "last_aux_loss", None)
        if moe_info is not None:
            moe_records.append(
                {
                    "aux": float(aux.detach().cpu()) if aux is not None else None,
                    "overflow": float(moe_info.get("overflow_rate", 0.0)),
                    "entropy": float(moe_info.get("router_entropy_scalar", 0.0)),
                }
            )
    details: Dict[str, Any] = {"steps": steps, "moe_records": moe_records}
    return x.detach().cpu(), details


def prepare_generation_inputs(
    prompt_ids: List[int],
    seq_len: int,
    gen_tokens: int,
    pad_id: int,
    mask_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = prompt_ids[: seq_len - gen_tokens]
    tokens = torch.full((seq_len,), pad_id, dtype=torch.long)
    tokens[: len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long)
    start = len(prompt_ids)
    end = min(seq_len, start + gen_tokens)
    mask = torch.zeros(seq_len, dtype=torch.bool)
    if start < end:
        tokens[start:end] = mask_id
        mask[start:end] = True
    return tokens, mask


@torch.no_grad()
def load_generation_model(ckpt_path: str, device: Optional[str] = None):
    resolved, info = resolve_ckpt_path(ckpt_path)
    if info:
        print(info)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer, cfg = load_model(resolved, torch_device)
    model.eval()
    return model, tokenizer, cfg

def scan_checkpoints(root: str, limit: int = 10) -> List[Path]:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return []
    files = [p for p in root_path.glob("**/*.pt") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if limit > 0:
        files = files[:limit]
    return files

def print_checkpoint_table(files: List[Path]) -> None:
    if not files:
        print("No checkpoints found.")
        return
    print("Available checkpoints (newest first):")
    for idx, path in enumerate(files, start=1):
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            size_mb = path.stat().st_size / (1024 * 1024)
        except FileNotFoundError:
            mtime = "?"
            size_mb = 0.0
        print(f"[{idx:2d}] {path} | {mtime} | {size_mb:.1f} MB")


@torch.no_grad()
def generate_text(
    ckpt_path: str,
    prompt: str,
    seq_len: int = 512,
    max_new_tokens: int = 128,
    steps: int = 8,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    device: Optional[str] = None,
    cached: Optional[Tuple[nn.Module, object]] = None,
    return_stats: bool = False,
) -> Any:
    if cached is None:
        model, tokenizer, _ = load_generation_model(ckpt_path, device)
    else:
        model, tokenizer = cached

    if isinstance(tokenizer, ByteTokenizer):
        prompt_ids = list(prompt.encode("utf-8", errors="replace"))
        pad_id = 0
        mask_id = tokenizer.mask_id if tokenizer.mask_id is not None else (tokenizer.vocab_size - 1)
        mask_id = int(mask_id)
    else:
        prompt_ids = tokenizer.encode(prompt, seq_len)
        pad_id = tokenizer.pad_id or 0
        mask_id = getattr(tokenizer, "mask_id", None)
        if mask_id is None:
            mask_id = getattr(tokenizer, "vocab_size", 0) - 1
        if mask_id is None or mask_id < 0:
            mask_id = 0
        mask_id = int(mask_id)
    gen_tokens = min(max_new_tokens, seq_len)
    tokens, mask = prepare_generation_inputs(prompt_ids, seq_len, gen_tokens, pad_id, mask_id)
    filled, details = iterative_fill_sample(
        model,
        tokens,
        mask,
        steps=steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    generated = filled.masked_select(mask)
    generated_ids = [int(i) for i in generated.tolist()]
    text = tokenizer.decode(generated_ids)
    stats = {
        "prompt": prompt,
        "prompt_len": len(prompt_ids),
        "generated_tokens": generated.tolist(),
        "generated_text": text.strip(),
        "seq_len": seq_len,
        "max_new_tokens": max_new_tokens,
        "steps": steps,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "mask_count": int(mask.sum().item()),
        "moe": details.get("moe_records", []),
        "timestamp": datetime.utcnow().isoformat(),
        "checkpoint": ckpt_path,
    }
    if stats["moe"]:
        aux_vals = [r["aux"] for r in stats["moe"] if r.get("aux") is not None]
        overflow_vals = [r.get("overflow", 0.0) for r in stats["moe"]]
        entropy_vals = [r.get("entropy", 0.0) for r in stats["moe"]]
        if aux_vals:
            stats["moe_aux_mean"] = sum(aux_vals) / len(aux_vals)
        stats["moe_overflow_mean"] = sum(overflow_vals) / len(overflow_vals)
        stats["moe_entropy_mean"] = sum(entropy_vals) / len(entropy_vals)
    if return_stats:
        return text.strip(), stats
    return text.strip()


def log_generation(log_path: str, payload: Dict[str, Any]) -> None:
    if not log_path:
        return
    log_file = Path(log_path).expanduser()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class ChatSession:
    def __init__(self, system_prompt: str, history: Optional[List[Tuple[str, str]]] = None):
        self.system_prompt = system_prompt.strip()
        self.history: List[Tuple[str, str]] = history[:] if history else []

    def append(self, user: str, assistant: str):
        self.history.append((user.strip(), assistant.strip()))

    def build_prompt(self, user_input: str) -> str:
        parts = [f"System: {self.system_prompt}"]
        for u, a in self.history:
            parts.append(f"User: {u}")
            parts.append(f"Assistant: {a}")
        parts.append(f"User: {user_input.strip()}")
        parts.append("Assistant:")
        return "\n".join(parts)


def chat_loop(args):
    session = ChatSession(args.system_prompt)
    model, tokenizer, _ = load_generation_model(args.ckpt, args.device)
    log_path = args.log.strip() if args.log else ""
    print("Fractal Chat — type 'exit' to quit")
    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break
        prompt = session.build_prompt(user)
        completion, stats = generate_text(
            args.ckpt,
            prompt,
            seq_len=args.seq_len,
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device,
            cached=(model, tokenizer),
            return_stats=True,
        )
        print(f"Assistant: {completion}")
        session.append(user, completion)
        if log_path:
            stats.update(
                {
                    "mode": "chat",
                    "turn": len(session.history),
                    "user_input": user,
                    "system_prompt": args.system_prompt,
                }
            )
            log_generation(log_path, stats)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fractal Neurons text generation/ chat")
    parser.add_argument("--ckpt", help="Path to checkpoint or run directory")
    parser.add_argument("--prompt", help="Single prompt for one-shot generation")
    parser.add_argument("--chat", action="store_true", help="Launch interactive chat loop")
    parser.add_argument("--scan", action="store_true", help="List checkpoints and exit if --prompt/--chat not provided")
    parser.add_argument("--runs_dir", default="runs", help="Directory to scan for checkpoints when --scan is used")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of checkpoints to display when scanning")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--system_prompt", default="You are a helpful assistant.")
    parser.add_argument("--log", default="", help="Path to JSONL log file for generations")
    args = parser.parse_args()

    if args.scan or not args.ckpt:
        files = scan_checkpoints(args.runs_dir, args.limit)
        print_checkpoint_table(files)
        if args.scan and not args.chat and not args.prompt:
            return
        if not args.ckpt:
            if not files:
                raise SystemExit("No checkpoints found. Provide --ckpt manually once available.")
            default_choice = 1
            choice = input(f"Select checkpoint [default {default_choice}]: ").strip()
            if not choice:
                choice_idx = default_choice
            else:
                try:
                    choice_idx = int(choice)
                except ValueError:
                    raise SystemExit("Invalid selection")
            if not (1 <= choice_idx <= len(files)):
                raise SystemExit("Selection out of range")
            args.ckpt = str(files[choice_idx - 1])

    if args.chat:
        if not args.ckpt:
            raise SystemExit("--ckpt required for chat mode")
        chat_loop(args)
    else:
        if not args.prompt:
            raise SystemExit("--prompt required for single generation mode")
        text, stats = generate_text(
            args.ckpt,
            args.prompt,
            seq_len=args.seq_len,
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device,
            return_stats=True,
        )
        print(text)
        if args.log:
            stats.update({"mode": "single"})
            log_generation(args.log, stats)


if __name__ == "__main__":
    main()
