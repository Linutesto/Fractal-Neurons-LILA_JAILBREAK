import argparse
import glob
import os
from typing import Optional, Dict, Any, Tuple

import torch

from .model import FractalModel, FractalModelConfig
from .tokenizer import TokenizerConfig, build_tokenizer, ByteTokenizer


def resolve_ckpt_path(spec: str) -> Tuple[str, Optional[str]]:
    """
    Resolve a checkpoint spec which can be a file or a directory.
    If it's a directory, find the newest *.pt inside (non-recursive, then recursive).
    Returns (resolved_path, info_message)
    """
    path = os.path.expanduser(spec)
    if os.path.isdir(path):
        # Prefer non-recursive *.pt first
        pts = sorted(glob.glob(os.path.join(path, "*.pt")), key=lambda p: os.path.getmtime(p), reverse=True)
        if not pts:
            pts = sorted(glob.glob(os.path.join(path, "**", "*.pt"), recursive=True), key=lambda p: os.path.getmtime(p), reverse=True)
        if not pts:
            raise FileNotFoundError(f"No .pt files found under directory: {path}")
        return pts[0], f"Resolved directory to latest checkpoint: {pts[0]}"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path, None


def load_model(ckpt_path: str, device: torch.device) -> FractalModel:
    obj = torch.load(ckpt_path, map_location=device)
    cfg: Dict[str, Any] = obj.get("cfg", {})
    model_cfg = dict(cfg.get("model", {}))
    tok_cfg = cfg.get("tokenizer", {})
    # Build tokenizer first (falls back to bytes)
    try:
        tcfg = TokenizerConfig(
            type=str(tok_cfg.get("type", "bytes")),
            name_or_path=tok_cfg.get("name_or_path"),
            truncation_side=str(tok_cfg.get("truncation_side", "right")),
        )
        tokenizer = build_tokenizer(tcfg)
    except Exception:
        tokenizer = ByteTokenizer()

    vocab_size = int(getattr(tokenizer, "vocab_size", model_cfg.get("vocab_size", 257)))
    model_cfg["vocab_size"] = vocab_size

    state_dict = obj["model"]
    moe_keys = [k for k in state_dict.keys() if k.startswith("moe.")]
    has_moe = len(moe_keys) > 0

    num_experts = int(model_cfg.get("num_experts", 0))
    if has_moe and num_experts <= 0:
        expert_idx = [int(k.split(".")[2]) for k in moe_keys if k.startswith("moe.experts.")]
        num_experts = (max(expert_idx) + 1) if expert_idx else 0
        model_cfg["num_experts"] = num_experts
    if not has_moe and num_experts > 0:
        # older checkpoint without MoE weights -> disable MoE to allow loading
        num_experts = 0
        model_cfg["num_experts"] = 0

    expert_hidden = int(model_cfg.get("expert_hidden", 0))
    if has_moe and expert_hidden <= 0:
        # infer width from first expert weight if available
        weight_key = next((k for k in moe_keys if k.endswith("0.weight")), None)
        if weight_key is not None:
            expert_hidden = state_dict[weight_key].shape[0]
            model_cfg["expert_hidden"] = expert_hidden

    fmc = FractalModelConfig(
        vocab_size=vocab_size,
        dim=int(model_cfg.get("dim", 128)),
        depth=int(model_cfg.get("depth", 6)),
        fanout=int(model_cfg.get("fanout", 10)),
        use_fp16=bool(model_cfg.get("use_fp16", False)),
        droppath_rate=float(model_cfg.get("droppath_rate", 0.0)),
        branch_dropout=float(model_cfg.get("branch_dropout", 0.0)),
        interconnect=bool(model_cfg.get("interconnect", True)),
        interconnect_heads=int(model_cfg.get("interconnect_heads", 2)),
        interconnect_dropout=float(model_cfg.get("interconnect_dropout", 0.0)),
        num_experts=num_experts,
        expert_hidden=int(model_cfg.get("expert_hidden", 0)),
        expert_top_k=int(model_cfg.get("expert_top_k", 2)),
        router_temperature=float(model_cfg.get("router_temperature", 1.0)),
        moe_aux_lambda=float(model_cfg.get("moe_aux_lambda", 0.0)),
        capacity_factor=float(model_cfg.get("capacity_factor", 1.1)),
    )
    model = FractalModel(fmc).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing parameters while loading: {missing}")
    if unexpected:
        print(f"[warn] unexpected parameters while loading: {unexpected}")
    model.eval()
    return model, tokenizer, cfg


def read_bytes_from_file(path: str, max_len: int) -> bytes:
    with open(path, "rb") as f:
        data = f.read(max_len)
    return data


def tokens_from_bytes(b: bytes, seq_len: int) -> torch.Tensor:
    t = torch.zeros(seq_len, dtype=torch.long)
    n = min(len(b), seq_len)
    if n > 0:
        t[:n] = torch.tensor(list(b[:n]), dtype=torch.long)
    return t


def iterative_fill(model: FractalModel, tokens: torch.Tensor, steps: int = 10, mask_rate: float = 0.15,
                   mask_id: Optional[int] = 256, vocab_size: int = 257,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    device = device or next(model.parameters()).device
    x = tokens.clone().to(device)
    B = 1
    x = x.unsqueeze(0)  # [1, T]
    T = x.shape[1]

    # initial mask
    mask = torch.rand(B, T, device=device) < mask_rate
    x_masked = x.clone()
    if mask_id is not None:
        x_masked[mask] = mask_id
    else:
        rand = torch.randint(0, vocab_size, x_masked.shape, device=device)
        x_masked[mask] = rand[mask]

    last_logits = None
    for _ in range(steps):
        with torch.no_grad():
            logits, _ = model(x_masked, loss_mask=mask, targets=x)
            last_logits = logits
            # fill masked with argmax
            pred = logits.argmax(dim=-1)
            x_masked[mask] = pred[mask]

    out = {
        "input": x.squeeze(0).detach().cpu(),
        "masked_input": (tokens.clone().fill_(mask_id)).where(mask.squeeze(0).cpu(), tokens),
        "filled": x_masked.squeeze(0).detach().cpu(),
        "mask": mask.squeeze(0).detach().cpu(),
        "logits": last_logits.squeeze(0).detach().cpu() if last_logits is not None else None,
    }
    return out


def masked_loss(model: FractalModel, tokens: torch.Tensor, mask_rate: float = 0.15, mask_id: int = 256,
                device: Optional[torch.device] = None) -> float:
    device = device or next(model.parameters()).device
    x = tokens.clone().unsqueeze(0).to(device)
    B, T = x.shape
    mask = torch.rand(B, T, device=device) < mask_rate
    inputs = x.clone()
    inputs[mask] = mask_id
    with torch.no_grad():
        _, loss = model(inputs, loss_mask=mask, targets=x)
    return float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file or a run directory")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="Input file to analyze")
    src.add_argument("--text", help="Raw text to analyze")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8, help="Fill steps")
    ap.add_argument("--mask_rate", type=float, default=0.15)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force_bytes", action="store_true", help="Treat file/text as bytes using byte tokenizer")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path, info = resolve_ckpt_path(args.ckpt)
    if info:
        print(info)
    model, tokenizer, cfg = load_model(ckpt_path, device)
    if args.force_bytes:
        from .tokenizer import ByteTokenizer
        tokenizer = ByteTokenizer()
    mask_id = getattr(tokenizer, "mask_id", None)
    vocab_size = int(getattr(tokenizer, "vocab_size", model.cfg.vocab_size))

    if args.file:
        if isinstance(tokenizer, ByteTokenizer) or args.force_bytes:
            raw = read_bytes_from_file(args.file, args.seq_len)
            tokens = tokens_from_bytes(raw, args.seq_len)
        else:
            with open(args.file, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(args.seq_len)
            ids = tokenizer.encode(text, args.seq_len)
            tokens = torch.tensor(ids, dtype=torch.long)
    else:
        text = args.text
        if isinstance(tokenizer, ByteTokenizer) or args.force_bytes:
            raw = text.encode("utf-8", errors="replace")[:args.seq_len]
            tokens = tokens_from_bytes(raw, args.seq_len)
        else:
            ids = tokenizer.encode(text, args.seq_len)
            tokens = torch.tensor(ids, dtype=torch.long)

    print("Running iterative fill...")
    res = iterative_fill(model, tokens, steps=args.steps, mask_rate=args.mask_rate, mask_id=mask_id, vocab_size=vocab_size, device=device)
    # Decode for display
    inp_dec = tokenizer.decode(res["input"].tolist())
    filled_dec = tokenizer.decode(res["filled"].tolist())

    print("Input (decoded):")
    print(inp_dec)
    print("\nReconstruction (decoded):")
    print(filled_dec)

    # Show a few positions that were masked and their top-k predictions
    logits = res.get("logits")
    mask = res.get("mask")
    if logits is not None and mask is not None:
        T = logits.shape[0]
        masked_positions = [i for i in range(T) if bool(mask[i])]
        show = masked_positions[: min(10, len(masked_positions))]
        print("\nTop-k predictions at sample masked positions:")
        for i in show:
            probs = torch.softmax(logits[i], dim=-1)
            topk = torch.topk(probs, k=min(args.topk, probs.numel()))
            pred_bytes = topk.indices.tolist()
            pred_chars = bytes([min(255, b) for b in pred_bytes]).decode("utf-8", errors="replace")
            print(f"pos {i:4d}: top{args.topk} byte ids {pred_bytes[:args.topk]} | chars '{pred_chars}'")

    mloss = masked_loss(model, tokens, mask_rate=args.mask_rate, mask_id=mask_id, device=device)
    print(f"\nApprox masked loss (@{args.mask_rate:.2f}): {mloss:.4f}")


if __name__ == "__main__":
    main()
