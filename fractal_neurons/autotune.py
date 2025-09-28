import argparse
import os
import time
from typing import Dict, Any

import torch

from .model import FractalModel, FractalModelConfig


def try_model(dim: int, depth: int, fanout: int, seq_len: int, batch_size: int, device: torch.device) -> bool:
    try:
        model = FractalModel(FractalModelConfig(dim=dim, depth=depth, fanout=fanout))
        model = model.to(device)
        x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x)
        del model, x, logits
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False
        raise


def autosize(device: str = "cuda", depth: int = 6, fanout: int = 10, max_seq: int = 2048) -> Dict[str, Any]:
    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError("Autosize currently supports CUDA only.")

    free, total = torch.cuda.mem_get_info(dev)
    total_gb = total / (1024**3)
    print(f"GPU memory total: {total_gb:.1f} GB")

    # Start with a reasonable upper bound and binary search
    dim = 512
    seq = max_seq
    bs = 64

    # reduce until it fits
    while not try_model(dim, depth, fanout, seq, bs, dev):
        if bs > 1:
            bs = max(1, bs // 2)
            continue
        if seq > 256:
            seq = max(256, seq // 2)
            continue
        if dim > 128:
            dim = max(128, dim // 2)
            continue
        break

    print(f"Autosized: dim={dim}, seq_len={seq}, batch_size={bs}")
    return {"dim": dim, "seq_len": seq, "batch_size": bs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--fanout", type=int, default=10)
    ap.add_argument("--max_seq", type=int, default=2048)
    args = ap.parse_args()

    conf = autosize(device=args.device, depth=args.depth, fanout=args.fanout, max_seq=args.max_seq)
    print(conf)


if __name__ == "__main__":
    main()

