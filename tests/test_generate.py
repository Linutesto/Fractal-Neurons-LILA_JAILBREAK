import os
import time

import torch

from fractal_neurons.model import FractalModel, FractalModelConfig
from fractal_neurons.generate import iterative_fill_sample, scan_checkpoints


def test_iterative_fill_sample_shapes():
    cfg = FractalModelConfig(vocab_size=64, dim=32, depth=2, fanout=2)
    model = FractalModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (32,))
    mask = torch.zeros(32, dtype=torch.bool)
    mask[-8:] = True
    filled, details = iterative_fill_sample(model, tokens, mask, steps=2, temperature=0.8, top_k=5)
    assert filled.shape == tokens.shape
    assert torch.equal(filled[~mask], tokens[~mask])
    assert details["steps"] == 2


def test_scan_checkpoints_returns_sorted(tmp_path):
    files = []
    for idx in range(3):
        path = tmp_path / f"test_{idx}.pt"
        with open(path, "wb") as f:
            f.write(b"torch")
        os.utime(path, (time.time() + idx, time.time() + idx))
        files.append(path)
    results = scan_checkpoints(str(tmp_path), limit=3)
    assert len(results) == 3
    # newest first
    assert results[0].stat().st_mtime >= results[1].stat().st_mtime
