import torch
from pathlib import Path

from fractal_neurons.model import FractalModel, FractalModelConfig
from fractal_neurons.infer import load_model


def _write_ckpt(path: Path, model: FractalModel, model_cfg):
    torch.save({"model": model.state_dict(), "cfg": {"model": model_cfg}}, path)


def test_load_model_handles_missing_moe(tmp_path):
    cfg = FractalModelConfig(vocab_size=64, dim=32, depth=2, fanout=2, num_experts=0)
    model = FractalModel(cfg)
    ckpt_path = tmp_path / "legacy.pt"
    model_cfg = {
        "dim": 32,
        "depth": 2,
        "fanout": 2,
        "num_experts": 4,  # stale config expecting experts
    }
    _write_ckpt(ckpt_path, model, model_cfg)
    loaded_model, tokenizer, _ = load_model(str(ckpt_path), torch.device("cpu"))
    assert getattr(loaded_model, "moe", None) is None


def test_load_model_infers_moe_count(tmp_path):
    cfg = FractalModelConfig(vocab_size=64, dim=32, depth=2, fanout=2, num_experts=3, expert_hidden=48)
    model = FractalModel(cfg)
    ckpt_path = tmp_path / "moe.pt"
    model_cfg = {
        "dim": 32,
        "depth": 2,
        "fanout": 2,
    }
    _write_ckpt(ckpt_path, model, model_cfg)
    loaded_model, tokenizer, _ = load_model(str(ckpt_path), torch.device("cpu"))
    assert getattr(loaded_model, "moe", None) is not None
    assert loaded_model.moe.num_experts == 3
