import torch

from fractal_neurons.model import FractalModel, FractalModelConfig


def test_moe_forward_and_aux_loss():
    cfg = FractalModelConfig(
        vocab_size=64,
        dim=48,
        depth=2,
        fanout=2,
        num_experts=3,
        expert_hidden=64,
        expert_top_k=2,
        router_temperature=0.9,
        capacity_factor=1.2,
        moe_aux_lambda=0.05,
    )
    model = FractalModel(cfg)
    batch, seq = 4, 12
    tokens = torch.randint(0, cfg.vocab_size, (batch, seq))
    mask = torch.ones(batch, seq, dtype=torch.bool)
    logits, loss = model(tokens, loss_mask=mask, targets=tokens)
    assert logits.shape == (batch, seq, cfg.vocab_size)
    assert loss is not None
    info = model.last_moe_info
    assert info is not None
    assert "aux_loss_scalar" in info
    assert info["aux_loss_scalar"].item() >= 0
    assert "overflow_rate" in info
    assert 0 <= info["overflow_rate"] <= 1
    assert model.last_aux_loss is not None
