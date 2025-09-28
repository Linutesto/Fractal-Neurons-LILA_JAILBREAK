import torch

from fractal_neurons.model import FractalModel, FractalModelConfig


def test_training_step_with_moe_aux_loss():
    torch.manual_seed(0)
    cfg = FractalModelConfig(
        vocab_size=32,
        dim=32,
        depth=2,
        fanout=2,
        num_experts=2,
        expert_hidden=48,
        expert_top_k=2,
        router_temperature=1.0,
        capacity_factor=1.1,
        moe_aux_lambda=0.05,
    )
    model = FractalModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch, seq = 3, 10
    tokens = torch.randint(0, cfg.vocab_size, (batch, seq))
    mask = torch.rand(batch, seq) < 0.4
    logits, loss = model(tokens, loss_mask=mask, targets=tokens)
    assert loss is not None
    aux = model.last_aux_loss
    assert aux is not None
    total_loss = loss + cfg.moe_aux_lambda * aux
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    assert not torch.isnan(total_loss.detach())
