import torch
from fractal_neurons.model import FractalModel, FractalModelConfig
from fractal_neurons.fmm import FractalMemoryMatrix

def test_forward_pass():
    cfg = FractalModelConfig(vocab_size=30522, dim=512, depth=6, fanout=8)
    model = FractalModel(cfg)
    x = torch.randint(0, 30522, (2, 128))
    out, loss = model(x, loss_mask=torch.ones_like(x), targets=x)
    assert out.shape[0] == 2
    assert loss.item() > 0

def test_moe_routing():
    cfg = FractalModelConfig(vocab_size=30522, dim=512, depth=6, fanout=8, num_experts=4)
    model = FractalModel(cfg)
    x = torch.randint(0, 30522, (2, 128))
    out, _ = model(x)
    assert out.shape[1] == 128