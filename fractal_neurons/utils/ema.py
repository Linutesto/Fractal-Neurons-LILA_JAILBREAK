from __future__ import annotations
from typing import Dict, Optional
import torch
from torch import nn

class EMAManager:
    """
    Minimal Exponential Moving Average manager for model parameters.
    - Keeps a shadow copy of parameters on CPU (fp32) by default
    - update(): shadow = decay * shadow + (1 - decay) * param
    - copy_to()/restore(): swap EMA params into the live model for eval/checkpoint
    - state_dict()/load_state_dict(): save/load EMA state
    """
    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[str] = None):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        cpu_device = torch.device("cpu")
        for name, p in model.named_parameters():
            if p.requires_grad:
                src = p.detach().float().to(cpu_device)  # store as fp32 on CPU
                self.shadow[name] = src.clone()
        self._device = device

    @torch.no_grad()
    def update(self, model: nn.Module):
        cpu_device = torch.device("cpu")
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().float().to(cpu_device).clone()
                continue
            shadow = self.shadow[name]
            new_val = p.detach().float().to(cpu_device)
            shadow.mul_(self.decay).add_(new_val, alpha=(1.0 - self.decay))

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """Temporarily load EMA params into model, backing up current ones."""
        self.backup.clear()
        for name, p in model.named_parameters():
            if name in self.shadow and p.requires_grad:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].to(p.device, dtype=p.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """Restore original params after copy_to()."""
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state):
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow", {})
        self.shadow = {k: v.clone().cpu() for k, v in shadow.items()}
