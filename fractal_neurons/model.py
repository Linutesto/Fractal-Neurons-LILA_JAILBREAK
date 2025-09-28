from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FractalModelConfig:
    vocab_size: int = 257  # 256 bytes + 1 [MASK]
    dim: int = 128
    depth: int = 6         # with fanout=10 → 111,111 nodes
    fanout: int = 10
    use_fp16: bool = False
    droppath_rate: float = 0.0
    branch_dropout: float = 0.0
    # Inter-layer interconnect across depth levels
    interconnect: bool = True
    interconnect_heads: int = 2
    interconnect_dropout: float = 0.0
    num_experts: int = 0
    expert_hidden: int = 0
    expert_top_k: int = 2
    router_temperature: float = 1.0
    moe_aux_lambda: float = 0.0
    capacity_factor: float = 1.1
    # FMM parameters
    use_fmm: bool = False
    fmm_max_nodes: int = 10000


class MoEBlock(nn.Module):
    """Top-k gated mixture of tiny expert MLPs over the global context."""

    def __init__(
        self,
        dim: int,
        hidden: int,
        num_experts: int,
        top_k: int = 2,
        temperature: float = 1.0,
        capacity_factor: float = 1.1,
    ):
        super().__init__()
        assert num_experts > 0
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.temperature = max(1e-3, float(temperature))
        self.capacity_factor = max(0.1, float(capacity_factor))
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # context: [B, D]
        logits = self.gate(context)  # [B, E]
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        top_vals, top_idx = torch.topk(scaled_logits, k=self.top_k, dim=-1)
        gate_scores = torch.softmax(top_vals, dim=-1)
        output = torch.zeros_like(context)
        dispatch = torch.zeros(self.num_experts, device=context.device, dtype=context.dtype)
        overflow = 0.0
        capacity = max(
            1,
            math.ceil(context.size(0) * self.top_k / float(self.num_experts) * self.capacity_factor),
        )
        total_slots = context.size(0) * self.top_k
        for expert_id in range(self.num_experts):
            mask = (top_idx == expert_id)
            if not mask.any():
                continue
            token_idx, ranks = mask.nonzero(as_tuple=True)
            scores = gate_scores[token_idx, ranks]
            if scores.numel() > capacity:
                top_scores, top_positions = torch.topk(scores, capacity)
                token_idx = token_idx[top_positions]
                ranks = ranks[top_positions]
                scores = top_scores
                overflow += float(mask.sum().item() - capacity)
            expert_in = context[token_idx]
            expert_out = self.experts[expert_id](expert_in)
            output.index_add_(0, token_idx, expert_out * scores.unsqueeze(-1))
            dispatch[expert_id] += float(token_idx.numel())
        info = {
            "topk_idx": top_idx.detach(),
            "gate_scores": gate_scores.detach(),
            "dispatch_counts": dispatch.detach(),
        }
        load = dispatch / max(1.0, dispatch.sum())
        importance = probs.mean(dim=0)
        aux_loss = (importance * load).sum() * self.num_experts
        entropy = -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1).mean()
        info.update(
            {
                "importance": importance.detach(),
                "load": load.detach(),
                "aux_loss": aux_loss,
                "aux_loss_scalar": aux_loss.detach(),
                "router_entropy": entropy,
                "router_entropy_scalar": entropy.detach(),
                "overflow_rate": float(overflow / max(1.0, total_slots)),
                "capacity": float(capacity),
            }
        )
        return output, info


class FractalNetwork(nn.Module):
    """
    Vectorized balanced f-ary fractal with shared parameters.
    Bottom-up aggregation with attention; parent update uses input context + child aggregate + depth bias.
    """

    def __init__(self, dim: int, depth: int, fanout: int, droppath_rate: float = 0.0, branch_dropout: float = 0.0,
                 interconnect: bool = True, interconnect_heads: int = 2, interconnect_dropout: float = 0.0):
        super().__init__()
        assert depth >= 1 and fanout >= 1
        self.dim = dim
        self.depth = depth
        self.fanout = fanout
        self.droppath_rate = droppath_rate
        self.branch_dropout = branch_dropout
        self.use_interconnect = interconnect

        # Shared projections
        self.proj_in = nn.Linear(dim, dim)
        self.proj_child = nn.Linear(dim, dim)
        self.depth_bias = nn.Parameter(torch.zeros(depth, dim))

        # Attention over children
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.out = nn.Linear(dim, dim)
        # level-wise gating
        self.level_gate = nn.Parameter(torch.zeros(depth))
        self.depth_gate = nn.Parameter(torch.zeros(depth, dim))

        # Lightweight attention-based interconnect across levels (root..leaves)
        if self.use_interconnect:
            self.inter_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=max(1, interconnect_heads),
                                                    dropout=interconnect_dropout, batch_first=True)
            self.inter_norm = nn.LayerNorm(dim)
            self.ic_gate = nn.Parameter(torch.zeros(1))

        # Precompute layer sizes (L-1 is leaves)
        self.layer_sizes = [1]
        for _ in range(1, depth):
            self.layer_sizes.append(self.layer_sizes[-1] * fanout)

    @torch.no_grad()
    def total_nodes(self) -> int:
        return sum(self.layer_sizes)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: [B, dim] sequence/contextual summary
        returns: [B, dim] global root representation
        """
        B, D = context.shape
        assert D == self.dim

        base = self.proj_in(context)  # [B, dim]

        # Work bottom-up: start from leaves (l = depth-1) where child agg = 0
        y_next = None  # children outputs for next iteration
        level_means = [None for _ in range(self.depth)]  # [B, dim] per level
        for l in reversed(range(self.depth)):
            n = self.layer_sizes[l]
            if l == self.depth - 1:
                # Leaves: no children
                child_agg = torch.zeros(B, n, self.dim, device=context.device, dtype=context.dtype)
            else:
                # Children outputs y_next: [B, n*F, dim] → group into F children per parent
                F_ = self.fanout
                yg = y_next.view(B, n, F_, self.dim)
                # Attention: q from (base + depth), k/v from child outputs
                q = self.q(base + self.depth_bias[l])  # [B, dim]
                q = q.view(B, 1, 1, self.dim).expand(B, n, F_, self.dim)
                k = self.k(yg)
                v = self.v(yg)
                att = (q * k).sum(-1) / math.sqrt(self.dim)  # [B, n, F]
                att = F.softmax(att, dim=-1).unsqueeze(-1)
                # Optional stochastic branch dropout (novel fractal sparsity)
                if self.training and self.branch_dropout > 0:
                    drop_mask = (torch.rand(B, n, F_, 1, device=context.device) > self.branch_dropout).float()
                    att = att * drop_mask
                    att = att / (att.sum(dim=2, keepdim=True) + 1e-6)
                # Apply depth_gate to attention output
                child_agg = torch.sigmoid(self.depth_gate[l]) * (att * v).sum(dim=2)  # [B, n, dim]

            # Parent update
            gate = torch.sigmoid(self.level_gate[l])
            upd = (
                base.view(B, 1, self.dim).expand(B, n, self.dim)
                + self.depth_bias[l].view(1, 1, self.dim)
                + self.proj_child(child_agg)
            ) * gate
            if self.training and self.droppath_rate > 0:
                # per-sample stochastic drop-path at this level
                keep = (torch.rand(B, 1, 1, device=context.device) > self.droppath_rate).float()
                upd = upd * keep
            h = F.relu(upd)
            y = self.out(h)  # [B, n, dim]
            # Track mean per level for interconnect without storing all nodes
            level_means[l] = y.mean(dim=1)
            y_next = y if l > 0 else y  # keep for next loop

        # y at l=0 has shape [B, 1, dim]
        root = y_next.view(B, self.dim)
        if self.use_interconnect:
            # Stack [B, depth, dim] and apply attention to allow information exchange across levels
            z = torch.stack(level_means, dim=1)  # [B, L, D]
            z2, _ = self.inter_attn(z, z, z)
            z2 = self.inter_norm(z + z2)
            # Update root token with inter-level context
            root = root + torch.sigmoid(self.ic_gate) * z2[:, 0, :]
        return root


from .fmm import FractalMemoryMatrix # Import FMM


class FractalModel(nn.Module):
    """
    Byte MLM with Fractal Network context. Inputs are byte tokens 0..255 and a [MASK]=256.
    - Token embeddings → mean pooled context → fractal network → per-token logits using context + token emb.
    """

    def __init__(self, cfg: FractalModelConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.dim = cfg.dim
        self.embed = nn.Embedding(self.vocab_size, self.dim)
        self.fractal = FractalNetwork(
            dim=cfg.dim,
            depth=cfg.depth,
            fanout=cfg.fanout,
            droppath_rate=cfg.droppath_rate,
            branch_dropout=cfg.branch_dropout,
            interconnect=cfg.interconnect,
            interconnect_heads=cfg.interconnect_heads,
            interconnect_dropout=cfg.interconnect_dropout,
        )
        self.num_experts = max(0, cfg.num_experts)
        self.moe_aux_lambda = float(cfg.moe_aux_lambda)
        self.last_moe_info: Optional[Dict[str, torch.Tensor]] = None
        self.last_aux_loss: Optional[torch.Tensor] = None
        if self.num_experts > 0:
            hidden = cfg.expert_hidden if cfg.expert_hidden > 0 else cfg.dim * 2
            top_k = max(1, min(cfg.expert_top_k, self.num_experts))
            self.moe = MoEBlock(
                cfg.dim,
                hidden,
                self.num_experts,
                top_k,
                temperature=cfg.router_temperature,
                capacity_factor=cfg.capacity_factor,
            )
        else:
            self.moe = None
        self.head = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.vocab_size),
        )
        # FMM Initialization
        self.fmm: Optional[FractalMemoryMatrix] = None
        if cfg.use_fmm:
            self.fmm = FractalMemoryMatrix(cfg.dim, cfg.fmm_max_nodes)

    @torch.no_grad()
    def total_nodes(self) -> int:
        return self.fractal.total_nodes()

    def forward(self, tokens: torch.Tensor, loss_mask: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None, t_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        tokens: [B, T] longs in [0..256]
        loss_mask: [B, T] bool, True where to compute loss
        targets: [B, T] longs, required if computing loss
        returns: (logits [B, T, V], loss)
        """
        B, T = tokens.shape
        x = self.embed(tokens)  # [B, T, dim]
        # Mean pool context (can be replaced with better encoders)
        context = x.mean(dim=1)  # [B, dim]

        # FMM Integration: Retrieve relevant memories and combine with context
        if self.fmm is not None and self.fmm.node_vectors:
            # For simplicity, retrieve for the first item in the batch
            # A more sophisticated approach would retrieve for each item or use a batched query
            query_vector = context[0].detach().cpu() # Use the first context vector as query
            retrieved_nodes = self.fmm.retrieve(query_vector, top_k=1) # Retrieve top 1 node
            if retrieved_nodes:
                # Combine retrieved memory with the original context
                # For simplicity, just add the semantic vector of the top retrieved node
                retrieved_vector = retrieved_nodes[0][0].semantic_vector.to(context.device)
                context = context + retrieved_vector.unsqueeze(0).expand_as(context)

        # QFP Integration: Add t_context to the context vector
        if t_context is not None:
            # Ensure t_context is broadcastable to context
            if t_context.dim() == 0: # Scalar
                context = context + t_context
            elif t_context.dim() == 1 and t_context.size(0) == context.size(0): # [B]
                context = context + t_context.unsqueeze(1)
            else:
                # Handle more complex t_context if needed, or raise error
                pass

        if self.moe is not None:
            moe_out, moe_info = self.moe(context)
            context = context + moe_out
            self.last_aux_loss = moe_info.get("aux_loss")
            info_log: Dict[str, torch.Tensor] = {}
            for key, value in moe_info.items():
                if key == "aux_loss":
                    continue
                if isinstance(value, torch.Tensor):
                    info_log[key] = value.detach()
                else:
                    info_log[key] = torch.tensor(value)
            self.last_moe_info = info_log
        else:
            self.last_aux_loss = None
            self.last_moe_info = None
        root = self.fractal(context)  # [B, dim]

        # FMM Integration: Add root representation to FMM
        if self.fmm is not None:
            # For simplicity, add each item in the batch as a separate node
            # In a more advanced FMM, context_anchor and parent_id would be more sophisticated
            for i in range(B):
                self.fmm.add_node(root[i].detach().cpu())

        # Combine token embedding with global context
        root_exp = root.unsqueeze(1).expand(B, T, self.dim)
        fused = x + root_exp
        logits = self.head(fused)  # [B, T, V]

        loss = None
        if targets is not None and loss_mask is not None:
            # Compute masked cross-entropy
            V = logits.shape[-1]
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            mask_flat = loss_mask.view(B * T)
            if mask_flat.any():
                selected = mask_flat.nonzero(as_tuple=False).squeeze(-1)
                if selected.numel() == 0:
                    loss = torch.tensor(0.0, device=logits.device)
                else:
                    chunk = 8192
                    total = selected.numel()
                    total_loss = logits_flat.new_tensor(0.0)
                    for start in range(0, total, chunk):
                        idx = selected[start : start + chunk]
                        logits_chunk = logits_flat.index_select(0, idx)
                        targets_chunk = targets_flat.index_select(0, idx)
                        total_loss = total_loss + F.cross_entropy(
                            logits_chunk,
                            targets_chunk,
                            reduction="sum",
                        )
                    loss = total_loss / float(total)
            else:
                loss = torch.tensor(0.0, device=logits.device)

        return logits, loss
