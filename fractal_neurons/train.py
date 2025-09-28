
from __future__ import annotations

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fractal_neurons.model import FractalModel, FractalModelConfig
from fractal_neurons.data import (
    build_dataloader,
    SystemDataConfig,
    TextDirStreamConfig,
    HFDatasetConfig,
)
from fractal_neurons.tokenizer import TokenizerConfig, build_tokenizer
from fractal_neurons.utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    get_latest_checkpoint,
)
from fractal_neurons.utils.progress import (
    ProgressLogger,
    SpeedEstimator,
    human_bytes,
    human_format,
)
from fractal_neurons.utils.sched import CosineDecay, get_grad_norm
from fractal_neurons.distill import load_generation_model

from fractal_neurons.evo import run_ga_step
from fractal_neurons.model import EMAManager

@dataclass
class TrainConfig:
    run_name: str = "fractal"
    device: str = "cuda"
    batch_size: int = 16
    steps: int = 50000
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_accum: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    torch_compile: bool = False
    tf32: bool = True
    log_every: int = 50
    ckpt_every: int = 1000
    out_dir: str = "runs"
    resume: bool = False
    verbose: bool = False
    ga_enable: bool = False
    ga_every: int = 500
    ga_population: int = 8
    ga_sigma: float = 0.05
    compile_mode: str = "default"
    prefetch_gpu: bool = False
    prefetch_gpu_batches: int = 2
    adam_fused: bool = False
    warmup_steps: int = 0
    lr_min: float = 0.0
    cosine_cycle: int = 0
    grad_clip: Optional[float] = None
    use_ema: bool = False
    ema_decay: float = 0.999
    seed: int = 1337
    deterministic: bool = False
    qfp_enable: bool = False
    qfp_time_complex_real: float = -2.999999
    qfp_time_complex_imag: float = -0.002189
    entropy_anchor: Optional[float] = None
    min_loss_delta: Optional[float] = None
    curriculum_mode: Optional[str] = None
    droppath_rate: float = 0.0
    branch_dropout: float = 0.0
    teacher_topk: Optional[int] = None
    supervisor_model: Optional[str] = None

def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Fractal Neurons Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to the training config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of dataloader workers")
    parser.add_argument("--init_ckpt", type=str, default=None, help="Initialize from a specific checkpoint")
    parser.add_argument("--entropy_anchor", type=float, default=None, help="Entropy anchor for curriculum learning")
    parser.add_argument("--min_loss_delta", type=float, default=None, help="Minimum loss delta for curriculum learning")
    parser.add_argument("--curriculum_mode", type=str, default=None, help="Curriculum learning mode")
    parser.add_argument("--droppath_rate", type=float, default=None, help="Drop path rate")
    parser.add_argument("--branch_dropout", type=float, default=None, help="Branch dropout rate")
    parser.add_argument("--teacher-topk", type=int, default=None, help="Teacher top-k for distillation")
    parser.add_argument("--supervisor_model", type=str, default=None, help="Supervisor model for distillation")
    args = parser.parse_args()

    if os.environ.get("SUPERVISOR_MODEL"):
        args.supervisor_model = os.environ.get("SUPERVISOR_MODEL")

    cfg = load_config(args.config)

    train_cfg = cfg.get("train", {})
    tc = TrainConfig(
        run_name=train_cfg.get("run_name", "fractal"),
        device=train_cfg.get("device", "cuda"),
        batch_size=train_cfg.get("batch_size", 16),
        steps=train_cfg.get("steps", 50000),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        grad_accum=train_cfg.get("grad_accum", 1),
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        prefetch_factor=train_cfg.get("prefetch_factor", 2),
        persistent_workers=train_cfg.get("persistent_workers", True),
        torch_compile=train_cfg.get("torch_compile", False),
        tf32=train_cfg.get("tf32", True),
        log_every=train_cfg.get("log_every", 50),
        ckpt_every=train_cfg.get("ckpt_every", 1000),
        out_dir=train_cfg.get("out_dir", "runs"),
        resume=train_cfg.get("resume", False),
        verbose=train_cfg.get("verbose", False),
        ga_enable=train_cfg.get("ga_enable", False),
        ga_every=train_cfg.get("ga_every", 500),
        ga_population=train_cfg.get("ga_population", 8),
        ga_sigma=train_cfg.get("ga_sigma", 0.05),
        compile_mode=train_cfg.get("compile_mode", "default"),
        prefetch_gpu=train_cfg.get("prefetch_gpu", False),
        prefetch_gpu_batches=train_cfg.get("prefetch_gpu_batches", 2),
        adam_fused=train_cfg.get("adam_fused", False),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        lr_min=train_cfg.get("lr_min", 0.0),
        cosine_cycle=train_cfg.get("cosine_cycle", 0),
        grad_clip=train_cfg.get("grad_clip"),
        use_ema=train_cfg.get("use_ema", False),
        ema_decay=train_cfg.get("ema_decay", 0.999),
        seed=train_cfg.get("seed", 1337),
        deterministic=train_cfg.get("deterministic", False),
        qfp_enable=train_cfg.get("qfp_enable", False),
        qfp_time_complex_real=train_cfg.get("qfp_time_complex_real", -2.999999),
        qfp_time_complex_imag=train_cfg.get("qfp_time_complex_imag", -0.002189),
        entropy_anchor=train_cfg.get("entropy_anchor"),
        min_loss_delta=train_cfg.get("min_loss_delta"),
        curriculum_mode=train_cfg.get("curriculum_mode"),
        droppath_rate=train_cfg.get("droppath_rate", 0.0),
        branch_dropout=train_cfg.get("branch_dropout", 0.0),
        teacher_topk=train_cfg.get("teacher_topk"),
        supervisor_model=train_cfg.get("supervisor_model"),
    )

    if args.verbose:
        tc.verbose = True
    if args.resume:
        tc.resume = True
    if args.device:
        tc.device = args.device
    if args.num_workers is not None:
        tc.num_workers = args.num_workers
    if args.entropy_anchor is not None:
        tc.entropy_anchor = args.entropy_anchor
    if args.min_loss_delta is not None:
        tc.min_loss_delta = args.min_loss_delta
    if args.curriculum_mode is not None:
        tc.curriculum_mode = args.curriculum_mode
    if args.droppath_rate is not None:
        tc.droppath_rate = args.droppath_rate
    if args.branch_dropout is not None:
        tc.branch_dropout = args.branch_dropout
    if args.teacher_topk is not None:
        tc.teacher_topk = args.teacher_topk
    if args.supervisor_model is not None:
        tc.supervisor_model = args.supervisor_model

    init_cfg = dict(cfg.get("init", {}) or {})
    if args.init_ckpt:
        init_cfg["from_checkpoint"] = args.init_ckpt

    # Tokenizer
    tok_cfg = cfg.get("tokenizer", {})
    tcfg = TokenizerConfig(
        type=tok_cfg.get("type", "hf"),
        name_or_path=tok_cfg.get("name_or_path"),
        truncation_side=tok_cfg.get("truncation_side", "right"),
        pad_to_multiple_of=tok_cfg.get("pad_to_multiple_of"),
    )
    tokenizer = build_tokenizer(tcfg)

    # Model config (vocab from tokenizer)
    model_cfg = cfg.setdefault("model", {})
    fmc = FractalModelConfig(
        vocab_size=int(getattr(tokenizer, "vocab_size", model_cfg.get("vocab_size", 257))),
        dim=int(model_cfg.get("dim", 128)),
        depth=int(model_cfg.get("depth", 6)),
        fanout=int(model_cfg.get("fanout", 10)),
        use_fp16=bool(model_cfg.get("use_fp16", False)),
        droppath_rate=tc.droppath_rate,
        branch_dropout=tc.branch_dropout,
        interconnect=bool(model_cfg.get("interconnect", True)),
        interconnect_heads=int(model_cfg.get("interconnect_heads", 2)),
        interconnect_dropout=float(model_cfg.get("interconnect_dropout", 0.0)),
        num_experts=int(model_cfg.get("num_experts", 0)),
        expert_hidden=int(model_cfg.get("expert_hidden", 0)),
        expert_top_k=int(model_cfg.get("expert_top_k", 2)),
        router_temperature=float(model_cfg.get("router_temperature", 1.0)),
        moe_aux_lambda=float(model_cfg.get("moe_aux_lambda", 0.0)),
        capacity_factor=float(model_cfg.get("capacity_factor", 1.1)),
    )
    model_cfg["vocab_size"] = fmc.vocab_size
    model_cfg.setdefault("router_temperature", fmc.router_temperature)
    model_cfg.setdefault("moe_aux_lambda", fmc.moe_aux_lambda)
    model_cfg.setdefault("capacity_factor", fmc.capacity_factor)

    model = FractalModel(fmc)

    model.head[2].weight = model.embed.weight

    # Device selection and validation
    device = tc.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Set train.device=cpu or install CUDA.")

    if tc.torch_compile:
        try:
            model = torch.compile(model, mode=tc.compile_mode)
        except Exception as e:
            print(f"torch.compile unavailable: {e}")

    model = model.to(device)

    supervisor_model = None
    if tc.supervisor_model:
        print(f"Loading supervisor model: {tc.supervisor_model}")
        supervisor_model, _, _ = load_generation_model(tc.supervisor_model, device)
        supervisor_model.eval()

    # Dataloader
    data_cfg = cfg.get("data", {})
    if data_cfg.get("source") == "system":
        ds_cfg = SystemDataConfig(
            include_globs=data_cfg.get("include_globs", ["**/*"]),
            exclude_globs=data_cfg.get("exclude_globs", []),
            seq_len=data_cfg.get("seq_len", 1536),
            stride=data_cfg.get("stride", 1536),
            allow_all=data_cfg.get("allow_all", False),
            root=data_cfg.get("root", "/"),
            max_file_mb=data_cfg.get("max_file_mb", 32),
            max_index_items=data_cfg.get("max_index_items", -1),
            verbose=tc.verbose,
        )
    elif data_cfg.get("source") == "textdir":
        ds_cfg = TextDirStreamConfig(
            root=data_cfg.get("text_root"),
            include_globs=data_cfg.get("text_globs"),
            exclude_globs=data_cfg.get("exclude_globs"),
            use_cache=data_cfg.get("text_use_cache", True),
            refresh_cache=data_cfg.get("text_refresh_cache", False),
            tokenize_in_workers=data_cfg.get("text_tokenize_in_workers", True),
            shuffle_buffer=data_cfg.get("text_shuffle_buffer", 2048),
            reshuffle_each_epoch=data_cfg.get("text_reshuffle_each_epoch", True),
            seed=data_cfg.get("text_seed"),
        )
    else:
        ds_cfg = HFDatasetConfig(
            path=data_cfg.get("hf_path"),
            name=data_cfg.get("hf_name"),
            split=data_cfg.get("hf_split"),
            streaming=data_cfg.get("hf_streaming", True),
            text_field=data_cfg.get("hf_text_field"),
        )

    dataloader = build_dataloader(
        ds_cfg,
        tokenizer,
        tc.batch_size,
        data_cfg.get("seq_len", 1536),
        data_cfg.get("stride", 1536),
        tc.num_workers,
        tc.pin_memory,
        tc.prefetch_factor,
        tc.persistent_workers,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc.lr,
        weight_decay=tc.weight_decay,
        fused=tc.adam_fused and device == "cuda",
    )

    # Scheduler
    scheduler = CosineDecay(optimizer, tc.steps // tc.grad_accum, tc.lr_min, tc.warmup_steps)

    # EMA Manager
    ema = EMAManager(model, tc.ema_decay) if tc.use_ema else None

    # Checkpoint loading
    step = 0
    if tc.resume:
        latest_ckpt = get_latest_checkpoint(os.path.join(tc.out_dir, tc.run_name))
        if latest_ckpt:
            try:
                step, model, optimizer, scheduler, ema = load_checkpoint(
                    latest_ckpt, model, optimizer, scheduler, ema
                )
                print(f"[resume] loaded {latest_ckpt} at step {step}")
            except Exception as e:
                print(f"[init] optimizer load failed: {e}")

    # Progress logger
    progress = ProgressLogger(tc.log_every, tc.steps, start_step=step)

    # Training loop
    for step in range(step, tc.steps):
        # Fetch batch
        batch = next(dataloader)
        tokens = batch["tokens"].to(device)
        targets = batch["targets"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        # Forward and backward passes
        student_stream = torch.cuda.Stream()
        teacher_stream = torch.cuda.Stream()

        with torch.cuda.stream(student_stream):
            with torch.amp.autocast(device_type="cuda", enabled=fmc.use_fp16 and device == "cuda"):
                logits, loss = model(tokens, loss_mask=loss_mask, targets=targets)

        if supervisor_model is not None:
            with torch.cuda.stream(teacher_stream):
                with torch.no_grad():
                    teacher_logits, _ = supervisor_model(tokens)

        torch.cuda.synchronize()

        # Distillation loss
        if supervisor_model is not None:
            if tc.teacher_topk:
                t_probs, t_idx = torch.topk(teacher_logits.softmax(-1), k=tc.teacher_topk, dim=-1)
                teacher_probs_sparse = torch.zeros_like(teacher_logits).scatter_(-1, t_idx, t_probs)
                teacher_log_probs = torch.log(teacher_probs_sparse + 1e-9)
            else:
                teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)

            kl_loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                teacher_log_probs,
                log_target=True,
                reduction="batchmean",
            )
            loss = loss + kl_loss

        # Aux loss for MoE
        aux_loss = getattr(model, "last_aux_loss", None)
        if aux_loss is not None and model.moe_aux_lambda > 0:
            loss = loss + model.moe_aux_lambda * aux_loss

        # Optimizer step
        loss = loss / tc.grad_accum
        loss.backward()

        if (step + 1) % tc.grad_accum == 0:
            if tc.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if ema:
                ema.update()

        # Logging
        progress.log(step, loss.item() * tc.grad_accum, scheduler.get_lr())

        # Checkpointing
        if (step + 1) % tc.ckpt_every == 0:
            save_checkpoint(
                os.path.join(tc.out_dir, tc.run_name, f"ckpt_{step+1}.pt"),
                step + 1,
                model,
                optimizer,
                scheduler,
                ema,
            )


if __name__ == "__main__":
    main()
